"""Vector store management module for Qdrant local persistent database."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import settings
from src.utils.logger import logger


def get_embedding_function(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    task_type: Optional[str] = None,
) -> GoogleGenerativeAIEmbeddings:
    """Instantiates GoogleGenerativeAIEmbeddings with configured model, API key, and task type."""
    key = api_key or settings.effective_api_key
    model = model_name or settings.embedding_model

    # Ensure model identifier follows expected format
    if not model.startswith("models/") and not model.startswith("gemini-"):
        model = f"models/{model}"

    kwargs: Dict[str, Any] = {
        "model": model,
        "google_api_key": key,
    }
    if task_type:
        kwargs["task_type"] = task_type

    return GoogleGenerativeAIEmbeddings(**kwargs)


class QdrantManager:
    """Manages Qdrant vector database connection, collections, and vector search."""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_dim: int = 768,
        in_memory: bool = False,
    ):
        """Initializes the Qdrant client and verifies collection availability."""
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.embedding_dim = embedding_dim
        self.in_memory = in_memory

        if self.in_memory:
            logger.info("Initializing in-memory Qdrant client for testing")
            self.client = QdrantClient(location=":memory:")
        else:
            db_path = Path(storage_path or settings.qdrant_storage_path)
            db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing persistent local Qdrant client at {db_path.resolve()}")
            self.client = QdrantClient(path=str(db_path))

    def ensure_collection(self, recreate: bool = False) -> None:
        """Ensures that the target collection exists with Cosine similarity metric."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            logger.info(f"Recreating collection '{self.collection_name}'...")
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={self.embedding_dim}, metric=Cosine)")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.embedding_dim,
                    distance=rest.Distance.COSINE,
                ),
            )
        else:
            logger.info(f"Qdrant collection '{self.collection_name}' is ready")

    def upsert_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
    ) -> int:
        """Upserts embedded documents into Qdrant collection."""
        if len(documents) != len(embeddings):
            raise ValueError(f"Document count ({len(documents)}) does not match embedding count ({len(embeddings)})")

        self.ensure_collection()

        points = []
        for idx, (doc, vec) in enumerate(zip(documents, embeddings)):
            point_id = idx + 1
            payload = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Successfully upserted {len(points)} FAQ documents into '{self.collection_name}'")
        return len(points)

    def similarity_search_with_score(
        self,
        query_vector: List[float],
        top_k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """Performs cosine similarity search against Qdrant collection."""
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results: List[Tuple[Document, float]] = []
        for hit in search_result.points:
            payload = hit.payload or {}
            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            score = float(hit.score)

            doc = Document(page_content=page_content, metadata=metadata)
            results.append((doc, score))

        return results

    def get_collection_count(self) -> int:
        """Returns the total number of points stored in the collection."""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def scroll_all_documents(self) -> List[Document]:
        """Fetches all documents from the collection for in-memory indexing.

        Scrolls through all points in the collection and reconstructs
        LangChain Document objects from stored payloads. Used by the BM25
        index to build a keyword search corpus.

        Returns:
            List of all Document objects stored in the collection.
        """
        documents: List[Document] = []
        offset = None

        try:
            while True:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in results:
                    payload = point.payload or {}
                    page_content = payload.get("page_content", "")
                    metadata = payload.get("metadata", {})
                    documents.append(
                        Document(page_content=page_content, metadata=metadata)
                    )

                if next_offset is None:
                    break
                offset = next_offset

            logger.debug(
                f"Scrolled {len(documents)} documents from '{self.collection_name}'"
            )
        except Exception as e:
            logger.error(f"Failed to scroll documents: {e}")

        return documents

    def upsert_learned_variant(
        self,
        query_text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> bool:
        """Upserts a single learned phrasing variant from positive user feedback into Qdrant.

        Generates a deterministic integer point ID so repeated identical queries do not
        create duplicate points.
        """
        self.ensure_collection()
        faq_id = metadata.get("id", "generic")
        clean_text = query_text.strip().lower()
        
        # 63-bit deterministic positive integer ID
        point_id = abs(hash(f"learned_{faq_id}_{clean_text}")) % (2**63 - 1)

        payload = {
            "page_content": query_text,
            "metadata": {
                **metadata,
                "is_learned_variant": True,
                "learned_from_query": query_text,
            },
        }

        point = rest.PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload,
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        logger.info(
            f"Self-Learning: Successfully indexed dynamic phrasing variant '{query_text}' for FAQ #{faq_id} into '{self.collection_name}' (Point ID: {point_id})"
        )
        return True

