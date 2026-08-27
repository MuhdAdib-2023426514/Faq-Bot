"""Ingestion indexer script that parses FAQ.md, computes embeddings, and indexes into Qdrant."""

import sys
from pathlib import Path
from typing import Optional

# Ensure project root directory is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.ingestion.parser import FAQParser
from src.vectorstore.qdrant_client import QdrantManager, get_embedding_function
from src.utils.logger import logger


def run_indexing(
    faq_file: Optional[str] = None,
    recreate_collection: bool = False,
    in_memory: bool = False,
) -> int:
    """Executes the full document ingestion pipeline.

    Parses the FAQ markdown file, generates embeddings via the shared
    embedding function, dynamically infers the vector dimension, and
    upserts all documents into Qdrant.

    Args:
        faq_file: Optional override path to the FAQ markdown file.
        recreate_collection: If True, drops and recreates the Qdrant
            collection before upserting. Defaults to False for safety.
        in_memory: If True, uses an ephemeral in-memory Qdrant instance
            (useful for testing).

    Returns:
        The number of documents successfully indexed.

    Raises:
        FileNotFoundError: If the FAQ file does not exist.
        Exception: If embedding generation or Qdrant upsert fails.
    """
    file_path = faq_file or settings.faq_file_path
    logger.info(f"Starting FAQ ingestion from: {file_path}")

    # 1. Parse FAQ
    parser = FAQParser(file_path=file_path)
    faq_items = parser.parse_file()
    documents = parser.to_langchain_documents(faq_items)

    if not documents:
        logger.warning(f"No FAQ items found in {file_path}")
        return 0

    logger.info(f"Extracted {len(documents)} structured documents from FAQ.")

    # 2. Compute Embeddings with asymmetric document task type
    embeddings_model = get_embedding_function(task_type="retrieval_document")
    logger.info(f"Generating document embeddings using model: {settings.embedding_model} (task_type=retrieval_document)")

    texts = [doc.page_content for doc in documents]

    try:
        embeddings = embeddings_model.embed_documents(texts)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

    # Dynamically infer dimension from actual embeddings
    embedding_dim = len(embeddings[0])
    logger.info(f"Generated {len(embeddings)} embeddings (dimension={embedding_dim}).")

    # 3. Upsert into Qdrant
    qdrant = QdrantManager(
        storage_path=settings.qdrant_storage_path,
        collection_name=settings.qdrant_collection_name,
        embedding_dim=embedding_dim,
        in_memory=in_memory,
    )
    qdrant.ensure_collection(recreate=recreate_collection)
    count = qdrant.upsert_documents(documents, embeddings)

    logger.info(f"Successfully indexed {count} FAQ items into Qdrant collection '{settings.qdrant_collection_name}'")
    return count


if __name__ == "__main__":
    try:
        total = run_indexing(recreate_collection=True)
        print(f"\n✅ Indexing completed successfully! Total indexed documents: {total}")
        sys.exit(0)
    except Exception as exc:
        print(f"\n❌ Indexing failed with error: {exc}", file=sys.stderr)
        sys.exit(1)
