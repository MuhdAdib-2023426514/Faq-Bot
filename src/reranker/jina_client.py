"""Jina AI Cross-Encoder Reranker Client."""

from typing import List, Optional, Tuple
import requests
from langchain_core.documents import Document

from config.settings import settings
from src.utils.logger import logger


class JinaReranker:
    """Client for Jina AI Multilingual Cross-Encoder Reranker API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 10.0,
    ):
        """Initializes the Jina Reranker client.

        Args:
            api_key: Optional override for Jina API key.
            model_name: Optional model override (defaults to jina-reranker-v2-base-multilingual).
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or settings.effective_jina_api_key
        self.model_name = model_name or settings.jina_rerank_model
        self.api_url = "https://api.jina.ai/v1/rerank"
        self.timeout = timeout

    def format_document_text(self, doc: Document) -> str:
        """Formats a Document with its category, question, and answer for rich reranking context.

        Args:
            doc: LangChain Document to format.

        Returns:
            Structured text string containing full FAQ context.
        """
        category = doc.metadata.get("category", "")
        question = doc.metadata.get("question", "")
        answer = doc.metadata.get("answer", doc.page_content)

        parts = []
        if category:
            parts.append(f"Kategori: {category}")
        if question:
            parts.append(f"Soalan: {question}")
        if answer:
            # Include up to first 600 characters of answer to keep context concise
            parts.append(f"Jawapan: {answer[:600].strip()}")

        return "\n".join(parts) if parts else doc.page_content

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_n: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Reranks candidate documents against a query using Jina Cross-Encoder API.

        Falls back gracefully to the original vector candidate ranking if the API
        key is not configured or if the API call encounters any network/HTTP error.

        Args:
            query: The user query string.
            candidates: List of (Document, original_score) tuples from vector retrieval.
            top_n: Number of top documents to return. Defaults to settings.top_k_results.

        Returns:
            List of (Document, rerank_score) tuples sorted by rerank_score descending.
        """
        limit = top_n or settings.top_k_results
        if not candidates:
            return []

        if not self.api_key:
            logger.info("Jina API key not configured; using vector similarity ranking.")
            return candidates[:limit]

        # Prepare formatted candidate document texts
        doc_texts = [self.format_document_text(doc) for doc, _ in candidates]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": doc_texts,
            "top_n": min(limit, len(candidates)),
        }

        try:
            logger.info(
                f"Calling Jina Reranker ({self.model_name}) for {len(candidates)} candidates..."
            )
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.warning(
                    f"Jina Reranker API returned HTTP {response.status_code}: {response.text[:200]}. "
                    "Falling back to vector similarity ranking."
                )
                return candidates[:limit]

            data = response.json()
            results = data.get("results", [])

            if not results:
                logger.warning("Jina Reranker returned empty results; falling back.")
                return candidates[:limit]

            reranked: List[Tuple[Document, float]] = []
            for item in results:
                idx = item.get("index")
                score = float(item.get("relevance_score", 0.0))
                if idx is not None and 0 <= idx < len(candidates):
                    orig_doc, _ = candidates[idx]
                    reranked.append((orig_doc, score))

            logger.info(
                f"Jina Reranker successfully reranked {len(reranked)} documents. "
                f"Top score: {reranked[0][1]:.4f}" if reranked else "No reranked documents"
            )
            return reranked

        except Exception as exc:
            logger.warning(
                f"Jina Reranker invocation failed ({exc}); falling back to vector similarity ranking."
            )
            return candidates[:limit]


def prune_tail_results(
    results: List[Tuple[Document, float]],
    score_margin: float = 0.25,
    min_score: float = 0.10,
) -> List[Tuple[Document, float]]:
    """Prunes low-confidence tail results to avoid context dilution in generation.

    Retains the top-ranked document and any subsequent documents whose score
    falls within `score_margin` of the top document's score and meets `min_score`.

    Args:
        results: List of (Document, score) tuples sorted by score descending.
        score_margin: Maximum score gap from top hit to retain a document.
        min_score: Absolute minimum score floor.

    Returns:
        Pruned list of (Document, score) tuples.
    """
    if not results:
        return []

    top_score = results[0][1]
    if top_score < min_score:
        return []

    pruned: List[Tuple[Document, float]] = []
    for doc, score in results:
        if score >= min_score and (top_score - score) <= score_margin:
            pruned.append((doc, score))

    if len(pruned) < len(results):
        logger.info(
            f"Adaptive pruning filtered {len(results)} candidates down to {len(pruned)} "
            f"(top score: {top_score:.4f}, margin: {score_margin:.2f})"
        )

    return pruned

