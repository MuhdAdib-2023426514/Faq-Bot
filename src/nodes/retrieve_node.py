"""Retrieval Node with Conversational Query Rewriting, Multi-Query Expansion, BM25, and Reranking.

Combines high-accuracy search signals:
1. Conversational Query Rewriting to resolve multi-turn coreferences
2. Dense vector search using Google Gemini Embeddings (primary signal)
3. Multi-Query reformulations for keyword and vocabulary expansion
4. BM25 keyword search with Malay affix awareness
5. Weighted Reciprocal Rank Fusion (RRF)
6. Jina AI Multilingual Cross-Encoder Reranking with Adaptive Score-Margin Pruning
"""

import threading
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import settings
from src.reranker.jina_client import JinaReranker, prune_tail_results
from src.retrieval import (
    BM25Index,
    generate_multi_queries,
    reciprocal_rank_fusion,
    rewrite_conversational_query,
)
from src.state import GraphState
from src.utils.logger import logger
from src.utils.normalizer import normalize_query
from src.vectorstore.qdrant_client import QdrantManager, get_embedding_function


# ---------------------------------------------------------------------------
# Stage-level performance instrumentation
# ---------------------------------------------------------------------------

class _Timer:
    """Context manager for logging stage-level latency."""

    __slots__ = ("name", "_start")

    def __init__(self, name: str) -> None:
        self.name = name
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        logger.debug(f"[{self.name}] completed in {elapsed_ms:.1f}ms")


# ---------------------------------------------------------------------------
# Cached singletons (avoid per-call object re-creation)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_retrieval_embedder():
    """Returns a cached embedding function configured for retrieval queries."""
    return get_embedding_function(task_type="retrieval_query")


@lru_cache(maxsize=1)
def _get_qdrant() -> QdrantManager:
    """Returns a cached QdrantManager singleton."""
    return QdrantManager(
        storage_path=settings.qdrant_storage_path,
        collection_name=settings.qdrant_collection_name,
    )


# ---------------------------------------------------------------------------
# BM25 index cache with count-based invalidation
# ---------------------------------------------------------------------------

_bm25_lock = threading.Lock()
_bm25_cache: Dict[str, object] = {"key": None, "index": None}


def _get_bm25_index(qdrant: QdrantManager) -> Optional[BM25Index]:
    """Returns a cached BM25 index, rebuilding only when the corpus size changes.

    Uses a lightweight count-based cache key. The index is rebuilt when the
    point count in Qdrant changes (e.g. after self-learning adds new variants).

    Args:
        qdrant: QdrantManager instance to scroll documents from.

    Returns:
        A BM25Index instance, or None if the collection is empty.
    """
    doc_count = qdrant.get_collection_count()
    cache_key = f"{qdrant.collection_name}:{doc_count}"

    with _bm25_lock:
        if _bm25_cache["key"] == cache_key and _bm25_cache["index"] is not None:
            return _bm25_cache["index"]

    # Build outside the lock to avoid blocking concurrent queries during scroll
    all_docs = qdrant.scroll_all_documents()
    if not all_docs:
        return None

    index = BM25Index(all_docs)
    logger.info(f"Rebuilt BM25 index ({doc_count} docs, collection: {qdrant.collection_name})")

    with _bm25_lock:
        _bm25_cache["key"] = cache_key
        _bm25_cache["index"] = index

    return index


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _deduplicate_by_faq_id(
    results: List[Tuple[Document, float]],
) -> List[Tuple[Document, float]]:
    """Deduplicates retrieved documents by FAQ ID, keeping the highest-scoring hit.

    With multi-question indexing, multiple variant vectors for the same FAQ
    item may appear in retrieval results. This function ensures each unique
    FAQ is represented at most once, retaining the variant with the highest
    similarity score.

    Args:
        results: List of (Document, score) tuples from similarity search.

    Returns:
        Deduplicated list of (Document, score) tuples, sorted by score descending.
    """
    seen: Dict[str, Tuple[Document, float]] = {}
    for doc, score in results:
        faq_id = doc.metadata.get("id", "")
        if faq_id not in seen or score > seen[faq_id][1]:
            seen[faq_id] = (doc, score)

    deduplicated = sorted(seen.values(), key=lambda x: x[1], reverse=True)

    if len(deduplicated) < len(results):
        logger.info(
            f"Deduplicated {len(results)} results to {len(deduplicated)} unique FAQ items."
        )

    return deduplicated


def _vector_search(
    qdrant: QdrantManager,
    embedder,
    query_text: str,
    top_k: int,
) -> List[Tuple[Document, float]]:
    """Embeds a query string and runs vector similarity search.

    Args:
        qdrant: QdrantManager instance for vector search.
        embedder: Embedding function with retrieval_query task type.
        query_text: Text to embed and search with.
        top_k: Maximum number of results to return.

    Returns:
        List of (Document, score) tuples from vector similarity search.
    """
    query_vector = embedder.embed_query(query_text)
    return qdrant.similarity_search_with_score(
        query_vector=query_vector,
        top_k=top_k,
        score_threshold=0.0,
    )


# ---------------------------------------------------------------------------
# Main retrieval node
# ---------------------------------------------------------------------------

def retrieve_node(state: GraphState) -> GraphState:
    """LangGraph node that retrieves relevant documents using a high-accuracy multi-signal pipeline.

    Pipeline:
    1. Conversational Query Rewriter: Resolves multi-turn references from chat history.
    2. Query Normalizer: Expands Malaysian chat slang and colloquial abbreviations.
    3. Dense Vector Search: Computes primary semantic similarity against Qdrant collection.
    4. Multi-Query Expansion: Generates alternative keyword and phrasing variants.
    5. BM25 Search: Performs lexical matching with Malay morphological awareness.
    6. Weighted RRF: Fuses candidate lists with priority weighting.
    7. FAQ ID Deduplication: Consolidates multiple hits per FAQ entry.
    8. Jina Cross-Encoder Reranking: Computes query-document cross-attention scores.
    9. Adaptive Score-Margin Pruning: Filters low-confidence tail results.
    10. Calibrated Relevance Scoring: Passes authoritative score to grade_node.
    """
    raw_question = state.get("question", "")
    chat_history = state.get("chat_history", [])
    logger.info(f"Initiating retrieval for question: '{raw_question}' (history turns: {len(chat_history)})")

    _EMPTY_RESULT: GraphState = {"documents": [], "relevance_score": 0.0}

    try:
        # 1. Conversational Query Rewriting (resolves context from previous turns)
        with _Timer("query_rewriting"):
            rewritten_query = rewrite_conversational_query(raw_question, chat_history)

        # 2. Slang and abbreviation normalization
        search_query = normalize_query(rewritten_query) or rewritten_query or raw_question
        if search_query.lower() != raw_question.lower():
            logger.info(f"Processed search query: '{raw_question}' -> '{search_query}'")

        embedder = _get_retrieval_embedder()
        qdrant = _get_qdrant()

        fetch_limit = max(
            settings.retrieval_candidate_count,
            settings.top_k_results * settings.fetch_limit_multiplier,
        )

        # Collect all retrieval signals with associated weights
        result_lists: List[List[Tuple[Document, float]]] = []
        weights: List[float] = []
        signal_names: List[str] = []

        # --- Signal 1: Primary Dense Vector Search ---
        with _Timer("vector_search"):
            primary_results = _vector_search(qdrant, embedder, search_query, fetch_limit)
        best_vector_score = primary_results[0][1] if primary_results else 0.0

        if primary_results:
            result_lists.append(primary_results)
            weights.append(settings.rrf_weight_primary_vector)
            signal_names.append(f"vector({len(primary_results)})")

        # --- Signal 2: Multi-Query Expansion Search ---
        # Optimization: Skip expensive LLM multi-query generation if primary vector
        # search is already high confidence (above configurable threshold)
        if settings.enable_multi_query and best_vector_score < settings.multi_query_skip_threshold:
            with _Timer("multi_query_expansion"):
                query_variants = generate_multi_queries(search_query)
                variants_with_results = 0
                for variant in query_variants:
                    if variant.lower() != search_query.lower():
                        variant_results = _vector_search(qdrant, embedder, variant, fetch_limit)
                        if variant_results:
                            result_lists.append(variant_results)
                            weights.append(settings.rrf_weight_multi_query)
                            variants_with_results += 1
                            if variant_results[0][1] > best_vector_score:
                                best_vector_score = variant_results[0][1]

            if variants_with_results > 0:
                signal_names.append(f"multi-query({variants_with_results}/{len(query_variants)} effective)")

        # --- Signal 3: BM25 Lexical Keyword Search ---
        if settings.enable_hybrid_search:
            with _Timer("bm25_search"):
                bm25_index = _get_bm25_index(qdrant)
                if bm25_index is not None:
                    bm25_results = bm25_index.search(query=search_query, top_k=fetch_limit)
                    if bm25_results:
                        result_lists.append(bm25_results)
                        weights.append(settings.rrf_weight_bm25)
                        signal_names.append(f"bm25({len(bm25_results)})")

        # --- Fuse all signals via Weighted RRF ---
        if len(result_lists) > 1:
            logger.info(f"Fusing {len(result_lists)} signals via Weighted RRF: {' + '.join(signal_names)}")
            with _Timer("rrf_fusion"):
                candidates = reciprocal_rank_fusion(
                    result_lists=result_lists,
                    weights=weights,
                    top_n=fetch_limit,
                )
        elif result_lists:
            candidates = result_lists[0]
        else:
            logger.info("No candidate documents retrieved from any signal.")
            return _EMPTY_RESULT

        # 7. Deduplicate candidates by unique FAQ ID
        deduplicated = _deduplicate_by_faq_id(candidates)

        # 8. Rerank candidates with Jina AI Cross-Encoder
        if settings.enable_reranker and deduplicated:
            rerank_limit = max(
                settings.top_k_results * settings.rerank_limit_multiplier,
                settings.rerank_limit_floor,
            )
            with _Timer("jina_rerank"):
                reranker = JinaReranker()
                top_results = reranker.rerank(
                    query=search_query,
                    candidates=deduplicated[:rerank_limit],
                    top_n=rerank_limit,
                )
            # Use cross-encoder score as authoritative score if available
            has_reranker_scores = bool(top_results and settings.effective_jina_api_key)
        else:
            top_results = deduplicated[: settings.top_k_results]
            has_reranker_scores = False

        if not top_results:
            return _EMPTY_RESULT

        # 9. Dynamic Score-Margin Pruning (removes noisy tail documents)
        if settings.enable_adaptive_pruning and has_reranker_scores:
            pruned_results = prune_tail_results(
                results=top_results,
                score_margin=settings.reranker_score_margin,
                min_score=settings.similarity_threshold * 0.5,
            )
            if pruned_results:
                top_results = pruned_results

        # Constrain to configured top_k_results
        top_results = top_results[: settings.top_k_results]

        documents = [doc for doc, _ in top_results]
        top_ranked_score = top_results[0][1]

        # 10. Calibrated Score Alignment: Reranker score if available, otherwise best vector similarity
        authoritative_score = top_ranked_score if has_reranker_scores else best_vector_score

        logger.info(
            f"Successfully retrieved {len(documents)} FAQ documents. "
            f"Authoritative relevance score: {authoritative_score:.4f} "
            f"(Cross-encoder: {top_ranked_score:.4f}, Best vector: {best_vector_score:.4f})"
        )
        return {
            "documents": documents,
            "relevance_score": authoritative_score,
        }

    except (ConnectionError, TimeoutError, OSError) as e:
        logger.error(f"Retrieval failed due to external service error: {e}")
        return _EMPTY_RESULT

    except Exception as e:
        logger.exception(f"Unexpected retrieval error: {e}")
        return _EMPTY_RESULT
