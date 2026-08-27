"""BM25 keyword search, Reciprocal Rank Fusion (RRF), and query expansion for hybrid retrieval.

Provides an in-memory BM25 index built from the Qdrant document corpus with
Malay morphological tokenization, weighted Reciprocal Rank Fusion (RRF), and
conversational query rewriting.
"""

import re
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Plus

from src.utils.logger import logger
from src.retrieval.query_expansion import (
    generate_multi_queries,
    rewrite_conversational_query,
)

def _stem_malay_word(word: str) -> List[str]:
    """Generates candidate root stems for Malay words using morpho-phonemic rules."""
    candidates = set()
    w = word.lower()

    # Suffix stripping: -kanlah, -kan, -lah, -kah, -nya, -an, -i
    suffixes = ("kanlah", "kan", "lah", "kah", "nya", "an", "i")
    stemmed_suffix = w
    for suf in suffixes:
        if stemmed_suffix.endswith(suf) and len(stemmed_suffix) > len(suf) + 2:
            stemmed_suffix = stemmed_suffix[: -len(suf)]
            candidates.add(stemmed_suffix)
            break

    # Apply prefix morpho-phonemic rules to both base and suffix-stripped word
    for base in [w, stemmed_suffix]:
        # pemb- / memb- (e.g. pembayaran -> bayar, pembatalan -> batal, membeli -> beli)
        if base.startswith("pemb") and len(base) > 5:
            candidates.add("b" + base[4:])
            candidates.add(base[4:])
        elif base.startswith("memb") and len(base) > 5:
            candidates.add("b" + base[4:])
            candidates.add(base[4:])
        # peny- / meny- (e.g. penyelesaian -> selesai)
        elif (base.startswith("peny") or base.startswith("meny")) and len(base) > 5:
            candidates.add("s" + base[4:])
        # memper- / per- (e.g. memperbaharui -> baharu)
        elif base.startswith("memper") and len(base) > 7:
            candidates.add(base[6:])
        elif base.startswith("per") and len(base) > 4:
            candidates.add(base[3:])
        # peng- / meng- (e.g. penggunaan -> guna)
        elif (base.startswith("peng") or base.startswith("meng")) and len(base) > 5:
            candidates.add(base[4:])
        # pen- / men- (e.g. pendaftaran -> daftar, menonton -> tonton)
        elif (base.startswith("pen") or base.startswith("men")) and len(base) > 4:
            candidates.add(base[3:])
            candidates.add("t" + base[3:])
        # ber- / ter- (e.g. berlangganan -> langgan, tergendala -> gendala)
        elif (base.startswith("ber") or base.startswith("ter")) and len(base) > 4:
            candidates.add(base[3:])
        # di- / ke- / se- / me-
        elif (base.startswith("di") or base.startswith("ke") or base.startswith("se") or base.startswith("me")) and len(base) > 3:
            candidates.add(base[2:])

    return [c for c in candidates if len(c) >= 3 and c != word]


def _tokenize_malay(text: str) -> List[str]:
    """Malay-optimized tokenizer with punctuation stripping and morphological expansion.

    Args:
        text: Raw text to tokenize.

    Returns:
        List of lowercase token strings including original and candidate root stems.
    """
    text = text.lower()
    # Remove punctuation except hyphens
    text = re.sub(r"[^\w\s\-]", " ", text)
    tokens = text.split()

    result_tokens: List[str] = []
    for t in tokens:
        if len(t) > 1 or t in {"x"}:
            result_tokens.append(t)
            for stem in _stem_malay_word(t):
                result_tokens.append(stem)

    return result_tokens


class BM25Index:
    """In-memory BM25 index over a list of LangChain Documents.

    Designed for FAQ corpora (< 2000 documents) where building the index
    in-memory is near-instant (< 10ms).
    """

    def __init__(self, documents: List[Document]):
        """Builds a BM25 index from the given documents.

        Args:
            documents: List of LangChain Documents whose ``page_content``
                will be tokenized and indexed.
        """
        self.documents = documents
        corpus = [_tokenize_malay(doc.page_content) for doc in documents]
        self.bm25 = BM25Plus(corpus)
        logger.debug(f"Built BM25Plus index over {len(documents)} documents.")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """Scores all documents against the query and returns top-k results.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of (Document, bm25_score) tuples sorted by score descending.
        """
        tokens = _tokenize_malay(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Pair documents with scores and sort descending
        scored = [(doc, float(score)) for doc, score in zip(self.documents, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Filter out zero-score documents and take top-k
        return [(doc, score) for doc, score in scored[:top_k] if score > 0.0]


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[Document, float]]],
    weights: Optional[List[float]] = None,
    k: int = 60,
    top_n: Optional[int] = None,
) -> List[Tuple[Document, float]]:
    """Fuses multiple ranked result lists using Weighted Reciprocal Rank Fusion (RRF).

    RRF assigns each document a score of ``weight * (1 / (k + rank))`` for each result
    list it appears in, then sums across lists. This produces a calibrated ranking
    that balances contributions from dense embeddings and keyword matching.

    Args:
        result_lists: List of ranked result lists, each containing
            (Document, score) tuples sorted by score descending.
        weights: Optional weighting multipliers for each result list.
            If None, all lists are weighted equally (1.0).
        k: RRF constant (default 60). Higher values smooth the influence
            of top-ranked documents relative to lower-ranked ones.
        top_n: Maximum number of fused results to return.

    Returns:
        List of (Document, rrf_score) tuples sorted by fused score descending.
    """
    if not result_lists:
        return []

    # Assign default equal weights if not specified or mismatched
    if not weights or len(weights) != len(result_lists):
        weights = [1.0] * len(result_lists)

    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for result_list, weight in zip(result_lists, weights):
        for rank, (doc, _score) in enumerate(result_list, start=1):
            faq_id = doc.metadata.get("id", "")
            doc_key = f"{faq_id}::{hash(doc.page_content)}"

            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (weight * (1.0 / (k + rank)))
            if doc_key not in doc_map:
                doc_map[doc_key] = doc

    # Sort by fused RRF score
    fused = [
        (doc_map[key], score)
        for key, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    if top_n is not None:
        fused = fused[:top_n]

    return fused


__all__ = [
    "BM25Index",
    "reciprocal_rank_fusion",
    "rewrite_conversational_query",
    "generate_multi_queries",
]
