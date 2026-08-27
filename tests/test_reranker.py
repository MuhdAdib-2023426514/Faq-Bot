"""Unit tests for the Jina Cross-Encoder Reranker module."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.reranker.jina_client import JinaReranker


def test_jina_reranker_format_document_text():
    """Test rich document formatting for cross-encoder context."""
    reranker = JinaReranker(api_key="mock_key")
    doc = Document(
        page_content="Langgan melalui web.",
        metadata={
            "category": "Langganan",
            "question": "Bagaimana langgan?",
            "answer": "Langgan melalui web rasmi.",
        },
    )
    formatted = reranker.format_document_text(doc)

    assert "Kategori: Langganan" in formatted
    assert "Soalan: Bagaimana langgan?" in formatted
    assert "Jawapan: Langgan melalui web rasmi." in formatted


def test_jina_reranker_fallback_when_no_api_key():
    """Test graceful fallback to vector similarity ordering when API key is empty."""
    reranker = JinaReranker(api_key="")
    doc1 = Document(page_content="Doc 1", metadata={"id": "faq_001"})
    doc2 = Document(page_content="Doc 2", metadata={"id": "faq_002"})
    candidates = [(doc1, 0.85), (doc2, 0.70)]

    results = reranker.rerank("test query", candidates, top_n=2)
    assert len(results) == 2
    assert results[0][0] == doc1
    assert results[0][1] == 0.85


@patch("requests.post")
def test_jina_reranker_successful_api_call(mock_post):
    """Test successful reranking with Jina API response."""
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "model": "jina-reranker-v2-base-multilingual",
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.40},
            ],
        },
    )

    reranker = JinaReranker(api_key="jina_test_key")
    doc1 = Document(page_content="Doc 1", metadata={"id": "faq_001"})
    doc2 = Document(page_content="Doc 2", metadata={"id": "faq_002"})
    candidates = [(doc1, 0.60), (doc2, 0.50)]

    results = reranker.rerank("test query", candidates, top_n=2)

    assert len(results) == 2
    # Document 2 should be ranked first due to index 1 having score 0.95
    assert results[0][0] == doc2
    assert results[0][1] == 0.95
    assert results[1][0] == doc1
    assert results[1][1] == 0.40


@patch("requests.post")
def test_jina_reranker_http_error_fallback(mock_post):
    """Test graceful fallback on HTTP error status."""
    mock_post.return_value = MagicMock(
        status_code=500,
        text="Internal Server Error",
    )

    reranker = JinaReranker(api_key="jina_test_key")
    doc1 = Document(page_content="Doc 1", metadata={"id": "faq_001"})
    candidates = [(doc1, 0.75)]

    results = reranker.rerank("test query", candidates, top_n=1)
    assert len(results) == 1
    assert results[0][0] == doc1
    assert results[0][1] == 0.75


def test_prune_tail_results_adaptive_margin():
    """Test adaptive score-margin pruning drops low-confidence tail results."""
    from src.reranker.jina_client import prune_tail_results

    doc1 = Document(page_content="Doc 1", metadata={"id": "faq_001"})
    doc2 = Document(page_content="Doc 2", metadata={"id": "faq_002"})
    doc3 = Document(page_content="Doc 3", metadata={"id": "faq_003"})

    # Top hit is 0.90, doc2 is 0.80 (within 0.25 margin), doc3 is 0.40 (outside 0.25 margin)
    candidates = [(doc1, 0.90), (doc2, 0.80), (doc3, 0.40)]

    pruned = prune_tail_results(candidates, score_margin=0.25, min_score=0.20)
    assert len(pruned) == 2
    assert pruned[0][0] == doc1
    assert pruned[1][0] == doc2


def test_prune_tail_results_below_min_score():
    """Test pruning returns empty list if top hit is below minimum threshold."""
    from src.reranker.jina_client import prune_tail_results

    doc1 = Document(page_content="Doc 1", metadata={"id": "faq_001"})
    candidates = [(doc1, 0.05)]

    pruned = prune_tail_results(candidates, score_margin=0.25, min_score=0.20)
    assert len(pruned) == 0

