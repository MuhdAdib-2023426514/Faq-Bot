"""Unit tests for query expansion, conversational query rewriting, and weighted RRF."""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.retrieval import (
    BM25Index,
    generate_multi_queries,
    reciprocal_rank_fusion,
    rewrite_conversational_query,
)


def test_rewrite_conversational_query_no_history():
    """Test that query rewriting immediately returns normalized query when history is empty."""
    raw_query = "xleh tgk video"
    # Should normalize without invoking LLM
    result = rewrite_conversational_query(raw_query, chat_history=[])
    assert result == "tidak boleh tonton video"


@patch("src.retrieval.query_expansion._get_llm")
def test_rewrite_conversational_query_with_history(mock_get_llm):
    """Test that query rewriting uses LLM to resolve pronouns when history is provided."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Bagaimana cara membuat pembayaran langganan TontonUp?")
    mock_get_llm.return_value = mock_llm

    history = [
        HumanMessage(content="Berapa harga langganan TontonUp?"),
        AIMessage(content="Harga langganan TontonUp ialah RM9.90 sebulan."),
    ]
    follow_up = "Macam mana nak bayar tu?"

    result = rewrite_conversational_query(follow_up, chat_history=history)

    assert result == "Bagaimana cara membuat pembayaran langganan TontonUp?"
    mock_llm.invoke.assert_called_once()


@patch("src.retrieval.query_expansion._get_llm")
def test_generate_multi_queries_success(mock_get_llm):
    """Test generating multi-query reformulations from JSON LLM output."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='["cara bayar tonton", "kaedah pembayaran akaun", "langganan tonton up bayar"]'
    )
    mock_get_llm.return_value = mock_llm

    queries = generate_multi_queries("Bagaimana membuat pembayaran?", num_variants=3)

    assert len(queries) == 3
    assert "cara bayar tonton" in queries
    assert "kaedah pembayaran akaun" in queries


def test_weighted_reciprocal_rank_fusion():
    """Test that weighted RRF correctly prioritizes higher-weighted signals."""
    doc1 = Document(page_content="Content A", metadata={"id": "faq_001"})
    doc2 = Document(page_content="Content B", metadata={"id": "faq_002"})

    # Signal 1 (e.g. Vector) ranks doc1 first
    list1 = [(doc1, 0.9), (doc2, 0.5)]
    # Signal 2 (e.g. Multi-Query) ranks doc2 first
    list2 = [(doc2, 0.8), (doc1, 0.6)]

    # Give Signal 1 higher weight (1.0 vs 0.2)
    fused = reciprocal_rank_fusion(
        result_lists=[list1, list2],
        weights=[1.0, 0.2],
        k=60,
    )

    assert len(fused) == 2
    # doc1 should win because Signal 1 has much higher weight
    assert fused[0][0].metadata["id"] == "faq_001"


def test_bm25_with_malay_morphology():
    """Test that BM25 indexing matches Malay root words with prefixes/suffixes."""
    doc1 = Document(page_content="Panduan pembayaran langganan bulanan", metadata={"id": "faq_001"})
    doc2 = Document(page_content="Maklumat akaun keselamatan kata laluan", metadata={"id": "faq_002"})
    doc3 = Document(page_content="Ralat siaran video tergendala", metadata={"id": "faq_003"})

    index = BM25Index([doc1, doc2, doc3])
    # Search with root word 'bayar' should match 'pembayaran'
    results = index.search("bayar", top_k=2)

    assert len(results) > 0
    assert results[0][0].metadata["id"] == "faq_001"

