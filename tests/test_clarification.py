"""Unit tests for the Clarification node."""

from langchain_core.documents import Document
from src.nodes.clarification_node import clarification_node
from src.state import GraphState


def test_clarification_node_with_documents():
    """Test clarification_node formats suggested questions from retrieved documents."""
    doc1 = Document(
        page_content="Panduan bayar",
        metadata={"id": "faq_001", "category": "Langganan", "question": "Bagaimana melanggan TontonUp?"},
    )
    doc2 = Document(
        page_content="Panduan batal",
        metadata={"id": "faq_002", "category": "Langganan", "question": "Bagaimana membatalkan langganan?"},
    )

    state: GraphState = {"documents": [doc1, doc2]}
    result = clarification_node(state)

    assert "generation" in result
    assert "Bagaimana melanggan TontonUp?" in result["generation"]
    assert "Bagaimana membatalkan langganan?" in result["generation"]
    assert len(result["sources"]) == 2
    assert result["sources"][0]["id"] == "faq_001"


def test_clarification_node_empty_documents():
    """Test clarification_node provides a polite generic prompt if no questions in docs."""
    state: GraphState = {"documents": []}
    result = clarification_node(state)

    assert "generation" in result
    assert "terperinci" in result["generation"]
    assert len(result["sources"]) == 0
