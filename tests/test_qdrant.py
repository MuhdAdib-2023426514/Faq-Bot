"""Unit tests for the Qdrant vector store manager."""

import pytest
from langchain_core.documents import Document
from src.vectorstore.qdrant_client import QdrantManager


def test_qdrant_in_memory_initialization():
    """Test that QdrantManager can initialize in-memory without errors."""
    qdrant = QdrantManager(in_memory=True, collection_name="test_collection", embedding_dim=3)
    qdrant.ensure_collection()
    
    assert qdrant.get_collection_count() == 0


def test_qdrant_upsert_and_search():
    """Test that documents can be upserted and retrieved."""
    qdrant = QdrantManager(in_memory=True, collection_name="test_search", embedding_dim=2)
    qdrant.ensure_collection()
    
    # Mock documents and embeddings
    docs = [
        Document(page_content="Apple is a fruit", metadata={"id": "1"}),
        Document(page_content="Dog is an animal", metadata={"id": "2"}),
    ]
    # Simple 2D vectors
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    
    # Test Upsert
    count = qdrant.upsert_documents(docs, embeddings)
    assert count == 2
    assert qdrant.get_collection_count() == 2
    
    # Test Search (Search for vector close to Apple)
    results = qdrant.similarity_search_with_score([0.9, 0.1], top_k=1)
    
    assert len(results) == 1
    retrieved_doc, score = results[0]
    
    assert retrieved_doc.metadata["id"] == "1"
    assert "Apple" in retrieved_doc.page_content
    assert score > 0.5  # Cosine similarity should be high
