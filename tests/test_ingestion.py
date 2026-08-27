"""Unit tests for the FAQ parsing and ingestion logic."""

import pytest
from unittest.mock import patch
from src.ingestion.parser import (
    FAQParser,
    infer_category,
    batch_infer_categories,
    batch_generate_question_variants,
)


def test_infer_category_fallback():
    """Test that infer_category returns a valid category even on LLM failure."""
    with patch(
        "src.ingestion.parser.batch_infer_categories",
        return_value=["Umum"],
    ):
        result = infer_category("test question", "test answer")
        assert isinstance(result, str)
        assert result == "Umum"


def test_batch_infer_categories_empty_input():
    """Test that batch_infer_categories handles an empty list gracefully."""
    result = batch_infer_categories([])
    assert result == []


def test_batch_generate_variants_empty_input():
    """Test that batch_generate_question_variants handles an empty list gracefully."""
    result = batch_generate_question_variants([])
    assert result == []


@patch("src.ingestion.parser._get_variants_count", return_value=0)
def test_batch_generate_variants_zero_count(mock_count):
    """Test that requesting 0 variants returns empty lists."""
    items = [{"question": "Test?", "answer": "Yes."}]
    result = batch_generate_question_variants(items, num_variants=0)
    assert result == [[]]


def test_faq_parser_cleaning():
    """Test that markdown characters are cleaned properly."""
    parser = FAQParser()
    dirty_text = r"\> This is a \*\*bold\*\* test \- with characters\_."
    clean_text = parser.clean_markdown_text(dirty_text)

    assert ">" in clean_text
    assert "-" in clean_text
    assert "_" in clean_text


def test_clean_markdown_extra_escapes():
    """Test additional escape sequences added in the refactor."""
    parser = FAQParser()
    text_with_brackets = r"See \[this link\] and \*emphasis\*"
    cleaned = parser.clean_markdown_text(text_with_brackets)

    assert "[this link]" in cleaned
    assert "*emphasis*" in cleaned


def test_clean_markdown_collapses_newlines():
    """Test that excessive blank lines are collapsed to one."""
    parser = FAQParser()
    text = "Line 1\n\n\n\n\nLine 2"
    cleaned = parser.clean_markdown_text(text)

    assert cleaned == "Line 1\n\nLine 2"


@patch("src.ingestion.parser.batch_infer_categories")
def test_parse_text_structure(mock_batch):
    """Test the parser can extract Q&A pairs from raw markdown."""
    mock_batch.return_value = ["Akaun & Keselamatan", "Langganan & Pembayaran"]

    raw_markdown = (
        "**Question:** How do I reset my password?\n"
        "**Answer:** Go to settings and click reset.\n\n"
        "Question: Can I cancel?\n"
        "Answer: Yes, you can cancel anytime."
    )

    parser = FAQParser()
    items = parser.parse_text(raw_markdown)

    assert len(items) == 2
    assert items[0].question == "How do I reset my password?"
    assert items[1].question == "Can I cancel?"

    # Verify batch was called once with both items
    mock_batch.assert_called_once()
    call_args = mock_batch.call_args[0][0]
    assert len(call_args) == 2
    assert call_args[0]["question"] == "How do I reset my password?"


@patch("src.ingestion.parser.batch_infer_categories")
def test_parse_text_zero_results_warning(mock_batch, caplog):
    """Test that a warning is logged when no FAQ items are found."""
    parser = FAQParser()
    items = parser.parse_text("This document has no FAQ headers whatsoever.")

    assert items == []
    mock_batch.assert_not_called()


@patch("src.ingestion.parser.batch_generate_question_variants")
@patch("src.ingestion.parser.batch_infer_categories")
def test_to_langchain_documents_with_variants(mock_categories, mock_variants):
    """Test conversion to LangChain Documents with multi-vector embedding and variants."""
    mock_categories.return_value = ["Umum"]
    mock_variants.return_value = [
        ["Variant Q1", "Variant Q2", "Variant Q3", "Variant Q4"]
    ]

    parser = FAQParser()
    raw_markdown = "**Question:** Test Q?\n**Answer:** Test A with details."
    items = parser.parse_text(raw_markdown)

    docs = parser.to_langchain_documents(items, enrich_variants=True)

    # 1 original + 1 answer + 4 variants = 6 documents
    assert len(docs) == 6

    # Original question document — pure question text (no category prefix)
    original = docs[0]
    assert original.page_content == "Test Q?"
    assert original.metadata["variant_type"] == "original"
    assert original.metadata["answer"] == "Test A with details."
    assert original.metadata["question"] == "Test Q?"
    assert original.metadata["category"] == "Umum"
    assert original.metadata["source"] == "FAQ.md"
    assert original.metadata["id"] == "faq_001"

    # Answer document — combined Q+A for semantic embedding
    answer_doc = docs[1]
    assert "Test Q?" in answer_doc.page_content
    assert "Test A with details." in answer_doc.page_content
    assert answer_doc.metadata["variant_type"] == "answer"
    assert answer_doc.metadata["id"] == "faq_001"

    # Variant documents — pure variant text (no category prefix)
    for i, variant_doc in enumerate(docs[2:], start=1):
        assert variant_doc.page_content == f"Variant Q{i}"
        assert variant_doc.metadata["variant_type"] == "synthetic"
        assert variant_doc.metadata["answer"] == "Test A with details."
        assert variant_doc.metadata["id"] == "faq_001"  # Same FAQ ID


@patch("src.ingestion.parser.batch_infer_categories")
def test_to_langchain_documents_no_variants(mock_categories):
    """Test conversion with enrich_variants=False still includes answer vector."""
    mock_categories.return_value = ["Umum"]

    parser = FAQParser()
    raw_markdown = "**Question:** Test Q?\n**Answer:** Test A."
    items = parser.parse_text(raw_markdown)

    docs = parser.to_langchain_documents(items, enrich_variants=False)

    # 1 original + 1 answer = 2 documents (no synthetic variants)
    assert len(docs) == 2
    assert "Test Q?" in docs[0].page_content
    assert docs[0].metadata["variant_type"] == "original"
    assert "Test A." in docs[1].page_content
    assert docs[1].metadata["variant_type"] == "answer"
    assert docs[1].metadata["answer"] == "Test A."


@patch("src.ingestion.parser.batch_infer_categories")
def test_to_langchain_documents_answer_in_metadata(mock_categories):
    """Test that full answer is stored in metadata for all document types."""
    mock_categories.return_value = ["Teknikal & Ralat"]

    parser = FAQParser()
    raw_markdown = "**Question:** Why error?\n**Answer:** Because of a bug in the system."
    items = parser.parse_text(raw_markdown)

    docs = parser.to_langchain_documents(items, enrich_variants=False)

    # 1 original + 1 answer = 2 documents
    assert len(docs) == 2

    # Question document: page_content is question with category
    q_doc = docs[0]
    assert "Why error?" in q_doc.page_content
    assert "bug" not in q_doc.page_content
    assert q_doc.metadata["answer"] == "Because of a bug in the system."

    # Answer document: page_content is answer text with category
    a_doc = docs[1]
    assert "bug" in a_doc.page_content
    assert a_doc.metadata["variant_type"] == "answer"
    # Full answer also in metadata (same as question doc)
    assert a_doc.metadata["answer"] == "Because of a bug in the system."
    assert a_doc.metadata["id"] == q_doc.metadata["id"]
