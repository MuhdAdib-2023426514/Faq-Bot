"""Ingestion package for FAQ parsing and indexing."""

from src.ingestion.parser import (
    FAQItem,
    FAQParser,
    infer_category,
    batch_infer_categories,
    batch_generate_question_variants,
)

__all__ = [
    "FAQItem",
    "FAQParser",
    "infer_category",
    "batch_infer_categories",
    "batch_generate_question_variants",
]
