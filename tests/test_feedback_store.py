"""Unit tests for feedback persistence and autonomous self-learning variant ingestion."""

import json
from pathlib import Path
import tempfile
import pytest

from config.settings import settings
from src.feedback.feedback_store import (
    get_feedback_analytics,
    get_recent_feedbacks,
    get_unresolved_queries,
    learn_variant_into_qdrant,
    record_user_feedback,
)
from src.vectorstore.qdrant_client import QdrantManager


@pytest.fixture
def temp_feedback_file(tmp_path):
    """Overrides feedback_storage_path to use a temporary file during testing."""
    test_file = str(tmp_path / "test_feedback.jsonl")
    orig_path = settings.feedback_storage_path
    settings.feedback_storage_path = test_file
    yield test_file
    settings.feedback_storage_path = orig_path


def test_record_feedback_upvote(temp_feedback_file):
    """Verifies that an upvote feedback record is serialized and saved correctly."""
    record = record_user_feedback(
        session_id="test-session-123",
        message_index=1,
        user_query="Bagaimana nak batalkan akaun?",
        assistant_response="Sila layari menu tetapan profil anda.",
        rating="up",
        sources=[{"id": "FAQ-03", "category": "Langganan", "question": "Bagaimana membatalkan langganan?"}],
        elapsed_seconds=1.2,
    )

    assert record["rating"] == "up"
    assert record["session_id"] == "test-session-123"
    assert Path(temp_feedback_file).exists()

    records = get_recent_feedbacks()
    assert len(records) == 1
    assert records[0]["user_query"] == "Bagaimana nak batalkan akaun?"


def test_record_feedback_downvote(temp_feedback_file):
    """Verifies that a downvote feedback record is tracked in unresolved queries."""
    record_user_feedback(
        session_id="test-session-456",
        message_index=1,
        user_query="Berapa kos pelan 2 tahun?",
        assistant_response="Maaf, tiada maklumat.",
        rating="down",
        sources=[],
        elapsed_seconds=0.8,
    )

    analytics = get_feedback_analytics()
    assert analytics["total_feedbacks"] == 1
    assert analytics["downvotes"] == 1
    assert analytics["upvotes"] == 0
    assert analytics["satisfaction_rate"] == 0.0

    unresolved = get_unresolved_queries()
    assert len(unresolved) == 1
    assert unresolved[0]["user_query"] == "Berapa kos pelan 2 tahun?"


def test_feedback_analytics_calculation(temp_feedback_file):
    """Verifies accurate analytics calculations with multiple feedback records."""
    record_user_feedback("s1", 1, "q1", "a1", "up")
    record_user_feedback("s1", 3, "q2", "a2", "up")
    record_user_feedback("s1", 5, "q3", "a3", "down")

    analytics = get_feedback_analytics()
    assert analytics["total_feedbacks"] == 3
    assert analytics["upvotes"] == 2
    assert analytics["downvotes"] == 1
    assert analytics["satisfaction_rate"] == 66.7


def test_upsert_learned_variant_in_qdrant():
    """Verifies that QdrantManager properly accepts and stores learned phrasing variants."""
    qdrant = QdrantManager(in_memory=True, collection_name="test_self_learn")
    qdrant.ensure_collection(recreate=True)

    dummy_vector = [0.05] * 768
    metadata = {"id": "FAQ-01", "category": "Langganan", "question": "Kenapa tak boleh tonton?"}

    success = qdrant.upsert_learned_variant(
        query_text="Kenapa video tak jalan lepas bayar?",
        embedding=dummy_vector,
        metadata=metadata,
    )
    assert success is True
    assert qdrant.get_collection_count() == 1

    # Repeating the exact same variant should overwrite deterministically without increasing count
    qdrant.upsert_learned_variant(
        query_text="Kenapa video tak jalan lepas bayar?",
        embedding=dummy_vector,
        metadata=metadata,
    )
    assert qdrant.get_collection_count() == 1
