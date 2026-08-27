"""Unit tests for Malaysian chat slang and colloquial query normalizer."""

import pytest
from src.utils.normalizer import normalize_query


def test_normalize_empty_query():
    """Test that empty or whitespace query returns empty string."""
    assert normalize_query("") == ""
    assert normalize_query("   ") == ""


def test_normalize_malay_slang_negations():
    """Test normalization of Malay slang negations."""
    assert normalize_query("xleh tgk tonton") == "tidak boleh tonton tonton"
    assert normalize_query("takleh log in") == "tidak boleh log in"
    assert normalize_query("xde video keluar") == "tiada video keluar"


def test_normalize_malay_interrogatives():
    """Test normalization of informal question words."""
    assert normalize_query("camne nk cancel sub") == "bagaimana nak batal langganan"
    assert normalize_query("cane bayor subscription") == "bagaimana bayar langganan"
    assert normalize_query("bape harga sebulan") == "berapa harga sebulan"


def test_normalize_account_terms():
    """Test normalization of account and credentials keywords."""
    assert normalize_query("lupa pw acc") == "lupa kata laluan akaun"
    assert normalize_query("tukar password hp") == "tukar kata laluan telefon"


def test_normalize_preserves_standard_text():
    """Test that standard formal text is preserved cleanly."""
    query = "Bagaimanakah cara untuk melanggan Tonton?"
    # case is lowercased and standardized
    normalized = normalize_query(query)
    assert "bagaimana" in normalized
    assert "melanggan" in normalized
    assert "tonton" in normalized
