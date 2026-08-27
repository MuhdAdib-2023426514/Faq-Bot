from pathlib import Path
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Centralized configuration schema for the FAQ RAG system."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # API Keys
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API Key for model inference and embeddings",
    )
    jina_api: str = Field(
        default="",
        description="Jina AI API Key for cross-encoder reranker inference",
    )
    jina_api_key: str = Field(
        default="",
        description="Alternative alias for Jina AI API Key",
    )

    # Model Configuration
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini LLM model identifier",
    )
    embedding_model: str = Field(
        default="gemini-embedding-2",
        description="Gemini Embedding model identifier",
    )
    jina_rerank_model: str = Field(
        default="jina-reranker-v2-base-multilingual",
        description="Jina AI Reranker model identifier",
    )

    # Qdrant Vector Store Configuration
    qdrant_storage_path: str = Field(
        default="./data/qdrant_db",
        description="Local persistent directory for Qdrant database",
    )
    qdrant_collection_name: str = Field(
        default="faq_collection",
        description="Name of the Qdrant vector collection",
    )

    # Retrieval & Pipeline Settings
    top_k_results: int = Field(
        default=3,
        description="Number of top documents to retrieve",
    )
    retrieval_candidate_count: int = Field(
        default=10,
        description="Number of initial candidate documents to fetch before reranking",
    )
    enable_reranker: bool = Field(
        default=True,
        description="Whether to use Jina AI cross-encoder reranker",
    )
    similarity_threshold: float = Field(
        default=0.35,
        description="Minimum similarity/relevance score threshold for relevant documents",
    )
    enable_hybrid_search: bool = Field(
        default=True,
        description="Whether to combine BM25 keyword search with vector search via RRF",
    )
    enable_query_rewriting: bool = Field(
        default=True,
        description="Whether to rewrite conversational follow-up queries using chat history",
    )
    enable_adaptive_pruning: bool = Field(
        default=True,
        description="Whether to prune low-confidence tail documents after reranking",
    )
    reranker_score_margin: float = Field(
        default=0.25,
        description="Maximum score difference from top hit allowed when pruning tail documents",
    )
    enable_multi_query: bool = Field(
        default=True,
        description="Whether to generate multiple query reformulations at query time",
    )
    multi_query_count: int = Field(
        default=3,
        description="Number of query variants to generate for multi-query retrieval",
    )
    enable_guardrails: bool = Field(
        default=True,
        description="Whether to activate prompt injection and safety guardrail checks",
    )
    confidence_high_threshold: float = Field(
        default=0.80,
        description="Threshold above which retrieval is considered high-confidence (direct answer)",
    )
    confidence_low_threshold: float = Field(
        default=0.50,
        description="Threshold below which retrieval is considered out-of-scope (refusal)",
    )
    max_query_length: int = Field(
        default=800,
        description="Maximum allowed character length for user queries to prevent DoS/ReDoS",
    )

    # Knowledge Base Path
    faq_file_path: str = Field(
        default="FAQ.md",
        description="Path to source FAQ markdown file",
    )
    feedback_storage_path: str = Field(
        default="./data/feedback_store.jsonl",
        description="Path to persistent JSONL log of user feedback for continuous self-learning",
    )
    enable_self_learning: bool = Field(
        default=True,
        description="Whether to automatically ingest positive feedback queries as new search vector variants into Qdrant",
    )

    # FAQ Classification Categories
    faq_categories: List[str] = Field(
        default=[
            "Langganan & Pembayaran",
            "Akaun & Keselamatan",
            "TV Tuisyen & Pendidikan",
            "Akses Antarabangsa",
            "Teknikal & Ralat",
            "Umum",
        ],
        description="Valid FAQ category labels for LLM-based classification",
    )
    question_variants_count: int = Field(
        default=4,
        description="Number of synthetic question variants to generate per FAQ item during ingestion",
    )

    @property
    def effective_api_key(self) -> str:
        """Returns the active Gemini API key from settings, environment, or Streamlit secrets."""
        key = self.gemini_api_key or ""
        if not key:
            import os
            key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                    key = str(st.secrets["GEMINI_API_KEY"])
            except Exception:
                pass
        return key.strip().strip('"').strip("'")

    @property
    def effective_jina_api_key(self) -> str:
        """Returns the active Jina AI API key from settings, environment, or Streamlit secrets."""
        key = self.jina_api or self.jina_api_key or ""
        if not key:
            import os
            key = os.environ.get("JINA_API", "") or os.environ.get("JINA_API_KEY", "")
        if not key:
            try:
                import streamlit as st
                if hasattr(st, "secrets"):
                    key = str(st.secrets.get("JINA_API", "") or st.secrets.get("JINA_API_KEY", ""))
            except Exception:
                pass
        return key.strip().strip('"').strip("'")


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton instance of Settings."""
    return Settings()


# Global settings singleton instance
settings = get_settings()
