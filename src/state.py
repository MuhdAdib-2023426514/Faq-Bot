"""LangGraph State definition for the RAG chatbot workflow."""

from typing import Any, Dict, List, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class GraphState(TypedDict, total=False):
    """Represents the complete state of the LangGraph RAG pipeline."""

    question: str
    chat_history: Sequence[BaseMessage]
    is_safe: bool
    guardrail_reason: Optional[str]
    documents: List[Document]
    relevance_score: float
    is_relevant: bool
    routing_intent: Optional[str]
    generation: str
    sources: List[Dict[str, Any]]

