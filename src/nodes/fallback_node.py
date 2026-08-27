"""Fallback Node for handling unanswerable, unsafe, or out-of-scope queries."""

from langchain_core.messages import AIMessage, HumanMessage
from src.state import GraphState
from src.utils.logger import logger


def fallback_node(state: GraphState) -> GraphState:
    """LangGraph node that generates safe fallback responses."""
    question = state.get("question", "")
    chat_history = list(state.get("chat_history", []))
    is_safe = state.get("is_safe", True)
    guardrail_reason = state.get("guardrail_reason", "")
    is_relevant = state.get("is_relevant", False)

    logger.info("Executing Fallback Node...")

    if not is_safe:
        generation = "Maaf, permintaan anda telah disekat kerana melanggar garis panduan keselamatan."
        if guardrail_reason:
            generation += f" (Sebab: {guardrail_reason})"
    elif not is_relevant:
        generation = (
            "Maaf, soalan anda nampaknya di luar skop FAQ Tonton, atau saya tidak menjumpai maklumat berkaitan.\n\n"
            "Sila layari portal sokongan pelanggan rasmi Tonton untuk bantuan lanjut."
        )
    else:
        generation = "Maaf, saya tidak dapat memproses permintaan anda pada masa ini."

    updated_history = list(chat_history) + [
        HumanMessage(content=question),
        AIMessage(content=generation),
    ]

    return {
        **state,
        "generation": generation,
        "chat_history": updated_history,
        "sources": [],
    }

