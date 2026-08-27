from typing import Any, Dict, List
from langchain_core.messages import AIMessage, HumanMessage
from src.state import GraphState
from src.utils.logger import logger


def clarification_node(state: GraphState) -> GraphState:
    """LangGraph node that generates a polite clarification request with suggested FAQ questions.

    Invoked when retrieval confidence is in the medium tier (0.50 - 0.80), indicating
    the user's query is partially related but ambiguous. Instead of guessing or abruptly
    failing, this node presents the closest matching FAQ titles for disambiguation.

    Args:
        state: Current GraphState containing retrieved documents.

    Returns:
        GraphState updated with clarification response and source attributions.
    """
    logger.info("Executing Clarification Node for ambiguous / partial match...")
    question = state.get("question", "")
    documents = state.get("documents", [])
    chat_history = list(state.get("chat_history", []))

    suggestions: List[str] = []
    sources: List[Dict[str, Any]] = []

    for doc in documents[:3]:
        doc_q = doc.metadata.get("question", "")
        if doc_q and doc_q not in suggestions:
            suggestions.append(doc_q)
            sources.append(
                {
                    "id": doc.metadata.get("id", "N/A"),
                    "category": doc.metadata.get("category", "Umum"),
                    "question": doc_q,
                }
            )

    if suggestions:
        numbered_options = "\n".join(f"{idx + 1}. {q}" for idx, q in enumerate(suggestions))
        generation = (
            "Saya tidak pasti maksud penuh soalan anda, tetapi berikut adalah beberapa topik FAQ berkaitan yang mungkin dapat membantu:\n\n"
            f"{numbered_options}\n\n"
            "Sila tanya soalan yang lebih spesifik atau pilih salah satu topik di atas untuk maklumat lanjut."
        )
    else:
        generation = (
            "Soalan anda agak umum atau kurang jelas. "
            "Bolehkah anda jelaskan dengan lebih terperinci mengenai langganan, akaun, atau masalah teknikal yang anda hadapi?"
        )

    updated_history = list(chat_history) + [
        HumanMessage(content=question),
        AIMessage(content=generation),
    ]

    return {
        **state,
        "generation": generation,
        "chat_history": updated_history,
        "sources": sources,
    }

