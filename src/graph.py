"""LangGraph workflow definition for the FAQ RAG pipeline with 3-tier confidence routing and InMemorySaver checkpointer."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.nodes.clarification_node import clarification_node
from src.nodes.fallback_node import fallback_node
from src.nodes.generate_node import generate_node
from src.nodes.grade_node import grade_node
from src.nodes.guardrail_node import guardrail_node
from src.nodes.retrieve_node import retrieve_node
from src.state import GraphState
from src.utils.logger import logger

# Global in-memory checkpointer instance for short-term session threads
checkpointer = InMemorySaver()


def route_after_guardrail(state: GraphState) -> str:
    """Conditional edge router after guardrail check."""
    if not state.get("is_safe", True):
        logger.info("Routing -> fallback (Input rejected by guardrails)")
        return "fallback"
    logger.info("Routing -> retrieve (Input passed guardrails)")
    return "retrieve"


def route_after_grade(state: GraphState) -> str:
    """Conditional edge router after document grading with 3-tier confidence branching."""
    intent = state.get("routing_intent", "out_of_scope")

    if intent == "high_confidence":
        logger.info("Routing -> generate (High confidence match >= 0.80)")
        return "generate"
    elif intent == "needs_clarification":
        logger.info("Routing -> clarification (Medium confidence match 0.50 - 0.80)")
        return "clarification"
    else:
        logger.info("Routing -> fallback (Out of scope / score < 0.50)")
        return "fallback"


def build_graph(saver: InMemorySaver = checkpointer) -> StateGraph:
    """Builds and compiles the LangGraph StateGraph workflow with checkpointer support."""
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("fallback", fallback_node)

    # Set Entry Point
    workflow.set_entry_point("guardrail")

    # Add Conditional Edges
    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "retrieve": "retrieve",
            "fallback": "fallback",
        },
    )

    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "clarification": "clarification",
            "fallback": "fallback",
        },
    )

    # Final terminal edges
    workflow.add_edge("generate", END)
    workflow.add_edge("clarification", END)
    workflow.add_edge("fallback", END)

    logger.info("LangGraph workflow compiled successfully with InMemorySaver checkpointer.")
    return workflow.compile(checkpointer=saver)


# Global graph instance
rag_app = build_graph()
