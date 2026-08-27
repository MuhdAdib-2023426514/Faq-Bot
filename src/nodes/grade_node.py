"""Grade Node for evaluating document relevance and 3-tier confidence routing."""

from config.settings import settings
from src.state import GraphState
from src.utils.logger import logger


def grade_node(state: GraphState) -> GraphState:
    """LangGraph node that evaluates retrieval confidence and determines 3-tier routing intent.

    Tiers:
    1. High Confidence (score >= confidence_high_threshold, e.g. >= 0.80):
       -> Routes to direct grounded answer generation (generate_node).
    2. Needs Clarification (confidence_low_threshold <= score < confidence_high_threshold, e.g. 0.50 - 0.80):
       -> Routes to interactive disambiguation / FAQ suggestions (clarification_node).
    3. Out of Scope (score < confidence_low_threshold, e.g. < 0.50 or empty docs):
       -> Routes to polite out-of-scope refusal (fallback_node).
    """
    documents = state.get("documents", [])
    relevance_score = state.get("relevance_score", 0.0)

    high_threshold = settings.confidence_high_threshold
    low_threshold = settings.confidence_low_threshold

    if not documents or relevance_score < low_threshold:
        intent = "out_of_scope"
        is_relevant = False
        logger.info(
            f"Grade evaluated -> OUT OF SCOPE (Score {relevance_score:.4f} < {low_threshold:.4f})"
        )
    elif relevance_score >= high_threshold:
        intent = "high_confidence"
        is_relevant = True
        logger.info(
            f"Grade evaluated -> HIGH CONFIDENCE (Score {relevance_score:.4f} >= {high_threshold:.4f})"
        )
    else:
        intent = "needs_clarification"
        is_relevant = True
        logger.info(
            f"Grade evaluated -> NEEDS CLARIFICATION ({low_threshold:.4f} <= Score {relevance_score:.4f} < {high_threshold:.4f})"
        )

    return {
        **state,
        "is_relevant": is_relevant,
        "routing_intent": intent,
    }
