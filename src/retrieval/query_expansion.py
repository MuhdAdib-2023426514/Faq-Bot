"""Query processing and expansion techniques for high-accuracy retrieval.

Provides conversational query rewriting to resolve multi-turn coreferences and
multi-query reformulations to broaden keyword and semantic coverage across the FAQ database.
"""

import json
import re
from typing import List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings
from src.utils.logger import logger
from src.utils.normalizer import normalize_query


def _get_llm() -> ChatGoogleGenerativeAI:
    """Returns a low-temperature Gemini instance for deterministic query processing."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.effective_api_key,
        temperature=0.0,
    )


def format_chat_history_for_prompt(chat_history: Sequence[BaseMessage], max_messages: int = 4) -> str:
    """Formats recent chat history messages into a clean text transcript.

    Args:
        chat_history: Sequence of BaseMessage objects (HumanMessage, AIMessage).
        max_messages: Maximum number of recent messages to include.

    Returns:
        Formatted string representing the dialogue history.
    """
    if not chat_history:
        return ""

    recent_messages = chat_history[-max_messages:]
    formatted_lines: List[str] = []

    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            formatted_lines.append(f"Pengguna: {msg.content}")
        elif isinstance(msg, AIMessage):
            # Truncate assistant messages to keep prompt focused
            content = str(msg.content)[:250].replace("\n", " ")
            formatted_lines.append(f"Pembantu: {content}")
        elif hasattr(msg, "content"):
            formatted_lines.append(f"Mesej: {msg.content}")

    return "\n".join(formatted_lines)


def rewrite_conversational_query(
    question: str,
    chat_history: Optional[Sequence[BaseMessage]] = None,
) -> str:
    """Resolves conversational pronouns and context into a standalone FAQ search query.

    If no chat history is present, returns the normalized question immediately with
    zero LLM latency. When history is present, instructs the LLM to rewrite ambiguous
    follow-up questions (e.g., 'Macam mana nak bayar tu?') into self-contained FAQ queries
    (e.g., 'Bagaimana cara membuat pembayaran langganan TontonUp?').

    Args:
        question: The user's latest input question.
        chat_history: Previous conversation turns.

    Returns:
        A self-contained query string in Bahasa Melayu.
    """
    if not question or not question.strip():
        return ""

    # If no history or query rewriting is disabled, return normalized question immediately
    if not chat_history or not settings.enable_query_rewriting:
        return normalize_query(question) or question.strip()

    history_text = format_chat_history_for_prompt(chat_history)
    if not history_text.strip():
        return normalize_query(question) or question.strip()

    prompt = (
        "Anda ialah pakar perumus carian FAQ untuk platform penstriman Tonton.\n\n"
        "Tugas anda: Berdasarkan sejarah perbualan dan soalan terkini pengguna, tulis semula "
        "soalan tersebut menjadi satu soalan carian FAQ yang LENGKAP dan BERDIRI SENDIRI (standalone) "
        "dalam Bahasa Melayu.\n\n"
        "Garis panduan:\n"
        "- Rujuk entiti atau topik daripada perbualan terdahulu jika soalan mengandungi kata ganti nama "
        "(cth: 'tu', 'ia', 'ini', 'langganan tersebut', 'masalah ini').\n"
        "- Jika soalan sudah lengkap dan jelas tanpa memerlukan konteks sejarah, kekalkan maksud asalnya.\n"
        "- JANGAN jawab soalan tersebut. Tulis HANYA soalan carian yang dirumus semula.\n"
        "- Jangan sertakan tanda petik, pengenalan, atau penjelasan.\n\n"
        f"Sejarah Perbualan:\n{history_text}\n\n"
        f"Soalan Terkini Pengguna: \"{question}\"\n\n"
        "Soalan Carian Standalone:"
    )

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        rewritten = response.content.strip().strip('"').strip("'")

        if rewritten:
            logger.info(
                f"Conversational query rewritten: '{question}' -> '{rewritten}' "
                f"(History: {len(chat_history)} messages)"
            )
            return rewritten

        return normalize_query(question) or question.strip()

    except Exception as e:
        logger.warning(f"Conversational query rewriting failed ({e}). Using normalized query.")
        return normalize_query(question) or question.strip()


def generate_multi_queries(
    question: str,
    num_variants: Optional[int] = None,
) -> List[str]:
    """Generates alternative query formulations for the same user intent.

    Produces N rephrased versions of the user's question using different
    vocabulary, formality levels, and phrasing structures. Each variant
    is embedded and searched independently, then results are fused via RRF.

    Args:
        question: The user's original/rewritten question.
        num_variants: Number of query variants to generate.
            Defaults to ``multi_query_count`` from settings.

    Returns:
        List of rephrased query strings. Returns empty list on failure or if disabled.
    """
    if not settings.enable_multi_query:
        return []

    n = num_variants if num_variants is not None else settings.multi_query_count

    if n <= 0 or not question or not question.strip():
        return []

    prompt = (
        "Anda ialah pakar pencarian FAQ untuk platform Tonton.\n\n"
        f"Soalan carian: \"{question}\"\n\n"
        f"Tulis {n} versi carian berbeza untuk soalan di atas bagi memaksimumkan ketepatan carian FAQ. "
        "Setiap versi mesti:\n"
        "- Mengekalkan maksud asal\n"
        "- Menggunakan sinonim dan perkataan berbeza (cth: langgan vs bayar, ralat vs masalah)\n"
        "- Campuran Bahasa Melayu formal, bahasa percakapan, dan carian kata kunci pendek (keyword)\n\n"
        "Pulangkan HANYA JSON array of strings, tanpa teks lain.\n"
        "Contoh: [\"cara bayar tonton\", \"langganan tonton up bayaran\", \"kaedah pembayaran akaun\"]"
    )

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        raw_text = response.content.strip()

        # Extract JSON array from response (handles markdown code fences)
        json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not json_match:
            logger.warning(
                f"Multi-query returned non-JSON response: {raw_text[:200]}"
            )
            return []

        parsed = json.loads(json_match.group())

        if not isinstance(parsed, list):
            logger.warning("Multi-query returned non-list JSON.")
            return []

        # Filter to valid strings, truncate to requested count
        variants = [v.strip() for v in parsed if isinstance(v, str) and v.strip()][:n]

        logger.info(
            f"Multi-query generated {len(variants)} variants for: '{question[:60]}...'"
        )
        return variants

    except Exception as e:
        logger.warning(f"Multi-query generation failed: {e}")
        return []
