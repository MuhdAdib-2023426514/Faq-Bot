"""Guardrail Node for input safety validation, bilingual prompt injection detection, and PII protection."""

import re
import unicodedata
from typing import List, Tuple
from config.settings import settings
from src.state import GraphState
from src.utils.logger import logger

# Bilingual Prompt Injection Patterns (English + Bahasa Melayu)
INJECTION_PATTERNS: List[re.Pattern] = [
    # English Injection & Jailbreak
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"(system\s+prompt|system\s+instructions?)\s+(override|reveal|leak|show|print|dump)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+(dan|developer|god|unrestricted|jailbreak)\s+mode", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(safety|filters?|rules?|guardrails?)", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    # Bahasa Melayu Injection
    re.compile(r"abaikan\s+(semua\s+)?(arahan|peraturan)\s+(terdahulu|sebelum\s+ini)", re.IGNORECASE),
    re.compile(r"(tunjuk|papar|bocorkan|cetak)\s+(prompt|arahan)\s+sistem", re.IGNORECASE),
    re.compile(r"anda\s+kini\s+dalam\s+mod\s+(pembangun|penggodam|tanpa\s+had)", re.IGNORECASE),
    re.compile(r"padam\s+(sekatan|had|peraturan\s+keselamatan)", re.IGNORECASE),
]

# Bilingual Harmful Intent Patterns (Contextual phrases instead of naive single words)
HARMFUL_PATTERNS: List[re.Pattern] = [
    # Weapons & Explosives
    re.compile(r"\b(how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|explosive|weapon))\b", re.IGNORECASE),
    re.compile(r"\b((cara|bagaimana)\s+(buat|bina|cipta|hasilkan)\s+(bom|senjata|bahan\s+letupan))\b", re.IGNORECASE),
    # Self-harm & Violence
    re.compile(r"\b(how\s+to\s+(commit\s+suicide|kill\s+(myself|someone)|harm\s+myself))\b", re.IGNORECASE),
    re.compile(r"\b((cara|nak|mahu)\s+(bunuh\s+diri|mencederakan\s+diri|bunuh\s+orang))\b", re.IGNORECASE),
    # Cyberattack & Credential Theft
    re.compile(r"\b(password\s*stealer|sql\s*injection|ddos\s*attack|hack\s*account)\b", re.IGNORECASE),
    re.compile(r"\b(curi\s*(kata\s*laluan|password|akaun)|godam\s*akaun)\b", re.IGNORECASE),
]

# Sensitive Personal Data (PII) Patterns (Malaysian MyKad IC, Full Credit Card Sequences)
PII_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b\d{6}-\d{2}-\d{4}\b"),               # Malaysian MyKad: 980101-14-1234
    re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),        # 16-digit Card Number: 4111 2222 3333 4444
]


def sanitize_input_text(text: str) -> str:
    """Normalizes unicode homoglyphs and collapses invisible control characters.

    Args:
        text: Raw user input.

    Returns:
        Sanitized and normalized text string.
    """
    if not text:
        return ""
    # Normalize unicode to NFKC (resolves full-width, compatibility characters)
    normalized = unicodedata.normalize("NFKC", text)
    # Remove control characters and zero-width spaces
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\ufeff]", " ", normalized)
    return cleaned.strip()


def check_input_safety(query: str) -> Tuple[bool, str]:
    """Validates the input query for safety, integrity, length, and security threats.

    Args:
        query: Raw user query string.

    Returns:
        Tuple of (is_safe: bool, reason: str).
    """
    if not query or not query.strip():
        return False, "Input query is empty."

    cleaned_query = sanitize_input_text(query)

    # 1. Length boundary check
    max_len = settings.max_query_length
    if len(cleaned_query) > max_len:
        logger.warning(f"Guardrail triggered: Query exceeded max length ({len(cleaned_query)} > {max_len})")
        return False, "Input length exceeds the allowable limit."

    # 2. Prompt Injection Detection (Bilingual)
    for pattern in INJECTION_PATTERNS:
        if pattern.search(cleaned_query):
            logger.warning(f"Guardrail triggered: Prompt injection pattern matched: '{pattern.pattern}'")
            return False, "Prompt injection attempt detected. Request blocked by security guardrails."

    # 3. Harmful / Toxic Intent Detection (Bilingual)
    for pattern in HARMFUL_PATTERNS:
        if pattern.search(cleaned_query):
            logger.warning(f"Guardrail triggered: Harmful intent pattern matched: '{pattern.pattern}'")
            return False, "Harmful or unsafe content detected. Request blocked by safety guardrails."

    # 4. Sensitive PII Detection (Protects user privacy)
    for pattern in PII_PATTERNS:
        if pattern.search(cleaned_query):
            logger.warning("Guardrail triggered: Sensitive PII (MyKad / Card number) detected in input.")
            return False, "Sensitive personal data detected. Request blocked for privacy protection."

    return True, ""


def guardrail_node(state: GraphState) -> GraphState:
    """LangGraph node that validates the safety and integrity of user input."""
    if not settings.enable_guardrails:
        logger.info("Guardrails disabled. Skipping.")
        return {**state, "is_safe": True, "guardrail_reason": None}

    question = state.get("question", "")
    is_safe, reason = check_input_safety(question)

    logger.info(f"Guardrail check: is_safe={is_safe}, reason={reason or 'None'}")
    return {
        **state,
        "is_safe": is_safe,
        "guardrail_reason": reason if not is_safe else None,
    }
