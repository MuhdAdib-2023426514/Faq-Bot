"""Query normalizer utility for colloquial Malay and search term normalization."""

import re
from typing import Dict

# Common Malaysian chat abbreviations, slang, and dialect mappings to standard terms
SLANG_MAP: Dict[str, str] = {
    # Negations & Capabilities
    r"\bxleh\b": "tidak boleh",
    r"\bx\s+leh\b": "tidak boleh",
    r"\btakleh\b": "tidak boleh",
    r"\btak\s+leh\b": "tidak boleh",
    r"\btak\s+bole\b": "tidak boleh",
    r"\bxde\b": "tiada",
    r"\bx\s+de\b": "tiada",
    r"\btakde\b": "tiada",
    r"\btak\s+de\b": "tiada",
    r"\btakda\b": "tiada",
    r"\bx\s+da\b": "tiada",
    r"\bx\s+dapat\b": "tidak dapat",
    r"\bxdpt\b": "tidak dapat",
    r"\btakdpt\b": "tidak dapat",
    r"\bx\b": "tidak",

    # Interrogatives (Questions)
    r"\bcane\b": "bagaimana",
    r"\bcamne\b": "bagaimana",
    r"\bcmne\b": "bagaimana",
    r"\bgymana\b": "bagaimana",
    r"\bmcm\s+mana\b": "bagaimana",
    r"\bmcm\s+mane\b": "bagaimana",
    r"\bbape\b": "berapa",
    r"\bbrapa\b": "berapa",
    r"\bpasal\s+apa\b": "kenapa",
    r"\bbile\b": "bila",

    # Account, Subscription & Media terms
    r"\bacc\b": "akaun",
    r"\bacct\b": "akaun",
    r"\bsubs\b": "langganan",
    r"\bsub\b": "langganan",
    r"\bsubscribe\b": "langgan",
    r"\bsubscription\b": "langganan",
    r"\bcancel\b": "batal",
    r"\btgk\b": "tonton",
    r"\btengok\b": "tonton",
    r"\bbyr\b": "bayar",
    r"\bbayor\b": "bayar",
    r"\bpayment\b": "pembayaran",
    r"\bpw\b": "kata laluan",
    r"\bpasswd\b": "kata laluan",
    r"\bpassword\b": "kata laluan",
    r"\bhp\b": "telefon",
    r"\bfon\b": "telefon",
    r"\bphone\b": "telefon",

    # Connectors & Adverbs
    r"\bsbb\b": "sebab",
    r"\bsb\b": "sebab",
    r"\bdh\b": "sudah",
    r"\bdah\b": "sudah",
    r"\butk\b": "untuk",
    r"\bkt\b": "di",
    r"\bkat\b": "di",
    r"\bdr\b": "daripada",
    r"\bdrpd\b": "daripada",
    r"\bdgn\b": "dengan",
    r"\bngan\b": "dengan",
    r"\bplak\b": "pula",
    r"\bpulak\b": "pula",
    r"\blak\b": "pula",
    r"\bjgk\b": "juga",
    r"\bjgak\b": "juga",
    r"\btau\b": "tahu",
    r"\btauke\b": "tahu",
    r"\bnk\b": "nak",
    r"\bklu\b": "kalau",
    r"\bkalo\b": "kalau",
    r"\bckp\b": "cakap",
}


def normalize_query(query: str) -> str:
    """Normalizes informal colloquial Malaysian chat query into standardized search terms.

    Args:
        query: Raw user input question.

    Returns:
        Cleaned, normalized string with expanded slang and standardized terms.
    """
    if not query or not query.strip():
        return ""

    normalized = query.lower()

    # Apply slang replacement dictionary using regex boundaries
    for pattern, replacement in SLANG_MAP.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    # Normalize excessive whitespaces
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized
