"""Logging configuration module for the FAQ RAG application."""

import logging
import sys


def setup_logger(name: str = "faq_rag") -> logging.Logger:
    """Configures and returns a structured logger with standardized formatting."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Suppress verbose upstream library advisories (e.g. Google GenAI SDK AFC warnings)
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger


logger = setup_logger("faq_rag")

