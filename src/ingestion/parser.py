"""FAQ Parser module for extracting structured Question-Answer documents from Markdown."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from src.utils.logger import logger


class FAQItem(BaseModel):
    """Structured data model for a single FAQ entry."""

    id: str = Field(description="Unique identifier for the FAQ item")
    question: str = Field(description="The customer's question")
    answer: str = Field(description="The detailed response and troubleshooting steps")
    category: str = Field(default="General", description="Topic category for the FAQ")
    source: str = Field(default="FAQ.md", description="Original source document name")


def _build_category_enum() -> List[str]:
    """Loads the valid FAQ categories from application settings.

    Lazy import avoids requiring a live config/API key at module import time,
    which keeps unit-test imports clean.
    """
    from config.settings import settings
    return list(settings.faq_categories)


def _get_llm():
    """Returns a ChatGoogleGenerativeAI instance.

    Lazy import keeps the module importable without a live API key.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from config.settings import settings

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.effective_api_key,
        temperature=0.0,
    )


def _get_variants_count() -> int:
    """Returns the configured number of question variants to generate.

    Lazy import keeps the module importable without a live config.
    """
    from config.settings import settings
    return settings.question_variants_count


def batch_infer_categories(
    items: List[Dict[str, str]],
) -> List[str]:
    """Classifies multiple FAQ items in a single LLM call.

    Args:
        items: List of dicts, each containing 'question' and 'answer' keys.

    Returns:
        List of category strings, one per input item. Falls back to 'Umum'
        for any item whose classification fails.
    """
    if not items:
        return []

    categories = _build_category_enum()
    fallback = categories[-1] if categories else "Umum"

    # Build a numbered list for the LLM
    items_block = "\n".join(
        f"{i+1}. Question: {item['question']}\n   Answer: {item['answer'][:300]}"
        for i, item in enumerate(items)
    )

    category_options = ", ".join(f'"{c}"' for c in categories)
    prompt = (
        "You are an expert classifier for the Tonton streaming platform FAQ.\n"
        f"Valid categories: [{category_options}]\n\n"
        "Classify each of the following FAQ items into exactly one category.\n"
        "Return ONLY a JSON array of category strings, one per item, in order.\n"
        f"Example for 2 items: [\"{categories[0]}\", \"{categories[1]}\"]\n\n"
        f"{items_block}"
    )

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        raw_text = response.content.strip()

        # Extract JSON array from response (handles markdown code fences)
        json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not json_match:
            logger.warning(f"LLM returned non-JSON response for batch classification: {raw_text[:200]}")
            return [fallback] * len(items)

        parsed: List[str] = json.loads(json_match.group())

        # Validate each category against allowed list
        validated = []
        for cat in parsed:
            if cat in categories:
                validated.append(cat)
            else:
                logger.warning(f"LLM returned unknown category '{cat}', falling back to '{fallback}'")
                validated.append(fallback)

        # Pad or truncate to match input length
        if len(validated) < len(items):
            logger.warning(
                f"LLM returned {len(validated)} categories for {len(items)} items, padding with '{fallback}'"
            )
            validated.extend([fallback] * (len(items) - len(validated)))
        elif len(validated) > len(items):
            validated = validated[: len(items)]

        return validated

    except Exception as e:
        logger.warning(f"Batch category inference failed: {e}. Falling back to '{fallback}' for all items.")
        return [fallback] * len(items)


def infer_category(question: str, answer: str) -> str:
    """Infers the category of a single FAQ item using the LLM.

    Backward-compatible wrapper around batch_infer_categories for single items.

    Args:
        question: The FAQ question text.
        answer: The FAQ answer text.

    Returns:
        The inferred category string.
    """
    results = batch_infer_categories([{"question": question, "answer": answer}])
    return results[0]


def batch_generate_question_variants(
    items: List[Dict[str, str]],
    num_variants: Optional[int] = None,
) -> List[List[str]]:
    """Generates synthetic question variants for multiple FAQ items in a single LLM call.

    For each FAQ item, produces N alternative questions that a user might ask
    and which should retrieve the same answer. Variants include colloquial Malay,
    symptom-based queries, and keyword-style searches.

    Args:
        items: List of dicts, each containing 'question' and 'answer' keys.
        num_variants: Number of variants to generate per item. Defaults to
            the ``question_variants_count`` setting.

    Returns:
        List of lists — one inner list of variant strings per input item.
        On failure, returns empty lists for all items.
    """
    if not items:
        return []

    n = num_variants if num_variants is not None else _get_variants_count()
    if n <= 0:
        return [[] for _ in items]

    # Build numbered FAQ list for the prompt
    items_block = "\n".join(
        f"{i+1}. Soalan Asal: {item['question']}\n   Jawapan: {item['answer'][:400]}"
        for i, item in enumerate(items)
    )

    prompt = (
        "You are a Malay-language FAQ enrichment specialist for the Tonton streaming platform.\n\n"
        f"For each of the {len(items)} FAQ items below, generate exactly {n} alternative questions "
        "that a user might realistically ask and that would be answered by the same FAQ answer.\n\n"
        "Requirements for each variant:\n"
        "- Mix of formal Bahasa Melayu AND colloquial/informal Malay (bahasa pasar)\n"
        "- Include symptom-based queries (e.g. 'tak boleh tengok', 'error muncul')\n"
        "- Include keyword-style short searches (e.g. 'cancel subscription Tonton')\n"
        "- Each variant must be distinct and not a trivial rephrasing\n\n"
        "Return ONLY a JSON array of arrays. Each inner array has exactly "
        f"{n} question strings.\n"
        f"Example for 2 FAQ items with {n} variants each:\n"
        f'[["variant1a", "variant1b", "variant1c", "variant1d"], '
        f'["variant2a", "variant2b", "variant2c", "variant2d"]]\n\n'
        f"{items_block}"
    )

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        raw_text = response.content.strip()

        # Extract JSON from response (handles markdown code fences)
        json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not json_match:
            logger.warning(
                f"LLM returned non-JSON response for variant generation: {raw_text[:200]}"
            )
            return [[] for _ in items]

        parsed = json.loads(json_match.group())

        # Validate structure: must be list of lists of strings
        if not isinstance(parsed, list):
            logger.warning("Variant generation returned non-list JSON.")
            return [[] for _ in items]

        result: List[List[str]] = []
        for i, item_variants in enumerate(parsed):
            if isinstance(item_variants, list):
                # Filter to only strings, truncate to n
                valid = [v for v in item_variants if isinstance(v, str)][:n]
                result.append(valid)
            else:
                logger.warning(f"Variant for item {i+1} is not a list, skipping.")
                result.append([])

        # Pad if LLM returned fewer items than expected
        while len(result) < len(items):
            result.append([])

        total_variants = sum(len(v) for v in result)
        logger.info(
            f"Generated {total_variants} question variants for {len(items)} FAQ items "
            f"({n} requested per item)."
        )
        return result

    except Exception as e:
        logger.warning(f"Batch variant generation failed: {e}. No variants will be created.")
        return [[] for _ in items]


class FAQParser:
    """Parses markdown FAQ files containing Question and Answer sections."""

    def __init__(self, file_path: Optional[str] = None):
        """Initializes the FAQParser with an optional file path.

        Args:
            file_path: Optional path to the FAQ markdown file.
        """
        self.file_path = file_path

    def clean_markdown_text(self, text: str) -> str:
        """Cleans escaped markdown characters and normalizes whitespace.

        Handles common markdown escape sequences and collapses excessive
        blank lines into a single blank line for cleaner document content.

        Args:
            text: Raw text potentially containing markdown escape sequences.

        Returns:
            Cleaned text with escapes resolved and whitespace normalized.
        """
        # Resolve escaped numbered lists: \\1\\. -> 1.
        text = re.sub(r'\\\\([0-9]+)\\\\\\.', r'\1.', text)
        # Resolve common markdown escape sequences
        text = text.replace(r'\>', '>').replace(r'\-', '-').replace(r'\_', '_')
        text = text.replace(r'\*', '*').replace(r'\[', '[').replace(r'\]', ']')
        # Collapse 3+ consecutive newlines into 2 (one blank line)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def parse_text(self, raw_content: str, source_name: str = "FAQ.md") -> List[FAQItem]:
        """Parses raw markdown text into structured FAQItem instances.

        Uses batch LLM classification to categorize all parsed items
        in a single API call instead of per-item calls.

        Args:
            raw_content: The full raw markdown content of the FAQ file.
            source_name: Name of the source document for metadata attribution.

        Returns:
            List of FAQItem instances extracted from the content.
        """
        # Split by **Question:** or Question: headers
        pattern = r"\*{0,2}Question:\*{0,2}\s*(.*?)(?=\*{0,2}Question:\*{0,2}|$)"
        matches = re.findall(pattern, raw_content, flags=re.DOTALL | re.IGNORECASE)

        if not matches:
            sample = raw_content[:500].replace('\n', '\\n')
            logger.warning(
                f"FAQ parser found 0 items in '{source_name}'. "
                f"Expected '**Question:**' or 'Question:' headers. "
                f"Content preview: {sample}"
            )
            return []

        # First pass: extract Q&A text pairs
        qa_pairs: List[Dict[str, str]] = []
        for block in matches:
            block = block.strip()
            if not block:
                continue

            ans_split = re.split(
                r"\*{0,2}Answer:\*{0,2}\s*", block, maxsplit=1, flags=re.IGNORECASE
            )
            if len(ans_split) == 2:
                q_text = self.clean_markdown_text(ans_split[0])
                a_text = self.clean_markdown_text(ans_split[1])
            else:
                lines = block.split("\n", 1)
                q_text = self.clean_markdown_text(lines[0])
                a_text = self.clean_markdown_text(lines[1]) if len(lines) > 1 else ""

            if not q_text:
                continue

            qa_pairs.append({"question": q_text, "answer": a_text})

        if not qa_pairs:
            logger.warning(f"All parsed blocks from '{source_name}' were empty after cleaning.")
            return []

        # Batch classify all items in one LLM call
        categories = batch_infer_categories(qa_pairs)

        # Build FAQItem list
        faq_items: List[FAQItem] = []
        for idx, (pair, category) in enumerate(zip(qa_pairs, categories), start=1):
            faq_id = f"faq_{idx:03d}"
            faq_items.append(
                FAQItem(
                    id=faq_id,
                    question=pair["question"],
                    answer=pair["answer"],
                    category=category,
                    source=source_name,
                )
            )

        logger.info(f"Successfully parsed {len(faq_items)} FAQ items from {source_name}")
        return faq_items

    def parse_file(self, file_path: Optional[str] = None) -> List[FAQItem]:
        """Reads and parses an FAQ markdown file from disk.

        Args:
            file_path: Optional override path. Falls back to instance path,
                       then to 'FAQ.md'.

        Returns:
            List of FAQItem instances parsed from the file.

        Raises:
            FileNotFoundError: If the resolved file path does not exist.
        """
        target_path = Path(file_path or self.file_path or "FAQ.md")
        if not target_path.exists():
            raise FileNotFoundError(f"FAQ file not found at path: {target_path.resolve()}")

        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.parse_text(content, source_name=target_path.name)

    def to_langchain_documents(
        self,
        faq_items: List[FAQItem],
        enrich_variants: bool = True,
        num_variants: Optional[int] = None,
    ) -> List[Document]:
        """Converts FAQItem list into LangChain Document objects with question-only embedding.

        Each FAQ item produces N+1 documents: 1 original question + N synthetic
        variants. All documents for the same FAQ share the same ``id`` in metadata
        and store the full answer in ``metadata["answer"]``.

        The ``page_content`` contains only the question text (for embedding),
        while the full answer is preserved in metadata for the generator node.

        Args:
            faq_items: List of structured FAQ items to convert.
            enrich_variants: If True, generates synthetic question variants
                via LLM. Set to False to skip variant generation (e.g. in tests).
            num_variants: Override for the number of variants per item.
                Defaults to the ``question_variants_count`` setting.

        Returns:
            List of LangChain Document objects. For N FAQ items with M variants
            each, returns up to N * (2 + M) documents (1 original Q + 1 answer + M variants).
        """
        if not faq_items:
            return []

        # Generate variants in a single batch LLM call
        all_variants: List[List[str]] = []
        if enrich_variants:
            qa_pairs = [
                {"question": item.question, "answer": item.answer}
                for item in faq_items
            ]
            all_variants = batch_generate_question_variants(qa_pairs, num_variants)
        else:
            all_variants = [[] for _ in faq_items]

        # Maximum answer length to embed alongside question for the answer vector.
        # Keeps the combined Q+A within a reasonable token budget for the embedding model.
        answer_embed_max_chars = 800

        documents: List[Document] = []
        for idx, item in enumerate(faq_items):
            # Shared metadata for all documents belonging to this FAQ
            base_metadata = {
                "id": item.id,
                "question": item.question,
                "answer": item.answer,
                "category": item.category,
                "source": item.source,
            }

            # 1. Original question document — pure question text for embedding
            documents.append(
                Document(
                    page_content=item.question,
                    metadata={**base_metadata, "variant_type": "original"},
                )
            )

            # 2. Answer document — combined Q+A captures the semantic relationship
            #    between what users ask and the actual answer content
            if item.answer and item.answer.strip():
                answer_text = item.answer[:answer_embed_max_chars].strip()
                documents.append(
                    Document(
                        page_content=f"{item.question}\n{answer_text}",
                        metadata={**base_metadata, "variant_type": "answer"},
                    )
                )

            # 3. Synthetic variant documents — pure variant text for embedding
            variants = all_variants[idx] if idx < len(all_variants) else []
            for variant_text in variants:
                if variant_text and variant_text.strip():
                    documents.append(
                        Document(
                            page_content=variant_text.strip(),
                            metadata={**base_metadata, "variant_type": "synthetic"},
                        )
                    )

        answer_docs = sum(1 for d in documents if d.metadata.get("variant_type") == "answer")
        variant_docs = sum(1 for d in documents if d.metadata.get("variant_type") == "synthetic")
        logger.info(
            f"Created {len(documents)} LangChain documents from {len(faq_items)} FAQ items "
            f"({len(faq_items)} originals, {answer_docs} answer vectors, {variant_docs} synthetic variants)."
        )
        return documents

