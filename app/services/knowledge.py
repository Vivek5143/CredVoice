"""
CredVoice — Knowledge Service (app/services/knowledge.py).

Loads the LAP FAQ knowledge base from knowledge/lap_faq.json and retrieves
the most relevant grounded answer for a customer question.

Design principles:
- Retrieval is purely deterministic keyword matching — no LLM, no embeddings,
  no external API calls required for Phase 3.
- If no grounded answer is found, returns a structured "no answer" result so
  the orchestrator can instruct the LLM to defer to a loan expert.
- Never fabricates interest rates, fees, approval guarantees, or eligibility
  thresholds that are not present in the approved FAQ data.
- The knowledge base path is configurable via app.config.settings.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FAQEntry:
    """A single entry from the knowledge base."""
    id: str
    tags: tuple[str, ...]    # keyword tags for matching
    question: str            # canonical question text
    answer: str              # approved answer text


@dataclass(frozen=True)
class KnowledgeResult:
    """
    Result returned by KnowledgeService.lookup().

    Attributes:
        found:   True if a grounded answer was located.
        entry:   The matched FAQEntry (None when found=False).
        answer:  Convenience accessor — the answer text, or the fallback phrase.
        score:   Relevance score (number of matched tag tokens); 0 when not found.
    """
    found: bool
    entry: Optional[FAQEntry]
    answer: str
    score: int

    # Fallback text injected into the prompt when no grounded answer exists.
    FALLBACK = (
        "I don't have the specific information for that question. "
        "A Senior Loan Expert will be able to answer it in detail during your consultation."
    )

    FALLBACK_HI = (
        "मेरे पास उस प्रश्न की विशिष्ट जानकारी नहीं है। "
        "एक वरिष्ठ ऋण विशेषज्ञ आपकी परामर्श के दौरान इसका उत्तर देंगे।"
    )


# ---------------------------------------------------------------------------
# Knowledge service
# ---------------------------------------------------------------------------

class KnowledgeService:
    """
    Retrieves grounded FAQ answers for customer questions.

    The FAQ JSON file is loaded once and cached. Retrieval uses normalised
    keyword overlap between the customer utterance and each entry's tags.

    Args:
        kb_path: Path to the knowledge base JSON file.
                 Defaults to settings.knowledge_base_path.
    """

    # Minimum number of tag tokens that must match before a result is returned.
    MIN_SCORE: int = 1

    def __init__(self, kb_path: Optional[str] = None) -> None:
        path = Path(kb_path or settings.knowledge_base_path)
        self._entries: list[FAQEntry] = self._load(path)
        logger.info(
            "KnowledgeService loaded %d FAQ entries from %s",
            len(self._entries),
            path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, utterance: str, language: str = "en") -> KnowledgeResult:
        """
        Find the best-matching FAQ entry for the customer's utterance.

        Matching algorithm:
        1. Normalise both the utterance and each entry's tags to lowercase tokens.
        2. Count tag-token overlaps with utterance tokens.
        3. Return the entry with the highest score if score >= MIN_SCORE.
        4. Otherwise return a no-answer result.

        Args:
            utterance: Raw customer utterance text.
            language:  "en" or "hi" — used to select fallback language.

        Returns:
            KnowledgeResult with found=True/False and the answer text.
        """
        if not utterance or not self._entries:
            return self._no_answer(language)

        tokens = self._tokenise(utterance)
        best_score = 0
        best_entry: Optional[FAQEntry] = None

        for entry in self._entries:
            score = self._score(tokens, entry.tags)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self.MIN_SCORE:
            logger.debug(
                "FAQ match: id=%s score=%d utterance=%.60s",
                best_entry.id,
                best_score,
                utterance,
            )
            return KnowledgeResult(
                found=True,
                entry=best_entry,
                answer=best_entry.answer,
                score=best_score,
            )

        logger.debug("No FAQ match for utterance: %.60s", utterance)
        return self._no_answer(language)

    def all_entries(self) -> list[FAQEntry]:
        """Return all loaded FAQ entries (useful for testing and auditing)."""
        return list(self._entries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> list[FAQEntry]:
        """Load and parse the FAQ JSON file."""
        if not path.exists():
            logger.warning("Knowledge base not found at %s — running with empty FAQ.", path)
            return []
        try:
            raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load knowledge base: %s", exc)
            return []

        entries: list[FAQEntry] = []
        for item in raw:
            try:
                entries.append(
                    FAQEntry(
                        id=str(item["id"]),
                        tags=tuple(str(t).lower() for t in item.get("tags", [])),
                        question=str(item["question"]),
                        answer=str(item["answer"]),
                    )
                )
            except (KeyError, TypeError) as exc:
                logger.warning("Skipping malformed FAQ entry: %s — %s", item, exc)

        return entries

    STOP_WORDS: set[str] = {
        "", "i", "a", "the", "is", "are", "to", "in", "of", "my", "me", "we",
        "it", "this", "that", "can", "will", "do", "does", "and", "or", "for",
        "on", "at", "an", "be", "by", "so", "if", "as", "up", "ha", "aap", "mai",
        "kya", "se", "ka", "ki", "ke"
    }

    @classmethod
    def _tokenise(cls, text: str) -> set[str]:
        """
        Normalise text to a set of lowercase tokens (words only, no punctuation).
        """
        normalised = text.lower()
        return set(re.split(r"[\s\W]+", normalised)) - cls.STOP_WORDS

    @classmethod
    def _score(cls, utterance_tokens: set[str], tags: tuple[str, ...]) -> int:
        """
        Count how many tag tokens (or multi-word tag substrings) appear in utterance_tokens.
        """
        score = 0
        for tag in tags:
            tag_words = [w for w in tag.split() if w not in cls.STOP_WORDS]
            if not tag_words:
                continue
            if len(tag_words) == 1:
                if tag_words[0] in utterance_tokens:
                    score += 1
            else:
                if all(w in utterance_tokens for w in tag_words):
                    score += len(tag_words)
        return score

    @staticmethod
    def _no_answer(language: str) -> KnowledgeResult:
        fallback = (
            KnowledgeResult.FALLBACK_HI
            if language == "hi"
            else KnowledgeResult.FALLBACK
        )
        return KnowledgeResult(found=False, entry=None, answer=fallback, score=0)


# ---------------------------------------------------------------------------
# Module-level singleton (lazy — constructed on first import)
# ---------------------------------------------------------------------------

_default_service: Optional[KnowledgeService] = None


def get_knowledge_service() -> KnowledgeService:
    """
    Return the module-level singleton KnowledgeService.

    Lazy construction means the FAQ is only loaded when first needed,
    and tests can safely import this module without side effects.
    """
    global _default_service
    if _default_service is None:
        _default_service = KnowledgeService()
    return _default_service


def reset_knowledge_service() -> None:
    """
    Reset the singleton (used in tests to inject a custom knowledge base path).
    """
    global _default_service
    _default_service = None
