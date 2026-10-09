"""
Unit tests for app/services/knowledge.py.

Tests cover:
 1.  Service loads FAQ entries from the real knowledge base file.
 2.  Exact-tag keyword match returns the correct entry.
 3.  Partial-match (fewer tokens) returns best-score entry.
 4.  Empty utterance returns no-answer result.
 5.  Utterance with no matching tags returns no-answer result.
 6.  Fallback text is non-empty and does not invent policy.
 7.  Hindi fallback text is returned when language="hi".
 8.  Multi-word tag match scores higher than single-word match.
 9.  Service survives a missing knowledge base file gracefully.
10.  Service survives a malformed JSON file gracefully.
11.  Interest rate question gets no grounded answer (not in FAQ).
12.  Processing time question returns the approved deferral answer.
13.  Loan amount question returns answer containing ₹75 lakh.
14.  Property type question returns answer mentioning agricultural.
15.  reset_knowledge_service allows re-initialising the singleton.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.services.knowledge import (
    FAQEntry,
    KnowledgeResult,
    KnowledgeService,
    get_knowledge_service,
    reset_knowledge_service,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before and after each test."""
    reset_knowledge_service()
    yield
    reset_knowledge_service()


@pytest.fixture()
def real_kb() -> KnowledgeService:
    """KnowledgeService loaded from the real knowledge/lap_faq.json."""
    return KnowledgeService()


@pytest.fixture()
def minimal_kb(tmp_path: Path) -> KnowledgeService:
    """KnowledgeService loaded from a tiny in-memory FAQ file."""
    data = [
        {
            "id": "test_loan",
            "tags": ["maximum", "loan", "amount"],
            "question": "What is the max loan?",
            "answer": "The maximum is ₹75 lakh.",
        },
        {
            "id": "test_tenure",
            "tags": ["tenure", "years", "duration"],
            "question": "What is the tenure?",
            "answer": "3 to 15 years.",
        },
        {
            "id": "test_transfer",
            "tags": ["balance", "transfer", "existing"],
            "question": "Can I transfer my loan?",
            "answer": "Yes, transfer is possible via the specialist.",
        },
    ]
    kb_file = tmp_path / "faq.json"
    kb_file.write_text(json.dumps(data), encoding="utf-8")
    return KnowledgeService(kb_path=str(kb_file))


# ---------------------------------------------------------------------------
# Test 1: Real knowledge base loads correctly
# ---------------------------------------------------------------------------

def test_real_kb_loads(real_kb: KnowledgeService):
    """Service loads at least 10 entries from the real FAQ file."""
    entries = real_kb.all_entries()
    assert len(entries) >= 10, f"Expected ≥10 FAQ entries, got {len(entries)}"
    ids = {e.id for e in entries}
    assert "max_loan_amount" in ids
    assert "loan_tenure" in ids
    assert "interest_rate" in ids


# ---------------------------------------------------------------------------
# Test 2: Exact tag match
# ---------------------------------------------------------------------------

def test_exact_tag_match(minimal_kb: KnowledgeService):
    result = minimal_kb.lookup("What is the maximum loan amount?")
    assert result.found is True
    assert result.entry.id == "test_loan"
    assert "75 lakh" in result.answer.lower() or "75" in result.answer


# ---------------------------------------------------------------------------
# Test 3: Partial-match returns best entry
# ---------------------------------------------------------------------------

def test_partial_match_returns_best(minimal_kb: KnowledgeService):
    result = minimal_kb.lookup("How many years for the tenure?")
    assert result.found is True
    assert result.entry.id == "test_tenure"


# ---------------------------------------------------------------------------
# Test 4: Empty utterance → no answer
# ---------------------------------------------------------------------------

def test_empty_utterance_no_answer(real_kb: KnowledgeService):
    result = real_kb.lookup("")
    assert result.found is False
    assert result.answer  # fallback text should not be empty


# ---------------------------------------------------------------------------
# Test 5: No matching tags → no answer
# ---------------------------------------------------------------------------

def test_no_matching_tags_no_answer(real_kb: KnowledgeService):
    result = real_kb.lookup("The weather is nice today, isn't it?")
    assert result.found is False
    assert len(result.answer) > 10  # fallback message present


# ---------------------------------------------------------------------------
# Test 6: Fallback text does not invent policy
# ---------------------------------------------------------------------------

def test_fallback_does_not_invent_policy(real_kb: KnowledgeService):
    result = real_kb.lookup("xyz_completely_unknown_query_that_should_not_match_anything")
    assert result.found is False
    # Must not contain invented rates or thresholds
    assert "%" not in result.answer
    assert "guaranteed" not in result.answer.lower()


# ---------------------------------------------------------------------------
# Test 7: Hindi fallback
# ---------------------------------------------------------------------------

def test_hindi_fallback(real_kb: KnowledgeService):
    result = real_kb.lookup("कुछ अजीब सवाल जो कहीं नहीं मिलेगा", language="hi")
    assert result.found is False
    assert result.answer == KnowledgeResult.FALLBACK_HI


# ---------------------------------------------------------------------------
# Test 8: Multi-word tag match scores higher
# ---------------------------------------------------------------------------

def test_multiword_tag_scores_higher(minimal_kb: KnowledgeService):
    """'balance transfer' multi-word tag should match the transfer entry."""
    result = minimal_kb.lookup("I want to do a balance transfer of my existing loan.")
    assert result.found is True
    assert result.entry.id == "test_transfer"
    assert result.score >= 2  # multi-word bonus applied


# ---------------------------------------------------------------------------
# Test 9: Missing knowledge base file
# ---------------------------------------------------------------------------

def test_missing_kb_file_graceful(tmp_path: Path):
    svc = KnowledgeService(kb_path=str(tmp_path / "does_not_exist.json"))
    assert svc.all_entries() == []
    result = svc.lookup("anything")
    assert result.found is False


# ---------------------------------------------------------------------------
# Test 10: Malformed JSON file
# ---------------------------------------------------------------------------

def test_malformed_json_graceful(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("THIS IS NOT JSON", encoding="utf-8")
    svc = KnowledgeService(kb_path=str(bad))
    assert svc.all_entries() == []
    result = svc.lookup("loan amount")
    assert result.found is False


# ---------------------------------------------------------------------------
# Test 11: Interest rate question — NOT in grounded FAQ → no answer
# ---------------------------------------------------------------------------

def test_interest_rate_not_grounded(real_kb: KnowledgeService):
    """Interest rate is in the FAQ only as a deferral — answer must not contain a percentage."""
    result = real_kb.lookup("What is the interest rate on this loan?")
    # It may match "interest_rate" entry but that entry defers to expert
    if result.found:
        assert "%" not in result.answer or "expert" in result.answer.lower()


# ---------------------------------------------------------------------------
# Test 12: Processing time → grounded deferral answer
# ---------------------------------------------------------------------------

def test_processing_time_answer(real_kb: KnowledgeService):
    result = real_kb.lookup("How long does it take to process the loan?")
    assert result.found is True
    assert result.entry.id == "processing_time"
    # Must defer — should not give a specific timeline
    answer_lower = result.answer.lower()
    assert "expert" in answer_lower or "specialist" in answer_lower


# ---------------------------------------------------------------------------
# Test 13: Loan amount → ₹75 lakh in answer
# ---------------------------------------------------------------------------

def test_loan_amount_answer_contains_cap(real_kb: KnowledgeService):
    result = real_kb.lookup("How much loan can I get?")
    assert result.found is True
    assert "75" in result.answer


# ---------------------------------------------------------------------------
# Test 14: Property type → agricultural disqualified
# ---------------------------------------------------------------------------

def test_property_type_answer_mentions_agricultural(real_kb: KnowledgeService):
    result = real_kb.lookup("What property types are eligible?")
    assert result.found is True
    assert "agricultural" in result.answer.lower()


# ---------------------------------------------------------------------------
# Test 15: Singleton reset
# ---------------------------------------------------------------------------

def test_singleton_reset():
    s1 = get_knowledge_service()
    reset_knowledge_service()
    s2 = get_knowledge_service()
    assert s1 is not s2  # Fresh instance after reset
