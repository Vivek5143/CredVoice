"""
Phase 3 integration tests — FAQ grounding, prompt consistency, and state preservation.

All tests use StubLLMAdapter (no network calls).
A StubKnowledgeService is injected where needed so tests do not depend on
knowledge base file content.

Scenarios:
 1.  FAQ question mid-qualification: answer grounded, state preserved, pending field asked next.
 2.  Multiple consecutive FAQ interruptions: qualification never regresses.
 3.  No FAQ match → fallback injected into prompt; state unchanged.
 4.  faq_answered flag is True when FAQ hit; False when no match.
 5.  English prompt contains the seven field labels.
 6.  Hindi prompt contains key Hindi terms.
 7.  Both prompts contain the extraction schema key list.
 8.  FAQ answer for interest rate does not contain a percentage.
 9.  State preserved across FAQ turn: fields collected before/after intact.
10.  Returning to pending fields after FAQ interruption.
11.  FAQ hit during VERIFICATION stage does not advance stage incorrectly.
12.  FAQ hit during DISQUALIFIED stage does not change stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.core import (
    StubLLMAdapter,
    TurnOrchestrator,
    create_orchestrator,
)
from app.agent.prompt_builder import build_system_prompt, _FIELD_LABELS_EN, _FIELD_LABELS_HI
from app.models.conversation import (
    ConversationState,
    EligibilityStatus,
    Language,
    PropertyType,
    Stage,
)
from app.services.knowledge import KnowledgeService, KnowledgeResult


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _StubKnowledge:
    """Minimal knowledge stub — always returns the configured result."""

    def __init__(self, found: bool = True, answer: str = "Test FAQ answer.", entry_id: str = "test"):
        self._result = KnowledgeResult(
            found=found,
            entry=None,
            answer=answer,
            score=1 if found else 0,
        )

    def lookup(self, utterance: str, language: str = "en") -> KnowledgeResult:
        return self._result

    def all_entries(self):
        return []


def _hit_knowledge(answer: str = "The tenure is 3 to 15 years.") -> _StubKnowledge:
    return _StubKnowledge(found=True, answer=answer)


def _miss_knowledge() -> _StubKnowledge:
    return _StubKnowledge(found=False, answer=KnowledgeResult.FALLBACK)


def _j(**fields) -> str:
    payload = json.dumps(fields, ensure_ascii=False)
    return f"Sure.\n```json\n{payload}\n```"


def _make(language: str = "en", knowledge=None) -> TurnOrchestrator:
    orc = create_orchestrator(
        customer_id="HC-P3-001",
        customer_name="Amit Kumar",
        language=language,
        knowledge_service=knowledge or _miss_knowledge(),
    )
    return orc


def _stub(*replies: str) -> StubLLMAdapter:
    return StubLLMAdapter(list(replies))


# ---------------------------------------------------------------------------
# Scenario 1: FAQ interruption — grounded answer, state preserved
# ---------------------------------------------------------------------------

def test_faq_interruption_state_preserved():
    """
    Customer answers property_type, then asks a FAQ question, then answers ownership.
    Qualification state must accumulate correctly across the FAQ turn.
    """
    orc = _make(knowledge=_hit_knowledge("Tenure is 3 to 15 years."))
    orc._llm = _stub(
        # T1 — verify
        _j(is_identity_confirmed=True, is_available=True),
        # T2 — property_type collected
        _j(property_type="residential"),
        # T3 — FAQ question (no qual fields extracted)
        _j(faq_answered=True),
        # T4 — ownership
        _j(ownership="sole"),
    )

    orc.process_turn("Yes, Amit here and I'm free.")
    orc.process_turn("My property is residential.")
    r3 = orc.process_turn("By the way, how many years is the tenure?")

    # FAQ was answered this turn
    assert r3.faq_answered is True
    # Property type still set from T2
    assert r3.state.qualification.property_type == PropertyType.RESIDENTIAL
    # Ownership not yet set
    assert r3.state.qualification.ownership is None
    # Stage remains QUALIFICATION
    assert r3.state.stage == Stage.QUALIFICATION

    r4 = orc.process_turn("I'm the sole owner.")
    assert r4.state.qualification.ownership.value == "sole"


# ---------------------------------------------------------------------------
# Scenario 2: Multiple consecutive FAQ interruptions
# ---------------------------------------------------------------------------

def test_multiple_faq_turns_no_regression():
    """Three FAQ turns in a row must not reset or corrupt qualification state."""
    orc = _make(knowledge=_hit_knowledge())
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(property_type="commercial"),
        _j(faq_answered=True),   # FAQ 1
        _j(faq_answered=True),   # FAQ 2
        _j(faq_answered=True),   # FAQ 3
        _j(ownership="joint"),
    )

    orc.process_turn("Yes, speaking.")
    orc.process_turn("Commercial property.")
    for _ in range(3):
        r = orc.process_turn("Another question about the loan...")
        assert r.state.qualification.property_type.value == "commercial"

    r_final = orc.process_turn("Joint ownership.")
    assert r_final.state.qualification.ownership.value == "joint"


# ---------------------------------------------------------------------------
# Scenario 3: No FAQ match → fallback; state unchanged
# ---------------------------------------------------------------------------

def test_no_faq_match_fallback_no_state_change():
    """When no FAQ matches, the fallback message is available and state is untouched."""
    orc = _make(knowledge=_miss_knowledge())
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(property_type="residential"),
        _j(),  # No extraction — unknown question
    )

    orc.process_turn("Yes, available.")
    orc.process_turn("Residential.")
    r3 = orc.process_turn("What is your head office address?")

    assert r3.faq_answered is False
    assert r3.state.qualification.property_type.value == "residential"


# ---------------------------------------------------------------------------
# Scenario 4: faq_answered flag accuracy
# ---------------------------------------------------------------------------

def test_faq_answered_flag_true_on_hit():
    orc = _make(knowledge=_hit_knowledge())
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(faq_answered=True),
    )
    orc.process_turn("Yes, it's me.")
    r = orc.process_turn("How much loan can I get?")
    assert r.faq_answered is True


def test_faq_answered_flag_false_on_miss():
    orc = _make(knowledge=_miss_knowledge())
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(),
    )
    orc.process_turn("Yes, it's me.")
    r = orc.process_turn("Completely off-topic ramble here.")
    assert r.faq_answered is False


# ---------------------------------------------------------------------------
# Scenario 5: English prompt contains all seven field labels
# ---------------------------------------------------------------------------

def test_english_prompt_contains_seven_field_labels():
    """build_system_prompt (English) must reference all seven qualification fields."""
    state = ConversationState(
        session_id="s1",
        customer_id="c1",
        language=Language.ENGLISH,
        stage=Stage.QUALIFICATION,
    )
    # Populate missing_fields so they appear in prompt
    state.missing_fields = list(_FIELD_LABELS_EN.keys())

    prompt = build_system_prompt(state)

    for label in _FIELD_LABELS_EN.values():
        assert label in prompt, f"English prompt missing label: {label}"


# ---------------------------------------------------------------------------
# Scenario 6: Hindi prompt contains key Hindi terms
# ---------------------------------------------------------------------------

def test_hindi_prompt_contains_key_terms():
    """build_system_prompt (Hindi) must reference Hindi field labels."""
    state = ConversationState(
        session_id="s2",
        customer_id="c2",
        language=Language.HINDI,
        stage=Stage.QUALIFICATION,
    )
    state.missing_fields = list(_FIELD_LABELS_HI.keys())

    prompt = build_system_prompt(state)

    for label in _FIELD_LABELS_HI.values():
        assert label in prompt, f"Hindi prompt missing label: {label}"


# ---------------------------------------------------------------------------
# Scenario 7: Both prompts contain extraction schema keys
# ---------------------------------------------------------------------------

def test_prompts_contain_extraction_keys():
    """Both English and Hindi prompts must embed the extraction schema."""
    required_keys = [
        "is_identity_confirmed", "property_type", "income_mode",
        "requested_tenure_years", "faq_answered",
    ]
    for lang in [Language.ENGLISH, Language.HINDI]:
        state = ConversationState(
            session_id="s3", customer_id="c3",
            language=lang, stage=Stage.QUALIFICATION,
        )
        prompt = build_system_prompt(state)
        for key in required_keys:
            assert key in prompt, f"Prompt ({lang.value}) missing extraction key: {key}"


# ---------------------------------------------------------------------------
# Scenario 8: Interest rate FAQ entry does not expose a percentage
# ---------------------------------------------------------------------------

def test_interest_rate_faq_no_percentage():
    """
    The FAQ answer for interest rate must not contain a specific percentage — it
    must defer to a loan expert.
    """
    from app.services.knowledge import KnowledgeService
    svc = KnowledgeService()
    result = svc.lookup("What is the interest rate on this loan?")
    if result.found:
        assert "%" not in result.answer, (
            f"Interest rate FAQ answer must not expose a specific percentage:\n{result.answer}"
        )


# ---------------------------------------------------------------------------
# Scenario 9: State preservation across a full FAQ turn
# ---------------------------------------------------------------------------

def test_state_accumulation_across_faq_and_qual_turns():
    """Fields collected before and after a FAQ turn are both present in final state."""
    orc = _make(knowledge=_hit_knowledge("Loan is multi-purpose."))
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(property_type="industrial", ownership="sole"),
        _j(faq_answered=True),               # FAQ turn
        _j(has_original_documents=True),
        _j(requested_loan_amount_inr=5000000),
        _j(occupation_type="salaried", income_mode="bank"),
        _j(property_market_value_inr=12000000),
        _j(requested_tenure_years=8),
    )

    orc.process_turn("Yes, it's me and I'm free.")
    orc.process_turn("Industrial property, sole owner.")
    orc.process_turn("What can I use this loan for?")  # FAQ
    orc.process_turn("Yes, I have original documents.")
    orc.process_turn("I need 50 lakh.")
    orc.process_turn("Salaried, bank income.")
    orc.process_turn("Property worth 1.2 crore.")
    rfinal = orc.process_turn("8 years.")

    q = rfinal.state.qualification
    assert q.property_type.value == "industrial"
    assert q.ownership.value == "sole"
    assert q.has_original_documents is True
    assert q.requested_loan_amount_inr == 5000000.0
    assert q.occupation_type.value == "salaried"
    assert q.income_mode.value == "bank"
    assert q.property_market_value_inr == 12000000.0
    assert q.requested_tenure_years == 8
    assert rfinal.action == "handoff"


# ---------------------------------------------------------------------------
# Scenario 10: Return to pending qualification field after FAQ interruption
# ---------------------------------------------------------------------------

def test_returns_to_pending_field_after_faq():
    """
    After a FAQ turn, the orchestrator still has the same missing_fields — the
    eligibility engine is called again on the next qualification turn and the
    missing fields list is updated correctly when the customer provides the answer.
    """
    orc = _make(knowledge=_hit_knowledge())
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(property_type="residential"),
        _j(faq_answered=True),          # FAQ — ownership still missing
        _j(ownership="joint"),          # Customer finally answers
    )

    orc.process_turn("Yes, me and available.")
    r2 = orc.process_turn("Residential.")
    assert "ownership" in r2.state.missing_fields

    r3 = orc.process_turn("By the way, is joint ownership allowed?")
    # After FAQ, ownership still pending
    assert "ownership" in r3.state.missing_fields

    r4 = orc.process_turn("Yes, joint ownership.")
    assert r4.state.qualification.ownership.value == "joint"
    # ownership should no longer be missing (once engine re-evaluates)
    assert "ownership" not in r4.state.missing_fields


# ---------------------------------------------------------------------------
# Scenario 11: FAQ during VERIFICATION does not advance stage
# ---------------------------------------------------------------------------

def test_faq_during_verification_no_stage_advance():
    """FAQ lookup during VERIFICATION must not push stage to QUALIFICATION."""
    orc = _make(knowledge=_hit_knowledge())
    orc._llm = _stub(
        # Customer asks FAQ before confirming identity
        _j(faq_answered=True),
    )

    r = orc.process_turn("What is the maximum loan amount?")
    # Identity not confirmed → still VERIFICATION
    assert r.state.stage == Stage.VERIFICATION


# ---------------------------------------------------------------------------
# Scenario 12: FAQ during DISQUALIFIED stage does not change stage
# ---------------------------------------------------------------------------

def test_faq_during_disqualified_no_stage_change():
    """After disqualification, a FAQ question must not change the stage."""
    orc = _make(knowledge=_hit_knowledge())
    # Force disqualified state
    orc.state.stage = Stage.DISQUALIFIED

    r = orc.process_turn("Can I still apply later?")
    assert r.state.stage == Stage.DISQUALIFIED
    assert r.action == "end"
