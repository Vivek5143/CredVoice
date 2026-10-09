"""
Multi-turn conversation scenario tests for CredVoice.

Each test drives the TurnOrchestrator through a scripted conversation using
StubLLMAdapter.  The LLM replies are pre-canned strings that contain the JSON
extraction blocks; the orchestrator processes them exactly as it would with a
real LLM response.  No network calls are made.

Scenarios covered:
 1. Happy path — eligible customer, linear flow
 2. Out-of-order answers — multiple fields in one turn
 3. Answer correction — customer revises a previous answer
 4. Callback request at verification
 5. Callback request during qualification
 6. Agricultural property — immediate disqualification
 7. Missing original documents — immediate disqualification
 8. Cash-only income — immediate disqualification
 9. Loan amount > ₹75 lakh, customer accepts cap → eligible
10. Loan amount > ₹75 lakh, customer declines cap → disqualified
11. Tenure out of range (< 3 years) → disqualified
12. Existing property loan → transfer routing
13. EMI reduction intent → transfer routing
14. Hindi language conversation (basic smoke test)
15. Wrong person / unverified identity → call ends
16. MAX_TURNS safety guard
"""

from __future__ import annotations

import json
import pytest

from app.agent.core import StubLLMAdapter, TurnOrchestrator, create_orchestrator
from app.models.conversation import (
    CallOutcome,
    EligibilityStatus,
    Language,
    Stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _j(**fields) -> str:
    """Build a minimal LLM reply string containing a JSON extraction block."""
    payload = json.dumps(fields, ensure_ascii=False)
    return f"Sure, understood.\n```json\n{payload}\n```"


def _make(customer_name: str = "Test Customer", language: str = "en") -> TurnOrchestrator:
    """Create an orchestrator with no LLM set (tests inject replies per-turn)."""
    return create_orchestrator(
        customer_id="HC-TEST-001",
        customer_name=customer_name,
        language=language,
        llm=None,
    )


def _stub(*replies: str) -> StubLLMAdapter:
    return StubLLMAdapter(list(replies))


# Full set of qualification replies for a clean eligible customer
_FULL_QUAL_REPLIES = [
    # T1 — identity + availability confirmed
    _j(is_identity_confirmed=True, is_available=True),
    # T2 — property type
    _j(property_type="residential"),
    # T3 — ownership
    _j(ownership="sole"),
    # T4 — documents available
    _j(has_original_documents=True),
    # T5 — loan amount (₹50 lakh, within cap)
    _j(requested_loan_amount_inr=5000000),
    # T6 — occupation + income mode
    _j(occupation_type="salaried", income_mode="bank"),
    # T7 — property market value
    _j(property_market_value_inr=10000000),
    # T8 — tenure
    _j(requested_tenure_years=10),
]


# ---------------------------------------------------------------------------
# Scenario 1: Happy path — linear eligible flow
# ---------------------------------------------------------------------------

def test_happy_path_eligible():
    """Full linear conversation reaches ELIGIBLE_HANDOFF with action='handoff'."""
    orc = _make()
    orc._llm = _stub(*_FULL_QUAL_REPLIES)

    # T1 — verification
    r1 = orc.process_turn("Yes, this is Priya.")
    assert r1.state.stage == Stage.QUALIFICATION

    # T2–T7 — qualification fields
    for utterance in [
        "My property is residential.",
        "I am the sole owner.",
        "Yes, I have the original documents.",
        "I need around 50 lakh.",
        "I am salaried and get paid to my bank account.",
        "The property value is around 1 crore.",
    ]:
        r = orc.process_turn(utterance)
        assert r.action in {"continue", "handoff"}

    # T8 — tenure (should trigger handoff)
    r8 = orc.process_turn("I want a 10-year loan.")
    assert r8.state.stage == Stage.ELIGIBLE_HANDOFF
    assert r8.action == "handoff"
    assert r8.state.eligibility_status == EligibilityStatus.ELIGIBLE
    assert r8.state.call_outcome == CallOutcome.QUALIFIED_HANDOFF


# ---------------------------------------------------------------------------
# Scenario 2: Out-of-order answers — multiple fields in one turn
# ---------------------------------------------------------------------------

def test_out_of_order_multi_field_turn():
    """
    Customer volunteers property type + ownership + documents in a single turn.
    All three should be captured and no duplicate questions asked.
    """
    orc = _make()
    orc._llm = _stub(
        # T1 — verification
        _j(is_identity_confirmed=True, is_available=True),
        # T2 — three fields at once
        _j(
            property_type="commercial",
            ownership="joint",
            has_original_documents=True,
        ),
        # T3 — remaining fields
        _j(requested_loan_amount_inr=4000000),
        _j(occupation_type="self_employed", income_mode="bank"),
        _j(property_market_value_inr=8000000),
        _j(requested_tenure_years=7),
    )

    orc.process_turn("Yes, it's me and I can talk now.")
    r2 = orc.process_turn("Commercial property, jointly owned, and I have all original papers.")

    q = r2.state.qualification
    assert q.property_type.value == "commercial"
    assert q.ownership.value == "joint"
    assert q.has_original_documents is True

    # Remaining fields still missing
    assert r2.state.eligibility_status == EligibilityStatus.INCOMPLETE

    orc.process_turn("I need 40 lakh.")
    orc.process_turn("Self-employed, income through bank.")
    orc.process_turn("Property worth about 80 lakh.")
    final = orc.process_turn("7 years please.")
    assert final.action == "handoff"


# ---------------------------------------------------------------------------
# Scenario 3: Answer correction
# ---------------------------------------------------------------------------

def test_answer_correction():
    """Customer corrects occupation from salaried to self_employed mid-conversation."""
    orc = _make()
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        # Initial wrong answer
        _j(occupation_type="salaried"),
        # Correction in next turn
        _j(occupation_type="self_employed", income_mode="bank"),
        _j(property_type="residential"),
        _j(ownership="sole"),
        _j(has_original_documents=True),
        _j(requested_loan_amount_inr=3000000),
        _j(property_market_value_inr=6000000),
        _j(requested_tenure_years=5),
    )

    orc.process_turn("Yes, speaking.")
    orc.process_turn("I am salaried.")  # Initial
    r3 = orc.process_turn("Actually, sorry, I'm self-employed and income is via bank.")
    assert r3.state.qualification.occupation_type.value == "self_employed"
    assert r3.state.qualification.income_mode.value == "bank"


# ---------------------------------------------------------------------------
# Scenario 4: Callback at verification
# ---------------------------------------------------------------------------

def test_callback_at_verification():
    """Customer busy during verification — session moves to CALLBACK_REQUESTED."""
    orc = _make()
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=False, callback_requested=True,
           callback_preferred_time="tomorrow 4pm"),
    )

    r = orc.process_turn("I'm driving right now, can you call me tomorrow at 4?")
    assert r.state.stage == Stage.CALLBACK_REQUESTED
    assert r.action == "callback"
    assert r.state.callback.callback_requested is True
    assert r.state.callback.preferred_time == "tomorrow 4pm"
    assert r.state.call_outcome == CallOutcome.CALLBACK_SCHEDULED


# ---------------------------------------------------------------------------
# Scenario 5: Callback during qualification
# ---------------------------------------------------------------------------

def test_callback_during_qualification():
    """Customer requests callback after verification, mid-qualification."""
    orc = _make()
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(property_type="residential"),
        # Customer suddenly busy
        _j(callback_requested=True, callback_preferred_time="evening 7pm"),
    )

    orc.process_turn("Yes it's me and I can talk.")
    orc.process_turn("Residential property.")
    r3 = orc.process_turn("Sorry, someone's at the door. Can you call me at 7 this evening?")
    assert r3.state.stage == Stage.CALLBACK_REQUESTED
    assert r3.action == "callback"


# ---------------------------------------------------------------------------
# Scenario 6: Agricultural property → immediate disqualification
# ---------------------------------------------------------------------------

def test_agricultural_property_disqualification():
    """Agricultural property disqualifies the customer immediately after the field is captured."""
    orc = _make()
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(property_type="agricultural"),
    )

    orc.process_turn("Yes, speaking, and yes go ahead.")
    r2 = orc.process_turn("It's agricultural land.")
    assert r2.state.stage == Stage.DISQUALIFIED
    assert r2.action == "end"
    assert r2.state.eligibility_status == EligibilityStatus.DISQUALIFIED


# ---------------------------------------------------------------------------
# Scenario 7: Missing original documents → disqualification
# ---------------------------------------------------------------------------

def test_missing_documents_disqualification():
    orc = _make()
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(property_type="residential", ownership="sole"),
        _j(has_original_documents=False),
    )

    orc.process_turn("Yes, this is correct.")
    orc.process_turn("Residential, sole owner.")
    r3 = orc.process_turn("No, the original documents are with my builder.")
    assert r3.state.stage == Stage.DISQUALIFIED
    assert r3.action == "end"


# ---------------------------------------------------------------------------
# Scenario 8: Cash income → disqualification
# ---------------------------------------------------------------------------

def test_cash_income_disqualification():
    orc = _make()
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(property_type="residential", ownership="sole", has_original_documents=True),
        _j(requested_loan_amount_inr=3000000),
        _j(occupation_type="self_employed", income_mode="cash"),
    )

    orc.process_turn("Yes, speaking.")
    orc.process_turn("Residential, sole, and yes I have original documents.")
    orc.process_turn("30 lakh please.")
    r4 = orc.process_turn("I'm self-employed and I receive cash payments.")
    assert r4.state.stage == Stage.DISQUALIFIED
    assert r4.action == "end"


# ---------------------------------------------------------------------------
# Scenario 9: Amount > ₹75 lakh, customer accepts cap → eligible
# ---------------------------------------------------------------------------

def test_amount_exceeds_cap_accepted():
    """Customer requests ₹1 crore, consents to proceed at ₹75 lakh → eligible."""
    orc = _make()
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(property_type="residential", ownership="sole", has_original_documents=True),
        _j(requested_loan_amount_inr=10000000),  # ₹1 crore
        _j(proceeds_at_cap=True),                # Accepts ₹75 lakh cap
        _j(occupation_type="salaried", income_mode="bank"),
        _j(property_market_value_inr=20000000),
        _j(requested_tenure_years=12),
    )

    orc.process_turn("Yes, it's me and I'm free.")
    orc.process_turn("Residential, sole owner, documents available.")
    r3 = orc.process_turn("I need 1 crore.")
    # Engine should flag incomplete — cap consent needed
    assert r3.state.qualification.requires_amount_cap_consent is True
    assert r3.state.eligibility_status == EligibilityStatus.INCOMPLETE

    r4 = orc.process_turn("Oh okay, yes I'm happy with 75 lakh.")
    # After consent — still incomplete (other fields outstanding)
    assert r4.state.qualification.proceeds_at_cap is True

    orc.process_turn("I'm salaried, income to bank.")
    orc.process_turn("Property worth around 2 crore.")
    rfinal = orc.process_turn("12 years.")
    assert rfinal.state.eligibility_status == EligibilityStatus.ELIGIBLE
    assert rfinal.action == "handoff"


# ---------------------------------------------------------------------------
# Scenario 10: Amount > ₹75 lakh, customer declines → disqualified
# ---------------------------------------------------------------------------

def test_amount_exceeds_cap_declined():
    orc = _make()
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(property_type="residential", ownership="sole", has_original_documents=True),
        _j(requested_loan_amount_inr=10000000),
        _j(proceeds_at_cap=False),  # Declines ₹75 lakh cap
        # No further replies needed — call ends after decline
        _j(),
    )

    orc.process_turn("Yes, available.")
    orc.process_turn("Residential, sole, documents ok.")
    orc.process_turn("I need 1 crore.")
    r4 = orc.process_turn("No, 75 lakh is not enough for me.")
    assert r4.state.stage == Stage.DISQUALIFIED
    assert r4.action == "end"


# ---------------------------------------------------------------------------
# Scenario 11: Tenure < 3 years → disqualified
# ---------------------------------------------------------------------------

def test_tenure_too_short_disqualified():
    orc = _make()
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(property_type="residential", ownership="sole", has_original_documents=True,
           requested_loan_amount_inr=3000000, occupation_type="salaried",
           income_mode="bank", property_market_value_inr=8000000),
        _j(requested_tenure_years=1),  # Out of range
    )

    orc.process_turn("Yes, speaking.")
    orc.process_turn("Residential, sole, docs available, need 30 lakh, salaried, bank, property 80 lakh.")
    r3 = orc.process_turn("I want a 1-year loan.")
    assert r3.state.stage == Stage.DISQUALIFIED
    assert r3.action == "end"


# ---------------------------------------------------------------------------
# Scenario 12: Existing property loan → transfer routing
# ---------------------------------------------------------------------------

def test_existing_loan_transfer_routing():
    orc = _make()
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(has_existing_property_loan=True),
    )

    orc.process_turn("Yes, it's me and I'm free.")
    r2 = orc.process_turn("Actually I already have a home loan from HDFC that I want to move here.")
    assert r2.state.stage == Stage.TRANSFER_ROUTED
    assert r2.action == "transfer"
    assert r2.state.eligibility_status == EligibilityStatus.TRANSFER_ROUTED
    assert r2.state.call_outcome == CallOutcome.TRANSFER_ROUTED


# ---------------------------------------------------------------------------
# Scenario 13: EMI reduction intent → transfer routing
# ---------------------------------------------------------------------------

def test_emi_reduction_transfer_routing():
    orc = _make()
    orc._llm = _stub(
        _j(is_identity_confirmed=True, is_available=True),
        _j(wants_emi_reduction=True),
    )

    orc.process_turn("Yes, speaking.")
    r2 = orc.process_turn("I want to reduce my current property loan EMI.")
    assert r2.state.stage == Stage.TRANSFER_ROUTED
    assert r2.action == "transfer"


# ---------------------------------------------------------------------------
# Scenario 14: Hindi language — basic smoke test
# ---------------------------------------------------------------------------

def test_hindi_language_smoke():
    """Verify the orchestrator runs correctly with Language.HINDI set."""
    orc = create_orchestrator(
        customer_id="HC-HI-001",
        customer_name="राज शर्मा",
        language="hi",
        llm=_stub(
            _j(is_identity_confirmed=True, is_available=True),
            _j(language_switch="hi", property_type="residential"),
        ),
    )
    assert orc.state.language == Language.HINDI
    r1 = orc.process_turn("हाँ, मैं राज बोल रहा हूँ।")
    assert r1.state.stage == Stage.QUALIFICATION
    r2 = orc.process_turn("मेरी संपत्ति आवासीय है।")
    assert r2.state.qualification.property_type.value == "residential"


# ---------------------------------------------------------------------------
# Scenario 15: Wrong person / identity not confirmed
# ---------------------------------------------------------------------------

def test_wrong_person_ends_call():
    """If identity is NOT confirmed, the session should move to ENDED."""
    orc = _make()
    orc._llm = _stub(
        _j(is_identity_confirmed=False),
    )

    r = orc.process_turn("I think you have the wrong number.")
    assert r.state.stage == Stage.ENDED
    assert r.action == "end"


# ---------------------------------------------------------------------------
# Scenario 16: MAX_TURNS safety guard
# ---------------------------------------------------------------------------

def test_max_turns_guard():
    """Orchestrator terminates after MAX_TURNS even if conversation has not concluded."""
    orc = _make()
    # Stub returns blank extraction (no state progress) every time
    orc._llm = _stub(*[_j() for _ in range(TurnOrchestrator.MAX_TURNS + 5)])

    # Seed with verification first
    orc.state.stage = Stage.QUALIFICATION  # Bypass verification for this test
    orc._turn_count = TurnOrchestrator.MAX_TURNS  # Simulate already at limit

    r = orc.process_turn("...")
    assert r.action == "end"
    assert r.state.stage == Stage.ENDED


# ---------------------------------------------------------------------------
# Scenario 17: Terminal stage guard — no further turns after handoff
# ---------------------------------------------------------------------------

def test_no_turns_after_terminal_stage():
    """After a terminal stage (e.g. DISQUALIFIED), further utterances return action=end."""
    orc = _make()
    orc.state.stage = Stage.DISQUALIFIED  # Force terminal state

    r = orc.process_turn("Wait, I changed my mind.")
    assert r.action == "end"
    # LLM should NOT have been called — _llm is None and no error raised
