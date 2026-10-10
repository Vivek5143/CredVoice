"""
CredVoice — Regression and Enforcement Tests for Bolna Flow Eligibility (tests/conversation/test_bolna_eligibility_enforcement.py).

Verifies the 5 issues identified from the Hindi test transcript:
1. Customer says 'क   ष    भ   म    ' (agricultural property) -> normalized to agricultural, immediate disqualification.
2. Customer says 'नकद नकद' (cash income) -> normalized to cash, immediate disqualification.
3. Customer requests a 40-year tenure -> rejected and clarified (3 to 15 years), not silently accepted.
4. Disqualification takes absolute precedence over EMI-reduction / transfer-routing.
5. Terminal state guard: after handoff or disqualification, the agent never restarts greeting or continues qualification.
"""

import pytest

from app.agent.core import StubLLMAdapter, TurnOrchestrator
from app.agent.prompt_builder import normalize_utterance_extraction
from app.integrations.bolna import BolnaIntegration
from app.models.conversation import (
    CallOutcome,
    ConversationState,
    DisqualificationReason,
    EligibilityStatus,
    ExistingLoanTransfer,
    Language,
    OwnershipType,
    PropertyType,
    QualificationState,
    Stage,
)
from app.services.eligibility import EligibilityEngine, update_conversation_eligibility
from app.services.session_manager import SessionManager


@pytest.fixture(autouse=True)
def mock_default_llm(monkeypatch):
    """Ensure all tests in this file use a mocked LLM and make zero network calls."""
    monkeypatch.setattr(
        "app.agent.core.create_llm_adapter",
        lambda: StubLLMAdapter(["Sure, understood. ```json {} ```"] * 30),
    )


# ---------------------------------------------------------------------------
# 1. Hindi and English Extraction Normalization Tests
# ---------------------------------------------------------------------------

def test_normalization_agricultural_property_hindi_variations():
    """Verify phonetic and spaced Hindi agricultural phrases normalize to property_type='agricultural'."""
    # Phonetic spaced ASR artifacts
    res1 = normalize_utterance_extraction("क   ष    भ   म    ")
    assert res1.get("property_type") == "agricultural"
    assert res1.get("language_switch") == "hi"

    # खेत की जमीन variation
    res2 = normalize_utterance_extraction("ख   त    क    जम   न ")
    assert res2.get("property_type") == "agricultural"

    # Standard Devanagari
    res3 = normalize_utterance_extraction("यह कृषि भूमि है")
    assert res3.get("property_type") == "agricultural"

    res4 = normalize_utterance_extraction("खेती की जमीन है")
    assert res4.get("property_type") == "agricultural"

    # English
    res5 = normalize_utterance_extraction("This is agricultural land")
    assert res5.get("property_type") == "agricultural"


def test_normalization_cash_income_hindi_variations():
    """Verify Hindi cash income phrases normalize to income_mode='cash'."""
    # Repeated / spaced cash
    res1 = normalize_utterance_extraction("नकद नकद")
    assert res1.get("income_mode") == "cash"
    assert res1.get("language_switch") == "hi"

    res2 = normalize_utterance_extraction("मुझे नगद मिलता है")
    assert res2.get("income_mode") == "cash"

    res3 = normalize_utterance_extraction("हाथ में रोकड़ मिलती है")
    assert res3.get("income_mode") == "cash"

    # English
    res4 = normalize_utterance_extraction("I receive cash in hand")
    assert res4.get("income_mode") == "cash"


def test_normalization_tenure_variations():
    """Verify tenure numbers in Hindi and English are extracted accurately."""
    res1 = normalize_utterance_extraction("40 years")
    assert res1.get("requested_tenure_years") == 40

    res2 = normalize_utterance_extraction("40 साल के लिए चाहिए")
    assert res2.get("requested_tenure_years") == 40

    res3 = normalize_utterance_extraction("10 वर्ष")
    assert res3.get("requested_tenure_years") == 10


# ---------------------------------------------------------------------------
# 2. Immediate Disqualification on Agricultural Property and Cash Income
# ---------------------------------------------------------------------------

def test_immediate_agricultural_disqualification_in_hindi():
    """Customer stating 'क   ष    भ   म    ' must trigger immediate disqualification, action='end'."""
    orchestrator = TurnOrchestrator(
        customer_id="HC-TEST-001",
        customer_name="राकेश शर्मा",
        language=Language.HINDI,
    )
    # Complete verification
    orchestrator.process_turn("हाँ, मैं राकेश शर्मा हूँ और बात करने का सही समय है।")
    assert orchestrator.state.stage == Stage.QUALIFICATION

    # Customer states agricultural land
    turn = orchestrator.process_turn("क   ष    भ   म    ")
    assert orchestrator.state.stage == Stage.DISQUALIFIED
    assert orchestrator.state.eligibility_status == EligibilityStatus.DISQUALIFIED
    assert DisqualificationReason.AGRICULTURAL_PROPERTY in orchestrator.state.disqualification_reasons
    assert turn.action == "end"
    assert "कृषि संपत्ति पर ऋण उपलब्ध नहीं है" in turn.agent_reply or "वर्तमान नीति के अनुसार" in turn.agent_reply
    assert orchestrator.state.call_outcome == CallOutcome.DISQUALIFIED


def test_immediate_cash_income_disqualification_in_hindi():
    """Customer stating 'नकद नकद' must trigger immediate disqualification, action='end'."""
    orchestrator = TurnOrchestrator(
        customer_id="HC-TEST-002",
        customer_name="सुरेश कुमार",
        language=Language.HINDI,
    )
    # Complete verification
    orchestrator.process_turn("हाँ, बात कर सकते हैं।")
    assert orchestrator.state.stage == Stage.QUALIFICATION

    # Customer states cash income
    turn = orchestrator.process_turn("नकद नकद")
    assert orchestrator.state.stage == Stage.DISQUALIFIED
    assert orchestrator.state.eligibility_status == EligibilityStatus.DISQUALIFIED
    assert DisqualificationReason.CASH_INCOME in orchestrator.state.disqualification_reasons
    assert turn.action == "end"
    assert "नकद आय पर ऋण उपलब्ध नहीं है" in turn.agent_reply or "बैंक खाते में प्राप्त होना अनिवार्य है" in turn.agent_reply


# ---------------------------------------------------------------------------
# 3. Invalid Tenure Rejected and Clarified (3 to 15 Years)
# ---------------------------------------------------------------------------

def test_invalid_tenure_rejected_and_clarified_without_disqualification():
    """Customer requesting 40-year tenure is rejected and clarified without disqualifying valid lead."""
    orchestrator = TurnOrchestrator(
        customer_id="HC-TEST-003",
        customer_name="Amit Patel",
        language=Language.ENGLISH,
    )
    orchestrator.process_turn("Yes, speaking.")
    assert orchestrator.state.stage == Stage.QUALIFICATION

    # Request 40 years
    turn = orchestrator.process_turn("I want a loan for 40 years.")
    # Must NOT disqualify the customer; must clarify and keep tenure missing
    assert orchestrator.state.stage == Stage.QUALIFICATION
    assert orchestrator.state.qualification.requested_tenure_years is None
    assert turn.action == "continue"
    assert "3 to 15 years" in turn.agent_reply or "between 3 and 15 years" in turn.agent_reply
    assert "40 years" in turn.agent_reply


def test_invalid_tenure_in_hindi_clarified():
    """Customer requesting 40-year tenure in Hindi receives Hindi clarification."""
    orchestrator = TurnOrchestrator(
        customer_id="HC-TEST-004",
        customer_name="अनिल वर्मा",
        language=Language.HINDI,
    )
    orchestrator.process_turn("हाँ, बात कर सकते हैं।")

    turn = orchestrator.process_turn("मुझे 40 साल के लिए चाहिए")
    assert orchestrator.state.stage == Stage.QUALIFICATION
    assert orchestrator.state.qualification.requested_tenure_years is None
    assert turn.action == "continue"
    assert "3 से 15 वर्ष" in turn.agent_reply


# ---------------------------------------------------------------------------
# 4. Priority: Disqualification vs Transfer Routing
# ---------------------------------------------------------------------------

def test_disqualification_takes_precedence_over_transfer_intent():
    """Customer with agricultural land who requests EMI reduction must be DISQUALIFIED, not routed to specialist."""
    state = QualificationState(
        property_type=PropertyType.AGRICULTURAL,  # Hard disqualifier
        ownership=OwnershipType.SOLE,
        has_original_documents=True,
        requested_loan_amount_inr=5000000.0,
        requested_tenure_years=10,
    )
    transfer = ExistingLoanTransfer(
        has_existing_property_loan=True,
        wants_emi_reduction=True,
    )
    result = EligibilityEngine.evaluate(qualification=state, transfer_intent=transfer)
    assert result.status == EligibilityStatus.DISQUALIFIED
    assert result.is_transfer_lead is False
    assert DisqualificationReason.AGRICULTURAL_PROPERTY in result.disqualification_reasons


def test_subsequent_emi_request_cannot_override_established_disqualification():
    """A customer already marked DISQUALIFIED cannot be transferred to specialist on subsequent turn."""
    conv_state = ConversationState(
        session_id="test-immutability-1",
        customer_id="HC-DISQ-01",
        stage=Stage.DISQUALIFIED,
        eligibility_status=EligibilityStatus.DISQUALIFIED,
    )
    conv_state.disqualification_reasons = [DisqualificationReason.AGRICULTURAL_PROPERTY]

    # Customer later says they want an EMI reduction
    conv_state.transfer_intent.wants_emi_reduction = True
    conv_state.transfer_intent.has_existing_property_loan = True

    updated_state = update_conversation_eligibility(conv_state)
    assert updated_state.stage == Stage.DISQUALIFIED
    assert updated_state.eligibility_status == EligibilityStatus.DISQUALIFIED


def test_clean_transfer_lead_routes_to_specialist():
    """Customer without disqualifications seeking EMI reduction routes to Loan-Transfer Specialist."""
    orchestrator = TurnOrchestrator(
        customer_id="HC-TRANSFER-001",
        customer_name="Pooja Mehta",
        language=Language.ENGLISH,
    )
    orchestrator.process_turn("Yes, speaking.")
    turn = orchestrator.process_turn("I already have a loan and want to reduce EMI with balance transfer.")
    assert orchestrator.state.stage == Stage.TRANSFER_ROUTED
    assert turn.action == "transfer"
    assert "loan-transfer specialist" in turn.agent_reply.lower()


# ---------------------------------------------------------------------------
# 5. Terminal State Guard & Greeting Restart Prevention
# ---------------------------------------------------------------------------

def test_terminal_session_prevents_greeting_restart():
    """Once a call reaches terminal stage, subsequent utterances never restart greeting or verification."""
    orchestrator = TurnOrchestrator(
        customer_id="HC-TERM-001",
        customer_name="Vikram Singh",
        language=Language.HINDI,
    )
    orchestrator.process_turn("हाँ, मैं विक्रम सिंह हूँ।")
    # Immediate disqualification
    orchestrator.process_turn("क   ष    भ   म    ")
    assert orchestrator.state.stage == Stage.DISQUALIFIED

    # Customer speaks again after terminal outcome
    post_terminal_turn = orchestrator.process_turn("क्या आप मुझे कुछ और बता सकते हैं?")
    assert post_terminal_turn.action == "end"
    assert orchestrator.state.stage == Stage.DISQUALIFIED
    assert "नमस्ते" in post_terminal_turn.agent_reply
    # Verify it does NOT restart greeting or say 'Vikram Singh' verification
    assert "विक्रम" not in post_terminal_turn.agent_reply


def test_bolna_webhook_terminal_session_does_not_restart_greeting():
    """Bolna webhook preserves terminal session state and enforces end_call=True."""
    session_mgr = SessionManager()
    bolna = BolnaIntegration(session_manager=session_mgr)
    call_id = "bolna-call-term-999"

    # Step 1: Start call
    bolna.process_webhook({
        "event": "call_started",
        "execution_id": call_id,
        "phone_number": "9876543210",
    })

    # Step 2: Verification
    bolna.process_webhook({
        "execution_id": call_id,
        "user_message": "हाँ, बात कर सकते हैं।",
    })

    # Step 3: Agricultural disqualification
    resp3, code3 = bolna.process_webhook({
        "execution_id": call_id,
        "user_message": "क   ष    भ   म    ",
    })
    assert code3 == 200
    assert resp3["action"] == "end"
    assert resp3["end_call"] is True
    assert resp3["stage"] == "disqualified"

    # Step 4: Late utterance arrives from Bolna
    resp4, code4 = bolna.process_webhook({
        "execution_id": call_id,
        "user_message": "हेलो?",
    })
    assert code4 == 200
    assert resp4["action"] == "end"
    assert resp4["end_call"] is True
    # Crucial: Must NOT re-greet or transition back to verification
    assert resp4["stage"] == "disqualified"
    assert "नमस्ते" in resp4["agent_response"]


def test_faq_head_office_address_does_not_mutate_property_type():
    """Unrelated question containing 'office' must not change property_type or missing_fields."""
    extracted = normalize_utterance_extraction("What is your head office address?")
    assert "property_type" not in extracted

    orchestrator = TurnOrchestrator(
        customer_id="HC-OFFICE-001",
        customer_name="Priya Sharma",
        language=Language.ENGLISH,
    )
    # Complete verification
    orchestrator.process_turn("Yes, speaking.")
    assert orchestrator.state.stage == Stage.QUALIFICATION

    # Collect residential property
    turn_prop = orchestrator.process_turn("It is a residential house.")
    assert orchestrator.state.qualification.property_type == PropertyType.RESIDENTIAL
    orig_missing = list(orchestrator.state.missing_fields)

    # Ask FAQ about head office address
    turn_faq = orchestrator.process_turn("What is your head office address?")
    assert orchestrator.state.qualification.property_type == PropertyType.RESIDENTIAL
    assert orchestrator.state.missing_fields == orig_missing
