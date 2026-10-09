"""
CredVoice — Phase 7 End-to-End Test Harness & 20 Scenario Suite (tests/e2e/test_e2e_scenarios.py).

Simulates the complete pipeline:
  Incoming Webhook -> Provider Integration -> Session Creation -> Transcript Processing ->
  LLM Extraction -> Deterministic Eligibility Evaluation -> Spoken Response -> Terminal Outcome.

Uses mocked provider events and StubLLMAdapter responses by default — zero network cost.
"""

import json
import pytest
from fastapi.testclient import TestClient

from app.agent.llm import StubLLMAdapter, LLMProviderError
from app.config import settings
from app.main import app
from app.models.conversation import CallOutcome, Stage
from app.services.session_manager import get_session_manager

client = TestClient(app)


# ---------------------------------------------------------------------------
# E2E Test Harness Helper
# ---------------------------------------------------------------------------

class E2ETestHarness:
    """Helper for executing multi-turn E2E call simulations through webhooks."""

    def __init__(self, provider: str = "retell", call_id: str = "E2E-CALL-01") -> None:
        self.provider = provider
        self.call_id = call_id
        self.session_mgr = get_session_manager()

    def start_call(self, customer_name: str = "Priya Sharma", customer_number: str = "+919876543210") -> dict:
        """Simulate call_started webhook event."""
        if self.provider == "retell":
            payload = {
                "event": "call_started",
                "call": {"call_id": self.call_id, "customer_number": customer_number}
            }
            res = client.post("/api/v1/webhooks/retell", json=payload)
        else:
            payload = {
                "event": "execution_started",
                "execution_id": self.call_id,
                "phone_number": customer_number
            }
            res = client.post("/api/v1/webhooks/bolna", json=payload)

        assert res.status_code == 200
        session = self.session_mgr.get_session(self.call_id)
        if session:
            session.orchestrator.state.verification.customer_name = customer_name
        return res.json()

    def set_llm_replies(self, replies: list[str]) -> None:
        """Inject pre-canned LLM replies into the call session."""
        session = self.session_mgr.get_session(self.call_id)
        assert session is not None, f"Session {self.call_id} must be started before setting LLM replies."
        session.orchestrator._llm = StubLLMAdapter(replies=replies)

    def send_turn(self, utterance: str, event_id: str = None, response_id: int = 1) -> dict:
        """Send a speech turn through the provider webhook."""
        if self.provider == "retell":
            payload = {
                "event": "user_transcript",
                "call_id": self.call_id,
                "user_transcript": utterance,
                "response_id": response_id,
            }
            if event_id:
                payload["event_id"] = event_id
            res = client.post("/api/v1/webhooks/retell", json=payload)
        else:
            payload = {
                "event": "user_transcript",
                "execution_id": self.call_id,
                "user_message": utterance,
            }
            if event_id:
                payload["event_id"] = event_id
            res = client.post("/api/v1/webhooks/bolna", json=payload)

        assert res.status_code == 200
        return res.json()

    def end_call(self) -> dict:
        """Simulate call_ended webhook event."""
        if self.provider == "retell":
            res = client.post("/api/v1/webhooks/retell", json={"event": "call_ended", "call": {"call_id": self.call_id}})
        else:
            res = client.post("/api/v1/webhooks/bolna", json={"event": "completed", "execution_id": self.call_id})
        assert res.status_code == 200
        return res.json()

    def get_session(self):
        return self.session_mgr.get_session(self.call_id)


# ---------------------------------------------------------------------------
# Scenario 1: Successful Verification & Full Qualification Pass
# ---------------------------------------------------------------------------

def test_e2e_01_eligible_qualification():
    harness = E2ETestHarness(provider="retell", call_id="E2E-01")
    harness.start_call(customer_name="Priya Sharma")

    harness.set_llm_replies([
        'Hello Priya! ```json {"is_identity_confirmed": true, "is_available": true} ```',
        'Got it. ```json {"property_type": "residential", "ownership": "sole", "has_original_documents": true, "requested_loan_amount_inr": 5000000, "occupation_type": "salaried", "income_mode": "bank", "property_market_value_inr": 12000000, "requested_tenure_years": 10} ```'
    ])

    t1 = harness.send_turn("Yes I am Priya Sharma, I can speak now.")
    assert t1["action"] == "continue"

    t2 = harness.send_turn("I have a residential property, sole ownership, original docs, want 50L loan, salaried bank income, value 1.2 crore, 10 years tenure.")
    assert t2["action"] == "handoff"
    assert t2["stage"] == Stage.ELIGIBLE_HANDOFF.value
    assert t2["end_call"] is True


# ---------------------------------------------------------------------------
# Scenario 2: Agricultural Property Disqualification
# ---------------------------------------------------------------------------

def test_e2e_02_agricultural_property_disqualification():
    harness = E2ETestHarness(provider="bolna", call_id="E2E-02")
    harness.start_call(customer_name="Ramesh")

    harness.set_llm_replies([
        'Hi Ramesh! ```json {"is_identity_confirmed": true, "is_available": true} ```',
        'Agricultural property is not eligible. ```json {"property_type": "agricultural"} ```'
    ])

    harness.send_turn("Yes speaking")
    t2 = harness.send_turn("It is agricultural land in Punjab.")

    assert t2["action"] == "end"
    assert t2["stage"] == Stage.DISQUALIFIED.value
    assert t2["end_call"] is True


# ---------------------------------------------------------------------------
# Scenario 3: Missing Original Documents Disqualification
# ---------------------------------------------------------------------------

def test_e2e_03_missing_original_documents():
    harness = E2ETestHarness(provider="retell", call_id="E2E-03")
    harness.start_call()

    harness.set_llm_replies([
        'Hi! ```json {"is_identity_confirmed": true, "is_available": true} ```',
        'Original docs needed. ```json {"property_type": "residential", "has_original_documents": false} ```'
    ])

    harness.send_turn("Yes available")
    t2 = harness.send_turn("I only have photocopies of my property papers, no original deed.")

    assert t2["action"] == "end"
    assert t2["stage"] == Stage.DISQUALIFIED.value


# ---------------------------------------------------------------------------
# Scenario 4: Cash-Only Income Disqualification
# ---------------------------------------------------------------------------

def test_e2e_04_cash_only_income():
    harness = E2ETestHarness(provider="bolna", call_id="E2E-04")
    harness.start_call()

    harness.set_llm_replies([
        'Hi! ```json {"is_identity_confirmed": true, "is_available": true} ```',
        'Cash income not eligible. ```json {"property_type": "commercial", "income_mode": "cash"} ```'
    ])

    harness.send_turn("Yes speaking")
    t2 = harness.send_turn("I get my income in cash only every month.")

    assert t2["action"] == "end"
    assert t2["stage"] == Stage.DISQUALIFIED.value


# ---------------------------------------------------------------------------
# Scenario 5: Loan Amount > ₹75 Lakh — Consent Accepted
# ---------------------------------------------------------------------------

def test_e2e_05_amount_above_75l_consent_accepted():
    harness = E2ETestHarness(provider="retell", call_id="E2E-05")
    harness.start_call()

    harness.set_llm_replies([
        'Hi! ```json {"is_identity_confirmed": true, "is_available": true} ```',
        'Our cap is 75 lakh. ```json {"requested_loan_amount_inr": 10000000} ```',
        'Great! ```json {"proceeds_at_cap": true, "property_type": "residential", "ownership": "sole", "has_original_documents": true, "occupation_type": "salaried", "income_mode": "bank", "property_market_value_inr": 20000000, "requested_tenure_years": 12} ```'
    ])

    harness.send_turn("Yes available")
    t2 = harness.send_turn("I need 1 crore rupees loan.")
    assert t2["action"] == "continue"

    t3 = harness.send_turn("Yes I agree to proceed at 75 lakh offer cap.")
    assert t3["action"] == "handoff"
    assert t3["stage"] == Stage.ELIGIBLE_HANDOFF.value


# ---------------------------------------------------------------------------
# Scenario 6: Loan Amount > ₹75 Lakh — Consent Declined
# ---------------------------------------------------------------------------

def test_e2e_06_amount_above_75l_consent_declined():
    harness = E2ETestHarness(provider="retell", call_id="E2E-06")
    harness.start_call()

    harness.set_llm_replies([
        'Hi! ```json {"is_identity_confirmed": true, "is_available": true} ```',
        'Our cap is 75 lakh. ```json {"requested_loan_amount_inr": 10000000} ```',
        'Sorry to hear that. ```json {"proceeds_at_cap": false} ```'
    ])

    harness.send_turn("Yes available")
    harness.send_turn("I need 1 crore loan")
    t3 = harness.send_turn("No 75 lakh is too small for me, I decline.")

    assert t3["action"] == "end"
    assert t3["stage"] == Stage.DISQUALIFIED.value


# ---------------------------------------------------------------------------
# Scenario 7: Tenure Outside Allowed Range
# ---------------------------------------------------------------------------

def test_e2e_07_tenure_outside_range():
    harness = E2ETestHarness(provider="bolna", call_id="E2E-07")
    harness.start_call()

    harness.set_llm_replies([
        'Hi! ```json {"is_identity_confirmed": true, "is_available": true} ```',
        'Tenure must be 3 to 15 years. ```json {"requested_tenure_years": 2} ```'
    ])

    harness.send_turn("Yes speaking")
    t2 = harness.send_turn("I want 2 years tenure only.")

    assert t2["action"] == "end"
    assert t2["stage"] == Stage.DISQUALIFIED.value


# ---------------------------------------------------------------------------
# Scenario 8: Existing Property Loan Transfer Routing
# ---------------------------------------------------------------------------

def test_e2e_08_existing_loan_transfer_routing():
    harness = E2ETestHarness(provider="retell", call_id="E2E-08")
    harness.start_call()

    harness.set_llm_replies([
        'Connecting you to transfer specialist. ```json {"is_identity_confirmed": true, "is_available": true, "has_existing_property_loan": true} ```'
    ])

    t1 = harness.send_turn("I already have a LAP loan with HDFC bank.")
    assert t1["action"] == "transfer"
    assert t1["stage"] == Stage.TRANSFER_ROUTED.value
    assert t1["end_call"] is True


# ---------------------------------------------------------------------------
# Scenario 9: EMI Reduction Request Routing
# ---------------------------------------------------------------------------

def test_e2e_09_emi_reduction_routing():
    harness = E2ETestHarness(provider="bolna", call_id="E2E-09")
    harness.start_call()

    harness.set_llm_replies([
        'Connecting you to transfer specialist. ```json {"is_identity_confirmed": true, "is_available": true, "wants_emi_reduction": true} ```'
    ])

    t1 = harness.send_turn("Can you help me reduce my current monthly EMI?")
    assert t1["action"] == "transfer"
    assert t1["stage"] == Stage.TRANSFER_ROUTED.value


# ---------------------------------------------------------------------------
# Scenario 10: Out-of-Order Multi-Field Turn
# ---------------------------------------------------------------------------

def test_e2e_10_out_of_order_multi_field():
    harness = E2ETestHarness(provider="retell", call_id="E2E-10")
    harness.start_call()

    harness.set_llm_replies([
        'Hi! ```json {"is_identity_confirmed": true, "is_available": true} ```',
        'Got those fields! ```json {"property_type": "industrial", "ownership": "joint", "has_original_documents": true, "requested_loan_amount_inr": 4000000} ```'
    ])

    harness.send_turn("Yes speaking")
    t2 = harness.send_turn("I have industrial joint property, original papers, and want 40L.")
    session = harness.get_session()

    assert session.orchestrator.state.qualification.property_type.value == "industrial"
    assert session.orchestrator.state.qualification.ownership.value == "joint"
    assert session.orchestrator.state.qualification.has_original_documents is True
    assert session.orchestrator.state.qualification.requested_loan_amount_inr == 4000000.0


# ---------------------------------------------------------------------------
# Scenario 11: Answer Correction
# ---------------------------------------------------------------------------

def test_e2e_11_answer_correction():
    harness = E2ETestHarness(provider="retell", call_id="E2E-11")
    harness.start_call()

    harness.set_llm_replies([
        'Hi! ```json {"is_identity_confirmed": true, "is_available": true} ```',
        'Recorded salaried. ```json {"property_type": "residential", "occupation_type": "salaried"} ```',
        'Updated to self-employed! ```json {"occupation_type": "self_employed", "income_mode": "bank"} ```'
    ])

    harness.send_turn("Yes available")
    harness.send_turn("I am salaried.")
    t3 = harness.send_turn("Actually sorry, I am self-employed and my income is deposited directly into my bank account.")

    session = harness.get_session()
    assert session.orchestrator.state.qualification.occupation_type.value == "self_employed"
    assert session.orchestrator.state.qualification.income_mode.value == "bank"


# ---------------------------------------------------------------------------
# Scenario 12: FAQ Question During Qualification
# ---------------------------------------------------------------------------

def test_e2e_12_faq_question_during_qualification():
    harness = E2ETestHarness(provider="retell", call_id="E2E-12")
    harness.start_call()

    harness.set_llm_replies([
        'Hi! ```json {"is_identity_confirmed": true, "is_available": true} ```',
        'Answered FAQ. ```json {"faq_answered": true} ```'
    ])

    harness.send_turn("Yes speaking")
    t2 = harness.send_turn("What properties can I use for this loan?")
    assert t2["action"] == "continue"
    session = harness.get_session()
    assert session.orchestrator.state.stage == Stage.QUALIFICATION


# ---------------------------------------------------------------------------
# Scenario 13: Callback Requested
# ---------------------------------------------------------------------------

def test_e2e_13_callback_requested():
    harness = E2ETestHarness(provider="bolna", call_id="E2E-13")
    harness.start_call()

    harness.set_llm_replies([
        'Scheduling callback. ```json {"is_identity_confirmed": true, "is_available": false, "callback_preferred_time": "tomorrow 3pm"} ```'
    ])

    t1 = harness.send_turn("I am driving right now, please call me back tomorrow at 3pm.")
    assert t1["action"] == "callback"
    assert t1["stage"] == Stage.CALLBACK_REQUESTED.value


# ---------------------------------------------------------------------------
# Scenario 14: Wrong Person Ends Call
# ---------------------------------------------------------------------------

def test_e2e_14_wrong_person_ends_call():
    harness = E2ETestHarness(provider="retell", call_id="E2E-14")
    harness.start_call()

    harness.set_llm_replies([
        'Sorry, wrong number. Goodbye. ```json {"is_identity_confirmed": false} ```'
    ])

    t1 = harness.send_turn("No, there is no Priya here. Wrong number.")
    assert t1["action"] == "end"
    assert t1["stage"] == Stage.ENDED.value


# ---------------------------------------------------------------------------
# Scenario 15: Bilingual English and Hindi Conversation Flows
# ---------------------------------------------------------------------------

def test_e2e_15_hindi_conversation_flow():
    harness = E2ETestHarness(provider="retell", call_id="E2E-15")
    harness.start_call()

    harness.set_llm_replies([
        'नमस्ते! ```json {"is_identity_confirmed": true, "is_available": true, "language_switch": "hi"} ```'
    ])

    t1 = harness.send_turn("हाँ मैं बात कर सकता हूँ, हिंदी में बात करें।")
    assert t1["action"] == "continue"
    session = harness.get_session()
    assert session.orchestrator.state.language.value == "hi"


# ---------------------------------------------------------------------------
# Scenario 16: Duplicate Webhook Delivery Idempotency
# ---------------------------------------------------------------------------

def test_e2e_16_duplicate_webhook_delivery():
    harness = E2ETestHarness(provider="retell", call_id="E2E-16")
    harness.start_call()

    harness.set_llm_replies([
        'Hi! ```json {"is_identity_confirmed": true, "is_available": true} ```'
    ])

    t1 = harness.send_turn("Yes speaking", event_id="DUPE-EVENT-100")
    t2 = harness.send_turn("Yes speaking", event_id="DUPE-EVENT-100")

    session = harness.get_session()
    assert session.orchestrator.turn_count == 1
    assert "listening" in t2["content"].lower()


# ---------------------------------------------------------------------------
# Scenario 17: Invalid Signature & Unauthorized Admin Requests
# ---------------------------------------------------------------------------

def test_e2e_17_invalid_signature_and_unauthorized_admin(monkeypatch):
    """Verify invalid webhook signatures and unauthorized admin calls return 401."""
    monkeypatch.setattr(settings, "verify_webhook_signatures", True)
    monkeypatch.setattr(settings, "retell_webhook_secret", "secret_key")
    monkeypatch.setattr(settings, "admin_api_key", "admin_pass")

    # Invalid Retell signature -> 401
    res_sig = client.post("/api/v1/webhooks/retell", json={"event": "call_started", "call": {"call_id": "SIG-TEST"}}, headers={"x-retell-signature": "bad_sig"})
    assert res_sig.status_code == 401

    # Unauthorized Call inspection -> 401
    res_admin = client.get("/api/v1/calls/E2E-01")
    assert res_admin.status_code == 401


# ---------------------------------------------------------------------------
# Scenario 18: LLM Timeout & Provider Failure Recovery
# ---------------------------------------------------------------------------

def test_e2e_18_llm_failure_recovery():
    harness = E2ETestHarness(provider="retell", call_id="E2E-18")
    harness.start_call()

    # Inject failing LLM adapter
    class FailingAdapter:
        def chat(self, *args, **kwargs):
            raise LLMProviderError("Provider timeout")

    session = harness.get_session()
    session.orchestrator._llm = FailingAdapter()

    t1 = harness.send_turn("Hello, can you hear me?")
    assert t1["action"] == "continue"
    assert "technical difficulty" in t1["content"].lower()


# ---------------------------------------------------------------------------
# Scenario 19: Terminal Call Protection
# ---------------------------------------------------------------------------

def test_e2e_19_terminal_call_protected():
    harness = E2ETestHarness(provider="retell", call_id="E2E-19")
    harness.start_call()

    harness.set_llm_replies([
        'Agri is disqualified. ```json {"is_identity_confirmed": true, "is_available": true, "property_type": "agricultural"} ```'
    ])

    t1 = harness.send_turn("Yes, it is agricultural land.")
    assert t1["action"] == "end"
    assert t1["stage"] == Stage.DISQUALIFIED.value

    # Subsequent turn after terminal stage
    t2 = harness.send_turn("Wait, what about my loan?")
    assert t2["action"] == "end"
    assert "goodbye" in t2["content"].lower()


# ---------------------------------------------------------------------------
# Scenario 20: Simultaneous Calls State Isolation
# ---------------------------------------------------------------------------

def test_e2e_20_simultaneous_calls_state_isolation():
    h1 = E2ETestHarness(provider="retell", call_id="SIMUL-CALL-A")
    h2 = E2ETestHarness(provider="bolna", call_id="SIMUL-CALL-B")

    h1.start_call(customer_name="Customer A")
    h2.start_call(customer_name="Customer B")

    h1.set_llm_replies(['Hello A ```json {"is_identity_confirmed": true, "is_available": true, "property_type": "residential"} ```'])
    h2.set_llm_replies(['Hello B ```json {"is_identity_confirmed": true, "is_available": true, "property_type": "commercial"} ```'])

    h1.send_turn("I have residential property")
    h2.send_turn("I have commercial property")

    s1 = h1.get_session()
    s2 = h2.get_session()

    assert s1.orchestrator.state.qualification.property_type.value == "residential"
    assert s2.orchestrator.state.qualification.property_type.value == "commercial"
    assert s1.orchestrator.state.session_id != s2.orchestrator.state.session_id
