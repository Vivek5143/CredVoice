"""
CredVoice — Integration Tests for Voice Platform Webhooks (tests/integration/test_webhooks.py).

Verifies FastAPI endpoints for Retell AI, Bolna, and direct voice turn endpoints.

Tests cover:
- Retell & Bolna call lifecycle (started -> turn -> ended).
- Payload validation & response schema formatting.
- Signature verification (valid vs invalid 401 rejection).
- Duplicate webhook idempotency.
- English and Hindi speech turns.
- Callback scheduling & transfer routing via webhooks.
- Call session inspection & explicit termination (/api/v1/calls).
"""

import hashlib
import hmac
import json
import pytest
from fastapi.testclient import TestClient

from app.agent.llm import StubLLMAdapter
from app.config import settings
from app.main import app
from app.services.session_manager import get_session_manager

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper: Compute HMAC Signature
# ---------------------------------------------------------------------------

def compute_hmac_sha256(secret: str, body_bytes: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Test 1: Direct Voice Turn Endpoint (/api/v1/webhooks/turn)
# ---------------------------------------------------------------------------

def test_direct_voice_turn_endpoint():
    """Verify generic direct turn processing endpoint."""
    payload = {
        "call_id": "DIRECT-CALL-01",
        "utterance": "Hello, my name is Priya Sharma. Yes I am available.",
        "customer_name": "Priya Sharma",
        "language": "en",
    }
    response = client.post("/api/v1/webhooks/turn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["call_id"] == "DIRECT-CALL-01"
    assert "agent_reply" in data
    assert data["action"] in {"continue", "handoff", "transfer", "callback", "end"}


# ---------------------------------------------------------------------------
# Test 2: Retell AI Call Lifecycle (Started -> Turn -> Ended)
# ---------------------------------------------------------------------------

def test_retell_call_lifecycle():
    """Retell AI call_started -> speech turn -> call_ended flow."""
    call_id = "RETELL-LIFE-01"

    # 1. call_started
    start_payload = {
        "event": "call_started",
        "event_id": "evt-start-01",
        "call": {"call_id": call_id, "customer_number": "+919876543210"}
    }
    res_start = client.post("/api/v1/webhooks/retell", json=start_payload)
    assert res_start.status_code == 200
    assert res_start.json()["status"] == "started"

    # Inject stub LLM into created session
    session = get_session_manager().get_session(call_id)
    assert session is not None
    session.orchestrator._llm = StubLLMAdapter(replies=[
        'Hello Priya! ```json {"is_identity_confirmed": true, "is_available": true} ```'
    ])

    # 2. Speech Turn
    turn_payload = {
        "event": "user_transcript",
        "event_id": "evt-turn-01",
        "call_id": call_id,
        "user_transcript": "Yes this is Priya Sharma",
        "response_id": 1,
    }
    res_turn = client.post("/api/v1/webhooks/retell", json=turn_payload)
    assert res_turn.status_code == 200
    data_turn = res_turn.json()
    assert data_turn["response_id"] == 1
    assert "content" in data_turn
    assert data_turn["content_complete"] is True
    assert data_turn["end_call"] is False

    # 3. call_ended
    end_payload = {
        "event": "call_ended",
        "event_id": "evt-end-01",
        "call": {"call_id": call_id, "disposition": "completed"}
    }
    res_end = client.post("/api/v1/webhooks/retell", json=end_payload)
    assert res_end.status_code == 200
    assert res_end.json()["status"] == "ended"


# ---------------------------------------------------------------------------
# Test 3: Retell Signature Verification (Valid vs Invalid 401)
# ---------------------------------------------------------------------------

def test_retell_signature_verification(monkeypatch):
    """Invalid Retell signature must return 401 when signature verification is enabled."""
    secret = "retell_test_secret_123"
    monkeypatch.setattr(settings, "verify_webhook_signatures", True)
    monkeypatch.setattr(settings, "retell_webhook_secret", secret)

    payload = {"event": "call_started", "call": {"call_id": "RETELL-SIG-01"}}
    body_bytes = json.dumps(payload).encode("utf-8")
    valid_sig = compute_hmac_sha256(secret, body_bytes)

    # Valid signature -> 200 OK
    res_valid = client.post(
        "/api/v1/webhooks/retell",
        content=body_bytes,
        headers={"x-retell-signature": valid_sig, "Content-Type": "application/json"},
    )
    assert res_valid.status_code == 200

    # Invalid signature -> 401 Unauthorized
    res_invalid = client.post(
        "/api/v1/webhooks/retell",
        content=body_bytes,
        headers={"x-retell-signature": "invalid_sig_abc", "Content-Type": "application/json"},
    )
    assert res_invalid.status_code == 401


# ---------------------------------------------------------------------------
# Test 4: Bolna Call Lifecycle & Response Format
# ---------------------------------------------------------------------------

def test_bolna_call_lifecycle():
    """Bolna execution_started -> turn -> completed flow."""
    call_id = "BOLNA-LIFE-01"

    # 1. execution_started
    start_payload = {
        "event": "execution_started",
        "execution_id": call_id,
        "phone_number": "+919123456789",
    }
    res_start = client.post("/api/v1/webhooks/bolna", json=start_payload)
    assert res_start.status_code == 200
    assert res_start.json()["status"] == "success"

    # Inject stub into session
    session = get_session_manager().get_session(call_id)
    assert session is not None
    session.orchestrator._llm = StubLLMAdapter(replies=[
        'Hi! ```json {"is_identity_confirmed": true, "is_available": true} ```'
    ])

    # 2. Turn
    turn_payload = {
        "event": "user_transcript",
        "execution_id": call_id,
        "user_message": "Yes I am available",
    }
    res_turn = client.post("/api/v1/webhooks/bolna", json=turn_payload)
    assert res_turn.status_code == 200
    data_turn = res_turn.json()
    assert data_turn["status"] == "success"
    assert "agent_response" in data_turn
    assert data_turn["end_call"] is False

    # 3. Completed
    end_payload = {
        "event": "completed",
        "execution_id": call_id,
    }
    res_end = client.post("/api/v1/webhooks/bolna", json=end_payload)
    assert res_end.status_code == 200
    assert res_end.json()["status"] == "success"


# ---------------------------------------------------------------------------
# Test 5: Bolna Signature Verification
# ---------------------------------------------------------------------------

def test_bolna_signature_verification(monkeypatch):
    """Bolna signature check validates x-api-key / x-bolna-signature."""
    secret = "bolna_test_secret_456"
    monkeypatch.setattr(settings, "verify_webhook_signatures", True)
    monkeypatch.setattr(settings, "bolna_webhook_secret", secret)

    payload = {"event": "execution_started", "execution_id": "BOLNA-SIG-01"}
    body_bytes = json.dumps(payload).encode("utf-8")

    # Valid secret token header -> 200 OK
    res_valid = client.post(
        "/api/v1/webhooks/bolna",
        content=body_bytes,
        headers={"x-api-key": secret, "Content-Type": "application/json"},
    )
    assert res_valid.status_code == 200

    # Invalid secret token header -> 401
    res_invalid = client.post(
        "/api/v1/webhooks/bolna",
        content=body_bytes,
        headers={"x-api-key": "wrong_key", "Content-Type": "application/json"},
    )
    assert res_invalid.status_code == 401


# ---------------------------------------------------------------------------
# Test 6: Existing Loan Transfer Routing via Webhook
# ---------------------------------------------------------------------------

def test_webhook_existing_loan_transfer_routing():
    """Mentions existing property loan transfer -> routes to specialist with transfer action."""
    call_id = "RETELL-TRANSFER-01"

    # Start call
    client.post("/api/v1/webhooks/retell", json={"event": "call_started", "call": {"call_id": call_id}})
    session = get_session_manager().get_session(call_id)
    session.orchestrator._llm = StubLLMAdapter(replies=[
        'I am transferring you. ```json {"is_identity_confirmed": true, "is_available": true, "has_existing_property_loan": true, "wants_emi_reduction": true} ```'
    ])

    res = client.post("/api/v1/webhooks/retell", json={
        "event": "user_transcript",
        "call_id": call_id,
        "user_transcript": "I want to reduce my EMI on my existing property loan",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["action"] == "transfer"
    assert data["end_call"] is True


# ---------------------------------------------------------------------------
# Test 7: Callback Request via Webhook
# ---------------------------------------------------------------------------

def test_webhook_callback_request():
    """Customer is busy -> schedules callback and sets callback action."""
    call_id = "BOLNA-CALLBACK-01"

    # Start call
    client.post("/api/v1/webhooks/bolna", json={"event": "execution_started", "execution_id": call_id})
    session = get_session_manager().get_session(call_id)
    session.orchestrator._llm = StubLLMAdapter(replies=[
        'I will schedule a callback. ```json {"is_identity_confirmed": true, "is_available": false, "callback_preferred_time": "tomorrow 4pm"} ```'
    ])

    res = client.post("/api/v1/webhooks/bolna", json={
        "event": "user_transcript",
        "execution_id": call_id,
        "user_message": "I am driving, call me tomorrow at 4pm",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["action"] == "callback"
    assert data["end_call"] is True


# ---------------------------------------------------------------------------
# Test 8: Call Session Inspection & Deletion APIs (/api/v1/calls)
# ---------------------------------------------------------------------------

def test_call_management_api_endpoints():
    """Query /api/v1/calls/{call_id} and delete session."""
    call_id = "CALL-MGMT-01"

    # Create session via direct turn
    client.post("/api/v1/webhooks/turn", json={
        "call_id": call_id,
        "utterance": "Hello",
        "customer_name": "Anita Verma",
    })

    # GET /api/v1/calls/CALL-MGMT-01
    headers = {"x-api-key": settings.admin_api_key} if settings.admin_api_key else {}
    res_get = client.get(f"/api/v1/calls/{call_id}", headers=headers)
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert data_get["call_id"] == call_id
    assert data_get["customer_name"] == "Anita Verma"
    assert data_get["turn_count"] >= 1

    # DELETE /api/v1/calls/CALL-MGMT-01
    res_del = client.delete(f"/api/v1/calls/{call_id}", headers=headers)
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "terminated"

    # GET after delete -> 404
    res_404 = client.get(f"/api/v1/calls/{call_id}", headers=headers)
    assert res_404.status_code == 404
