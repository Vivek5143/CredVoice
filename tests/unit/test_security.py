"""
CredVoice — Security & Reliability Unit Tests (tests/unit/test_security.py).

Verifies PII redaction, fail-closed signature verification in production,
unauthorized admin endpoint protection, request body size limits, and Redis fallback.
"""

import json
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.integrations.bolna import verify_bolna_signature
from app.integrations.retell import verify_retell_signature
from app.main import app
from app.services.session_manager import RedisSessionStore
from app.utils.logging import redact_pii

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: PII Redaction Utility
# ---------------------------------------------------------------------------

def test_redact_pii_sanitization():
    """Verify phone numbers, emails, and API keys are redacted from log outputs."""
    raw_log = "Customer Priya Sharma with phone +919876543210 and email priya@example.com using key sk-proj-1234567890abcdef1234."
    redacted = redact_pii(raw_log)

    assert "+919876543210" not in redacted
    assert "priya@example.com" not in redacted
    assert "sk-proj-1234567890abcdef1234" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_SECRET]" in redacted


# ---------------------------------------------------------------------------
# Test 2: Fail-Closed Webhook Signature in Production
# ---------------------------------------------------------------------------

def test_retell_signature_fails_closed_in_production(monkeypatch):
    """Retell webhook must reject missing/unconfigured secret requests in production."""
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "retell_webhook_secret", "")

    # Missing signature -> False (fails closed)
    assert verify_retell_signature(b"{}", signature_header=None) is False
    assert verify_retell_signature(b"{}", signature_header="sig123") is False


def test_bolna_signature_fails_closed_in_production(monkeypatch):
    """Bolna webhook must reject missing/unconfigured secret requests in production."""
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "bolna_webhook_secret", "")

    # Missing signature -> False (fails closed)
    assert verify_bolna_signature(b"{}", signature_header=None) is False
    assert verify_bolna_signature(b"{}", signature_header="sig123") is False


# ---------------------------------------------------------------------------
# Test 3: Unauthorized Admin Call Inspection & Deletion Protection
# ---------------------------------------------------------------------------

def test_admin_call_endpoints_auth_required(monkeypatch):
    """Admin key protection rejects unauthorized calls API requests."""
    monkeypatch.setattr(settings, "admin_api_key", "secret_admin_key_789")

    # Unauthorized GET -> 401
    res_get = client.get("/api/v1/calls/TEST-CALL-01")
    assert res_get.status_code == 401
    assert "Invalid or missing admin API key" in res_get.json()["detail"]

    # Unauthorized DELETE -> 401
    res_del = client.delete("/api/v1/calls/TEST-CALL-01")
    assert res_del.status_code == 401

    # Authorized GET with valid header -> 404 (because call doesn't exist, but auth passed!)
    res_auth = client.get("/api/v1/calls/TEST-CALL-01", headers={"x-api-key": "secret_admin_key_789"})
    assert res_auth.status_code == 404


# ---------------------------------------------------------------------------
# Test 4: Payload Body Size Limit Middleware (413 Payload Too Large)
# ---------------------------------------------------------------------------

def test_request_body_size_limit():
    """Overly large payload (> 1MB) must be rejected with 413 Payload Too Large."""
    large_payload = "x" * (1048576 + 500)  # > 1MB
    res = client.post(
        "/api/v1/webhooks/turn",
        content=large_payload,
        headers={"Content-Type": "application/json", "Content-Length": str(len(large_payload))},
    )
    assert res.status_code == 413
    assert "exceeds maximum allowed size" in res.json()["detail"]


# ---------------------------------------------------------------------------
# Test 5: Redis Session Store Fallback
# ---------------------------------------------------------------------------

def test_redis_session_store_graceful_fallback():
    """RedisSessionStore falls back to InMemorySessionStore when Redis is unavailable."""
    store = RedisSessionStore(redis_url="redis://localhost:9999/0")  # Invalid Redis port
    assert store._client is None  # Connection failed gracefully

    # Store operations continue working via in-memory fallback
    from app.agent.core import create_orchestrator
    from app.services.session_manager import CallSession
    orch = create_orchestrator("HC-TEST", "Ramesh", "en")
    sess = CallSession("CALL-REDIS-FALLBACK", orch)

    store.save(sess)
    retrieved = store.get("CALL-REDIS-FALLBACK")
    assert retrieved is not None
    assert retrieved.call_id == "CALL-REDIS-FALLBACK"


def test_redis_session_store_fails_closed_in_production(monkeypatch):
    """RedisSessionStore raises RuntimeError in production mode when Redis is unreachable."""
    monkeypatch.setattr(settings, "app_env", "production")
    with pytest.raises(RuntimeError, match="Production Redis connection failed"):
        RedisSessionStore(redis_url="redis://localhost:9999/0")

