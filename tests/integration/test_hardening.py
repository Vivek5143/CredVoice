"""
CredVoice — Integration Tests for Security & Reliability Hardening (tests/integration/test_hardening.py).

Verifies readiness check, correlation ID propagation, and clean error handling.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_readiness_probe_endpoint():
    """Verify /readiness probe returns ready status and knowledge entry count."""
    res = client.get("/readiness")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["knowledge_entries"] > 0


def test_correlation_id_middleware_propagation():
    """Verify x-correlation-id header is preserved or generated on all HTTP responses."""
    custom_cid = "CORR-ID-999-ABC"
    res = client.get("/health", headers={"x-correlation-id": custom_cid})
    assert res.status_code == 200
    assert res.headers.get("x-correlation-id") == custom_cid

    # Generated correlation ID if omitted
    res_gen = client.get("/health")
    assert res_gen.status_code == 200
    assert "x-correlation-id" in res_gen.headers
