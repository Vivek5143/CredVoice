"""
CredVoice — Phase 7 Opt-In Live Sandbox Integration Tests (tests/integration/test_live_sandbox.py).

These tests connect to live LLM providers and/or voice provider webhooks when explicitly enabled.
DISABLED BY DEFAULT to prevent unintended API costs or external network dependencies during standard CI.

To enable:
  $env:RUN_LIVE_SANDBOX_TESTS="true"
  $env:OPENAI_API_KEY="sk-..."
  python -m pytest tests/integration/test_live_sandbox.py -v
"""

import os
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.agent.llm import OpenAILLMAdapter

RUN_LIVE_SANDBOX = os.getenv("RUN_LIVE_SANDBOX_TESTS", "false").lower() in ("true", "1", "yes")

client = TestClient(app)


@pytest.mark.skipif(not RUN_LIVE_SANDBOX, reason="Live sandbox tests disabled by default. Set RUN_LIVE_SANDBOX_TESTS=true to run.")
def test_live_llm_sandbox_call():
    """Verify live LLM extraction against configured OpenAI API endpoint."""
    api_key = settings.openai_api_key.get_secret_value()
    assert api_key and api_key != "mock_key", "Valid OPENAI_API_KEY required for live sandbox test."

    adapter = OpenAILLMAdapter()
    system_prompt = "You are a helpful assistant. Output JSON only."
    messages = [{"role": "user", "content": "I want a 50 lakh loan against residential property."}]

    response = adapter.chat(messages=messages, system_prompt=system_prompt)
    assert response is not None
    assert len(response.text) > 0


@pytest.mark.skipif(not RUN_LIVE_SANDBOX, reason="Live sandbox tests disabled by default. Set RUN_LIVE_SANDBOX_TESTS=true to run.")
def test_live_retell_webhook_sandbox():
    """Verify live Retell webhook endpoint behavior with live signature if configured."""
    res = client.post(
        "/api/v1/webhooks/retell",
        json={"event": "call_started", "call": {"call_id": "SANDBOX-RETELL-01", "customer_number": "+919876543210"}},
    )
    assert res.status_code in (200, 401)
