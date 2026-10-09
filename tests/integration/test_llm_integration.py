"""
CredVoice — Live Integration Test for Production LLM Adapter (tests/integration/test_llm_integration.py).

This test connects to the real OpenAI API if a valid OPENAI_API_KEY is present
in the environment or .env file. Otherwise, it is automatically skipped.

To run this test:
  $ set OPENAI_API_KEY=sk-...
  $ python -m pytest tests/integration/test_llm_integration.py -v
"""

import os
import pytest
from app.agent.llm import OpenAILLMAdapter, LLMError

# Determine if a real OpenAI API key and explicit opt-in flag are present
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
RUN_LIVE_OPT_IN = os.getenv("RUN_LIVE_LLM_TESTS", "false").lower() in ("true", "1", "yes")
IS_REAL_KEY = (
    bool(API_KEY)
    and API_KEY != "your_openai_api_key_here"
    and not API_KEY.startswith("sk-test")
    and RUN_LIVE_OPT_IN
)


@pytest.mark.skipif(
    not IS_REAL_KEY,
    reason="Live integration test requires valid OPENAI_API_KEY and RUN_LIVE_LLM_TESTS=true in environment.",
)
def test_live_openai_llm_adapter_chat():
    """Live call to OpenAI ChatCompletions API — verifies real response structure."""
    adapter = OpenAILLMAdapter(api_key=API_KEY)
    system_prompt = (
        "You are CredVoice, a voice assistant for Home Credit LAP qualification. "
        "Return extraction JSON inside a ```json ``` block."
    )
    user_messages = [
        {"role": "user", "content": "Hello! My name is Priya Sharma. Yes, I am available to talk."}
    ]

    try:
        response_text = adapter.chat(system_prompt=system_prompt, messages=user_messages)
        assert response_text is not None
        assert len(response_text) > 0
        assert "```json" in response_text or "is_identity_confirmed" in response_text or "Priya" in response_text
    except LLMError as exc:
        pytest.fail(f"Live OpenAI API call failed: {exc}")
