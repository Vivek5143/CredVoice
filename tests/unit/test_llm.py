"""
CredVoice — Unit Tests for Production LLM Adapter (tests/unit/test_llm.py).

Verifies OpenAILLMAdapter, StubLLMAdapter, custom exception hierarchy, secret masking,
extract_json_from_response robustness, error fallbacks, and factory instantiation.

All tests mock network interactions — zero API calls or live credentials required.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from app.agent.llm import (
    LLMAdapter,
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    OpenAILLMAdapter,
    StubLLMAdapter,
    create_llm_adapter,
    mask_secret,
)
from app.agent.prompt_builder import extract_json_from_response
from app.agent.core import TurnOrchestrator
from app.models.conversation import Language, Stage


# ---------------------------------------------------------------------------
# Test 1: Secret Masking Helper
# ---------------------------------------------------------------------------

def test_mask_secret():
    """Verify secrets are masked safely for logging and error reporting."""
    assert mask_secret(None) == "<EMPTY>"
    assert mask_secret("") == "<EMPTY>"
    assert mask_secret("short") == "****"
    assert mask_secret("sk-proj-1234567890abcdef") == "sk-p...cdef"


# ---------------------------------------------------------------------------
# Test 2: Missing API Key Configuration Error
# ---------------------------------------------------------------------------

def test_openai_adapter_missing_api_key(monkeypatch):
    """OpenAILLMAdapter must raise LLMConfigurationError if key is missing or dummy."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError) as exc_info:
        OpenAILLMAdapter(api_key="")
    assert "OpenAI API key is missing" in str(exc_info.value)

    with pytest.raises(LLMConfigurationError):
        OpenAILLMAdapter(api_key="your_openai_api_key_here")


# ---------------------------------------------------------------------------
# Test 3: Successful OpenAI Response (Mocked)
# ---------------------------------------------------------------------------

@patch("openai.OpenAI")
def test_openai_adapter_successful_chat(mock_openai_class):
    """OpenAILLMAdapter formats messages properly and returns LLM output text."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Sure! ```json {\"is_identity_confirmed\": true} ```"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    adapter = OpenAILLMAdapter(api_key="sk-testkey1234567890")
    result = adapter.chat(
        system_prompt="You are CredVoice.",
        messages=[{"role": "user", "content": "Yes this is Priya Sharma"}],
    )

    assert result == "Sure! ```json {\"is_identity_confirmed\": true} ```"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == adapter.model
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["content"] == "Yes this is Priya Sharma"


# ---------------------------------------------------------------------------
# Test 4: OpenAI Timeout Error Mapping
# ---------------------------------------------------------------------------

@patch("openai.OpenAI")
def test_openai_adapter_timeout_error(mock_openai_class):
    """APITimeoutError must be caught and re-raised as LLMTimeoutError."""
    import openai

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.APITimeoutError(request=MagicMock())
    mock_openai_class.return_value = mock_client

    adapter = OpenAILLMAdapter(api_key="sk-testkey1234567890")
    with pytest.raises(LLMTimeoutError) as exc_info:
        adapter.chat("System prompt", [{"role": "user", "content": "Hi"}])
    assert "timed out" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 5: OpenAI Rate Limit Error Mapping
# ---------------------------------------------------------------------------

@patch("openai.OpenAI")
def test_openai_adapter_rate_limit_error(mock_openai_class):
    """RateLimitError must be caught and re-raised as LLMRateLimitError."""
    import openai

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
        message="Rate limit reached", response=mock_response, body={}
    )
    mock_openai_class.return_value = mock_client

    adapter = OpenAILLMAdapter(api_key="sk-testkey1234567890")
    with pytest.raises(LLMRateLimitError):
        adapter.chat("System prompt", [{"role": "user", "content": "Hi"}])


# ---------------------------------------------------------------------------
# Test 6: OpenAI General Provider Error Mapping
# ---------------------------------------------------------------------------

@patch("openai.OpenAI")
def test_openai_adapter_provider_error(mock_openai_class):
    """APIError / InternalServerError must be caught and re-raised as LLMProviderError."""
    import openai

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.APIError(
        message="Internal Error", request=MagicMock(), body={}
    )
    mock_openai_class.return_value = mock_client

    adapter = OpenAILLMAdapter(api_key="sk-testkey1234567890")
    with pytest.raises(LLMProviderError):
        adapter.chat("System prompt", [{"role": "user", "content": "Hi"}])


# ---------------------------------------------------------------------------
# Test 7: Extraction Robustness — Malformed JSON & Non-Dict Handling
# ---------------------------------------------------------------------------

def test_extract_json_from_response_robustness():
    """Verify JSON extraction handles malformed JSON, strings, arrays, empty inputs gracefully."""
    assert extract_json_from_response("") == {}
    assert extract_json_from_response(None) == {}
    assert extract_json_from_response("Plain text without JSON") == {}

    # Malformed JSON block
    assert extract_json_from_response("Here is JSON: ```json {invalid json} ```") == {}

    # Valid JSON but non-dict (array)
    assert extract_json_from_response("```json [1, 2, 3] ```") == {}

    # Valid JSON but non-dict (string)
    assert extract_json_from_response('```json "hello" ```') == {}

    # Valid dictionary JSON block
    parsed = extract_json_from_response('```json {"property_type": "residential"} ```')
    assert parsed == {"property_type": "residential"}


# ---------------------------------------------------------------------------
# Test 8: Missing Fields in LLM Response
# ---------------------------------------------------------------------------

def test_extraction_missing_fields_safe():
    """Extraction with missing optional fields must not crash prompt_builder."""
    parsed = extract_json_from_response('```json {"callback_requested": true} ```')
    assert parsed.get("callback_requested") is True
    assert "property_type" not in parsed
    assert "requested_loan_amount_inr" not in parsed


# ---------------------------------------------------------------------------
# Test 9: Factory Function `create_llm_adapter`
# ---------------------------------------------------------------------------

def test_create_llm_adapter_factory(monkeypatch):
    """Factory correctly instantiates StubLLMAdapter or OpenAILLMAdapter."""
    stub_adapter = create_llm_adapter(provider="stub")
    assert isinstance(stub_adapter, StubLLMAdapter)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-testkey1234567890")
    with patch("openai.OpenAI"):
        openai_adapter = create_llm_adapter(provider="openai")
        assert isinstance(openai_adapter, OpenAILLMAdapter)

    with pytest.raises(LLMConfigurationError):
        create_llm_adapter(provider="unsupported_provider_xyz")


# ---------------------------------------------------------------------------
# Test 10: TurnOrchestrator Fallback on LLM Error
# ---------------------------------------------------------------------------

class FailingLLMAdapter(LLMAdapter):
    """Adapter that intentionally raises an LLMError."""
    def chat(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        raise LLMProviderError("Simulated LLM connection failure")


def test_orchestrator_handles_llm_failure_gracefully():
    """TurnOrchestrator catches LLM exception and returns graceful fallback result."""
    failing_llm = FailingLLMAdapter()
    orchestrator = TurnOrchestrator(
        customer_id="HC-FAIL-01",
        customer_name="Ramesh Kumar",
        language=Language.ENGLISH,
        llm=failing_llm,
    )
    result = orchestrator.process_turn("Hello, I want to talk about my loan.")
    assert "technical difficulty" in result.agent_reply.lower()
    assert result.action == "continue"
    assert result.extraction == {}


# ---------------------------------------------------------------------------
# Test 11: Hindi Language Fallback Response on LLM Error
# ---------------------------------------------------------------------------

def test_orchestrator_hindi_llm_failure_fallback():
    """Hindi session returns Hindi fallback phrase when LLM fails."""
    failing_llm = FailingLLMAdapter()
    orchestrator = TurnOrchestrator(
        customer_id="HC-FAIL-HI",
        customer_name="Suresh Patel",
        language=Language.HINDI,
        llm=failing_llm,
    )
    result = orchestrator.process_turn("नमस्ते")
    assert "तकनीकी समस्या" in result.agent_reply
    assert result.action == "continue"


# ---------------------------------------------------------------------------
# Test 12: Async Chat Method `achat`
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
@patch("openai.OpenAI")
async def test_openai_adapter_async_chat(mock_openai_class, mock_async_openai_class):
    """OpenAILLMAdapter.achat executes asynchronously."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Async response ```json {} ```"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_async_client = MagicMock()
    mock_async_client.chat.completions.create = pytest.mark.asyncio(
        MagicMock(return_value=mock_response)
    )

    # Make the async call return a coroutine
    async def mock_create(*args, **kwargs):
        return mock_response

    mock_async_client.chat.completions.create = mock_create
    mock_async_openai_class.return_value = mock_async_client

    adapter = OpenAILLMAdapter(api_key="sk-testkey1234567890")
    result = await adapter.achat("System", [{"role": "user", "content": "Hi"}])
    assert result == "Async response ```json {} ```"


# ---------------------------------------------------------------------------
# Test 13: LLM Cannot Override Deterministic Business Rules
# ---------------------------------------------------------------------------

def test_deterministic_eligibility_prevents_llm_override():
    """LLM attempts to claim eligible in extraction, but cash income rule disqualifies."""
    stub_llm = StubLLMAdapter(replies=[
        'Hi Ramesh! ```json {"is_identity_confirmed": true, "is_available": true} ```',
        'Got it. ```json {"property_type": "residential", "ownership": "sole", "has_original_documents": true, "requested_loan_amount_inr": 5000000, "occupation_type": "self_employed", "income_mode": "cash", "property_market_value_inr": 10000000, "requested_tenure_years": 10, "is_eligible": true} ```',
    ])
    orchestrator = TurnOrchestrator(
        customer_id="HC-OVERRIDE",
        customer_name="Ramesh",
        language=Language.ENGLISH,
        llm=stub_llm,
    )
    orchestrator.process_turn("Yes I am Ramesh")
    res2 = orchestrator.process_turn("I have residential property, original docs, 50L loan, cash income, 10 yrs")
    assert res2.state.stage == Stage.DISQUALIFIED
    assert res2.action == "end"

