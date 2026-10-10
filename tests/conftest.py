"""
Global pytest configuration and fixtures for CredVoice test suite.
Ensures hermetic testing with mocked LLM adapters and prevents unintended live API calls.
"""

import pytest
from app.agent.llm import StubLLMAdapter


@pytest.fixture(autouse=True)
def isolate_llm_calls_in_tests(monkeypatch, request):
    """
    Ensure all tests run hermetically with mocked LLM adapters unless specifically
    testing LLM adapter implementation/configuration in test_llm.py.
    """
    if "test_llm" in request.module.__name__:
        return

    monkeypatch.setattr(
        "app.agent.core.create_llm_adapter",
        lambda: StubLLMAdapter(["Sure, understood. ```json {} ```"] * 50),
    )
