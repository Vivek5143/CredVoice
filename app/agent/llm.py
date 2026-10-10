"""
CredVoice — LLM Adapter Module (app/agent/llm.py).

Provides production LLM integration using OpenAI (or configured provider),
stub adapter for testing, custom exception hierarchy, secret masking, and
factory instantiation.

Design contract:
- API credentials are read from environment variables or Settings — never hardcoded.
- Credentials / API keys are masked in any logs or exception messages.
- The interface chat(system_prompt, messages) -> str is preserved across implementations.
- Production adapter handles timeouts, rate limits, provider errors, and invalid API keys gracefully.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exception Hierarchy
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base exception for all LLM adapter errors."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when required API keys or settings are missing or invalid."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM API request times out."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limits or quotas are exceeded."""
    pass


class LLMProviderError(LLMError):
    """Raised when the LLM provider returns a server or connection error."""
    pass


# ---------------------------------------------------------------------------
# Helper: Mask API Keys / Secrets
# ---------------------------------------------------------------------------

def mask_secret(secret: Optional[str]) -> str:
    """Return a masked representation of a secret string (e.g. 'sk-proj-...1234')."""
    if not secret:
        return "<EMPTY>"
    secret_str = str(secret).strip()
    if len(secret_str) <= 8:
        return "****"
    return f"{secret_str[:4]}...{secret_str[-4:]}"


# ---------------------------------------------------------------------------
# Base Interface
# ---------------------------------------------------------------------------

class LLMAdapter:
    """
    Abstract interface for LLM interaction.

    Concrete implementations call real provider APIs (OpenAI, Anthropic, Gemini)
    or pre-canned stubs for testing.
    """

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Send a chat completion request to the LLM.

        Args:
            system_prompt: Turn-specific system prompt string.
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts.

        Returns:
            Raw text output from the LLM (expected to contain JSON extraction).

        Raises:
            LLMError subclass on failure.
        """
        raise NotImplementedError("Subclasses must implement chat().")

    async def achat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """Asynchronous chat completion request."""
        return self.chat(system_prompt, messages)


# ---------------------------------------------------------------------------
# Stub Adapter (Testing & Development)
# ---------------------------------------------------------------------------

class StubLLMAdapter(LLMAdapter):
    """
    Test stub: returns pre-programmed replies without network calls.

    Inject via TurnOrchestrator(llm=StubLLMAdapter(replies=[...])).
    replies are consumed in order; the last reply repeats if exhausted.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._index = 0

    def chat(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        if not self._replies:
            return "Hello, how can I help you today? ```json {} ```"
        reply = self._replies[min(self._index, len(self._replies) - 1)]
        self._index += 1
        return reply

    async def achat(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        return self.chat(system_prompt, messages)


# ---------------------------------------------------------------------------
# OpenAI Production Adapter
# ---------------------------------------------------------------------------

class OpenAILLMAdapter(LLMAdapter):
    """
    Production LLM adapter using the official OpenAI Python SDK.

    Args:
        api_key: OpenAI API key. Defaults to settings.openai_api_key or OPENAI_API_KEY env var.
        model: Model name. Defaults to settings.llm_model ('gpt-4o-mini').
        timeout_seconds: Request timeout in seconds.
        max_retries: Number of automatic SDK retries for transient errors.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        temperature: float = 0.2,
    ) -> None:
        resolved_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or getattr(settings, "openai_api_key", "")
        )
        if not resolved_key or resolved_key.strip() in ("", "your_openai_api_key_here"):
            raise LLMConfigurationError(
                "OpenAI API key is missing or invalid. Set OPENAI_API_KEY in environment or .env file."
            )

        self.api_key = resolved_key.strip()
        self.model = model or getattr(settings, "llm_model", "gpt-4o-mini")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature

        logger.info(
            "Initializing OpenAILLMAdapter with model=%s, key=%s",
            self.model,
            mask_secret(self.api_key),
        )

        try:
            import openai
            self._openai_lib = openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )
            self.async_client = openai.AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'openai' package is not installed. Install it via 'pip install openai'."
            ) from exc
        except Exception as exc:
            raise LLMConfigurationError(
                f"Failed to initialize OpenAI client: {exc}"
            ) from exc

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Send a synchronous chat completion request to OpenAI ChatCompletions API.
        """
        formatted_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in {"user", "assistant", "system"}:
                formatted_messages.append({"role": role, "content": content})

        try:
            logger.debug(
                "Sending OpenAI request: model=%s, message_count=%d",
                self.model,
                len(formatted_messages),
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
            )
            if not response.choices:
                raise LLMProviderError("OpenAI returned an empty response choices list.")

            content = response.choices[0].message.content or ""
            return content

        except self._openai_lib.AuthenticationError as exc:
            logger.error("OpenAI authentication error: %s", exc)
            raise LLMConfigurationError("Invalid or expired OpenAI API key.") from exc

        except self._openai_lib.PermissionDeniedError as exc:
            logger.error("OpenAI permission denied error: %s", exc)
            raise LLMConfigurationError("Permission denied for requested OpenAI model/resource.") from exc

        except self._openai_lib.APITimeoutError as exc:
            logger.error("OpenAI request timed out after %.1fs: %s", self.timeout_seconds, exc)
            raise LLMTimeoutError(f"OpenAI API request timed out after {self.timeout_seconds}s.") from exc

        except self._openai_lib.RateLimitError as exc:
            logger.error("OpenAI rate limit / quota exceeded: %s", exc)
            raise LLMRateLimitError("OpenAI API rate limit or quota exceeded.") from exc

        except self._openai_lib.APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise LLMProviderError(f"OpenAI API error: {exc}") from exc

        except Exception as exc:
            logger.error("Unexpected error during OpenAI API call: %s", exc)
            raise LLMProviderError(f"LLM call failed: {exc}") from exc

    async def achat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Send an asynchronous chat completion request to OpenAI ChatCompletions API.
        """
        formatted_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in {"user", "assistant", "system"}:
                formatted_messages.append({"role": role, "content": content})

        try:
            logger.debug(
                "Sending async OpenAI request: model=%s, message_count=%d",
                self.model,
                len(formatted_messages),
            )
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
            )
            if not response.choices:
                raise LLMProviderError("OpenAI returned an empty response choices list.")

            content = response.choices[0].message.content or ""
            return content

        except self._openai_lib.AuthenticationError as exc:
            logger.error("OpenAI authentication error: %s", exc)
            raise LLMConfigurationError("Invalid or expired OpenAI API key.") from exc

        except self._openai_lib.PermissionDeniedError as exc:
            logger.error("OpenAI permission denied error: %s", exc)
            raise LLMConfigurationError("Permission denied for requested OpenAI model/resource.") from exc

        except self._openai_lib.APITimeoutError as exc:
            logger.error("OpenAI request timed out after %.1fs: %s", self.timeout_seconds, exc)
            raise LLMTimeoutError(f"OpenAI API request timed out after {self.timeout_seconds}s.") from exc

        except self._openai_lib.RateLimitError as exc:
            logger.error("OpenAI rate limit / quota exceeded: %s", exc)
            raise LLMRateLimitError("OpenAI API rate limit or quota exceeded.") from exc

        except self._openai_lib.APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise LLMProviderError(f"OpenAI API error: {exc}") from exc

        except Exception as exc:
            logger.error("Unexpected error during async OpenAI API call: %s", exc)
            raise LLMProviderError(f"LLM call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Gemini Production Adapter
# ---------------------------------------------------------------------------

class GeminiLLMAdapter(LLMAdapter):
    """
    Production LLM adapter using Google's official google-genai Python SDK.

    Args:
        api_key: Gemini API key. Defaults to GEMINI_API_KEY env var or settings.gemini_api_key.
                 Credentials are never hardcoded or emitted in logs.
        model: Model name. Defaults to settings.llm_model or 'gemini-2.0-flash'.
               If the configured model looks like an OpenAI model (e.g. gpt-4o-mini),
               it falls back to 'gemini-2.0-flash' automatically.
        timeout_seconds: Request timeout in seconds (sync: passed via HttpOptions; async: asyncio.wait_for).
        temperature: Sampling temperature (0.0–1.0).
    """

    _PLACEHOLDER = "your_gemini_api_key_here"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 15.0,
        temperature: float = 0.2,
    ) -> None:
        resolved_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or getattr(settings, "gemini_api_key", "")
        )
        if not resolved_key or resolved_key.strip() in ("", self._PLACEHOLDER):
            raise LLMConfigurationError(
                "Gemini API key is missing or invalid. "
                "Set GEMINI_API_KEY in environment or .env file."
            )

        self._api_key = resolved_key.strip()

        # Resolve model: if the configured model is clearly an OpenAI model, default to Gemini
        raw_model = model or getattr(settings, "llm_model", "gemini-2.0-flash")
        self.model = raw_model if not raw_model.startswith("gpt-") else "gemini-2.0-flash"

        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

        logger.info(
            "Initializing GeminiLLMAdapter with model=%s, key=%s",
            self.model,
            mask_secret(self._api_key),
        )

        try:
            from google import genai as _genai
            from google.genai import types as _types
            self._genai = _genai
            self._types = _types
            self._client = _genai.Client(api_key=self._api_key)
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'google-genai' package is not installed. "
                "Install it via 'pip install google-genai'."
            ) from exc
        except Exception as exc:
            raise LLMConfigurationError(
                f"Failed to initialize Gemini client: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Message format helpers
    # ------------------------------------------------------------------

    def _build_gemini_contents(
        self, messages: list[dict[str, str]]
    ) -> list[Any]:
        """
        Convert OpenAI-style role/content messages into google-genai Content objects.

        - "user"      → role="user"
        - "assistant" → role="model"
        - "system"    → skipped (goes into system_instruction instead)
        """
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                continue  # system messages handled via system_instruction
            contents.append(
                self._types.Content(
                    role=gemini_role,
                    parts=[self._types.Part(text=text)],
                )
            )
        return contents

    # ------------------------------------------------------------------
    # Synchronous chat
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Send a synchronous chat completion request to Gemini GenerateContent API.

        Raises:
            LLMConfigurationError: Invalid/missing API key or auth failure.
            LLMTimeoutError:       Request exceeded timeout_seconds.
            LLMRateLimitError:     Quota or rate limit exceeded.
            LLMProviderError:      Any other Gemini API error.
        """
        contents = self._build_gemini_contents(messages)

        try:
            logger.debug(
                "Sending Gemini request: model=%s, turn_count=%d",
                self.model,
                len(contents),
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature,
                    http_options=self._types.HttpOptions(
                        timeout=int(self.timeout_seconds * 1000),
                    ),
                ),
            )
            text = response.text
            if text is None:
                raise LLMProviderError("Gemini returned an empty response.")
            return text

        except LLMError:
            raise  # re-raise our own typed exceptions unchanged

        except Exception as exc:
            self._raise_mapped_exception(exc)

    # ------------------------------------------------------------------
    # Asynchronous chat
    # ------------------------------------------------------------------

    async def achat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Send an asynchronous chat completion request to Gemini via the aio client.

        asyncio.wait_for wraps the coroutine so that timeout_seconds is enforced
        even when the SDK does not natively surface asyncio.TimeoutError.
        """
        import asyncio

        contents = self._build_gemini_contents(messages)

        try:
            logger.debug(
                "Sending async Gemini request: model=%s, turn_count=%d",
                self.model,
                len(contents),
            )
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self._types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.temperature,
                    ),
                ),
                timeout=self.timeout_seconds,
            )
            text = response.text
            if text is None:
                raise LLMProviderError("Gemini returned an empty async response.")
            return text

        except asyncio.TimeoutError as exc:
            logger.error(
                "Gemini async request timed out after %.1fs", self.timeout_seconds
            )
            raise LLMTimeoutError(
                f"Gemini API async request timed out after {self.timeout_seconds}s."
            ) from exc

        except LLMError:
            raise

        except Exception as exc:
            self._raise_mapped_exception(exc)

    # ------------------------------------------------------------------
    # Exception mapping — no credentials in messages
    # ------------------------------------------------------------------

    def _raise_mapped_exception(self, exc: Exception) -> None:
        """
        Translate google-genai SDK exceptions into our LLMError hierarchy.

        Inspects only the exception type name and lowercase message tokens;
        never logs the raw API key or any secret value.
        """
        exc_type = type(exc).__name__
        exc_str = str(exc).lower()

        if any(tok in exc_str for tok in ("timeout", "deadline", "timed out")):
            logger.error("Gemini request timed out: %s", exc_type)
            raise LLMTimeoutError(
                f"Gemini API request timed out after {self.timeout_seconds}s."
            ) from exc

        if any(tok in exc_str for tok in (
            "api_key", "invalid api key", "unauthenticated",
            "permission", "401", "403",
        )):
            logger.error("Gemini authentication/permission error: %s", exc_type)
            raise LLMConfigurationError("Invalid or expired Gemini API key.") from exc

        if any(tok in exc_str for tok in ("quota", "rate", "429", "resource_exhausted")):
            logger.error("Gemini rate limit exceeded: %s", exc_type)
            raise LLMRateLimitError("Gemini API rate limit or quota exceeded.") from exc

        logger.error("Gemini API error (%s): %s", exc_type, exc_str)
        raise LLMProviderError(f"Gemini API error: {exc_type}") from exc


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

def create_llm_adapter(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMAdapter:
    """
    Factory function for instantiating the configured LLM adapter.

    Args:
        provider: Provider name ("openai", "gemini", "stub"). Defaults to settings.llm_provider.
        model: Model name. Defaults to settings.llm_model.
        api_key: Provider API key. Defaults to the provider-specific env var or settings field.

    Returns:
        Concrete LLMAdapter instance.

    Raises:
        LLMConfigurationError: If provider is unsupported or credentials missing.
    """
    resolved_provider = (provider or getattr(settings, "llm_provider", "openai")).lower()

    if resolved_provider == "openai":
        return OpenAILLMAdapter(api_key=api_key, model=model)
    elif resolved_provider == "gemini":
        return GeminiLLMAdapter(api_key=api_key, model=model)
    elif resolved_provider in {"stub", "mock", "test"}:
        return StubLLMAdapter(replies=[])
    else:
        raise LLMConfigurationError(
            f"Unsupported LLM provider: '{resolved_provider}'. "
            "Supported: 'openai', 'gemini', 'stub'."
        )

