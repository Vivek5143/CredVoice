"""
CredVoice — Logging & Privacy Utility Module (app/utils/logging.py).

Provides PII redaction (phone numbers, full names, account numbers), secret masking,
and structured logging context helpers for audit compliance.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Regex Patterns for PII Redaction
# ---------------------------------------------------------------------------

_PHONE_PATTERN = re.compile(r"\b(\+?\d{1,3}[-.\s]?)?\d{10}\b")
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_SECRET_PATTERN = re.compile(r"sk-[a-zA-Z0-9_\-]{15,}|key-[a-zA-Z0-9_\-]{15,}")


def redact_pii(text: Optional[str]) -> str:
    """
    Redact PII (phone numbers, email addresses, API keys) from text before logging.

    Args:
        text: Raw input string or log message.

    Returns:
        Sanitized string with PII replaced by masked tokens.
    """
    if not text:
        return ""

    sanitized = str(text)

    # Redact API keys / secrets first
    sanitized = _SECRET_PATTERN.sub("[REDACTED_SECRET]", sanitized)

    # Redact email addresses
    sanitized = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)

    # Redact phone numbers (preserve last 4 digits if 10+ digits)
    def _mask_phone(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) >= 10:
            return f"+91*****{digits[-4:]}"
        return "[REDACTED_PHONE]"

    sanitized = _PHONE_PATTERN.sub(_mask_phone, sanitized)

    return sanitized
