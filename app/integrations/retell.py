"""
CredVoice — Retell AI Integration Module (app/integrations/retell.py).

Provides payload parsing, signature verification, event handling, and custom LLM
response formatting for Retell AI voice platform webhooks.

Design contract:
- HMAC SHA256 signature verification validates incoming requests when enabled.
- Retell custom LLM response contract (content, content_complete, end_call) is strictly enforced.
- Integrates cleanly with app.services.session_manager.SessionManager.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from app.config import settings
from app.services.session_manager import SessionManager, get_session_manager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HMAC Signature Verification
# ---------------------------------------------------------------------------

def verify_retell_signature(
    body_bytes: bytes,
    signature_header: Optional[str],
    secret: Optional[str] = None,
) -> bool:
    """
    Verify Retell AI HMAC-SHA256 request signature.

    Args:
        body_bytes: Raw HTTP request body.
        signature_header: Header value from 'x-retell-signature' or 'x-signature'.
        secret: Webhook secret or API key. Defaults to settings.retell_webhook_secret or retell_api_key.

    Returns:
        True if signature is valid or if verification is disabled; False otherwise.
    """
    resolved_secret = (
        secret
        or getattr(settings, "retell_webhook_secret", "")
        or getattr(settings, "retell_api_key", "")
    )

    should_verify = settings.should_verify_signatures or bool(secret)

    if not should_verify:
        return True

    if not signature_header or not resolved_secret or resolved_secret in ("", "your_retell_webhook_secret_here", "your_retell_api_key_here"):
        logger.warning("Retell signature verification failed: signature header or secret is missing or unconfigured.")
        return False

    try:
        expected = hmac.new(
            resolved_secret.encode("utf-8"),
            body_bytes,
            hashlib.sha256,
        ).hexdigest()

        # Handle potential prefixing (e.g., v1=...)
        sig_value = signature_header
        if "=" in signature_header:
            sig_value = signature_header.split("=")[-1]

        is_valid = hmac.compare_digest(expected.lower(), sig_value.lower())
        if not is_valid:
            logger.warning("Invalid Retell signature: expected=%s, got=%s", expected, sig_value)
        return is_valid
    except Exception as exc:
        logger.error("Error during Retell signature verification: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Payload Models
# ---------------------------------------------------------------------------

class RetellCallData(BaseModel):
    """Retell nested call object."""
    call_id: str
    customer_number: Optional[str] = None
    agent_id: Optional[str] = None
    disposition: Optional[str] = None
    transcript: Optional[str] = None
    recording_url: Optional[str] = None


class RetellWebhookPayload(BaseModel):
    """Incoming Retell webhook event payload."""
    event: Optional[str] = Field(default=None, description="Event name, e.g., 'call_started', 'call_ended', 'user_transcript'")
    event_id: Optional[str] = Field(default=None, description="Unique event identifier for idempotency")
    call_id: Optional[str] = None
    call: Optional[RetellCallData] = None
    transcript: Optional[str] = None
    user_transcript: Optional[str] = None
    interaction_type: Optional[str] = None
    response_id: Optional[int] = 0


# ---------------------------------------------------------------------------
# Event Processor
# ---------------------------------------------------------------------------

class RetellIntegration:
    """
    Handles event routing and response generation for Retell AI.
    """

    def __init__(self, session_manager: Optional[SessionManager] = None) -> None:
        self.session_mgr = session_manager or get_session_manager()

    def process_webhook(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Process an incoming Retell payload and return (response_dict, status_code).
        """
        event_name = payload.get("event") or payload.get("interaction_type") or "response_required"
        event_id = payload.get("event_id") or payload.get("interaction_id")

        # Extract call_id from top-level or nested call dict
        call_id = payload.get("call_id")
        if not call_id and isinstance(payload.get("call"), dict):
            call_id = payload["call"].get("call_id")

        if not call_id:
            logger.error("Retell payload missing call_id: %s", payload)
            return {"error": "Missing call_id in payload"}, 400

        # Handle lifecycle events
        if event_name in {"call_started", "call_start"}:
            customer_name = "Valued Customer"
            customer_id = "HC-RETELL-CUST"
            if isinstance(payload.get("call"), dict) and payload["call"].get("customer_number"):
                customer_id = f"HC-TEL-{payload['call']['customer_number']}"

            self.session_mgr.get_or_create_session(
                call_id=call_id,
                customer_id=customer_id,
                customer_name=customer_name,
                language=settings.default_language,
            )
            return {
                "status": "started",
                "call_id": call_id,
                "message": "Call session initialized.",
            }, 200

        if event_name in {"call_ended", "call_completion", "call_analyzed"}:
            orchestrator = self.session_mgr.end_session(call_id)
            outcome = orchestrator.state.call_outcome.value if orchestrator and orchestrator.state.call_outcome else "ended"
            return {
                "status": "ended",
                "call_id": call_id,
                "outcome": outcome,
            }, 200

        # Handle turn / speech interaction
        utterance = (
            payload.get("user_transcript")
            or payload.get("transcript")
            or payload.get("args", {}).get("user_transcript")
            or ""
        )

        turn_result = self.session_mgr.process_turn(
            call_id=call_id,
            customer_utterance=utterance,
            event_id=event_id,
            language=settings.default_language,
        )

        should_end = turn_result.action in {"end", "handoff", "transfer", "callback"}

        response_body = {
          "response_id": payload.get("response_id", 0),
          "content": turn_result.agent_reply,
          "content_complete": True,
          "end_call": should_end,
          "action": turn_result.action,
          "stage": turn_result.state.stage.value,
        }

        return response_body, 200
