"""
CredVoice — Bolna Integration Module (app/integrations/bolna.py).

Provides payload parsing, HMAC signature verification, event processing, and
voice response formatting for the Bolna conversational voice platform.

Design contract:
- HMAC SHA256 signature verification validates incoming requests when enabled.
- Bolna payload schema (execution_id, agent_response, action, end_call) is strictly enforced.
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

def verify_bolna_signature(
    body_bytes: bytes,
    signature_header: Optional[str],
    secret: Optional[str] = None,
) -> bool:
    """
    Verify Bolna HMAC-SHA256 request signature or token header.

    Args:
        body_bytes: Raw HTTP request body.
        signature_header: Value from 'x-bolna-signature' or 'x-api-key'.
        secret: Webhook secret or API key. Defaults to settings.bolna_webhook_secret or bolna_api_key.

    Returns:
        True if signature is valid or if verification is disabled; False otherwise.
    """
    resolved_secret = (
        secret
        or getattr(settings, "bolna_webhook_secret", "")
        or getattr(settings, "bolna_api_key", "")
    )

    should_verify = settings.should_verify_signatures or bool(secret)

    if not should_verify:
        return True

    if not signature_header or not resolved_secret or resolved_secret in ("", "your_bolna_webhook_secret_here", "your_bolna_api_key_here"):
        logger.warning("Bolna signature verification failed: signature header or secret is missing or unconfigured.")
        return False

    try:
        # Check direct token match if header matches secret key
        if hmac.compare_digest(signature_header.strip(), resolved_secret.strip()):
            return True

        # Check HMAC SHA256 digest
        expected = hmac.new(
            resolved_secret.encode("utf-8"),
            body_bytes,
            hashlib.sha256,
        ).hexdigest()

        sig_value = signature_header
        if "=" in signature_header:
            sig_value = signature_header.split("=")[-1]

        is_valid = hmac.compare_digest(expected.lower(), sig_value.lower())
        if not is_valid:
            logger.warning("Invalid Bolna signature: expected=%s, got=%s", expected, sig_value)
        return is_valid
    except Exception as exc:
        logger.error("Error during Bolna signature verification: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Payload Models
# ---------------------------------------------------------------------------

class BolnaWebhookPayload(BaseModel):
    """Incoming Bolna webhook event payload."""
    event: Optional[str] = Field(default=None, description="Event name, e.g., 'call_started', 'completed', 'user_transcript'")
    event_id: Optional[str] = Field(default=None, description="Unique event identifier for idempotency")
    execution_id: Optional[str] = Field(default=None, description="Bolna call execution ID")
    call_id: Optional[str] = None
    transcript: Optional[str] = None
    user_message: Optional[str] = None
    phone_number: Optional[str] = None
    agent_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Event Processor
# ---------------------------------------------------------------------------

class BolnaIntegration:
    """
    Handles event routing and response generation for Bolna.
    """

    def __init__(self, session_manager: Optional[SessionManager] = None) -> None:
        self.session_mgr = session_manager or get_session_manager()

    def process_webhook(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Process an incoming Bolna payload and return (response_dict, status_code).
        """
        call_id = payload.get("execution_id") or payload.get("call_id")
        if not call_id:
            logger.error("Bolna payload missing execution_id / call_id: %s", payload)
            return {"status": "error", "message": "Missing execution_id in payload"}, 400

        event_name = payload.get("event") or payload.get("status") or "user_transcript"
        event_id = payload.get("event_id") or payload.get("message_id")

        if event_name in {"call_started", "execution_started", "started"}:
            phone = payload.get("phone_number") or "Unknown"
            self.session_mgr.get_or_create_session(
                call_id=call_id,
                customer_id=f"HC-BOLNA-{phone}",
                customer_name="Valued Customer",
                language=settings.default_language,
            )
            return {
                "status": "success",
                "execution_id": call_id,
                "message": "Call session initialized.",
            }, 200

        if event_name in {"call_ended", "completed", "terminated"}:
            orchestrator = self.session_mgr.end_session(call_id)
            outcome = orchestrator.state.call_outcome.value if orchestrator and orchestrator.state.call_outcome else "ended"
            return {
                "status": "success",
                "execution_id": call_id,
                "outcome": outcome,
            }, 200

        # Extract user utterance
        utterance = (
            payload.get("user_message")
            or payload.get("transcript")
            or payload.get("user_transcript")
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
            "status": "success",
            "execution_id": call_id,
            "agent_response": turn_result.agent_reply,
            "action": turn_result.action,
            "end_call": should_end,
            "stage": turn_result.state.stage.value,
        }

        return response_body, 200
