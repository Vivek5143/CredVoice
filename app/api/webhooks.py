"""
CredVoice — Webhooks API Router (app/api/webhooks.py).

Provides FastAPI endpoints for Retell AI, Bolna, and generic voice turn processing.

Design contract:
- Validates request payloads and HMAC signatures safely.
- Non-blocking async endpoints delegates turn processing to SessionManager.
- Handles unknown call IDs, malformed JSON, duplicate webhooks, and provider errors.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.config import settings
from app.integrations.bolna import BolnaIntegration, verify_bolna_signature
from app.integrations.retell import RetellIntegration, verify_retell_signature
from app.services.session_manager import get_session_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Direct Voice Turn Request / Response Models
# ---------------------------------------------------------------------------

class VoiceTurnRequest(BaseModel):
    """Payload for generic /direct turn simulation endpoint."""
    call_id: str = Field(description="Unique call session identifier")
    utterance: str = Field(description="Transcribed customer speech input")
    customer_id: str = Field(default="HC-CUST-DEFAULT", description="Customer ID")
    customer_name: str = Field(default="Valued Customer", description="Customer full name")
    language: str = Field(default="en", description="Conversation language ('en' or 'hi')")
    event_id: Optional[str] = Field(default=None, description="Optional turn event ID for deduplication")


class VoiceTurnResponse(BaseModel):
    """Response returned for direct turn simulation."""
    call_id: str
    agent_reply: str
    action: str
    stage: str
    faq_answered: bool


# ---------------------------------------------------------------------------
# Webhook Endpoints
# ---------------------------------------------------------------------------

@router.post("/retell", summary="Retell AI Voice Platform Webhook", tags=["Voice Webhooks"])
async def retell_webhook(
    request: Request,
    x_retell_signature: Optional[str] = Header(None, alias="x-retell-signature"),
    x_signature: Optional[str] = Header(None, alias="x-signature"),
):
    """
    Webhook endpoint for Retell AI call lifecycle events and custom LLM turns.
    """
    body_bytes = await request.body()
    sig = x_retell_signature or x_signature

    if getattr(settings, "verify_webhook_signatures", False):
        if not verify_retell_signature(body_bytes, sig):
            logger.warning("Rejected Retell webhook request: invalid signature.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing Retell webhook signature.",
            )

    try:
        payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception as exc:
        logger.error("Malformed Retell webhook JSON payload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload.",
        ) from exc

    integration = RetellIntegration()
    response_body, status_code = integration.process_webhook(payload)
    return Response(
        content=json.dumps(response_body),
        status_code=status_code,
        media_type="application/json",
    )


@router.post("/bolna", summary="Bolna Voice Platform Webhook", tags=["Voice Webhooks"])
async def bolna_webhook(
    request: Request,
    x_bolna_signature: Optional[str] = Header(None, alias="x-bolna-signature"),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    """
    Webhook endpoint for Bolna voice call lifecycle events and speech turns.
    """
    body_bytes = await request.body()
    sig = x_bolna_signature or x_api_key

    if getattr(settings, "verify_webhook_signatures", False):
        if not verify_bolna_signature(body_bytes, sig):
            logger.warning("Rejected Bolna webhook request: invalid signature.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing Bolna webhook signature.",
            )

    try:
        payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception as exc:
        logger.error("Malformed Bolna webhook JSON payload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload.",
        ) from exc

    integration = BolnaIntegration()
    response_body, status_code = integration.process_webhook(payload)
    return Response(
        content=json.dumps(response_body),
        status_code=status_code,
        media_type="application/json",
    )


@router.post("/turn", response_model=VoiceTurnResponse, summary="Direct Voice Turn Processing", tags=["Voice Webhooks"])
async def direct_voice_turn(req: VoiceTurnRequest):
    """
    Generic endpoint for processing customer turns directly (ideal for testing & simulations).
    """
    session_mgr = get_session_manager()
    turn_result = session_mgr.process_turn(
        call_id=req.call_id,
        customer_utterance=req.utterance,
        event_id=req.event_id,
        customer_id=req.customer_id,
        customer_name=req.customer_name,
        language=req.language,
    )

    return VoiceTurnResponse(
        call_id=req.call_id,
        agent_reply=turn_result.agent_reply,
        action=turn_result.action,
        stage=turn_result.state.stage.value,
        faq_answered=turn_result.faq_answered,
    )
