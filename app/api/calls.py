"""
CredVoice — Call Management API Router (app/api/calls.py).

Provides endpoints to inspect call session state, query transcript histories,
and terminate active calls.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

import hmac

from app.config import settings
from app.services.session_manager import get_session_manager

logger = logging.getLogger(__name__)

router = APIRouter()


def verify_admin_auth(x_api_key: Optional[str] = Header(None, alias="x-api-key")) -> None:
    """
    Verify admin API key header for protected call management endpoints.
    In production or when admin_api_key is set, missing or mismatched key is rejected with 401.
    """
    admin_key = getattr(settings, "admin_api_key", "").strip()
    if not admin_key:
        if getattr(settings, "is_production", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Admin authentication is unconfigured in production environment.",
            )
        return  # Allowed in dev mode when admin_key is unconfigured

    if not x_api_key or not hmac.compare_digest(x_api_key.strip(), admin_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin API key ('x-api-key').",
        )


class CallSessionSummary(BaseModel):
    """Summary representation of an active call session."""
    call_id: str
    customer_id: str
    customer_name: str
    stage: str
    action: str
    turn_count: int
    language: str
    is_terminal: bool
    call_outcome: Optional[str] = None


@router.get("/{call_id}", response_model=CallSessionSummary, summary="Get Call Session Details", tags=["Call Management"])
async def get_call_session(
    call_id: str,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    """
    Retrieve current state and summary of an active call session.
    """
    verify_admin_auth(x_api_key)
    session_mgr = get_session_manager()
    session = session_mgr.get_session(call_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call session '{call_id}' not found.",
        )

    orch = session.orchestrator
    state = orch.state
    action = orch._derive_action()

    return CallSessionSummary(
        call_id=call_id,
        customer_id=session.customer_id,
        customer_name=session.customer_name,
        stage=state.stage.value,
        action=action,
        turn_count=orch.turn_count,
        language=state.language.value,
        is_terminal=orch._is_terminal_stage(),
        call_outcome=state.call_outcome.value if state.call_outcome else None,
    )


@router.delete("/{call_id}", summary="Terminate Call Session", tags=["Call Management"])
async def terminate_call_session(
    call_id: str,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    """
    Explicitly terminate and remove an active call session.
    """
    verify_admin_auth(x_api_key)
    session_mgr = get_session_manager()
    orch = session_mgr.end_session(call_id)
    if not orch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call session '{call_id}' not found.",
        )

    return {
        "status": "terminated",
        "call_id": call_id,
        "final_stage": orch.state.stage.value,
    }
