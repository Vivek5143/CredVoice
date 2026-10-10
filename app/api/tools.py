"""
CredVoice — Custom Tools API Router (app/api/tools.py).

Provides custom tool execution endpoints for conversational voice platforms such as Bolna.
Hosts the deterministic LAP qualification tool: `check_lap_eligibility`.

Design contract:
- Directly evaluates business criteria via EligibilityEngine.evaluate().
- Never trusts an LLM-supplied eligibility assertion.
- Enforces secure API key authentication (fails closed in production).
- Returns customer-safe, deterministic outcomes and handoff readiness.
- Never logs API credentials or sensitive customer financial information.
"""

from __future__ import annotations

import hmac
import logging
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.config import settings
from app.models.conversation import (
    DisqualificationReason,
    EligibilityStatus,
    ExistingLoanTransfer,
    IncomeMode,
    OccupationType,
    OwnershipType,
    PropertyType,
    QualificationState,
)
from app.services.eligibility import EligibilityEngine, EligibilityResult

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Authentication Helper for Bolna Custom Tools
# ---------------------------------------------------------------------------

def verify_tool_auth(
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None, alias="authorization"),
) -> None:
    """
    Validate incoming Bolna custom-tool API key header.

    Supported headers:
    - 'x-api-key: <KEY>' (standard Bolna custom tool header)
    - 'Authorization: Bearer <KEY>'

    Security behavior:
    - Fails closed with 401 in production if credentials are unconfigured or invalid.
    - Validates against settings.bolna_api_key (or settings.admin_api_key).
    - Uses constant-time comparison (hmac.compare_digest).
    - Never logs incoming key or secret values.
    """
    configured_key = (
        getattr(settings, "bolna_api_key", "").strip()
        or getattr(settings, "admin_api_key", "").strip()
    )

    # In production, authentication MUST be configured
    if getattr(settings, "is_production", False) or getattr(settings, "verify_webhook_signatures", False):
        if not configured_key or configured_key in ("", "your_bolna_api_key_here", "your_admin_api_key_here"):
            logger.error("Rejecting tool request: tool authentication key is unconfigured in secure environment.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tool authentication is unconfigured in secure environment.",
            )

    # In dev mode, if no key is configured at all, permit development calls
    if not configured_key or configured_key in ("", "your_bolna_api_key_here", "your_admin_api_key_here"):
        return

    # Extract token from x-api-key or Authorization header
    token = x_api_key.strip() if x_api_key else None
    if not token and authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
        elif len(parts) == 1:
            token = parts[0]

    if not token or not hmac.compare_digest(token, configured_key):
        logger.warning("Rejected tool request: invalid or missing authentication key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing tool API key ('x-api-key').",
        )


# ---------------------------------------------------------------------------
# Request and Response Models
# ---------------------------------------------------------------------------

class CheckLAPEligibilityRequest(BaseModel):
    """
    Request model for the Bolna custom tool 'check_lap_eligibility'.

    Accepts the 7 LAP qualification parameters, ₹75 Lakh cap consent, and transfer intent.
    All fields default to None to support partial state evaluations during multi-turn calls.
    """
    property_type: Optional[PropertyType] = Field(
        default=None,
        description="Property collateral type (residential, commercial, industrial, agricultural, or other)."
    )
    ownership: Optional[OwnershipType] = Field(
        default=None,
        description="Property ownership type (sole, joint, or unapproved)."
    )
    has_original_documents: Optional[bool] = Field(
        default=None,
        description="Whether original property registry documents are available."
    )
    requested_loan_amount_inr: Optional[float] = Field(
        default=None,
        description="Desired loan amount in INR (absolute rupees, e.g. 5000000 for ₹50 Lakh)."
    )
    proceeds_at_cap: Optional[bool] = Field(
        default=None,
        description="Customer consent to proceed at ₹75 Lakh cap if requested amount exceeds ₹75 Lakh."
    )
    occupation_type: Optional[OccupationType] = Field(
        default=None,
        description="Customer occupation (salaried, self_employed, unemployed, or other)."
    )
    income_mode: Optional[IncomeMode] = Field(
        default=None,
        description="Mode of income receipt (bank = bank transfer/cheque; cash = cash in hand only)."
    )
    property_market_value_inr: Optional[float] = Field(
        default=None,
        description="Customer's estimated property market value in INR (must be > 0)."
    )
    requested_tenure_years: Optional[int] = Field(
        default=None,
        description="Requested loan tenure in years (valid range: 3 to 15 years inclusive)."
    )
    has_existing_property_loan: Optional[bool] = Field(
        default=None,
        description="True if customer has an active property loan with another bank/NBFC."
    )
    wants_emi_reduction: Optional[bool] = Field(
        default=None,
        description="True if customer seeks balance transfer or EMI reduction on an existing loan."
    )


class CheckLAPEligibilityResponse(BaseModel):
    """
    Stable response model returned to Bolna's custom tool executor.
    """
    status: EligibilityStatus = Field(
        description="Deterministic status: eligible, disqualified, incomplete, or transfer_routed."
    )
    is_eligible: bool = Field(
        description="True ONLY if all 7 qualification rules pass without disqualification."
    )
    handoff_ready: bool = Field(
        description="True if all data points are verified and lead is ready for Senior Loan Specialist handoff."
    )
    is_transfer_lead: bool = Field(
        description="True if lead should be routed to Balance Transfer / EMI reduction specialist."
    )
    requires_cap_consent: bool = Field(
        description="True if requested loan amount exceeds ₹75 Lakh and requires customer consent."
    )
    disqualification_reasons: List[str] = Field(
        default_factory=list,
        description="List of specific policy disqualification reasons, if any."
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of mandatory parameters still needed to complete qualification."
    )
    summary_explanation: str = Field(
        description="Concise, policy-grounded summary of the qualification result."
    )
    recommended_spoken_reply: str = Field(
        description="Customer-safe spoken response suitable for the voice agent to speak directly."
    )


# ---------------------------------------------------------------------------
# Spoken Reply Helper
# ---------------------------------------------------------------------------

def _derive_spoken_reply(result: EligibilityResult, req: CheckLAPEligibilityRequest) -> str:
    """Generate a safe, policy-compliant spoken phrase for the Bolna agent."""
    if result.status == EligibilityStatus.TRANSFER_ROUTED:
        return (
            "I see that you have an existing property loan or are looking to reduce your current EMI. "
            "Let me connect you directly with our Balance Transfer Specialist who can assist you."
        )

    if result.status == EligibilityStatus.DISQUALIFIED:
        # Provide specific polite reasons without promising alternative loan terms
        if DisqualificationReason.AGRICULTURAL_PROPERTY in result.disqualification_reasons:
            return (
                "Thank you for sharing that. Home Credit Loan Against Property accepts residential, "
                "commercial, or industrial properties. Agricultural land is not eligible under this program."
            )
        if DisqualificationReason.CASH_INCOME in result.disqualification_reasons:
            return (
                "Thank you for the information. Home Credit requires salary or business income to be "
                "received directly into a bank account. We cannot proceed with cash income."
            )
        if DisqualificationReason.MISSING_DOCUMENTS in result.disqualification_reasons:
            return (
                "Original property registry documents are mandatory for verification. "
                "Since originals are unavailable, we cannot proceed with qualification at this time."
            )
        if DisqualificationReason.LOAN_AMOUNT_EXCEEDED_DECLINED in result.disqualification_reasons:
            return (
                "Under our current policy, the maximum loan amount is ₹75 Lakh. Since you do not wish "
                "to proceed with this limit, we are unable to process this application."
            )
        if DisqualificationReason.TENURE_OUT_OF_RANGE in result.disqualification_reasons:
            return (
                "Home Credit Loan Against Property offers tenure between 3 and 15 years. "
                "Your requested tenure falls outside this eligible range."
            )
        return (
            "Thank you for providing the details. Based on our current policy guidelines, "
            "we are unable to qualify this property for Home Credit Loan Against Property."
        )

    if result.status == EligibilityStatus.INCOMPLETE:
        if result.requires_cap_consent and req.proceeds_at_cap is None:
            return (
                "Our maximum loan amount under this offer is ₹75 Lakh. "
                "Would you be comfortable proceeding with a ₹75 Lakh loan?"
            )
        missing_count = len(result.missing_fields)
        return (
            f"Thank you. To check your eligibility, we still need {missing_count} remaining detail(s). "
            "Could you please share those?"
        )

    # Status is ELIGIBLE
    return (
        "Congratulations! Based on the details provided, you qualify for Home Credit Loan Against Property "
        "of up to ₹75 Lakh. I am connecting you with our Senior Loan Specialist to proceed with your application."
    )


# ---------------------------------------------------------------------------
# Endpoint Implementation
# ---------------------------------------------------------------------------

@router.post(
    "/check-lap-eligibility",
    response_model=CheckLAPEligibilityResponse,
    summary="Bolna Custom Tool: Check LAP Eligibility",
    tags=["Custom Tools"],
)
async def check_lap_eligibility(
    req: CheckLAPEligibilityRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    """
    Custom tool endpoint for Bolna voice agents to evaluate customer qualification deterministically.

    Calls EligibilityEngine.evaluate() directly. No LLM output can override the deterministic rules.
    """
    verify_tool_auth(x_api_key=x_api_key, authorization=authorization)

    # 1. Build QualificationState from incoming payload
    qualification = QualificationState(
        property_type=req.property_type,
        ownership=req.ownership,
        has_original_documents=req.has_original_documents,
        requested_loan_amount_inr=req.requested_loan_amount_inr,
        proceeds_at_cap=req.proceeds_at_cap,
        occupation_type=req.occupation_type,
        income_mode=req.income_mode,
        property_market_value_inr=req.property_market_value_inr,
        requested_tenure_years=req.requested_tenure_years,
    )

    # 2. Build transfer intent if provided
    transfer_intent: Optional[ExistingLoanTransfer] = None
    if req.has_existing_property_loan is not None or req.wants_emi_reduction is not None:
        transfer_intent = ExistingLoanTransfer(
            has_existing_property_loan=req.has_existing_property_loan,
            wants_emi_reduction=req.wants_emi_reduction,
        )

    # 3. Evaluate deterministically via EligibilityEngine
    result: EligibilityResult = EligibilityEngine.evaluate(
        qualification=qualification,
        transfer_intent=transfer_intent,
    )

    # 4. Formulate spoken reply
    spoken_reply = _derive_spoken_reply(result, req)

    # 5. Return structured response
    return CheckLAPEligibilityResponse(
        status=result.status,
        is_eligible=result.status == EligibilityStatus.ELIGIBLE,
        handoff_ready=result.handoff_ready,
        is_transfer_lead=result.is_transfer_lead,
        requires_cap_consent=result.requires_cap_consent,
        disqualification_reasons=[r.value for r in result.disqualification_reasons],
        missing_fields=result.missing_fields,
        summary_explanation=result.summary_explanation,
        recommended_spoken_reply=spoken_reply,
    )
