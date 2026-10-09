"""
CredVoice Deterministic Eligibility Engine.

Independent business logic service evaluating the seven mandatory LAP qualification rules.
Zero reliance on LLM output for final eligibility assessment or disqualification decisions.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.config import settings
from app.models.conversation import (
    QualificationState,
    ExistingLoanTransfer,
    ConversationState,
    EligibilityStatus,
    DisqualificationReason,
    PropertyType,
    OwnershipType,
    IncomeMode,
    OccupationType,
    Stage,
)


class EligibilityResult(BaseModel):
    """Structured result returned by the deterministic eligibility engine."""

    status: EligibilityStatus
    disqualification_reasons: List[DisqualificationReason] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    requires_cap_consent: bool = False
    is_transfer_lead: bool = False
    handoff_ready: bool = False
    summary_explanation: str = ""


class EligibilityEngine:
    """
    Deterministic rule engine evaluating qualification for Home Credit LAP (up to ₹75 Lakh).
    """

    MAX_LOAN_AMOUNT: float = settings.max_loan_amount_inr  # 75,000,000 INR
    MIN_TENURE: int = settings.min_tenure_years            # 3 years
    MAX_TENURE: int = settings.max_tenure_years            # 15 years

    @classmethod
    def evaluate(
        cls,
        qualification: QualificationState,
        transfer_intent: Optional[ExistingLoanTransfer] = None
    ) -> EligibilityResult:
        """
        Evaluate eligibility based on the 7 criteria and existing loan transfer intent.

        Args:
            qualification: Current collected qualification data.
            transfer_intent: Existing loan / EMI reduction details.

        Returns:
            EligibilityResult with calculated status, reasons, and missing fields.
        """
        disqualifications: List[DisqualificationReason] = []
        missing_fields: List[str] = []
        requires_cap_consent = False
        is_transfer_lead = False

        # 1. Existing Loan / EMI Reduction Transfer Routing Check
        if transfer_intent:
            if transfer_intent.has_existing_property_loan is True or transfer_intent.wants_emi_reduction is True:
                return EligibilityResult(
                    status=EligibilityStatus.TRANSFER_ROUTED,
                    disqualification_reasons=[],
                    missing_fields=[],
                    requires_cap_consent=False,
                    is_transfer_lead=True,
                    handoff_ready=False,
                    summary_explanation="Customer has existing property loan or seeks EMI reduction. Route to Loan-Transfer Specialist."
                )

        # 2. Immediate Disqualification Checks (Deterministic Mandatory Failure Rules)

        # Rule 1: Property Type (Agricultural disqualifies)
        if qualification.property_type == PropertyType.AGRICULTURAL:
            disqualifications.append(DisqualificationReason.AGRICULTURAL_PROPERTY)
        elif qualification.property_type is None:
            missing_fields.append("property_type")

        # Rule 2: Ownership (Sole or Joint allowed)
        if qualification.ownership == OwnershipType.UNAPPROVED:
            disqualifications.append(DisqualificationReason.UNQUALIFIED_OWNERSHIP)
        elif qualification.ownership is None:
            missing_fields.append("ownership")

        # Rule 3: Original Documents (Must be available)
        if qualification.has_original_documents is False:
            disqualifications.append(DisqualificationReason.MISSING_DOCUMENTS)
        elif qualification.has_original_documents is None:
            missing_fields.append("has_original_documents")

        # Rule 4: Loan Amount Cap (Max ₹75 Lakh)
        if qualification.requested_loan_amount_inr is not None:
            if qualification.requested_loan_amount_inr > cls.MAX_LOAN_AMOUNT:
                requires_cap_consent = True
                qualification.requires_amount_cap_consent = True
                if qualification.proceeds_at_cap is False:
                    disqualifications.append(DisqualificationReason.LOAN_AMOUNT_EXCEEDED_DECLINED)
                elif qualification.proceeds_at_cap is None:
                    missing_fields.append("proceeds_at_cap")
        else:
            missing_fields.append("requested_loan_amount_inr")

        # Rule 5: Occupation & Income Mode (Salaried/Self-Employed, Income mode MUST be bank)
        if qualification.income_mode == IncomeMode.CASH:
            disqualifications.append(DisqualificationReason.CASH_INCOME)
        elif qualification.income_mode is None:
            missing_fields.append("income_mode")

        if qualification.occupation_type == OccupationType.UNEMPLOYED:
            disqualifications.append(DisqualificationReason.UNQUALIFIED_OCCUPATION)
        elif qualification.occupation_type is None:
            missing_fields.append("occupation_type")

        # Rule 6: Property Market Value (Record customer's estimate; no minimum threshold invented)
        if qualification.property_market_value_inr is None:
            missing_fields.append("property_market_value_inr")
        elif qualification.property_market_value_inr <= 0:
            # Unrealistic non-positive market value input
            missing_fields.append("property_market_value_inr")

        # Rule 7: Tenure (3 to 15 years inclusive)
        if qualification.requested_tenure_years is not None:
            if qualification.requested_tenure_years < cls.MIN_TENURE or qualification.requested_tenure_years > cls.MAX_TENURE:
                disqualifications.append(DisqualificationReason.TENURE_OUT_OF_RANGE)
        else:
            missing_fields.append("requested_tenure_years")

        # 3. Final Result Classification

        # If any disqualification condition is met, customer is DISQUALIFIED
        if len(disqualifications) > 0:
            reasons_str = ", ".join([r.value for r in disqualifications])
            return EligibilityResult(
                status=EligibilityStatus.DISQUALIFIED,
                disqualification_reasons=disqualifications,
                missing_fields=missing_fields,
                requires_cap_consent=requires_cap_consent,
                is_transfer_lead=False,
                handoff_ready=False,
                summary_explanation=f"Customer disqualified due to: {reasons_str}."
            )

        # If there are missing fields or unconfirmed cap consent, qualification is INCOMPLETE
        if len(missing_fields) > 0:
            return EligibilityResult(
                status=EligibilityStatus.INCOMPLETE,
                disqualification_reasons=[],
                missing_fields=missing_fields,
                requires_cap_consent=requires_cap_consent,
                is_transfer_lead=False,
                handoff_ready=False,
                summary_explanation=f"Qualification incomplete. Remaining missing fields: {', '.join(missing_fields)}."
            )

        # All 7 criteria met and no disqualifications -> ELIGIBLE
        return EligibilityResult(
            status=EligibilityStatus.ELIGIBLE,
            disqualification_reasons=[],
            missing_fields=[],
            requires_cap_consent=requires_cap_consent,
            is_transfer_lead=False,
            handoff_ready=True,
            summary_explanation="Customer successfully qualified for Home Credit Loan Against Property. Ready for Senior Expert handoff."
        )


def update_conversation_eligibility(state: ConversationState) -> ConversationState:
    """
    Evaluate conversation state against eligibility engine and update state fields in-place.

    Args:
        state: ConversationState object to evaluate and update.

    Returns:
        Updated ConversationState object.
    """
    result = EligibilityEngine.evaluate(
        qualification=state.qualification,
        transfer_intent=state.transfer_intent
    )

    state.eligibility_status = result.status
    state.disqualification_reasons = result.disqualification_reasons
    state.missing_fields = result.missing_fields

    # Update conversation stage based on result
    if result.status == EligibilityStatus.TRANSFER_ROUTED:
        state.stage = Stage.TRANSFER_ROUTED
    elif result.status == EligibilityStatus.DISQUALIFIED:
        state.stage = Stage.DISQUALIFIED
    elif result.status == EligibilityStatus.ELIGIBLE:
        state.stage = Stage.ELIGIBLE_HANDOFF

    return state
