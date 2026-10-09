"""
CredVoice Conversation State & API Data Models.

Defines Pydantic models and Enums for customer verification, qualification fields,
eligibility assessments, transfer routing, and call stage management.
Unanswered fields remain None to cleanly distinguish missing values from negative responses.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported conversation languages."""
    ENGLISH = "en"
    HINDI = "hi"


class PropertyType(str, Enum):
    """Permitted and non-permitted property types for LAP."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    AGRICULTURAL = "agricultural"
    OTHER = "other"


class OwnershipType(str, Enum):
    """Property ownership structures."""
    SOLE = "sole"
    JOINT = "joint"
    UNAPPROVED = "unapproved"


class IncomeMode(str, Enum):
    """Mode of receiving income/salary."""
    BANK = "bank"  # Bank transfer, cheque, online deposit
    CASH = "cash"  # Cash in hand only


class OccupationType(str, Enum):
    """Customer occupation status."""
    SALARIED = "salaried"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"
    OTHER = "other"


class Stage(str, Enum):
    """Current stage of the voice conversation lifecycle."""
    VERIFICATION = "verification"
    QUALIFICATION = "qualification"
    ELIGIBLE_HANDOFF = "eligible_handoff"
    DISQUALIFIED = "disqualified"
    TRANSFER_ROUTED = "transfer_routed"
    CALLBACK_REQUESTED = "callback_requested"
    ENDED = "ended"


class EligibilityStatus(str, Enum):
    """Overall status resulting from deterministic rule evaluation."""
    INCOMPLETE = "incomplete"
    ELIGIBLE = "eligible"
    DISQUALIFIED = "disqualified"
    TRANSFER_ROUTED = "transfer_routed"


class DisqualificationReason(str, Enum):
    """Specific reasons for immediate disqualification."""
    AGRICULTURAL_PROPERTY = "agricultural_property"
    MISSING_DOCUMENTS = "missing_original_documents"
    CASH_INCOME = "cash_only_income"
    LOAN_AMOUNT_EXCEEDED_DECLINED = "loan_amount_exceeded_and_declined_cap"
    TENURE_OUT_OF_RANGE = "tenure_out_of_range"
    UNQUALIFIED_OWNERSHIP = "unqualified_ownership"
    UNQUALIFIED_OCCUPATION = "unqualified_occupation"


class CallOutcome(str, Enum):
    """Final business outcome of the call."""
    QUALIFIED_HANDOFF = "qualified_handoff"
    DISQUALIFIED = "disqualified"
    TRANSFER_ROUTED = "transfer_routed"
    CALLBACK_SCHEDULED = "callback_scheduled"
    NOT_AVAILABLE = "not_available"
    WRONG_NUMBER = "wrong_number"


class CustomerVerification(BaseModel):
    """Customer verification and immediate call availability status."""
    is_identity_verified: Optional[bool] = Field(
        default=None,
        description="Whether the customer confirmed their identity."
    )
    is_available_to_speak: Optional[bool] = Field(
        default=None,
        description="Whether the customer is available for the call right now."
    )
    customer_name: Optional[str] = Field(
        default=None,
        description="Name of the pre-approved customer."
    )


class QualificationState(BaseModel):
    """
    The seven eligibility data points required for Home Credit LAP qualification.
    All fields default to None until explicitly collected from the customer.
    """
    property_type: Optional[PropertyType] = Field(
        default=None,
        description="Type of property offered as collateral (Residential/Commercial/Industrial vs Agricultural)."
    )
    ownership: Optional[OwnershipType] = Field(
        default=None,
        description="Property ownership type (Sole or Joint)."
    )
    has_original_documents: Optional[bool] = Field(
        default=None,
        description="Whether original property documents are available for verification."
    )
    requested_loan_amount_inr: Optional[float] = Field(
        default=None,
        description="Customer's desired loan amount in INR."
    )
    requires_amount_cap_consent: bool = Field(
        default=False,
        description="True if requested amount > ₹75 Lakh and requires customer consent for ₹75 Lakh cap."
    )
    proceeds_at_cap: Optional[bool] = Field(
        default=None,
        description="Customer's explicit consent to proceed at ₹75 Lakh when requested > ₹75 Lakh."
    )
    occupation_type: Optional[OccupationType] = Field(
        default=None,
        description="Occupation category (Salaried or Self-Employed)."
    )
    income_mode: Optional[IncomeMode] = Field(
        default=None,
        description="Mode of income receipt (Bank transfer vs Cash)."
    )
    property_market_value_inr: Optional[float] = Field(
        default=None,
        description="Customer's estimated market value of the property in INR."
    )
    requested_tenure_years: Optional[int] = Field(
        default=None,
        description="Requested loan tenure in years (3-15 years inclusive)."
    )


class ExistingLoanTransfer(BaseModel):
    """Fields tracking existing property loan transfer or EMI reduction intent."""
    has_existing_property_loan: Optional[bool] = Field(
        default=None,
        description="Whether customer already has an active property loan with another lender."
    )
    wants_emi_reduction: Optional[bool] = Field(
        default=None,
        description="Whether customer is seeking to balance transfer to reduce existing EMIs."
    )


class CallbackPreference(BaseModel):
    """Details captured when customer requests a callback."""
    callback_requested: bool = False
    preferred_time: Optional[str] = Field(
        default=None,
        description="Date and time requested by the customer for callback."
    )
    contact_number: Optional[str] = Field(
        default=None,
        description="Phone number for callback."
    )


class ConversationState(BaseModel):
    """Complete structured conversation state for a session."""
    session_id: str
    customer_id: str
    language: Language = Language.ENGLISH
    stage: Stage = Stage.VERIFICATION
    verification: CustomerVerification = Field(default_factory=CustomerVerification)
    qualification: QualificationState = Field(default_factory=QualificationState)
    transfer_intent: ExistingLoanTransfer = Field(default_factory=ExistingLoanTransfer)
    callback: CallbackPreference = Field(default_factory=CallbackPreference)
    eligibility_status: EligibilityStatus = EligibilityStatus.INCOMPLETE
    disqualification_reasons: List[DisqualificationReason] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    call_outcome: Optional[CallOutcome] = None
