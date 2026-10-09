"""
Unit tests for CredVoice Deterministic Eligibility Engine.

Validates the 7 qualification rules, immediate disqualification conditions,
amount capping consent flow, and transfer specialist routing logic.
"""

import pytest
from app.models.conversation import (
    QualificationState,
    ExistingLoanTransfer,
    PropertyType,
    OwnershipType,
    IncomeMode,
    OccupationType,
    EligibilityStatus,
    DisqualificationReason,
)
from app.services.eligibility import EligibilityEngine


def test_incomplete_qualification():
    """Test that a partially filled qualification state returns INCOMPLETE."""
    state = QualificationState(
        property_type=PropertyType.RESIDENTIAL,
        ownership=OwnershipType.SOLE,
        has_original_documents=True,
        requested_loan_amount_inr=5000000.0,
        # Missing occupation_type, income_mode, property_market_value_inr, requested_tenure_years
    )
    result = EligibilityEngine.evaluate(state)
    assert result.status == EligibilityStatus.INCOMPLETE
    assert result.handoff_ready is False
    assert len(result.missing_fields) > 0
    assert "income_mode" in result.missing_fields
    assert "requested_tenure_years" in result.missing_fields


def test_eligible_customer():
    """Test that meeting all 7 eligibility criteria returns ELIGIBLE and handoff_ready=True."""
    state = QualificationState(
        property_type=PropertyType.RESIDENTIAL,
        ownership=OwnershipType.SOLE,
        has_original_documents=True,
        requested_loan_amount_inr=5000000.0,
        occupation_type=OccupationType.SALARIED,
        income_mode=IncomeMode.BANK,
        property_market_value_inr=10000000.0,
        requested_tenure_years=10,
    )
    result = EligibilityEngine.evaluate(state)
    assert result.status == EligibilityStatus.ELIGIBLE
    assert result.handoff_ready is True
    assert len(result.disqualification_reasons) == 0
    assert len(result.missing_fields) == 0


def test_agricultural_property_disqualification():
    """Test that an agricultural property immediately disqualifies the customer."""
    state = QualificationState(
        property_type=PropertyType.AGRICULTURAL,
        ownership=OwnershipType.SOLE,
        has_original_documents=True,
        requested_loan_amount_inr=3000000.0,
        occupation_type=OccupationType.SELF_EMPLOYED,
        income_mode=IncomeMode.BANK,
        property_market_value_inr=8000000.0,
        requested_tenure_years=5,
    )
    result = EligibilityEngine.evaluate(state)
    assert result.status == EligibilityStatus.DISQUALIFIED
    assert result.handoff_ready is False
    assert DisqualificationReason.AGRICULTURAL_PROPERTY in result.disqualification_reasons


def test_missing_original_documents_disqualification():
    """Test that missing original property documents causes immediate disqualification."""
    state = QualificationState(
        property_type=PropertyType.COMMERCIAL,
        ownership=OwnershipType.JOINT,
        has_original_documents=False,  # Documents not available
        requested_loan_amount_inr=4000000.0,
        occupation_type=OccupationType.SALARIED,
        income_mode=IncomeMode.BANK,
        property_market_value_inr=9000000.0,
        requested_tenure_years=7,
    )
    result = EligibilityEngine.evaluate(state)
    assert result.status == EligibilityStatus.DISQUALIFIED
    assert result.handoff_ready is False
    assert DisqualificationReason.MISSING_DOCUMENTS in result.disqualification_reasons


def test_cash_only_income_disqualification():
    """Test that cash-only income mode causes immediate disqualification."""
    state = QualificationState(
        property_type=PropertyType.RESIDENTIAL,
        ownership=OwnershipType.SOLE,
        has_original_documents=True,
        requested_loan_amount_inr=2000000.0,
        occupation_type=OccupationType.SELF_EMPLOYED,
        income_mode=IncomeMode.CASH,  # Cash-only income
        property_market_value_inr=6000000.0,
        requested_tenure_years=5,
    )
    result = EligibilityEngine.evaluate(state)
    assert result.status == EligibilityStatus.DISQUALIFIED
    assert result.handoff_ready is False
    assert DisqualificationReason.CASH_INCOME in result.disqualification_reasons


def test_loan_amount_exceeding_75_lakh_requires_consent():
    """Test that requesting > ₹75 Lakh requires explicit customer consent for ₹75 Lakh cap."""
    state = QualificationState(
        property_type=PropertyType.INDUSTRIAL,
        ownership=OwnershipType.SOLE,
        has_original_documents=True,
        requested_loan_amount_inr=10000000.0,  # ₹1 Crore (exceeds ₹75 Lakh)
        proceeds_at_cap=None,  # Consent not yet asked/given
        occupation_type=OccupationType.SALARIED,
        income_mode=IncomeMode.BANK,
        property_market_value_inr=20000000.0,
        requested_tenure_years=12,
    )
    result = EligibilityEngine.evaluate(state)
    assert result.requires_cap_consent is True
    assert result.status == EligibilityStatus.INCOMPLETE
    assert "proceeds_at_cap" in result.missing_fields


def test_loan_amount_exceeding_75_lakh_declined_disqualifies():
    """Test that declining consent for ₹75 Lakh cap disqualifies the customer."""
    state = QualificationState(
        property_type=PropertyType.RESIDENTIAL,
        ownership=OwnershipType.SOLE,
        has_original_documents=True,
        requested_loan_amount_inr=10000000.0,  # Exceeds cap
        proceeds_at_cap=False,  # Explicitly declined to proceed at ₹75 Lakh
        occupation_type=OccupationType.SALARIED,
        income_mode=IncomeMode.BANK,
        property_market_value_inr=20000000.0,
        requested_tenure_years=12,
    )
    result = EligibilityEngine.evaluate(state)
    assert result.status == EligibilityStatus.DISQUALIFIED
    assert DisqualificationReason.LOAN_AMOUNT_EXCEEDED_DECLINED in result.disqualification_reasons


def test_loan_amount_exceeding_75_lakh_accepted_qualifies():
    """Test that accepting consent for ₹75 Lakh cap allows eligibility when other rules pass."""
    state = QualificationState(
        property_type=PropertyType.RESIDENTIAL,
        ownership=OwnershipType.SOLE,
        has_original_documents=True,
        requested_loan_amount_inr=10000000.0,  # Exceeds cap
        proceeds_at_cap=True,  # Agreed to proceed at ₹75 Lakh
        occupation_type=OccupationType.SALARIED,
        income_mode=IncomeMode.BANK,
        property_market_value_inr=20000000.0,
        requested_tenure_years=12,
    )
    result = EligibilityEngine.evaluate(state)
    assert result.status == EligibilityStatus.ELIGIBLE
    assert result.handoff_ready is True


def test_tenure_out_of_range_disqualifies():
    """Test that tenure < 3 years or > 15 years causes disqualification."""
    # Case A: Tenure < 3 years
    state_low = QualificationState(
        property_type=PropertyType.RESIDENTIAL,
        ownership=OwnershipType.SOLE,
        has_original_documents=True,
        requested_loan_amount_inr=3000000.0,
        occupation_type=OccupationType.SALARIED,
        income_mode=IncomeMode.BANK,
        property_market_value_inr=8000000.0,
        requested_tenure_years=2,  # Too low
    )
    res_low = EligibilityEngine.evaluate(state_low)
    assert res_low.status == EligibilityStatus.DISQUALIFIED
    assert DisqualificationReason.TENURE_OUT_OF_RANGE in res_low.disqualification_reasons

    # Case B: Tenure > 15 years
    state_high = QualificationState(
        property_type=PropertyType.RESIDENTIAL,
        ownership=OwnershipType.SOLE,
        has_original_documents=True,
        requested_loan_amount_inr=3000000.0,
        occupation_type=OccupationType.SALARIED,
        income_mode=IncomeMode.BANK,
        property_market_value_inr=8000000.0,
        requested_tenure_years=20,  # Too high
    )
    res_high = EligibilityEngine.evaluate(state_high)
    assert res_high.status == EligibilityStatus.DISQUALIFIED
    assert DisqualificationReason.TENURE_OUT_OF_RANGE in res_high.disqualification_reasons


def test_existing_property_loan_transfer_routing():
    """Test that existing property loan or EMI reduction intent routes to Loan-Transfer Specialist."""
    state = QualificationState(
        property_type=PropertyType.RESIDENTIAL,
        ownership=OwnershipType.SOLE,
        has_original_documents=True,
        requested_loan_amount_inr=5000000.0,
        occupation_type=OccupationType.SALARIED,
        income_mode=IncomeMode.BANK,
        property_market_value_inr=10000000.0,
        requested_tenure_years=10,
    )
    transfer_intent = ExistingLoanTransfer(
        has_existing_property_loan=True,
        wants_emi_reduction=True
    )
    result = EligibilityEngine.evaluate(qualification=state, transfer_intent=transfer_intent)
    assert result.status == EligibilityStatus.TRANSFER_ROUTED
    assert result.is_transfer_lead is True
    assert result.handoff_ready is False


def test_property_value_recorded_without_invented_minimum_threshold():
    """Test that any positive property market value is recorded without enforcing unstated thresholds."""
    state = QualificationState(
        property_type=PropertyType.RESIDENTIAL,
        ownership=OwnershipType.SOLE,
        has_original_documents=True,
        requested_loan_amount_inr=1000000.0,
        occupation_type=OccupationType.SALARIED,
        income_mode=IncomeMode.BANK,
        property_market_value_inr=1500000.0,  # Modest estimate recorded
        requested_tenure_years=5,
    )
    result = EligibilityEngine.evaluate(state)
    assert result.status == EligibilityStatus.ELIGIBLE
    assert "property_market_value_inr" not in result.missing_fields
