"""
CredVoice — Unit and Integration Tests for Custom Tools API (tests/unit/test_tools.py).

Verifies the Bolna custom tool endpoint: POST /api/v1/tools/check-lap-eligibility.
Covers:
- Eligible customer payload (all 7 rules met)
- Agricultural property disqualification
- Cash-only income mode disqualification
- Missing original documents disqualification
- Loan amount exceeding ₹75 Lakh with and without cap consent
- Missing and partial fields (incomplete qualification)
- Invalid field values (validation error)
- Existing property loan and EMI reduction transfer routing
- Authentication: valid API key, missing API key, invalid API key, Bearer token
- Production fail-closed security
- Secret sanitization: ensures credentials are never returned in errors or leaked
"""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models.conversation import EligibilityStatus

AUTH_KEY = getattr(settings, "bolna_api_key", "") or "test-bolna-key"
client = TestClient(app, headers={"x-api-key": AUTH_KEY})

ENDPOINT = "/api/v1/tools/check-lap-eligibility"


# ---------------------------------------------------------------------------
# Fixture: Base Eligible Payload
# ---------------------------------------------------------------------------

@pytest.fixture
def eligible_payload() -> dict:
    """A valid payload that meets all 7 LAP qualification criteria."""
    return {
        "property_type": "residential",
        "ownership": "sole",
        "has_original_documents": True,
        "requested_loan_amount_inr": 5000000.0,  # ₹50 Lakh
        "proceeds_at_cap": None,
        "occupation_type": "salaried",
        "income_mode": "bank",
        "property_market_value_inr": 12000000.0,  # ₹1.2 Crore
        "requested_tenure_years": 10,
        "has_existing_property_loan": False,
        "wants_emi_reduction": False,
    }


# ---------------------------------------------------------------------------
# Test 1: Eligible Customer
# ---------------------------------------------------------------------------

def test_check_lap_eligibility_eligible(eligible_payload):
    """Fully compliant customer returns ELIGIBLE, handoff_ready=True, no disqualifications."""
    response = client.post(ENDPOINT, json=eligible_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == EligibilityStatus.ELIGIBLE.value
    assert data["is_eligible"] is True
    assert data["handoff_ready"] is True
    assert data["is_transfer_lead"] is False
    assert data["requires_cap_consent"] is False
    assert data["disqualification_reasons"] == []
    assert data["missing_fields"] == []
    assert "qualified" in data["summary_explanation"].lower()
    assert "congratulations" in data["recommended_spoken_reply"].lower()


# ---------------------------------------------------------------------------
# Test 2: Agricultural Property Disqualification
# ---------------------------------------------------------------------------

def test_check_lap_eligibility_agricultural_disqualification(eligible_payload):
    """Agricultural property collateral causes immediate disqualification."""
    eligible_payload["property_type"] = "agricultural"
    response = client.post(ENDPOINT, json=eligible_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == EligibilityStatus.DISQUALIFIED.value
    assert data["is_eligible"] is False
    assert data["handoff_ready"] is False
    assert "agricultural_property" in data["disqualification_reasons"]
    assert "agricultural" in data["recommended_spoken_reply"].lower()


# ---------------------------------------------------------------------------
# Test 3: Cash Income Disqualification
# ---------------------------------------------------------------------------

def test_check_lap_eligibility_cash_income_disqualification(eligible_payload):
    """Cash-only income mode causes immediate disqualification."""
    eligible_payload["income_mode"] = "cash"
    response = client.post(ENDPOINT, json=eligible_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == EligibilityStatus.DISQUALIFIED.value
    assert data["is_eligible"] is False
    assert data["handoff_ready"] is False
    assert "cash_only_income" in data["disqualification_reasons"]
    assert "cash" in data["recommended_spoken_reply"].lower()


# ---------------------------------------------------------------------------
# Test 4: Missing Original Documents Disqualification
# ---------------------------------------------------------------------------

def test_check_lap_eligibility_missing_documents_disqualification(eligible_payload):
    """Unavailable original registry documents causes immediate disqualification."""
    eligible_payload["has_original_documents"] = False
    response = client.post(ENDPOINT, json=eligible_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == EligibilityStatus.DISQUALIFIED.value
    assert data["is_eligible"] is False
    assert "missing_original_documents" in data["disqualification_reasons"]
    assert "original" in data["recommended_spoken_reply"].lower()


# ---------------------------------------------------------------------------
# Test 5: Loan Amount Exceeds ₹75 Lakh - Cap Consent Scenarios
# ---------------------------------------------------------------------------

def test_check_lap_eligibility_exceeds_cap_consent_accepted(eligible_payload):
    """Amount > ₹75L with proceeds_at_cap=True qualifies at ₹75 Lakh cap."""
    eligible_payload["requested_loan_amount_inr"] = 9000000.0  # ₹90 Lakh
    eligible_payload["proceeds_at_cap"] = True

    response = client.post(ENDPOINT, json=eligible_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == EligibilityStatus.ELIGIBLE.value
    assert data["is_eligible"] is True
    assert data["handoff_ready"] is True
    assert data["requires_cap_consent"] is True
    assert data["disqualification_reasons"] == []


def test_check_lap_eligibility_exceeds_cap_consent_declined(eligible_payload):
    """Amount > ₹75L with proceeds_at_cap=False disqualifies."""
    eligible_payload["requested_loan_amount_inr"] = 9000000.0  # ₹90 Lakh
    eligible_payload["proceeds_at_cap"] = False

    response = client.post(ENDPOINT, json=eligible_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == EligibilityStatus.DISQUALIFIED.value
    assert data["is_eligible"] is False
    assert "loan_amount_exceeded_and_declined_cap" in data["disqualification_reasons"]


def test_check_lap_eligibility_exceeds_cap_consent_pending(eligible_payload):
    """Amount > ₹75L with proceeds_at_cap=None returns INCOMPLETE requesting consent."""
    eligible_payload["requested_loan_amount_inr"] = 9000000.0  # ₹90 Lakh
    eligible_payload["proceeds_at_cap"] = None

    response = client.post(ENDPOINT, json=eligible_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == EligibilityStatus.INCOMPLETE.value
    assert data["is_eligible"] is False
    assert data["requires_cap_consent"] is True
    assert "proceeds_at_cap" in data["missing_fields"]
    assert "75 lakh" in data["recommended_spoken_reply"].lower()


# ---------------------------------------------------------------------------
# Test 6: Incomplete / Missing Fields Handling
# ---------------------------------------------------------------------------

def test_check_lap_eligibility_partial_payload():
    """Empty or partial request returns INCOMPLETE with list of missing fields."""
    partial_payload = {
        "property_type": "residential",
        "ownership": "sole",
    }
    response = client.post(ENDPOINT, json=partial_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == EligibilityStatus.INCOMPLETE.value
    assert data["is_eligible"] is False
    assert data["handoff_ready"] is False
    assert len(data["missing_fields"]) > 0
    assert "has_original_documents" in data["missing_fields"]
    assert "requested_loan_amount_inr" in data["missing_fields"]


# ---------------------------------------------------------------------------
# Test 7: Invalid Field Types / Enums Validation Error
# ---------------------------------------------------------------------------

def test_check_lap_eligibility_invalid_enum_validation():
    """Invalid enum value triggers 422 Unprocessable Entity error safely."""
    invalid_payload = {
        "property_type": "invalid_property_type_xyz",
    }
    response = client.post(ENDPOINT, json=invalid_payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 8: Transfer Routing (Existing Loan / EMI Reduction)
# ---------------------------------------------------------------------------

def test_check_lap_eligibility_existing_loan_transfer(eligible_payload):
    """has_existing_property_loan=True immediately routes to Balance Transfer Specialist."""
    eligible_payload["has_existing_property_loan"] = True
    response = client.post(ENDPOINT, json=eligible_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == EligibilityStatus.TRANSFER_ROUTED.value
    assert data["is_transfer_lead"] is True
    assert data["is_eligible"] is False
    assert data["handoff_ready"] is False
    assert "transfer" in data["recommended_spoken_reply"].lower()


def test_check_lap_eligibility_emi_reduction_transfer(eligible_payload):
    """wants_emi_reduction=True immediately routes to Balance Transfer Specialist."""
    eligible_payload["wants_emi_reduction"] = True
    response = client.post(ENDPOINT, json=eligible_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == EligibilityStatus.TRANSFER_ROUTED.value
    assert data["is_transfer_lead"] is True
    assert data["is_eligible"] is False
    assert data["handoff_ready"] is False


# ---------------------------------------------------------------------------
# Test 9: Authentication & Security
# ---------------------------------------------------------------------------

def test_tool_auth_valid_api_key_header(monkeypatch, eligible_payload):
    """Valid x-api-key header successfully authenticates."""
    secret_key = "bolna-tool-test-key-12345"
    monkeypatch.setattr(settings, "bolna_api_key", secret_key)

    response = client.post(
        ENDPOINT,
        json=eligible_payload,
        headers={"x-api-key": secret_key},
    )
    assert response.status_code == 200
    assert response.json()["status"] == EligibilityStatus.ELIGIBLE.value


def test_tool_auth_valid_bearer_token(monkeypatch, eligible_payload):
    """Valid Authorization: Bearer <key> header successfully authenticates."""
    secret_key = "bolna-tool-bearer-key-999"
    monkeypatch.setattr(settings, "bolna_api_key", secret_key)

    bearer_client = TestClient(app)
    response = bearer_client.post(
        ENDPOINT,
        json=eligible_payload,
        headers={"Authorization": f"Bearer {secret_key}"},
    )
    assert response.status_code == 200


def test_tool_auth_invalid_api_key_rejected(monkeypatch, eligible_payload):
    """Incorrect API key header returns 401 Unauthorized."""
    secret_key = "bolna-tool-test-key-12345"
    monkeypatch.setattr(settings, "bolna_api_key", secret_key)

    response = client.post(
        ENDPOINT,
        json=eligible_payload,
        headers={"x-api-key": "wrong_key_xyz"},
    )
    assert response.status_code == 401
    assert "invalid or missing" in response.json()["detail"].lower()
    # Confirm secret is NEVER revealed in error body
    assert secret_key not in response.text


def test_tool_auth_missing_api_key_when_configured(monkeypatch, eligible_payload):
    """Missing API key when bolna_api_key is set returns 401 Unauthorized."""
    secret_key = "bolna-tool-test-key-12345"
    monkeypatch.setattr(settings, "bolna_api_key", secret_key)

    unauthed_client = TestClient(app)
    response = unauthed_client.post(ENDPOINT, json=eligible_payload)
    assert response.status_code == 401
    assert "invalid or missing" in response.json()["detail"].lower()


def test_tool_auth_fails_closed_in_production(monkeypatch, eligible_payload):
    """In production mode without configured bolna_api_key, request fails closed with 401."""
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "bolna_api_key", "")
    monkeypatch.setattr(settings, "admin_api_key", "")

    unauthed_client = TestClient(app)
    response = unauthed_client.post(ENDPOINT, json=eligible_payload)
    assert response.status_code == 401
    assert "unconfigured" in response.json()["detail"].lower()
