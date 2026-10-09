"""
Integration tests for CredVoice API Gateway endpoints.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint status and basic project metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "CredVoice"
    assert "version" in data


def test_health_endpoint():
    """Test GET /health returns status healthy and runtime configuration."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "CredVoice"
    assert "voice_provider" in data
    assert "llm_provider" in data
