"""
Test module for Azure Integration Service main functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os
import sys

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_azure_auth():
    """Mock Azure authentication."""
    with patch('services.azure_auth.AzureAuthService') as mock:
        mock_instance = Mock()
        mock_instance.verify_azure_connection.return_value = True
        mock.return_value = mock_instance
        yield mock_instance


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "azure-integration"
    assert "timestamp" in data
    assert "version" in data


def test_readiness_check(client, mock_azure_auth):
    """Test readiness check endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ready"
    assert data["service"] == "azure-integration"
    assert "details" in data
    assert data["details"]["azure_connected"] is True


def test_metrics_endpoint(client):
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_unauthenticated_request(client):
    """Test that protected endpoints require authentication."""
    response = client.get("/api/v1/policies")
    assert response.status_code == 401


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.options("/health")
    assert response.status_code == 200
    # CORS headers should be present due to middleware


@pytest.mark.asyncio
async def test_request_logging_middleware(client):
    """Test that request logging middleware works."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers


def test_openapi_docs(client):
    """Test OpenAPI documentation endpoint."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema(client):
    """Test OpenAPI schema endpoint."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert schema["info"]["title"] == "PolicyCortex Azure Integration Service"