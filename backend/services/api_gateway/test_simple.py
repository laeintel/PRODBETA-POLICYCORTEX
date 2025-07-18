"""
Simple baseline test for API Gateway.
"""

import pytest
from fastapi.testclient import TestClient
from main_simple import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_readiness_check():
    """Test readiness check endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "timestamp" in data

def test_404_endpoint():
    """Test non-existent endpoint."""
    response = client.get("/nonexistent")
    assert response.status_code == 404

if __name__ == "__main__":
    pytest.main([__file__, "-v"])