"""
Simple baseline test for Azure Integration Service.
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

def test_metrics_endpoint():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "content-type" in response.headers
    # Basic check that it looks like Prometheus metrics
    assert "# TYPE" in response.text or "# HELP" in response.text

if __name__ == "__main__":
    pytest.main([__file__, "-v"])