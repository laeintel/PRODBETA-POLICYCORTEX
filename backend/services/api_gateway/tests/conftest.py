"""
Test configuration and fixtures for API Gateway service.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth import AuthManager
from circuit_breaker import CircuitBreaker
from main import app
from rate_limiter import RateLimiter


@pytest.fixture
def client():
    """Create test client for API Gateway."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_auth_manager():
    """Mock AuthManager for testing."""
    with patch("main.auth_manager") as mock_auth:
        mock_auth.verify_token = AsyncMock(
            return_value={"id": "test-user-id", "email": "test@example.com", "name": "Test User"}
        )
        yield mock_auth


@pytest.fixture
def mock_rate_limiter():
    """Mock RateLimiter for testing."""
    with patch("main.rate_limiter") as mock_limiter:
        mock_limiter.is_allowed = AsyncMock(return_value=(True, None))
        yield mock_limiter


@pytest.fixture
def mock_circuit_breaker():
    """Mock CircuitBreaker for testing."""
    mock_cb = MagicMock()
    mock_cb.can_execute.return_value = True
    mock_cb.record_success = MagicMock()
    mock_cb.record_failure = MagicMock()
    mock_cb.state = "CLOSED"
    return mock_cb


@pytest.fixture
def mock_service_registry(mock_circuit_breaker):
    """Mock service registry for testing."""
    return {
        "azure-integration": {
            "url": "http://localhost:8001",
            "health_path": "/health",
            "timeout": 30,
            "circuit_breaker": mock_circuit_breaker,
        },
        "ai-engine": {
            "url": "http://localhost:8002",
            "health_path": "/health",
            "timeout": 60,
            "circuit_breaker": mock_circuit_breaker,
        },
    }


@pytest.fixture
def mock_httpx_response():
    """Mock httpx response for testing."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.content = b'{"status": "success"}'
    mock_response.text = '{"status": "success"}'
    return mock_response


@pytest.fixture
def auth_headers():
    """Authentication headers for testing."""
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def invalid_auth_headers():
    """Invalid authentication headers for testing."""
    return {"Authorization": "Bearer invalid-token"}


@pytest.fixture
def mock_prometheus_metrics():
    """Mock Prometheus metrics for testing."""
    with patch("main.REQUEST_COUNT") as mock_counter, patch(
        "main.REQUEST_DURATION"
    ) as mock_histogram, patch("main.SERVICE_REQUESTS") as mock_service_counter:
        # Mock counter methods
        mock_counter.labels.return_value.inc = MagicMock()
        mock_histogram.observe = MagicMock()
        mock_service_counter.labels.return_value.inc = MagicMock()

        yield {
            "request_count": mock_counter,
            "request_duration": mock_histogram,
            "service_requests": mock_service_counter,
        }
