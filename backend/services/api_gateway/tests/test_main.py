"""
Unit tests for API Gateway main module.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.services.api_gateway.main import (
    app,
    check_rate_limit,
    proxy_request,
    verify_authentication,
)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "api-gateway"
        assert "timestamp" in data
        assert "version" in data

    @patch("backend.services.api_gateway.main.httpx.AsyncClient")
    def test_readiness_check_all_services_healthy(self, mock_client, client):
        """Test readiness check when all services are healthy."""
        # Mock healthy responses from all services
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "details" in data

    @patch("backend.services.api_gateway.main.httpx.AsyncClient")
    def test_readiness_check_some_services_unhealthy(self, mock_client, client):
        """Test readiness check when some services are unhealthy."""
        # Mock unhealthy responses from some services
        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        response = client.get("/ready")
        assert response.status_code == 503

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"


class TestAuthentication:
    """Test authentication functionality."""

    def test_verify_authentication_success(self, mock_auth_manager):
        """Test successful authentication."""
        # This would be tested in an async context
        pass

    def test_verify_authentication_no_credentials(self, client):
        """Test authentication failure with no credentials."""
        response = client.get("/api/v1/gateway/services")
        assert response.status_code == 401
        data = response.json()
        assert "Authentication required" in data["detail"]

    def test_verify_authentication_invalid_token(self, client):
        """Test authentication failure with invalid token."""
        headers = {"Authorization": "Bearer invalid-token"}

        with patch("backend.services.api_gateway.main.auth_manager") as mock_auth:
            mock_auth.verify_token = AsyncMock(side_effect=Exception("Invalid token"))
            response = client.get("/api/v1/gateway/services", headers=headers)
            assert response.status_code == 401

    def test_skip_authentication_for_public_endpoints(self, client):
        """Test that public endpoints skip authentication."""
        public_endpoints = ["/health", "/ready", "/metrics"]

        for endpoint in public_endpoints:
            response = client.get(endpoint)
            assert response.status_code in [200, 503]  # 503 for readiness check that might fail


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limit_allowed(self, mock_rate_limiter):
        """Test rate limiting when request is allowed."""
        # Mock request object
        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.state.user = {"id": "test-user"}

        # Should not raise exception
        await check_rate_limit(mock_request)

        mock_rate_limiter.is_allowed.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, mock_rate_limiter):
        """Test rate limiting when limit is exceeded."""
        mock_rate_limiter.is_allowed = AsyncMock(return_value=(False, "2023-01-01T00:00:00Z"))

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.state.user = {"id": "test-user"}

        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit(mock_request)

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value.detail)


class TestProxyRequest:
    """Test request proxying functionality."""

    @pytest.mark.asyncio
    async def test_proxy_request_success(self, mock_service_registry, mock_httpx_response):
        """Test successful request proxying."""
        with patch(
            "backend.services.api_gateway.main.SERVICE_REGISTRY", mock_service_registry
        ), patch("backend.services.api_gateway.main.httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.request.return_value = mock_httpx_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            mock_request = MagicMock()
            mock_request.headers = {"user-agent": "test"}
            mock_request.state.request_id = "test-request-id"
            mock_request.body = AsyncMock(return_value=b"test body")
            mock_request.query_params = {}

            result = await proxy_request("azure-integration", "/test", mock_request, "GET")

            assert result["status_code"] == 200
            assert result["content"] == b'{"status": "success"}'
            mock_client_instance.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_proxy_request_service_not_found(self, mock_service_registry):
        """Test proxy request with non-existent service."""
        with patch("backend.services.api_gateway.main.SERVICE_REGISTRY", mock_service_registry):
            mock_request = MagicMock()

            with pytest.raises(HTTPException) as exc_info:
                await proxy_request("non-existent-service", "/test", mock_request, "GET")

            assert exc_info.value.status_code == 404
            assert "Service 'non-existent-service' not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_proxy_request_circuit_breaker_open(self, mock_service_registry):
        """Test proxy request when circuit breaker is open."""
        # Mock circuit breaker in open state
        mock_service_registry["azure-integration"][
            "circuit_breaker"
        ].can_execute.return_value = False

        with patch("backend.services.api_gateway.main.SERVICE_REGISTRY", mock_service_registry):
            mock_request = MagicMock()

            with pytest.raises(HTTPException) as exc_info:
                await proxy_request("azure-integration", "/test", mock_request, "GET")

            assert exc_info.value.status_code == 503
            assert "currently unavailable" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_proxy_request_network_error(self, mock_service_registry):
        """Test proxy request with network error."""
        with patch(
            "backend.services.api_gateway.main.SERVICE_REGISTRY", mock_service_registry
        ), patch("backend.services.api_gateway.main.httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.request.side_effect = Exception("Network error")
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            mock_request = MagicMock()
            mock_request.headers = {"user-agent": "test"}
            mock_request.state.request_id = "test-request-id"
            mock_request.body = AsyncMock(return_value=b"test body")
            mock_request.query_params = {}

            with pytest.raises(HTTPException) as exc_info:
                await proxy_request("azure-integration", "/test", mock_request, "GET")

            assert exc_info.value.status_code == 503
            assert "request failed" in str(exc_info.value.detail)

            # Verify circuit breaker recorded failure
            mock_service_registry["azure-integration"][
                "circuit_breaker"
            ].record_failure.assert_called_once()


class TestServiceRoutes:
    """Test service route endpoints."""

    def test_azure_proxy_route(self, client, auth_headers):
        """Test Azure integration proxy route."""
        with patch("backend.services.api_gateway.main.proxy_request") as mock_proxy, patch(
            "backend.services.api_gateway.main.verify_authentication"
        ) as mock_auth, patch(
            "backend.services.api_gateway.main.check_rate_limit"
        ) as mock_rate_limit:
            mock_auth.return_value = {"id": "test-user"}
            mock_rate_limit.return_value = None
            mock_proxy.return_value = {
                "status_code": 200,
                "content": b'{"success": true}',
                "headers": {"content-type": "application/json"},
            }

            response = client.get("/api/v1/azure/resources", headers=auth_headers)
            assert response.status_code == 200
            mock_proxy.assert_called_once()

    def test_ai_proxy_route(self, client, auth_headers):
        """Test AI engine proxy route."""
        with patch("backend.services.api_gateway.main.proxy_request") as mock_proxy, patch(
            "backend.services.api_gateway.main.verify_authentication"
        ) as mock_auth, patch(
            "backend.services.api_gateway.main.check_rate_limit"
        ) as mock_rate_limit:
            mock_auth.return_value = {"id": "test-user"}
            mock_rate_limit.return_value = None
            mock_proxy.return_value = {
                "status_code": 200,
                "content": b'{"success": true}',
                "headers": {"content-type": "application/json"},
            }

            response = client.get("/api/v1/ai/models", headers=auth_headers)
            assert response.status_code == 200
            mock_proxy.assert_called_once()

    def test_conversation_proxy_route(self, client, auth_headers):
        """Test conversation proxy route."""
        with patch("backend.services.api_gateway.main.proxy_request") as mock_proxy, patch(
            "backend.services.api_gateway.main.verify_authentication"
        ) as mock_auth, patch(
            "backend.services.api_gateway.main.check_rate_limit"
        ) as mock_rate_limit:
            mock_auth.return_value = {"id": "test-user"}
            mock_rate_limit.return_value = None
            mock_proxy.return_value = {
                "status_code": 200,
                "content": b'{"success": true}',
                "headers": {"content-type": "application/json"},
            }

            response = client.get("/api/v1/chat/conversations", headers=auth_headers)
            assert response.status_code == 200
            mock_proxy.assert_called_once()


class TestManagementEndpoints:
    """Test gateway management endpoints."""

    def test_get_service_registry(self, client, auth_headers, mock_service_registry):
        """Test get service registry endpoint."""
        with patch(
            "backend.services.api_gateway.main.SERVICE_REGISTRY", mock_service_registry
        ), patch("backend.services.api_gateway.main.verify_authentication") as mock_auth:
            mock_auth.return_value = {"id": "test-user"}

            response = client.get("/api/v1/gateway/services", headers=auth_headers)
            assert response.status_code == 200

            data = response.json()
            assert "azure-integration" in data
            assert "ai-engine" in data
            assert data["azure-integration"]["status"] == "healthy"

    def test_get_gateway_metrics(self, client, auth_headers, mock_prometheus_metrics):
        """Test get gateway metrics endpoint."""
        with patch("backend.services.api_gateway.main.verify_authentication") as mock_auth, patch(
            "backend.services.api_gateway.main.SERVICE_REGISTRY", {}
        ) as mock_registry:
            mock_auth.return_value = {"id": "test-user"}

            response = client.get("/api/v1/gateway/metrics", headers=auth_headers)
            assert response.status_code == 200

            data = response.json()
            assert "total_requests" in data
            assert "service_requests" in data
            assert "circuit_breaker_states" in data


class TestMiddleware:
    """Test middleware functionality."""

    def test_request_logging_middleware(self, client, mock_prometheus_metrics):
        """Test request logging middleware."""
        with patch("backend.services.api_gateway.main.logger") as mock_logger:
            response = client.get("/health")
            assert response.status_code == 200

            # Verify logging calls
            mock_logger.info.assert_called()

            # Verify metrics were updated
            mock_prometheus_metrics["request_count"].labels.assert_called()
            mock_prometheus_metrics["request_duration"].observe.assert_called()

    def test_request_logging_middleware_error(self, client, mock_prometheus_metrics):
        """Test request logging middleware with error."""
        with patch("backend.services.api_gateway.main.logger") as mock_logger:
            # This would cause an error in the handler
            with patch(
                "backend.services.api_gateway.main.health_check",
                side_effect=Exception("Test error"),
            ):
                response = client.get("/health")
                assert response.status_code == 500

                # Verify error logging
                mock_logger.error.assert_called()


class TestErrorHandling:
    """Test error handling."""

    def test_global_exception_handler(self, client):
        """Test global exception handler."""
        with patch(
            "backend.services.api_gateway.main.health_check", side_effect=Exception("Test error")
        ):
            response = client.get("/health")
            assert response.status_code == 500

            data = response.json()
            assert "error" in data
            assert "Internal server error" in data["error"]

    def test_http_exception_handling(self, client):
        """Test HTTP exception handling."""
        # Test with invalid endpoint
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404
