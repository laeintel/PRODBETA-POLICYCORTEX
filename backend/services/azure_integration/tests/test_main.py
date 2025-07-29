"""
Unit tests for Azure Integration service main module.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from backend.services.azure_integration.main import app


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "azure-integration"
        assert "timestamp" in data
        assert "version" in data

    def test_readiness_check_success(self, client, mock_azure_auth):
        """Test readiness check when Azure connection is healthy."""
        mock_azure_auth.verify_azure_connection.return_value = True

        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["details"]["azure_connected"] is True

    def test_readiness_check_failure(self, client, mock_azure_auth):
        """Test readiness check when Azure connection fails."""
        mock_azure_auth.verify_azure_connection.return_value = False

        response = client.get("/ready")
        assert response.status_code == 503

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""

    def test_azure_login_success(self, client, azure_auth_request, mock_azure_auth):
        """Test successful Azure login."""
        mock_azure_auth.authenticate.return_value = {
            "access_token": "test-access-token",
            "expires_in": 3600,
            "token_type": "Bearer"
        }

        response = client.post("/auth/login", json=azure_auth_request)
        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "test-access-token"
        assert data["expires_in"] == 3600

    def test_azure_login_failure(self, client, azure_auth_request, mock_azure_auth):
        """Test failed Azure login."""
        mock_azure_auth.authenticate.side_effect = Exception("Authentication failed")

        response = client.post("/auth/login", json=azure_auth_request)
        assert response.status_code == 401
        data = response.json()
        assert "Authentication failed" in data["detail"]

    def test_refresh_token_success(self, client, mock_azure_auth):
        """Test successful token refresh."""
        mock_azure_auth.refresh_token.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600
        }

        response = client.post("/auth/refresh?refresh_token=test-refresh-token")
        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "new-access-token"

    def test_refresh_token_failure(self, client, mock_azure_auth):
        """Test failed token refresh."""
        mock_azure_auth.refresh_token.side_effect = Exception("Token refresh failed")

        response = client.post("/auth/refresh?refresh_token=invalid-token")
        assert response.status_code == 401
        data = response.json()
        assert "Token refresh failed" in data["detail"]


class TestPolicyEndpoints:
    """Test policy management endpoints."""

    def test_list_policies(self, client, auth_headers, mock_policy_service, subscription_id):
        """Test listing policies."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/policies?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "Test Policy 1"

    def test_list_policies_with_filters(
        self,
        client,
        auth_headers,
        mock_policy_service,
        subscription_id
    ):
        """Test listing policies with filters."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/policies?subscription_id = (
                    {subscription_id}&resource_group=test-rg&policy_type=custom",
                )
                headers=auth_headers
            )
            assert response.status_code == 200
            mock_policy_service.list_policies.assert_called_once_with(
                subscription_id=subscription_id,
                resource_group="test-rg",
                policy_type="custom"
            )

    def test_get_policy(
        self,
        client,
        auth_headers,
        mock_policy_service,
        subscription_id,
        policy_id
    ):
        """Test getting a specific policy."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/policies/{policy_id}?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Test Policy 1"
            mock_policy_service.get_policy.assert_called_once_with(
                subscription_id=subscription_id,
                policy_id=policy_id
            )

    def test_create_policy(
        self,
        client,
        auth_headers,
        mock_policy_service,
        subscription_id,
        policy_request
    ):
        """Test creating a new policy."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.post(
                f"/api/v1/policies?subscription_id={subscription_id}",
                json=policy_request,
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "New Policy"
            mock_policy_service.create_policy.assert_called_once()

    def test_update_policy(
        self,
        client,
        auth_headers,
        mock_policy_service,
        subscription_id,
        policy_id,
        policy_request
    ):
        """Test updating an existing policy."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.put(
                f"/api/v1/policies/{policy_id}?subscription_id={subscription_id}",
                json=policy_request,
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Updated Policy"
            mock_policy_service.update_policy.assert_called_once()

    def test_delete_policy(
        self,
        client,
        auth_headers,
        mock_policy_service,
        subscription_id,
        policy_id
    ):
        """Test deleting a policy."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.delete(
                f"/api/v1/policies/{policy_id}?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert f"Policy {policy_id} deleted successfully" in data["message"]
            mock_policy_service.delete_policy.assert_called_once()

    def test_get_policy_compliance(
        self,
        client,
        auth_headers,
        mock_policy_service,
        subscription_id,
        policy_id
    ):
        """Test getting policy compliance."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/policies/{policy_id}/compliance?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["compliant"] is True
            mock_policy_service.get_policy_compliance.assert_called_once()

    def test_policy_endpoint_error_handling(
        self,
        client,
        auth_headers,
        mock_policy_service,
        subscription_id
    ):
        """Test policy endpoint error handling."""
        mock_policy_service.list_policies.side_effect = Exception("Azure API error")

        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/policies?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 500
            data = response.json()
            assert "Failed to list policies" in data["detail"]


class TestRBACEndpoints:
    """Test RBAC management endpoints."""

    def test_list_roles(self, client, auth_headers, mock_rbac_service, subscription_id):
        """Test listing roles."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/rbac/roles?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "Contributor"

    def test_list_role_assignments(self, client, auth_headers, mock_rbac_service, subscription_id):
        """Test listing role assignments."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/rbac/assignments?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["principalId"] == "user-id"

    def test_create_role_assignment(
        self,
        client,
        auth_headers,
        mock_rbac_service,
        subscription_id,
        rbac_request
    ):
        """Test creating a role assignment."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.post(
                f"/api/v1/rbac/assignments?subscription_id={subscription_id}",
                json=rbac_request,
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "created"
            mock_rbac_service.create_role_assignment.assert_called_once()

    def test_delete_role_assignment(
        self,
        client,
        auth_headers,
        mock_rbac_service,
        subscription_id,
        role_assignment_id
    ):
        """Test deleting a role assignment."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.delete(
                f"/api/v1/rbac/assignments/{role_assignment_id}?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert f"Role assignment {role_assignment_id} deleted successfully" in data["message"]
            mock_rbac_service.delete_role_assignment.assert_called_once()


class TestCostEndpoints:
    """Test cost management endpoints."""

    def test_get_cost_usage(self, client, auth_headers, mock_cost_service, subscription_id):
        """Test getting cost usage."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/costs/usage?subscription_id = (
                    {subscription_id}&start_date=2023-01-01&end_date=2023-01-31",
                )
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 226.25
            mock_cost_service.get_usage_details.assert_called_once()

    def test_get_cost_forecast(self, client, auth_headers, mock_cost_service, subscription_id):
        """Test getting cost forecast."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/costs/forecast?subscription_id={subscription_id}&forecast_days=30",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert "forecast" in data
            mock_cost_service.get_cost_forecast.assert_called_once()

    def test_list_budgets(self, client, auth_headers, mock_cost_service, subscription_id):
        """Test listing budgets."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/costs/budgets?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "Test Budget"

    def test_get_cost_recommendations(
        self,
        client,
        auth_headers,
        mock_cost_service,
        subscription_id
    ):
        """Test getting cost recommendations."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/costs/recommendations?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["type"] == "shutdown"


class TestNetworkEndpoints:
    """Test network management endpoints."""

    def test_list_networks(self, client, auth_headers, mock_network_service, subscription_id):
        """Test listing virtual networks."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/networks?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "test-vnet"

    def test_list_network_security_groups(
        self,
        client,
        auth_headers,
        mock_network_service,
        subscription_id
    ):
        """Test listing network security groups."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/networks/security-groups?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "test-nsg"

    def test_analyze_network_security(
        self,
        client,
        auth_headers,
        mock_network_service,
        subscription_id
    ):
        """Test analyzing network security."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/networks/security-analysis?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["score"] == 85
            assert "vulnerabilities" in data
            assert "recommendations" in data


class TestResourceEndpoints:
    """Test resource management endpoints."""

    def test_list_resources(self, client, auth_headers, mock_resource_service, subscription_id):
        """Test listing resources."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/resources?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "test-vm"

    def test_get_resource(self, client, auth_headers, mock_resource_service, resource_id):
        """Test getting a specific resource."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/resources/{resource_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "test-vm"

    def test_list_resource_groups(
        self,
        client,
        auth_headers,
        mock_resource_service,
        subscription_id
    ):
        """Test listing resource groups."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                f"/api/v1/resources/groups?subscription_id={subscription_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "test-rg"

    def test_update_resource_tags(self, client, auth_headers, mock_resource_service, resource_id):
        """Test updating resource tags."""
        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            tags = {"environment": "production", "team": "backend"}
            response = client.post(
                f"/api/v1/resources/tags/{resource_id}",
                json=tags,
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert "Tags updated successfully" in data["message"]
            mock_resource_service.update_resource_tags.assert_called_once()


class TestMiddleware:
    """Test middleware functionality."""

    def test_request_logging_middleware(self, client, mock_prometheus_metrics):
        """Test request logging middleware."""
        with patch("backend.services.azure_integration.main.logger") as mock_logger:
            response = client.get("/health")
            assert response.status_code == 200

            # Verify logging calls
            mock_logger.info.assert_called()

            # Verify metrics were updated
            mock_prometheus_metrics["request_count"].labels.assert_called()
            mock_prometheus_metrics["request_duration"].observe.assert_called()

    def test_azure_auth_middleware(self, client):
        """Test Azure authentication middleware."""
        # This would test the middleware that handles Azure auth
        # Implementation depends on the actual middleware
        pass


class TestErrorHandling:
    """Test error handling."""

    def test_global_exception_handler(self, client):
        """Test global exception handler."""
        with patch(
            "backend.services.azure_integration.main.health_check",
            side_effect=Exception("Test error")
        ):
            response = client.get("/health")
            assert response.status_code == 500

            data = response.json()
            assert "error" in data
            assert "Internal server error" in data["error"]

    def test_authentication_required(self, client, subscription_id):
        """Test that authentication is required for protected endpoints."""
        response = client.get(f"/api/v1/policies?subscription_id={subscription_id}")
        assert response.status_code == 401

    def test_invalid_subscription_id(self, client, auth_headers, mock_policy_service):
        """Test handling of invalid subscription ID."""
        mock_policy_service.list_policies.side_effect = Exception("Invalid subscription")

        with patch("backend.services.azure_integration.main.get_current_user") as mock_user:
            mock_user.return_value = {"id": "test-user"}

            response = client.get(
                "/api/v1/policies?subscription_id=invalid-sub",
                headers=auth_headers
            )
            assert response.status_code == 500
