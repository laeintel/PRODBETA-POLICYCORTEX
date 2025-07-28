"""
Test configuration and fixtures for Azure Integration service.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from backend.services.azure_integration.main import app


@pytest.fixture
def client():
    """Create test client for Azure Integration service."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_azure_auth():
    """Mock Azure authentication."""
    with patch("backend.services.azure_integration.services.AzureAuthService") as mock_auth:
        mock_instance = MagicMock()
        mock_auth.return_value = mock_instance

        # Mock authentication methods
        mock_instance.authenticate = AsyncMock(return_value={
            "access_token": "test-access-token",
            "expires_in": 3600,
            "token_type": "Bearer"
        })
        mock_instance.verify_azure_connection = AsyncMock(return_value=True)
        mock_instance.refresh_token = AsyncMock(return_value={
            "access_token": "new-access-token",
            "expires_in": 3600
        })

        yield mock_instance


@pytest.fixture
def mock_policy_service():
    """Mock PolicyManagementService."""
    with patch("backend.services.azure_integration.services.PolicyManagementService") as mock_service:
        mock_instance = MagicMock()
        mock_service.return_value = mock_instance

        # Mock policy operations
        mock_instance.list_policies = AsyncMock(return_value=[
            {
                "id": "policy-1",
                "name": "Test Policy 1",
                "type": "custom",
                "description": "Test policy description"
            }
        ])
        mock_instance.get_policy = AsyncMock(return_value={
            "id": "policy-1",
            "name": "Test Policy 1",
            "type": "custom",
            "description": "Test policy description"
        })
        mock_instance.create_policy = AsyncMock(return_value={
            "id": "new-policy-id",
            "name": "New Policy",
            "status": "created"
        })
        mock_instance.update_policy = AsyncMock(return_value={
            "id": "policy-1",
            "name": "Updated Policy",
            "status": "updated"
        })
        mock_instance.delete_policy = AsyncMock(return_value=True)
        mock_instance.get_policy_compliance = AsyncMock(return_value={
            "compliant": True,
            "violations": []
        })

        yield mock_instance


@pytest.fixture
def mock_rbac_service():
    """Mock RBACManagementService."""
    with patch("backend.services.azure_integration.services.RBACManagementService") as mock_service:
        mock_instance = MagicMock()
        mock_service.return_value = mock_instance

        # Mock RBAC operations
        mock_instance.list_roles = AsyncMock(return_value=[
            {
                "id": "role-1",
                "name": "Contributor",
                "type": "BuiltInRole"
            }
        ])
        mock_instance.list_role_assignments = AsyncMock(return_value=[
            {
                "id": "assignment-1",
                "principalId": "user-id",
                "roleDefinitionId": "role-1",
                "scope": "/subscriptions/test-sub"
            }
        ])
        mock_instance.create_role_assignment = AsyncMock(return_value={
            "id": "new-assignment-id",
            "status": "created"
        })
        mock_instance.delete_role_assignment = AsyncMock(return_value=True)

        yield mock_instance


@pytest.fixture
def mock_cost_service():
    """Mock CostManagementService."""
    with patch("backend.services.azure_integration.services.CostManagementService") as mock_service:
        mock_instance = MagicMock()
        mock_service.return_value = mock_instance

        # Mock cost operations
        mock_instance.get_usage_details = AsyncMock(return_value={
            "costs": [
                {"date": "2023-01-01", "cost": 100.50, "currency": "USD"},
                {"date": "2023-01-02", "cost": 125.75, "currency": "USD"}
            ],
            "total": 226.25
        })
        mock_instance.get_cost_forecast = AsyncMock(return_value={
            "forecast": [
                {"date": "2023-01-03", "cost": 110.00, "currency": "USD"},
                {"date": "2023-01-04", "cost": 115.25, "currency": "USD"}
            ]
        })
        mock_instance.list_budgets = AsyncMock(return_value=[
            {
                "id": "budget-1",
                "name": "Test Budget",
                "amount": 1000.00,
                "spent": 500.00
            }
        ])
        mock_instance.get_cost_recommendations = AsyncMock(return_value=[
            {
                "type": "shutdown",
                "resource": "vm-1",
                "potential_savings": 200.00
            }
        ])

        yield mock_instance


@pytest.fixture
def mock_network_service():
    """Mock NetworkManagementService."""
    with patch("backend.services.azure_integration.services.NetworkManagementService") as mock_service:
        mock_instance = MagicMock()
        mock_service.return_value = mock_instance

        # Mock network operations
        mock_instance.list_virtual_networks = AsyncMock(return_value=[
            {
                "id": "vnet-1",
                "name": "test-vnet",
                "location": "eastus",
                "addressSpace": {"addressPrefixes": ["10.0.0.0/16"]}
            }
        ])
        mock_instance.list_network_security_groups = AsyncMock(return_value=[
            {
                "id": "nsg-1",
                "name": "test-nsg",
                "location": "eastus",
                "securityRules": []
            }
        ])
        mock_instance.analyze_network_security = AsyncMock(return_value={
            "vulnerabilities": [],
            "recommendations": [],
            "score": 85
        })

        yield mock_instance


@pytest.fixture
def mock_resource_service():
    """Mock ResourceManagementService."""
    with patch("backend.services.azure_integration.services.ResourceManagementService") as mock_service:
        mock_instance = MagicMock()
        mock_service.return_value = mock_instance

        # Mock resource operations
        mock_instance.list_resources = AsyncMock(return_value=[
            {
                "id": "resource-1",
                "name": "test-vm",
                "type": "Microsoft.Compute/virtualMachines",
                "location": "eastus",
                "resourceGroup": "test-rg"
            }
        ])
        mock_instance.get_resource = AsyncMock(return_value={
            "id": "resource-1",
            "name": "test-vm",
            "type": "Microsoft.Compute/virtualMachines",
            "location": "eastus",
            "resourceGroup": "test-rg"
        })
        mock_instance.list_resource_groups = AsyncMock(return_value=[
            {
                "id": "rg-1",
                "name": "test-rg",
                "location": "eastus"
            }
        ])
        mock_instance.update_resource_tags = AsyncMock(return_value=True)

        yield mock_instance


@pytest.fixture
def auth_headers():
    """Authentication headers for testing."""
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def azure_auth_request():
    """Sample Azure authentication request."""
    return {
        "tenant_id": "test-tenant-id",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret"
    }


@pytest.fixture
def policy_request():
    """Sample policy request."""
    return {
        "name": "Test Policy",
        "description": "Test policy description",
        "type": "custom",
        "rules": [
            {
                "field": "type",
                "operator": "equals",
                "value": "Microsoft.Compute/virtualMachines"
            }
        ],
        "effects": ["deny"]
    }


@pytest.fixture
def rbac_request():
    """Sample RBAC request."""
    return {
        "principalId": "user-id",
        "roleDefinitionId": "role-definition-id",
        "scope": "/subscriptions/test-subscription"
    }


@pytest.fixture
def mock_prometheus_metrics():
    """Mock Prometheus metrics for testing."""
    with patch("backend.services.azure_integration.main.REQUEST_COUNT") as mock_counter, \
         patch("backend.services.azure_integration.main.REQUEST_DURATION") as mock_histogram, \
         patch("backend.services.azure_integration.main.AZURE_API_CALLS") as mock_api_counter, \
         patch("backend.services.azure_integration.main.AZURE_API_DURATION") as mock_api_histogram:

        # Mock counter methods
        mock_counter.labels.return_value.inc = MagicMock()
        mock_histogram.observe = MagicMock()
        mock_api_counter.labels.return_value.inc = MagicMock()
        mock_api_histogram.labels.return_value.observe = MagicMock()

        yield {
            "request_count": mock_counter,
            "request_duration": mock_histogram,
            "api_calls": mock_api_counter,
            "api_duration": mock_api_histogram
        }


@pytest.fixture
def mock_current_user():
    """Mock current user for testing."""
    return {
        "id": "test-user-id",
        "email": "test@example.com",
        "name": "Test User",
        "roles": ["user"]
    }


@pytest.fixture
def mock_azure_sdk_clients():
    """Mock Azure SDK clients."""
    with patch("azure.identity.DefaultAzureCredential") as mock_credential, \
         patch("azure.mgmt.resource.ResourceManagementClient") as mock_resource_client, \
         patch("azure.mgmt.authorization.AuthorizationManagementClient") as mock_auth_client, \
         patch("azure.mgmt.consumption.ConsumptionManagementClient") as mock_consumption_client, \
         patch("azure.mgmt.network.NetworkManagementClient") as mock_network_client, \
         patch("azure.mgmt.monitor.MonitorManagementClient") as mock_monitor_client:

        # Mock credential
        mock_credential.return_value = MagicMock()

        # Mock clients
        mock_resource_instance = MagicMock()
        mock_resource_client.return_value = mock_resource_instance

        mock_auth_instance = MagicMock()
        mock_auth_client.return_value = mock_auth_instance

        mock_consumption_instance = MagicMock()
        mock_consumption_client.return_value = mock_consumption_instance

        mock_network_instance = MagicMock()
        mock_network_client.return_value = mock_network_instance

        mock_monitor_instance = MagicMock()
        mock_monitor_client.return_value = mock_monitor_instance

        yield {
            "credential": mock_credential,
            "resource_client": mock_resource_instance,
            "auth_client": mock_auth_instance,
            "consumption_client": mock_consumption_instance,
            "network_client": mock_network_instance,
            "monitor_client": mock_monitor_instance
        }


@pytest.fixture
def subscription_id():
    """Test subscription ID."""
    return "test-subscription-id"


@pytest.fixture
def resource_group_name():
    """Test resource group name."""
    return "test-resource-group"


@pytest.fixture
def policy_id():
    """Test policy ID."""
    return "test-policy-id"


@pytest.fixture
def role_assignment_id():
    """Test role assignment ID."""
    return "test-role-assignment-id"


@pytest.fixture
def resource_id():
    """Test resource ID."""
    return "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm"
