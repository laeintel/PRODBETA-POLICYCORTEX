"""
Shared test configuration and fixtures for all backend services.
"""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Generator, AsyncGenerator
import uuid
from datetime import datetime, timedelta

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.shared.config import get_settings
from backend.shared.database import Base, get_async_db


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
        )
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def db_session():
    """Create a test database session."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def override_get_db(db_session):
    """Override database dependency for testing."""
    def _override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()
    return _override_get_db


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.debug = True
    settings.service.service_version = "test"
    settings.service.service_host = "localhost"
    settings.service.service_port = 8000
    settings.database.database_url = "sqlite:///./test.db"
    settings.security.cors_origins = ["*"]
    settings.security.cors_methods = ["*"]
    settings.security.cors_headers = ["*"]
    settings.security.rate_limit_per_minute = 100
    settings.monitoring.log_level = "INFO"
    settings.azure_integration_url = "http://localhost:8001"
    settings.ai_engine_url = "http://localhost:8002"
    settings.data_processing_url = "http://localhost:8003"
    settings.conversation_url = "http://localhost:8004"
    settings.notification_url = "http://localhost:8005"
    return settings


@pytest.fixture
def mock_user():
    """Mock user for authentication testing."""
    return {
        "id": "test-user-id",
        "email": "test@example.com",
        "name": "Test User",
        "roles": ["user"],
        "tenant_id": "test-tenant"
    }


@pytest.fixture
def mock_admin_user():
    """Mock admin user for authentication testing."""
    return {
        "id": "test-admin-id",
        "email": "admin@example.com",
        "name": "Test Admin",
        "roles": ["admin"],
        "tenant_id": "test-tenant"
    }


@pytest.fixture
def mock_request_id():
    """Mock request ID."""
    return str(uuid.uuid4())


@pytest.fixture
def mock_azure_credentials():
    """Mock Azure credentials."""
    return {
        "tenant_id": "test-tenant-id",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "subscription_id": "test-subscription-id"
    }


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for external API calls."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_policy_data():
    """Sample policy data for testing."""
    return {
        "id": "test-policy-id",
        "name": "Test Policy",
        "description": "Test policy description",
        "type": "custom",
        "rules": [
            {
                "field": "resourceType",
                "operator": "equals",
                "value": "Microsoft.Compute/virtualMachines"
            }
        ],
        "effects": ["deny"],
        "parameters": {},
        "metadata": {
            "created_by": "test-user",
            "created_at": "2023-01-01T00:00:00Z"
        }
    }


@pytest.fixture
def sample_resource_data():
    """Sample resource data for testing."""
    return {
        "id": "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm",
        "name": "test-vm",
        "type": "Microsoft.Compute/virtualMachines",
        "location": "eastus",
        "resourceGroup": "test-rg",
        "subscriptionId": "test-sub",
        "properties": {
            "provisioningState": "Succeeded",
            "vmId": "test-vm-id",
            "hardwareProfile": {
                "vmSize": "Standard_B2s"
            },
            "storageProfile": {
                "osDisk": {
                    "osType": "Linux",
                    "name": "test-vm-disk",
                    "createOption": "FromImage"
                }
            },
            "osProfile": {
                "computerName": "test-vm",
                "adminUsername": "azureuser"
            },
            "networkProfile": {
                "networkInterfaces": [
                    {
                        "id": "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Network/networkInterfaces/test-vm-nic"
                    }
                ]
            }
        },
        "tags": {
            "environment": "test",
            "project": "policycortex"
        }
    }


@pytest.fixture
def sample_cost_data():
    """Sample cost data for testing."""
    return {
        "timeframe": "Monthly",
        "granularity": "Daily",
        "aggregation": {
            "totalCost": {
                "name": "PreTaxCost",
                "function": "Sum"
            }
        },
        "rows": [
            [20230101, 150.75, "USD"],
            [20230102, 165.25, "USD"],
            [20230103, 142.50, "USD"]
        ],
        "columns": [
            {"name": "Date", "type": "Number"},
            {"name": "PreTaxCost", "type": "Number"},
            {"name": "Currency", "type": "String"}
        ]
    }


@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing."""
    return {
        "session_id": "test-session-id",
        "user_id": "test-user-id",
        "message": "Show me my Azure virtual machines",
        "intent": "list_resources",
        "entities": [
            {
                "entity": "resource_type",
                "value": "virtual_machines",
                "confidence": 0.95
            }
        ],
        "context": {
            "previous_messages": [],
            "current_subscription": "test-subscription-id"
        }
    }


@pytest.fixture
def sample_notification_data():
    """Sample notification data for testing."""
    return {
        "id": "test-notification-id",
        "type": "email",
        "subject": "Test Notification",
        "message": "This is a test notification",
        "recipients": ["test@example.com"],
        "priority": "normal",
        "metadata": {
            "source": "test",
            "category": "alert"
        }
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    with patch("redis.Redis") as mock_redis_class:
        mock_instance = MagicMock()
        mock_redis_class.return_value = mock_instance

        # Mock async operations
        mock_instance.get = AsyncMock(return_value=None)
        mock_instance.set = AsyncMock(return_value=True)
        mock_instance.delete = AsyncMock(return_value=True)
        mock_instance.exists = AsyncMock(return_value=False)
        mock_instance.expire = AsyncMock(return_value=True)
        mock_instance.incr = AsyncMock(return_value=1)
        mock_instance.hget = AsyncMock(return_value=None)
        mock_instance.hset = AsyncMock(return_value=True)
        mock_instance.hgetall = AsyncMock(return_value={})

        yield mock_instance


@pytest.fixture
def mock_azure_sdk():
    """Mock Azure SDK clients."""
    with patch("azure.identity.DefaultAzureCredential") as mock_credential, \
        patch("azure.mgmt.resource.ResourceManagementClient") as mock_resource_client, \
        patch("azure.mgmt.authorization.AuthorizationManagementClient") as mock_auth_client, \
        patch("azure.mgmt.consumption.ConsumptionManagementClient") as mock_consumption_client, \
        patch("azure.mgmt.network.NetworkManagementClient") as mock_network_client:

        # Mock credential
        mock_credential.return_value = MagicMock()

        # Mock resource client
        mock_resource_instance = MagicMock()
        mock_resource_client.return_value = mock_resource_instance

        # Mock auth client
        mock_auth_instance = MagicMock()
        mock_auth_client.return_value = mock_auth_instance

        # Mock consumption client
        mock_consumption_instance = MagicMock()
        mock_consumption_client.return_value = mock_consumption_instance

        # Mock network client
        mock_network_instance = MagicMock()
        mock_network_client.return_value = mock_network_instance

        yield {
            "credential": mock_credential,
            "resource_client": mock_resource_instance,
            "auth_client": mock_auth_instance,
            "consumption_client": mock_consumption_instance,
            "network_client": mock_network_instance
        }


@pytest.fixture
def mock_ai_models():
    """Mock AI models for testing."""
    with patch("backend.services.ai_engine.services.model_manager.ModelManager") as mock_model_manager:
        mock_instance = MagicMock()
        mock_model_manager.return_value = mock_instance

        # Mock model operations
        mock_instance.load_model = AsyncMock(return_value=True)
        mock_instance.predict = AsyncMock(return_value={"prediction": "test", "confidence": 0.95})
        mock_instance.train_model = AsyncMock(return_value={"task_id": "test-task-id"})
        mock_instance.get_model_info = AsyncMock(
            return_value={"name": "test-model",
            "version": "1.0"}
        )

        yield mock_instance


@pytest.fixture
def mock_email_service():
    """Mock email service for testing."""
    with patch("smtplib.SMTP") as mock_smtp:
        mock_instance = MagicMock()
        mock_smtp.return_value = mock_instance
        mock_instance.send_message = MagicMock(return_value={})
        yield mock_instance


@pytest.fixture
def mock_sms_service():
    """Mock SMS service for testing."""
    with patch("twilio.rest.Client") as mock_twilio:
        mock_instance = MagicMock()
        mock_twilio.return_value = mock_instance
        mock_instance.messages.create = MagicMock(return_value=MagicMock(sid="test-sms-id"))
        yield mock_instance


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing."""
    mock_ws = MagicMock()
    mock_ws.accept = AsyncMock()
    mock_ws.send_json = AsyncMock()
    mock_ws.receive_json = AsyncMock()
    mock_ws.close = AsyncMock()
    return mock_ws


@pytest.fixture
def mock_background_tasks():
    """Mock background tasks for testing."""
    mock_tasks = MagicMock()
    mock_tasks.add_task = MagicMock()
    return mock_tasks


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, json_data=None, status_code=200, headers=None, text=""):
        self.json_data = json_data or {}
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self.content = text.encode() if isinstance(text, str) else text

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=MagicMock(),
                response=self
            )


@pytest.fixture
def mock_response():
    """Mock HTTP response factory."""
    return MockResponse


# Test utilities
def create_test_client(app, dependencies_override=None):
    """Create a test client with optional dependency overrides."""
    if dependencies_override:
        for dependency, override in dependencies_override.items():
            app.dependency_overrides[dependency] = override

    client = TestClient(app)

    # Clean up overrides after test
    def cleanup():
        app.dependency_overrides.clear()

    client.cleanup = cleanup
    return client


def assert_response_success(response, expected_status=200):
    """Assert that a response is successful."""
    assert response.status_code == expected_status
    if response.headers.get("content-type", "").startswith("application/json"):
        data = response.json()
        assert "error" not in data or data.get("success", True)


def assert_response_error(response, expected_status=400):
    """Assert that a response is an error."""
    assert response.status_code == expected_status
    if response.headers.get("content-type", "").startswith("application/json"):
        data = response.json()
        assert "error" in data or "detail" in data


def create_mock_auth_token(user_data=None):
    """Create a mock authentication token."""
    if user_data is None:
        user_data = {
            "id": "test-user-id",
            "email": "test@example.com",
            "name": "Test User"
        }

    # In a real implementation, this would be a JWT token
    # For testing, we'll just return a simple string
    return "test-auth-token"


# Async test utilities
async def async_assert_response_success(response, expected_status=200):
    """Assert that an async response is successful."""
    assert response.status_code == expected_status
    if hasattr(response, 'json'):
        data = await response.json() if asyncio.iscoroutine(response.json()) else response.json()
        assert "error" not in data or data.get("success", True)


async def async_assert_response_error(response, expected_status=400):
    """Assert that an async response is an error."""
    assert response.status_code == expected_status
    if hasattr(response, 'json'):
        data = await response.json() if asyncio.iscoroutine(response.json()) else response.json()
        assert "error" in data or "detail" in data
