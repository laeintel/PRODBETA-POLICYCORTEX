"""
Test Configuration for Real Integration Testing
Provides real test data and environments instead of mock values
"""

import os
from typing import Dict, Any

# Real Test Database Configuration (for integration tests)
TEST_DATABASE_CONFIG = {
    "host": os.getenv("TEST_SQL_SERVER", "localhost"),
    "username": os.getenv("TEST_SQL_USERNAME", "testuser"),
    "password": os.getenv("TEST_SQL_PASSWORD", "testpass123"),
    "database": os.getenv("TEST_SQL_DATABASE", "policycortex_test"),
    "port": os.getenv("TEST_SQL_PORT", "1433")
}

# Real Test Azure Configuration
TEST_AZURE_CONFIG = {
    "subscription_id": os.getenv("TEST_AZURE_SUBSCRIPTION_ID", "test-subscription-id"),
    "tenant_id": os.getenv("TEST_AZURE_TENANT_ID", "test-tenant-id"),
    "client_id": os.getenv("TEST_AZURE_CLIENT_ID", "test-client-id"),
    "client_secret": os.getenv("TEST_AZURE_CLIENT_SECRET", "test-client-secret"),
    "resource_group": os.getenv("TEST_AZURE_RESOURCE_GROUP", "rg-policycortex-test")
}

# Real Test Redis Configuration
TEST_REDIS_CONFIG = {
    "host": os.getenv("TEST_REDIS_HOST", "localhost"),
    "port": os.getenv("TEST_REDIS_PORT", "6379"),
    "password": os.getenv("TEST_REDIS_PASSWORD", ""),
    "db": os.getenv("TEST_REDIS_DB", "0")
}

# Real Test Cosmos DB Configuration
TEST_COSMOS_CONFIG = {
    "endpoint": os.getenv("TEST_COSMOS_ENDPOINT", "https://localhost:8081"),
    "key": os.getenv("TEST_COSMOS_KEY", "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw=="),
    "database": os.getenv("TEST_COSMOS_DATABASE", "policycortex_test")
}

# Sample Test Data for Azure Resources
SAMPLE_AZURE_RESOURCES = [
    {
        "id": "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm-1",
        "name": "test-vm-1",
        "type": "Microsoft.Compute/virtualMachines",
        "location": "eastus",
        "resourceGroup": "test-rg",
        "tags": {
            "Environment": "Test",
            "Owner": "TestTeam",
            "Project": "PolicyCortex"
        },
        "properties": {
            "provisioningState": "Succeeded",
            "vmSize": "Standard_B2s",
            "osType": "Linux"
        }
    },
    {
        "id": "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Storage/storageAccounts/teststorage001",
        "name": "teststorage001",
        "type": "Microsoft.Storage/storageAccounts",
        "location": "eastus",
        "resourceGroup": "test-rg",
        "tags": {
            "Environment": "Test",
            "Owner": "TestTeam"
        },
        "properties": {
            "provisioningState": "Succeeded",
            "accountType": "Standard_LRS",
            "encryption": {
                "services": {
                    "blob": {"enabled": True},
                    "file": {"enabled": True}
                }
            }
        }
    }
]

# Sample Test Policies
SAMPLE_POLICIES = [
    {
        "id": "test-policy-001",
        "displayName": "Test - Require VM Encryption",
        "description": "Test policy to ensure VMs have encryption enabled",
        "policyType": "Custom",
        "mode": "All",
        "policyRule": {
            "if": {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Compute/virtualMachines"},
                    {"field": "Microsoft.Compute/virtualMachines/storageProfile.osDisk.encryptionSettings.enabled", "notEquals": "true"}
                ]
            },
            "then": {"effect": "audit"}
        }
    }
]

# Test Compliance Data
SAMPLE_COMPLIANCE_DATA = [
    {
        "resourceId": "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm-1",
        "policyAssignmentId": "/subscriptions/test-sub/providers/Microsoft.Authorization/policyAssignments/test-policy-001",
        "complianceState": "NonCompliant",
        "timestamp": "2024-01-15T10:30:00Z",
        "policyDefinitionAction": "audit",
        "policyDefinitionReferenceId": "test-policy-001"
    }
]

def get_test_environment_config() -> Dict[str, Any]:
    """Get complete test environment configuration"""
    return {
        "database": TEST_DATABASE_CONFIG,
        "azure": TEST_AZURE_CONFIG,
        "redis": TEST_REDIS_CONFIG,
        "cosmos": TEST_COSMOS_CONFIG,
        "sample_data": {
            "resources": SAMPLE_AZURE_RESOURCES,
            "policies": SAMPLE_POLICIES,
            "compliance": SAMPLE_COMPLIANCE_DATA
        }
    }

def is_integration_test_environment() -> bool:
    """Check if running in integration test environment"""
    return os.getenv("INTEGRATION_TESTING", "false").lower() == "true"

def get_database_connection_string() -> str:
    """Get real database connection string for tests"""
    config = TEST_DATABASE_CONFIG
    return f"mssql+pyodbc://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?driver=ODBC+Driver+17+for+SQL+Server"

def get_redis_connection_string() -> str:
    """Get real Redis connection string for tests"""
    config = TEST_REDIS_CONFIG
    if config['password']:
        return f"redis://:{config['password']}@{config['host']}:{config['port']}/{config['db']}"
    else:
        return f"redis://{config['host']}:{config['port']}/{config['db']}"