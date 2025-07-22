#!/bin/bash
# Import existing Azure resources into Terraform state
# This script is designed to run in GitHub Actions workflow

set -e

# Get environment from command line argument
ENVIRONMENT=${1:-dev}

echo "======================================================"
echo "Importing existing Azure resources into Terraform state"
echo "Environment: $ENVIRONMENT"
echo "======================================================"

# Set environment variables for Service Principal authentication
export ARM_USE_CLI=true
SUBSCRIPTION_ID="9f16cc88-89ce-49ba-a96d-308ed3169595"
APP_RG="rg-policycortex-app-$ENVIRONMENT"
NETWORK_RG="rg-policycortex-network-$ENVIRONMENT"

# Function to safely import resources
import_resource() {
    local resource_type=$1
    local resource_id=$2
    local azure_id=$3
    
    echo "Importing $resource_type..."
    if terraform import "$resource_id" "$azure_id" 2>/dev/null; then
        echo "✓ Successfully imported $resource_type"
    else
        echo "⚠ $resource_type already exists in state or import failed"
    fi
}

# Import Key Vault Access Policy
import_resource "Key Vault Access Policy" \
    "azurerm_key_vault_access_policy.current_client" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.KeyVault/vaults/kvpolicycortex${ENVIRONMENT}v2/objectId/178e2973-bb20-49da-ab80-0d1ddc7b0649"

# Import Log Analytics Workspace
import_resource "Log Analytics Workspace" \
    "azurerm_log_analytics_workspace.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.OperationalInsights/workspaces/law-policycortex-$ENVIRONMENT"

# Import User Assigned Identity
import_resource "User Assigned Identity" \
    "azurerm_user_assigned_identity.container_apps" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id-policycortex-$ENVIRONMENT"

# Import Virtual Network
import_resource "Virtual Network" \
    "module.networking.azurerm_virtual_network.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/virtualNetworks/policycortex-$ENVIRONMENT-vnet"

# Import Network Security Groups
for nsg in "private_endpoints" "container_apps" "data_services" "ai_services" "app_gateway"; do
    import_resource "NSG $nsg" \
        "module.networking.azurerm_network_security_group.subnet_nsgs[\"$nsg\"]" \
        "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/networkSecurityGroups/policycortex-$ENVIRONMENT-nsg-$nsg"
done

# Import Route Table
import_resource "Route Table" \
    "module.networking.azurerm_route_table.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/routeTables/policycortex-$ENVIRONMENT-rt"

# Import Network Watcher
import_resource "Network Watcher" \
    "module.networking.azurerm_network_watcher.main[0]" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/networkWatchers/policycortex-$ENVIRONMENT-nw"

# Import Private DNS Zones
import_resource "Internal DNS Zone" \
    "module.networking.azurerm_private_dns_zone.internal" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/privateDnsZones/policycortex.internal"

import_resource "SQL DNS Zone" \
    "module.networking.azurerm_private_dns_zone.sql" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/privateDnsZones/privatelink.database.windows.net"

import_resource "Cosmos DNS Zone" \
    "module.networking.azurerm_private_dns_zone.cosmos" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/privateDnsZones/privatelink.documents.azure.com"

import_resource "Redis DNS Zone" \
    "module.networking.azurerm_private_dns_zone.redis" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/privateDnsZones/privatelink.redis.cache.windows.net"

import_resource "Cognitive Services DNS Zone" \
    "module.networking.azurerm_private_dns_zone.cognitive" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/privateDnsZones/privatelink.cognitiveservices.azure.com"

import_resource "ML DNS Zone" \
    "module.networking.azurerm_private_dns_zone.ml" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/privateDnsZones/privatelink.api.azureml.ms"

import_resource "OpenAI DNS Zone" \
    "module.networking.azurerm_private_dns_zone.openai" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/privateDnsZones/privatelink.openai.azure.com"

# Import Key Vault Secrets
echo ""
echo "Importing Key Vault Secrets..."

# Main.tf secrets
import_resource "JWT Secret Key" \
    "azurerm_key_vault_secret.jwt_secret_key" \
    "https://kvpolicycortex${ENVIRONMENT}v2.vault.azure.net/secrets/jwt-secret-key"

import_resource "Managed Identity Client ID Secret" \
    "azurerm_key_vault_secret.managed_identity_client_id" \
    "https://kvpolicycortex${ENVIRONMENT}v2.vault.azure.net/secrets/managed-identity-client-id"

import_resource "Storage Account Name Secret" \
    "azurerm_key_vault_secret.storage_account_name" \
    "https://kvpolicycortex${ENVIRONMENT}v2.vault.azure.net/secrets/storage-account-name"

import_resource "Application Insights Connection String Secret" \
    "azurerm_key_vault_secret.application_insights_connection_string" \
    "https://kvpolicycortex${ENVIRONMENT}v2.vault.azure.net/secrets/application-insights-connection-string"

# Data services module secrets  
import_resource "SQL Admin Password Secret" \
    "module.data_services.azurerm_key_vault_secret.sql_admin_password" \
    "https://kvpolicycortex${ENVIRONMENT}v2.vault.azure.net/secrets/sql-admin-password"

import_resource "Redis Connection String Secret" \
    "module.data_services.azurerm_key_vault_secret.redis_connection_string" \
    "https://kvpolicycortex${ENVIRONMENT}v2.vault.azure.net/secrets/redis-connection-string"

# AI services module secrets
import_resource "Cognitive Services Key Secret" \
    "module.ai_services.azurerm_key_vault_secret.cognitive_services_key" \
    "https://kvpolicycortex${ENVIRONMENT}v2.vault.azure.net/secrets/cognitive-services-key"

import_resource "Cognitive Services Endpoint Secret" \
    "module.ai_services.azurerm_key_vault_secret.cognitive_services_endpoint" \
    "https://kvpolicycortex${ENVIRONMENT}v2.vault.azure.net/secrets/cognitive-services-endpoint"

# Import Private Endpoints
echo ""
echo "Importing Private Endpoints..."

import_resource "Cosmos Private Endpoint" \
    "module.data_services.azurerm_private_endpoint.cosmos" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/privateEndpoints/policycortex-cosmos-pe-$ENVIRONMENT"

import_resource "Redis Private Endpoint" \
    "module.data_services.azurerm_private_endpoint.redis" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/privateEndpoints/policycortex-redis-pe-$ENVIRONMENT"

import_resource "Cognitive Services Private Endpoint" \
    "module.ai_services.azurerm_private_endpoint.cognitive" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/privateEndpoints/policycortex-cognitive-pe-$ENVIRONMENT"

import_resource "EventGrid Private Endpoint" \
    "module.ai_services.azurerm_private_endpoint.eventgrid" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/privateEndpoints/policycortex-eventgrid-pe-$ENVIRONMENT"

# Import Additional Key Vault Secrets that may exist
import_resource "Cosmos Connection String Secret" \
    "module.data_services.azurerm_key_vault_secret.cosmos_connection_string" \
    "https://kvpolicycortex${ENVIRONMENT}v2.vault.azure.net/secrets/cosmos-connection-string"

# Import EventGrid Topic
echo ""
echo "Importing EventGrid Topic..."

import_resource "ML Operations EventGrid Topic" \
    "module.ai_services.azurerm_eventgrid_topic.ml_operations" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.EventGrid/topics/policycortex-ml-events-$ENVIRONMENT"

# Import role assignments (attempt to get actual IDs dynamically)
echo ""
echo "Attempting to import role assignments..."
echo "Note: Role assignments may need specific IDs from Azure"

# Try to get role assignment IDs for Key Vault
ROLE_ASSIGNMENTS=$(az role assignment list \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.KeyVault/vaults/kvpolicycortex${ENVIRONMENT}v2" \
    --query "[].id" -o tsv 2>/dev/null || echo "")

if [ -n "$ROLE_ASSIGNMENTS" ]; then
    for assignment_id in $ROLE_ASSIGNMENTS; do
        # Extract the assignment GUID from the full ID
        assignment_guid=$(echo "$assignment_id" | grep -oE '[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$')
        if [ -n "$assignment_guid" ]; then
            import_resource "Role Assignment $assignment_guid" \
                "azurerm_role_assignment.key_vault_admin_current_client" \
                "$assignment_id"
            break  # Only import the first one as key_vault_admin_current_client
        fi
    done
else
    echo "⚠ Could not retrieve role assignments - may need manual import"
fi

# Import container apps KeyVault role assignment
echo ""
echo "Importing container apps KeyVault role assignment..."

# Get the container apps role assignment for KeyVault
CONTAINER_ROLE_ASSIGNMENTS=$(az role assignment list \
    --assignee "$(az identity show --name id-policycortex-$ENVIRONMENT --resource-group $APP_RG --query principalId -o tsv 2>/dev/null)" \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.KeyVault/vaults/kvpolicycortex${ENVIRONMENT}v2" \
    --query "[].id" -o tsv 2>/dev/null || echo "")

if [ -n "$CONTAINER_ROLE_ASSIGNMENTS" ]; then
    for assignment_id in $CONTAINER_ROLE_ASSIGNMENTS; do
        import_resource "Container Apps KeyVault Role Assignment" \
            "azurerm_role_assignment.container_apps_keyvault" \
            "$assignment_id"
        break  # Only import the first one
    done
else
    echo "⚠ Could not retrieve container apps role assignments"
fi

# Handle soft-deleted Cognitive Services account
echo ""
echo "Checking for soft-deleted Cognitive Services account..."

# First try to purge the soft-deleted account
echo "Attempting to purge soft-deleted cognitive services account..."
az cognitiveservices account purge \
    --name "policycortex-cognitive-$ENVIRONMENT" \
    --resource-group "$APP_RG" \
    --location "East US" 2>/dev/null || echo "Account may not be soft-deleted or already purged"

echo "Waiting 30 seconds for purge to complete..."
sleep 30

echo ""
echo "======================================================"
echo "Import process completed"
echo "======================================================"
echo ""
echo "Run 'terraform plan' to verify the imported state"