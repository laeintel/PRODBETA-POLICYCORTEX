#!/bin/bash
# Import existing Azure resources into Terraform state
# This script is designed to run in GitHub Actions workflow

set -e

echo "======================================================"
echo "Importing existing Azure resources into Terraform state"
echo "======================================================"

# Set environment variables for Service Principal authentication
export ARM_USE_CLI=true
SUBSCRIPTION_ID="9f16cc88-89ce-49ba-a96d-308ed3169595"
APP_RG="rg-policycortex-app-dev"
NETWORK_RG="rg-policycortex-network-dev"

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
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.KeyVault/vaults/kvpolicycortexdevv2/objectId/178e2973-bb20-49da-ab80-0d1ddc7b0649"

# Import Log Analytics Workspace
import_resource "Log Analytics Workspace" \
    "azurerm_log_analytics_workspace.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.OperationalInsights/workspaces/law-policycortex-dev"

# Import User Assigned Identity
import_resource "User Assigned Identity" \
    "azurerm_user_assigned_identity.container_apps" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id-policycortex-dev"

# Import Virtual Network
import_resource "Virtual Network" \
    "module.networking.azurerm_virtual_network.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/virtualNetworks/policycortex-dev-vnet"

# Import Network Security Groups
for nsg in "private_endpoints" "container_apps" "data_services" "ai_services" "app_gateway"; do
    import_resource "NSG $nsg" \
        "module.networking.azurerm_network_security_group.subnet_nsgs[\"$nsg\"]" \
        "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/networkSecurityGroups/policycortex-dev-nsg-$nsg"
done

# Import Route Table
import_resource "Route Table" \
    "module.networking.azurerm_route_table.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/routeTables/policycortex-dev-rt"

# Import Network Watcher
import_resource "Network Watcher" \
    "module.networking.azurerm_network_watcher.main[0]" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/networkWatchers/policycortex-dev-nw"

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

# Import role assignments (attempt to get actual IDs dynamically)
echo ""
echo "Attempting to import role assignments..."
echo "Note: Role assignments may need specific IDs from Azure"

# Try to get role assignment IDs for Key Vault
ROLE_ASSIGNMENTS=$(az role assignment list \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.KeyVault/vaults/kvpolicycortexdevv2" \
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

echo ""
echo "======================================================"
echo "Import process completed"
echo "======================================================"
echo ""
echo "Run 'terraform plan' to verify the imported state"