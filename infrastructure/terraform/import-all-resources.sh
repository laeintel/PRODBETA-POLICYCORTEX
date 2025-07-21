#!/bin/bash
# Comprehensive Terraform Import Script for PolicyCortex
# This script imports all existing Azure resources into Terraform state

echo "=== PolicyCortex Comprehensive Resource Import Script ==="
echo "This script will import ALL existing Azure resources into Terraform state"
echo "WARNING: This may take several minutes to complete"
echo ""

# Check if we're in the correct directory
if [ ! -f "main.tf" ]; then
    echo "Error: Please run this script from the infrastructure/terraform directory"
    exit 1
fi

# Set Azure CLI environment
export ARM_USE_CLI=true

# Define subscription and resource group info
SUBSCRIPTION_ID="9f16cc88-89ce-49ba-a96d-308ed3169595"
APP_RG="rg-policycortex-app-dev"
NETWORK_RG="rg-policycortex-network-dev"

echo "=== Importing Main Resources ==="

# Import Key Vault access policy
echo "Importing Key Vault access policy..."
if terraform import azurerm_key_vault_access_policy.current_client "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.KeyVault/vaults/kvpolicycortexdevv2/objectId/178e2973-bb20-49da-ab80-0d1ddc7b0649" 2>/dev/null; then
    echo "✓ Key Vault access policy imported"
else
    echo "! Key Vault access policy import failed or already exists"
fi

# Import Log Analytics Workspace
echo "Importing Log Analytics Workspace..."
if terraform import azurerm_log_analytics_workspace.main "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.OperationalInsights/workspaces/law-policycortex-dev" 2>/dev/null; then
    echo "✓ Log Analytics Workspace imported"
else
    echo "! Log Analytics Workspace import failed or already exists"
fi

# Import User Assigned Identity
echo "Importing User Assigned Identity..."
if terraform import azurerm_user_assigned_identity.container_apps "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$APP_RG/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id-policycortex-dev" 2>/dev/null; then
    echo "✓ User Assigned Identity imported"
else
    echo "! User Assigned Identity import failed or already exists"
fi

echo ""
echo "=== Importing Networking Resources ==="

# Import Virtual Network
echo "Importing Virtual Network..."
if terraform import "module.networking.azurerm_virtual_network.main" "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/virtualNetworks/policycortex-dev-vnet" 2>/dev/null; then
    echo "✓ Virtual Network imported"
else
    echo "! Virtual Network import failed or already exists"
fi

# Import Network Security Groups
declare -A nsgs=(
    ["data_services"]="policycortex-dev-nsg-data_services"
    ["private_endpoints"]="policycortex-dev-nsg-private_endpoints"
    ["container_apps"]="policycortex-dev-nsg-container_apps"
    ["ai_services"]="policycortex-dev-nsg-ai_services"
    ["app_gateway"]="policycortex-dev-nsg-app_gateway"
)

for key in "${!nsgs[@]}"; do
    nsg_resource="${nsgs[$key]}"
    echo "Importing NSG: $nsg_resource..."
    if terraform import "module.networking.azurerm_network_security_group.subnet_nsgs[\"$key\"]" "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/networkSecurityGroups/$nsg_resource" 2>/dev/null; then
        echo "✓ NSG $nsg_resource imported"
    else
        echo "! NSG $nsg_resource import failed or already exists"
    fi
done

# Import Route Table
echo "Importing Route Table..."
if terraform import "module.networking.azurerm_route_table.main" "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/routeTables/policycortex-dev-rt" 2>/dev/null; then
    echo "✓ Route Table imported"
else
    echo "! Route Table import failed or already exists"
fi

# Import Network Watcher
echo "Importing Network Watcher..."
if terraform import "module.networking.azurerm_network_watcher.main[0]" "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/networkWatchers/policycortex-dev-nw" 2>/dev/null; then
    echo "✓ Network Watcher imported"
else
    echo "! Network Watcher import failed or already exists"
fi

# Import Private DNS Zones
declare -A dns_zones=(
    ["internal"]="policycortex.internal"
    ["sql"]="privatelink.database.windows.net"
    ["cosmos"]="privatelink.documents.azure.com"
    ["redis"]="privatelink.redis.cache.windows.net"
    ["cognitive"]="privatelink.cognitiveservices.azure.com"
    ["ml"]="privatelink.api.azureml.ms"
    ["openai"]="privatelink.openai.azure.com"
)

for name in "${!dns_zones[@]}"; do
    zone="${dns_zones[$name]}"
    echo "Importing DNS Zone: $zone..."
    if terraform import "module.networking.azurerm_private_dns_zone.$name" "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$NETWORK_RG/providers/Microsoft.Network/privateDnsZones/$zone" 2>/dev/null; then
        echo "✓ DNS Zone $zone imported"
    else
        echo "! DNS Zone $zone import failed or already exists"
    fi
done

echo ""
echo "=== Import Phase 1 Complete ==="
echo "Now we need to handle role assignments and other complex resources..."
echo ""
echo "Next steps:"
echo "1. Run: terraform plan -var-file=environments/dev/terraform.tfvars"
echo "2. Look for any remaining 'already exists' errors"
echo "3. Use 'az role assignment list' to get role assignment IDs for import"
echo ""

echo "=== Running Terraform Plan to Check Status ==="
terraform plan -var-file=environments/dev/terraform.tfvars