#!/bin/bash
# Script to import existing Azure resources into Terraform state

set -euo pipefail

# Get environment from argument
ENVIRONMENT=${1:-dev}
echo "Importing resources for environment: $ENVIRONMENT"

# Get subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "Subscription ID: $SUBSCRIPTION_ID"

# Resource group name
RESOURCE_GROUP="rg-policycortex-$ENVIRONMENT"

# Function to check if resource exists in state
resource_in_state() {
    terraform state show "$1" &>/dev/null
}

# Function to check if resource exists in Azure
resource_exists_in_azure() {
    local resource_type=$1
    local resource_name=$2
    
    case $resource_type in
        "resource_group")
            az group show --name "$resource_name" &>/dev/null
            ;;
        "storage_account")
            az storage account show --name "$resource_name" &>/dev/null
            ;;
        "key_vault")
            az keyvault show --name "$resource_name" &>/dev/null
            ;;
        "container_registry")
            az acr show --name "$resource_name" &>/dev/null
            ;;
        "log_analytics_workspace")
            az monitor log-analytics workspace show --resource-group "$RESOURCE_GROUP" --workspace-name "$resource_name" &>/dev/null
            ;;
        "application_insights")
            az monitor app-insights component show --resource-group "$RESOURCE_GROUP" --app "$resource_name" &>/dev/null
            ;;
        "container_app_environment")
            az containerapp env show --name "$resource_name" --resource-group "$RESOURCE_GROUP" &>/dev/null
            ;;
        "managed_identity")
            az identity show --name "$resource_name" --resource-group "$RESOURCE_GROUP" &>/dev/null
            ;;
        "virtual_network")
            az network vnet show --name "$resource_name" --resource-group "$RESOURCE_GROUP" &>/dev/null
            ;;
        "network_security_group")
            az network nsg show --name "$resource_name" --resource-group "$RESOURCE_GROUP" &>/dev/null
            ;;
        "route_table")
            az network route-table show --name "$resource_name" --resource-group "$RESOURCE_GROUP" &>/dev/null
            ;;
        "private_dns_zone")
            az network private-dns zone show --name "$resource_name" --resource-group "$RESOURCE_GROUP" &>/dev/null
            ;;
        *)
            return 1
            ;;
    esac
}

# Import resource if it exists in Azure but not in state
import_if_exists() {
    local terraform_resource=$1
    local azure_resource_id=$2
    local resource_type=$3
    local resource_name=$4
    
    if resource_exists_in_azure "$resource_type" "$resource_name"; then
        echo "Resource $resource_name exists in Azure"
        if ! resource_in_state "$terraform_resource"; then
            echo "Importing $terraform_resource..."
            terraform import "$terraform_resource" "$azure_resource_id" || echo "Import failed for $terraform_resource"
        else
            echo "$terraform_resource already in state"
        fi
    else
        echo "Resource $resource_name does not exist in Azure"
    fi
}

# Main imports
echo "Starting resource imports..."

# Resource Group
import_if_exists "azurerm_resource_group.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" \
    "resource_group" \
    "$RESOURCE_GROUP"

# Storage Account
import_if_exists "azurerm_storage_account.app_storage" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/stpolicycortex${ENVIRONMENT}stg" \
    "storage_account" \
    "stpolicycortex${ENVIRONMENT}stg"

# Key Vault
import_if_exists "azurerm_key_vault.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/kvpolicycortex$ENVIRONMENT" \
    "key_vault" \
    "kvpolicycortex$ENVIRONMENT"

# Container Registry
import_if_exists "azurerm_container_registry.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ContainerRegistry/registries/crpolicycortex$ENVIRONMENT" \
    "container_registry" \
    "crpolicycortex$ENVIRONMENT"

# Log Analytics Workspace
import_if_exists "azurerm_log_analytics_workspace.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.OperationalInsights/workspaces/law-policycortex-$ENVIRONMENT" \
    "log_analytics_workspace" \
    "law-policycortex-$ENVIRONMENT"

# Application Insights
import_if_exists "azurerm_application_insights.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Insights/components/ai-policycortex-$ENVIRONMENT" \
    "application_insights" \
    "ai-policycortex-$ENVIRONMENT"

# Container Apps Environment
import_if_exists "azurerm_container_app_environment.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.App/managedEnvironments/cae-policycortex-$ENVIRONMENT" \
    "container_app_environment" \
    "cae-policycortex-$ENVIRONMENT"

# Managed Identity
import_if_exists "azurerm_user_assigned_identity.container_apps" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id-policycortex-$ENVIRONMENT" \
    "managed_identity" \
    "id-policycortex-$ENVIRONMENT"

# Networking resources
# Virtual Network
import_if_exists "module.networking.azurerm_virtual_network.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/virtualNetworks/policycortex-$ENVIRONMENT-vnet" \
    "virtual_network" \
    "policycortex-$ENVIRONMENT-vnet"

# Network Security Groups
for nsg_type in "container_apps" "app_gateway"; do
    import_if_exists "module.networking.azurerm_network_security_group.subnet_nsgs[\"$nsg_type\"]" \
        "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/networkSecurityGroups/policycortex-$ENVIRONMENT-nsg-$nsg_type" \
        "network_security_group" \
        "policycortex-$ENVIRONMENT-nsg-$nsg_type"
done

# Route Table
import_if_exists "module.networking.azurerm_route_table.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/routeTables/policycortex-$ENVIRONMENT-rt" \
    "route_table" \
    "policycortex-$ENVIRONMENT-rt"

# Private DNS Zone
import_if_exists "module.networking.azurerm_private_dns_zone.internal" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/privateDnsZones/policycortex.internal" \
    "private_dns_zone" \
    "policycortex.internal"

# Container Apps
for service in "api_gateway" "azure_integration" "ai_engine" "data_processing" "conversation" "notification" "frontend"; do
    terraform_name=$(echo $service | tr '_' '-')
    import_if_exists "azurerm_container_app.$service" \
        "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.App/containerApps/ca-$terraform_name-$ENVIRONMENT" \
        "container_app" \
        "ca-$terraform_name-$ENVIRONMENT"
done

echo "Import process completed"
echo "Current state:"
terraform state list || echo "No resources in state"