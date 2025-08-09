#!/bin/bash
# Script to import existing Azure resources into Terraform state
# This prevents "already exists" errors when resources were created outside Terraform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get environment from argument or default to dev
ENV="${1:-dev}"
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-$2}"

if [ -z "$SUBSCRIPTION_ID" ]; then
    echo -e "${RED}Error: AZURE_SUBSCRIPTION_ID not set${NC}"
    exit 1
fi

echo -e "${GREEN}Starting resource import for environment: $ENV${NC}"
echo "Subscription ID: $SUBSCRIPTION_ID"

# Ensure correct subscription context
az account set --subscription "$SUBSCRIPTION_ID" >/dev/null 2>&1 || true

# Function to try importing a resource
import_resource() {
    local resource_type=$1
    local resource_address=$2
    local resource_id=$3
    
    echo -e "${YELLOW}Checking $resource_type...${NC}"
    
    # Check if resource exists in state
    if terraform state show "$resource_address" 2>/dev/null; then
        echo "  ✓ Already in state"
    else
        # Check if the resource exists in Azure first
        if resource_exists "$resource_id"; then
            echo "  → Resource exists in Azure, importing..."
            # Try to import with more verbose output for debugging
            if terraform import "$resource_address" "$resource_id"; then
                echo -e "  ${GREEN}✓ Imported successfully${NC}"
            else
                echo -e "  ${RED}✗ Import failed - resource exists but couldn't be imported${NC}"
                echo "  Resource ID: $resource_id"
            fi
        else
            echo -e "  ${YELLOW}⚠ Resource not found in Azure${NC}"
        fi
    fi
}

# Generic existence check that doesn't require provider-specific CLI extensions
resource_exists() {
    local resource_id=$1
    # Handle Git Bash on Windows path translation issue
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        # Use double slash to prevent Git Bash from treating it as a Unix path
        resource_id="//${resource_id#/}"
    fi
    az resource show --ids "$resource_id" >/dev/null 2>&1
}

# Resource Group
RG_NAME="rg-cortex-${ENV}"
echo -e "\n${GREEN}Checking Resource Group: $RG_NAME${NC}"
if az group show --name "$RG_NAME" 2>/dev/null; then
    import_resource "Resource Group" \
        "azurerm_resource_group.main" \
        "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}"
fi

# Storage Account for tfstate
TFSTATE_RG="rg-tfstate-cortex-${ENV}"
HASH=$(echo -n "$GITHUB_REPOSITORY" | sha1sum | cut -c1-6)
TFSTATE_SA="sttfcortex${ENV}${HASH}"
echo -e "\n${GREEN}Checking TFState Storage Account: $TFSTATE_SA${NC}"
if az storage account show --name "$TFSTATE_SA" --resource-group "$TFSTATE_RG" 2>/dev/null; then
    echo "  ✓ TFState storage exists (managed separately)"
fi

# PostgreSQL Flexible Server
PSQL_NAME="psql-cortex-${ENV}"
echo -e "\n${GREEN}Checking PostgreSQL Server: $PSQL_NAME${NC}"
if az postgres flexible-server show --name "$PSQL_NAME" --resource-group "$RG_NAME" 2>/dev/null; then
    import_resource "PostgreSQL Server" \
        "azurerm_postgresql_flexible_server.main" \
        "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.DBforPostgreSQL/flexibleServers/${PSQL_NAME}"
    
    # PostgreSQL Database
    import_resource "PostgreSQL Database" \
        "azurerm_postgresql_flexible_server_database.main" \
        "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.DBforPostgreSQL/flexibleServers/${PSQL_NAME}/databases/policycortex"
fi

# CosmosDB Account - check with unique suffix
echo -e "\n${GREEN}Checking CosmosDB Accounts...${NC}"
COSMOS_ACCOUNTS=$(az cosmosdb list --resource-group "$RG_NAME" --query "[?starts_with(name, 'cosmos-cortex-${ENV}')].name" -o tsv 2>/dev/null || true)
if [ ! -z "$COSMOS_ACCOUNTS" ]; then
    for COSMOS_NAME in $COSMOS_ACCOUNTS; do
        import_resource "CosmosDB Account" \
            "azurerm_cosmosdb_account.main" \
            "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.DocumentDB/databaseAccounts/${COSMOS_NAME}"
        
        # CosmosDB Database
        import_resource "CosmosDB Database" \
            "azurerm_cosmosdb_sql_database.main" \
            "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.DocumentDB/databaseAccounts/${COSMOS_NAME}/sqlDatabases/policycortex"
    done
fi

# Container Registry
echo -e "\n${GREEN}Checking Container Registry...${NC}"
ACR_ACCOUNTS=$(az acr list --resource-group "$RG_NAME" --query "[?starts_with(name, 'crcortex${ENV}')].name" -o tsv 2>/dev/null || true)
if [ ! -z "$ACR_ACCOUNTS" ]; then
    for ACR_NAME in $ACR_ACCOUNTS; do
        import_resource "Container Registry" \
            "azurerm_container_registry.main" \
            "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.ContainerRegistry/registries/${ACR_NAME}"
    done
fi

# Key Vault
echo -e "\n${GREEN}Checking Key Vault...${NC}"
KV_ACCOUNTS=$(az keyvault list --resource-group "$RG_NAME" --query "[?starts_with(name, 'kv-cortex-${ENV}')].name" -o tsv 2>/dev/null || true)
if [ ! -z "$KV_ACCOUNTS" ]; then
    for KV_NAME in $KV_ACCOUNTS; do
        import_resource "Key Vault" \
            "azurerm_key_vault.main" \
            "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.KeyVault/vaults/${KV_NAME}"
    done
fi

# Virtual Network
VNET_NAME="vnet-cortex-${ENV}"
echo -e "\n${GREEN}Checking Virtual Network: $VNET_NAME${NC}"
if az network vnet show --name "$VNET_NAME" --resource-group "$RG_NAME" 2>/dev/null; then
    import_resource "Virtual Network" \
        "azurerm_virtual_network.main" \
        "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.Network/virtualNetworks/${VNET_NAME}"
    
    # Subnet
    import_resource "Subnet" \
        "azurerm_subnet.main" \
        "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.Network/virtualNetworks/${VNET_NAME}/subnets/subnet-app"
fi

# Log Analytics Workspace
LOG_NAME="log-cortex-${ENV}"
echo -e "\n${GREEN}Checking Log Analytics Workspace: $LOG_NAME${NC}"
if az monitor log-analytics workspace show --name "$LOG_NAME" --resource-group "$RG_NAME" 2>/dev/null; then
    import_resource "Log Analytics Workspace" \
        "azurerm_log_analytics_workspace.main" \
        "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.OperationalInsights/workspaces/${LOG_NAME}"
fi

# Application Insights
APPI_NAME="appi-cortex-${ENV}"
echo -e "\n${GREEN}Checking Application Insights: $APPI_NAME${NC}"
if az monitor app-insights component show --app "$APPI_NAME" --resource-group "$RG_NAME" 2>/dev/null; then
    import_resource "Application Insights" \
        "azurerm_application_insights.main" \
        "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.Insights/components/${APPI_NAME}"
fi

# Container Apps Environment (via az resource)
CAE_NAME="cae-cortex-${ENV}"
CAE_ID="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.App/managedEnvironments/${CAE_NAME}"
echo -e "\n${GREEN}Checking Container Apps Environment: $CAE_NAME${NC}"

# First check if resource exists in state and remove if it's stale
if terraform state show "azurerm_container_app_environment.main" 2>/dev/null; then
    # Verify the resource still exists in Azure
    if ! resource_exists "$CAE_ID"; then
        echo "  → Resource in state but not in Azure, removing from state..."
        terraform state rm "azurerm_container_app_environment.main" 2>/dev/null || true
    fi
fi

# Now check if it exists in Azure and needs importing
if resource_exists "$CAE_ID"; then
    import_resource "Container Apps Environment" \
        "azurerm_container_app_environment.main" \
        "$CAE_ID"
fi

# Container App - Core (via az resource)
CA_CORE_NAME="ca-cortex-core-${ENV}"
CA_CORE_ID="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.App/containerApps/${CA_CORE_NAME}"
echo -e "\n${GREEN}Checking Container App Core: $CA_CORE_NAME${NC}"

# First check if resource exists in state and remove if it's stale
if terraform state show "azurerm_container_app.core" 2>/dev/null; then
    # Verify the resource still exists in Azure
    if ! resource_exists "$CA_CORE_ID"; then
        echo "  → Resource in state but not in Azure, removing from state..."
        terraform state rm "azurerm_container_app.core" 2>/dev/null || true
    fi
fi

# Now check if it exists in Azure
if resource_exists "$CA_CORE_ID"; then
    # Check if the Container App is in a failed state
    PROVISIONING_STATE=$(az containerapp show --name "$CA_CORE_NAME" --resource-group "$RG_NAME" --query "properties.provisioningState" -o tsv 2>/dev/null || echo "Unknown")
    
    if [ "$PROVISIONING_STATE" == "Failed" ]; then
        echo "  → Container App exists but is in Failed state, deleting..."
        az containerapp delete --name "$CA_CORE_NAME" --resource-group "$RG_NAME" --yes >/dev/null 2>&1
        echo "  → Failed Container App deleted, will be recreated by Terraform"
    else
        import_resource "Container App Core" \
            "azurerm_container_app.core" \
            "$CA_CORE_ID"
    fi
fi

# Container App - Frontend (via az resource)
CA_FRONTEND_NAME="ca-cortex-frontend-${ENV}"
CA_FRONTEND_ID="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.App/containerApps/${CA_FRONTEND_NAME}"
echo -e "\n${GREEN}Checking Container App Frontend: $CA_FRONTEND_NAME${NC}"

# First check if resource exists in state and remove if it's stale
if terraform state show "azurerm_container_app.frontend" 2>/dev/null; then
    # Verify the resource still exists in Azure
    if ! resource_exists "$CA_FRONTEND_ID"; then
        echo "  → Resource in state but not in Azure, removing from state..."
        terraform state rm "azurerm_container_app.frontend" 2>/dev/null || true
    fi
fi

# Now check if it exists in Azure
if resource_exists "$CA_FRONTEND_ID"; then
    # Check if the Container App is in a failed state
    PROVISIONING_STATE=$(az containerapp show --name "$CA_FRONTEND_NAME" --resource-group "$RG_NAME" --query "properties.provisioningState" -o tsv 2>/dev/null || echo "Unknown")
    
    if [ "$PROVISIONING_STATE" == "Failed" ]; then
        echo "  → Container App exists but is in Failed state, deleting..."
        az containerapp delete --name "$CA_FRONTEND_NAME" --resource-group "$RG_NAME" --yes >/dev/null 2>&1
        echo "  → Failed Container App deleted, will be recreated by Terraform"
    else
        import_resource "Container App Frontend" \
            "azurerm_container_app.frontend" \
            "$CA_FRONTEND_ID"
    fi
fi

# Service Bus (prod only)
if [ "$ENV" == "prod" ]; then
    SB_NAME="sb-cortex-${ENV}"
    echo -e "\n${GREEN}Checking Service Bus: $SB_NAME${NC}"
    if az servicebus namespace show --name "$SB_NAME" --resource-group "$RG_NAME" 2>/dev/null; then
        import_resource "Service Bus Namespace" \
            "azurerm_servicebus_namespace.main[0]" \
            "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}/providers/Microsoft.ServiceBus/namespaces/${SB_NAME}"
    fi
fi

echo -e "\n${GREEN}Import scan complete!${NC}"
echo "You can now run 'terraform plan' to see the current state."