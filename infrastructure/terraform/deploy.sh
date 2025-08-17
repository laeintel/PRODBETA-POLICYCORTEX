#!/bin/bash
# Terraform deployment script with automatic resource import

set -e

echo "🚀 PolicyCortex Terraform Deployment Script"
echo "==========================================="

# Configuration
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-205b477d-17e7-4b3b-92c1-32cf02626b78}"
ENVIRONMENT="${TF_VAR_environment:-dev}"
RESOURCE_GROUP="rg-cortex-${ENVIRONMENT}"

echo "Environment: $ENVIRONMENT"
echo "Subscription: $SUBSCRIPTION_ID"
echo "Resource Group: $RESOURCE_GROUP"
echo ""

# Function to check if resource exists
resource_exists() {
    local resource_id=$1
    az resource show --ids "$resource_id" &>/dev/null
    return $?
}

# Function to import resource if it exists
import_if_exists() {
    local terraform_address=$1
    local azure_resource_id=$2
    local resource_type=$3
    
    echo -n "Checking $resource_type... "
    
    if resource_exists "$azure_resource_id"; then
        echo "Found! Importing to Terraform state..."
        
        # Check if already in state
        if terraform state show "$terraform_address" &>/dev/null; then
            echo "  Already in Terraform state, skipping import"
        else
            terraform import "$terraform_address" "$azure_resource_id" || {
                echo "  Warning: Import failed (may already be managed)"
            }
        fi
    else
        echo "Not found (will be created)"
    fi
}

# Initialize Terraform
echo "📦 Initializing Terraform..."
terraform init -upgrade

# Import existing resources
echo ""
echo "🔍 Checking for existing Azure resources..."

# Import resource group
import_if_exists \
    "azurerm_resource_group.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" \
    "Resource Group"

# Import Container Apps Environment
import_if_exists \
    "azurerm_container_app_environment.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.App/managedEnvironments/cae-cortex-$ENVIRONMENT" \
    "Container Apps Environment"

# Import Container Apps
import_if_exists \
    "azurerm_container_app.core" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.App/containerApps/ca-cortex-core-$ENVIRONMENT" \
    "Core Container App"

import_if_exists \
    "azurerm_container_app.frontend" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.App/containerApps/ca-cortex-frontend-$ENVIRONMENT" \
    "Frontend Container App"

# Import Log Analytics Workspace
import_if_exists \
    "azurerm_log_analytics_workspace.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.OperationalInsights/workspaces/log-cortex-$ENVIRONMENT" \
    "Log Analytics Workspace"

# Import Application Insights
import_if_exists \
    "azurerm_application_insights.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Insights/components/appi-cortex-$ENVIRONMENT" \
    "Application Insights"

# Import Storage Account (need to find the exact name with suffix)
STORAGE_ACCOUNT=$(az storage account list --resource-group "$RESOURCE_GROUP" --query "[?starts_with(name, 'stcortex$ENVIRONMENT')].name | [0]" -o tsv 2>/dev/null || echo "")
if [ -n "$STORAGE_ACCOUNT" ]; then
    import_if_exists \
        "azurerm_storage_account.main" \
        "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$STORAGE_ACCOUNT" \
        "Storage Account"
fi

# Import Container Registry
ACR_NAME=$(az acr list --resource-group "$RESOURCE_GROUP" --query "[?starts_with(name, 'crcortex$ENVIRONMENT')].name | [0]" -o tsv 2>/dev/null || echo "")
if [ -n "$ACR_NAME" ]; then
    import_if_exists \
        "azurerm_container_registry.main" \
        "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ContainerRegistry/registries/$ACR_NAME" \
        "Container Registry"
fi

# Import Key Vault
KV_NAME=$(az keyvault list --resource-group "$RESOURCE_GROUP" --query "[?starts_with(name, 'kv-cortex-$ENVIRONMENT')].name | [0]" -o tsv 2>/dev/null || echo "")
if [ -n "$KV_NAME" ]; then
    import_if_exists \
        "azurerm_key_vault.main" \
        "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$KV_NAME" \
        "Key Vault"
fi

# Import PostgreSQL Server
PSQL_NAME=$(az postgres flexible-server list --resource-group "$RESOURCE_GROUP" --query "[?starts_with(name, 'psql-cortex-$ENVIRONMENT')].name | [0]" -o tsv 2>/dev/null || echo "")
if [ -n "$PSQL_NAME" ]; then
    import_if_exists \
        "azurerm_postgresql_flexible_server.main" \
        "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.DBforPostgreSQL/flexibleServers/$PSQL_NAME" \
        "PostgreSQL Server"
    
    # Import database
    import_if_exists \
        "azurerm_postgresql_flexible_server_database.main" \
        "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.DBforPostgreSQL/flexibleServers/$PSQL_NAME/databases/policycortex" \
        "PostgreSQL Database"
fi

# Import Cosmos DB
COSMOS_NAME=$(az cosmosdb list --resource-group "$RESOURCE_GROUP" --query "[?starts_with(name, 'cosmos-cortex-$ENVIRONMENT')].name | [0]" -o tsv 2>/dev/null || echo "")
if [ -n "$COSMOS_NAME" ]; then
    import_if_exists \
        "azurerm_cosmosdb_account.main" \
        "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.DocumentDB/databaseAccounts/$COSMOS_NAME" \
        "Cosmos DB Account"
    
    # Import Cosmos Database
    import_if_exists \
        "azurerm_cosmosdb_sql_database.main" \
        "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.DocumentDB/databaseAccounts/$COSMOS_NAME/sqlDatabases/policycortex" \
        "Cosmos DB Database"
fi

# Import Virtual Network
import_if_exists \
    "azurerm_virtual_network.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/virtualNetworks/vnet-cortex-$ENVIRONMENT" \
    "Virtual Network"

# Import Subnet
import_if_exists \
    "azurerm_subnet.main" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/virtualNetworks/vnet-cortex-$ENVIRONMENT/subnets/subnet-app" \
    "Subnet"

# Import Cognitive Services (OpenAI)
import_if_exists \
    "azurerm_cognitive_account.openai" \
    "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.CognitiveServices/accounts/cogao-cortex-$ENVIRONMENT" \
    "Azure OpenAI"

echo ""
echo "✅ Import check complete!"
echo ""

# Validate Terraform configuration
echo "🔍 Validating Terraform configuration..."
terraform validate

# Create Terraform plan
echo ""
echo "📋 Creating Terraform plan..."
terraform plan -out=tfplan -var="environment=$ENVIRONMENT"

# Show plan summary
echo ""
echo "📊 Plan Summary:"
terraform show -no-color tfplan | grep -E "will be created|will be destroyed|will be updated" | sort | uniq -c || echo "No changes detected"

# Ask for confirmation
echo ""
echo "⚠️  The above resources will be created/modified/destroyed."
read -p "Do you want to apply these changes? (yes/no): " -r
echo ""

if [[ $REPLY =~ ^[Yy][Ee][Ss]|[Yy]$ ]]; then
    echo "🚀 Applying Terraform changes..."
    terraform apply tfplan
    
    echo ""
    echo "✅ Deployment complete!"
    echo ""
    echo "📌 Outputs:"
    terraform output
else
    echo "❌ Deployment cancelled"
    exit 1
fi