#!/bin/bash

# Script to fix all container apps with proper environment variables

echo "Getting secrets from Key Vault..."

# Get required secrets
COSMOS_CONN=$(az keyvault secret show --vault-name kvpolicycortexdev --name cosmos-connection-string --query "value" -o tsv)
COSMOS_ENDPOINT=$(echo "$COSMOS_CONN" | grep -o 'AccountEndpoint=[^;]*' | cut -d'=' -f2)
COSMOS_KEY=$(echo "$COSMOS_CONN" | grep -o 'AccountKey=[^;]*' | cut -d'=' -f2)
REDIS_CONN=$(az keyvault secret show --vault-name kvpolicycortexdev --name redis-connection-string --query "value" -o tsv)
SQL_PASSWORD=$(az keyvault secret show --vault-name kvpolicycortexdev --name sql-admin-password --query "value" -o tsv)
COGNITIVE_KEY=$(az keyvault secret show --vault-name kvpolicycortexdev --name cognitive-services-key --query "value" -o tsv)
COGNITIVE_ENDPOINT=$(az keyvault secret show --vault-name kvpolicycortexdev --name cognitive-services-endpoint --query "value" -o tsv)

echo "Updating container apps..."

# Define common environment variables
ENV_VARS="ENVIRONMENT=development \
DEBUG=false \
TESTING=false \
SQL_SERVER=policycortex-sql-dev.database.windows.net \
SQL_DATABASE=policycortex_dev \
SQL_USERNAME=sqladmin \
SQL_PASSWORD=${SQL_PASSWORD} \
SQL_PORT=1433 \
SQL_DRIVER=ODBC Driver 18 for SQL Server \
AZURE_COSMOS_ENDPOINT=${COSMOS_ENDPOINT} \
AZURE_COSMOS_KEY=${COSMOS_KEY} \
AZURE_COSMOS_DATABASE=policycortex \
REDIS_URL=${REDIS_CONN} \
AZURE_SUBSCRIPTION_ID=9f16cc88-89ce-49ba-a96d-308ed3169595 \
AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7 \
AZURE_RESOURCE_GROUP=rg-policycortex-dev \
AZURE_LOCATION=eastus \
AZURE_KEY_VAULT_NAME=kvpolicycortexdev \
AZURE_KEY_VAULT_URL=https://kvpolicycortexdev.vault.azure.net/ \
AZURE_STORAGE_ACCOUNT_NAME=stpolicycortexdevstg \
AZURE_SERVICE_BUS_NAMESPACE=policycortex-dev-sb \
AZURE_COGNITIVE_SERVICES_KEY=${COGNITIVE_KEY} \
AZURE_COGNITIVE_SERVICES_ENDPOINT=${COGNITIVE_ENDPOINT} \
JWT_SECRET_KEY=development-secret-key-change-in-production \
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30 \
LOG_LEVEL=info \
ENABLE_COST_OPTIMIZATION=true \
ENABLE_POLICY_AUTOMATION=true \
ENABLE_RBAC_ANALYSIS=true"

# List of container apps to update
APPS=("ca-api-gateway-dev" "ca-azure-integration-dev" "ca-ai-engine-dev" "ca-data-processing-dev" "ca-conversation-dev" "ca-notification-dev")

for app in "${APPS[@]}"; do
    echo "Updating $app..."
    az containerapp update \
        --name "$app" \
        --resource-group rg-policycortex-dev \
        --set-env-vars $ENV_VARS
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully updated $app"
    else
        echo "❌ Failed to update $app"
    fi
done

echo "All updates completed. Waiting for containers to restart..."
sleep 60

echo "Checking health of all services..."
for app in "${APPS[@]}"; do
    echo "Checking $app..."
    az containerapp show --name "$app" --resource-group rg-policycortex-dev --query "properties.runningStatus" -o tsv
done

echo "Script completed!"