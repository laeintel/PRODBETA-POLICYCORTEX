#!/bin/bash
# Script to populate Azure Key Vault with PolicyCortex secrets
# This script creates/updates all secrets needed by the Container Apps

set -e

# Configuration
ENVIRONMENT=${1:-"dev"}
RESOURCE_GROUP="rg-pcx-app-${ENVIRONMENT}"
KEY_VAULT_NAME="kv-pcx-${ENVIRONMENT}"

echo "🔐 Setting up Key Vault secrets for PolicyCortex ${ENVIRONMENT} environment"
echo "Resource Group: ${RESOURCE_GROUP}"
echo "Key Vault: ${KEY_VAULT_NAME}"

# Check if Key Vault exists
if ! az keyvault show --name "${KEY_VAULT_NAME}" --resource-group "${RESOURCE_GROUP}" &>/dev/null; then
    echo "❌ Key Vault ${KEY_VAULT_NAME} not found in resource group ${RESOURCE_GROUP}"
    echo "Please ensure the infrastructure is deployed first."
    exit 1
fi

echo "✅ Key Vault ${KEY_VAULT_NAME} found"

# Function to set or update a secret
set_secret() {
    local secret_name=$1
    local secret_value=$2
    local description=$3
    
    echo "Setting secret: ${secret_name} (${description})"
    az keyvault secret set \
        --vault-name "${KEY_VAULT_NAME}" \
        --name "${secret_name}" \
        --value "${secret_value}" \
        --output none
}

# Function to generate a random secret
generate_secret() {
    openssl rand -base64 32
}

# Function to get existing secret or generate new one
get_or_generate_secret() {
    local secret_name=$1
    
    # Try to get existing secret
    if existing_secret=$(az keyvault secret show --vault-name "${KEY_VAULT_NAME}" --name "${secret_name}" --query "value" -o tsv 2>/dev/null); then
        echo "${existing_secret}"
    else
        generate_secret
    fi
}

echo "🔑 Setting up authentication secrets..."

# JWT Secret Key
jwt_secret=$(get_or_generate_secret "jwt-secret")
set_secret "jwt-secret" "${jwt_secret}" "JWT token signing key"

# Encryption Key
encryption_key=$(get_or_generate_secret "encryption-key")
set_secret "encryption-key" "${encryption_key}" "Data encryption key"

echo "🌐 Setting up Azure AD secrets..."

# Azure AD Configuration (you'll need to replace these with real values)
# These are example values - replace with your actual Azure AD app registration
AZURE_CLIENT_ID=${AZURE_CLIENT_ID:-"e8c5b8a0-123e-4567-8901-234567890123"}
AZURE_TENANT_ID=${AZURE_TENANT_ID:-"9ef5b184-d371-462a-bc75-5024ce8baff7"}

set_secret "azure-client-id" "${AZURE_CLIENT_ID}" "Azure AD Application Client ID"
set_secret "azure-tenant-id" "${AZURE_TENANT_ID}" "Azure AD Tenant ID"

echo "🗄️ Setting up database and storage secrets..."

# Get Cosmos DB connection details
echo "Retrieving Cosmos DB details..."
cosmos_accounts=$(az cosmosdb list --resource-group "${RESOURCE_GROUP}" --query "[].{name:name}" -o tsv)
if [ -n "${cosmos_accounts}" ]; then
    cosmos_account=$(echo "${cosmos_accounts}" | head -n 1)
    cosmos_endpoint="https://${cosmos_account}.documents.azure.com:443/"
    cosmos_key=$(az cosmosdb keys list --name "${cosmos_account}" --resource-group "${RESOURCE_GROUP}" --query "primaryMasterKey" -o tsv)
    
    set_secret "cosmos-endpoint" "${cosmos_endpoint}" "Cosmos DB endpoint"
    set_secret "cosmos-key" "${cosmos_key}" "Cosmos DB primary key"
    
    # Cosmos connection string
    cosmos_connection_string="AccountEndpoint=${cosmos_endpoint};AccountKey=${cosmos_key};"
    set_secret "cosmos-connection-string" "${cosmos_connection_string}" "Cosmos DB connection string"
else
    echo "⚠️  No Cosmos DB found, using placeholder values"
    set_secret "cosmos-endpoint" "https://placeholder-cosmos.documents.azure.com:443/" "Cosmos DB endpoint (placeholder)"
    set_secret "cosmos-key" "placeholder-cosmos-key" "Cosmos DB key (placeholder)"
    set_secret "cosmos-connection-string" "AccountEndpoint=https://placeholder-cosmos.documents.azure.com:443/;AccountKey=placeholder-key;" "Cosmos DB connection string (placeholder)"
fi

# Get Redis connection string
echo "Retrieving Redis details..."
redis_caches=$(az redis list --resource-group "${RESOURCE_GROUP}" --query "[].{name:name}" -o tsv)
if [ -n "${redis_caches}" ]; then
    redis_name=$(echo "${redis_caches}" | head -n 1)
    redis_key=$(az redis list-keys --name "${redis_name}" --resource-group "${RESOURCE_GROUP}" --query "primaryKey" -o tsv)
    redis_hostname=$(az redis show --name "${redis_name}" --resource-group "${RESOURCE_GROUP}" --query "hostName" -o tsv)
    redis_port=$(az redis show --name "${redis_name}" --resource-group "${RESOURCE_GROUP}" --query "port" -o tsv)
    redis_ssl_port=$(az redis show --name "${redis_name}" --resource-group "${RESOURCE_GROUP}" --query "sslPort" -o tsv)
    
    redis_connection_string="${redis_hostname}:${redis_ssl_port},password=${redis_key},ssl=True,abortConnect=False"
    set_secret "redis-connection-string" "${redis_connection_string}" "Redis connection string"
else
    echo "⚠️  No Redis found, using placeholder value"
    set_secret "redis-connection-string" "localhost:6379,password=placeholder-redis-key" "Redis connection string (placeholder)"
fi

# Get Storage Account details
echo "Retrieving Storage Account details..."
storage_accounts=$(az storage account list --resource-group "${RESOURCE_GROUP}" --query "[].{name:name}" -o tsv)
if [ -n "${storage_accounts}" ]; then
    storage_account=$(echo "${storage_accounts}" | head -n 1)
    set_secret "storage-account-name" "${storage_account}" "Storage account name"
    
    # Get storage connection string
    storage_connection_string=$(az storage account show-connection-string --name "${storage_account}" --resource-group "${RESOURCE_GROUP}" --query "connectionString" -o tsv)
    set_secret "storage-connection-string" "${storage_connection_string}" "Storage account connection string"
else
    echo "⚠️  No Storage Account found, using placeholder value"
    set_secret "storage-account-name" "placeholder-storage" "Storage account name (placeholder)"
    set_secret "storage-connection-string" "DefaultEndpointsProtocol=https;AccountName=placeholder;AccountKey=placeholder;" "Storage connection string (placeholder)"
fi

echo "🧠 Setting up AI services secrets..."

# Get Cognitive Services details
cognitive_accounts=$(az cognitiveservices account list --resource-group "${RESOURCE_GROUP}" --query "[].{name:name}" -o tsv)
if [ -n "${cognitive_accounts}" ]; then
    cognitive_account=$(echo "${cognitive_accounts}" | head -n 1)
    cognitive_key=$(az cognitiveservices account keys list --name "${cognitive_account}" --resource-group "${RESOURCE_GROUP}" --query "key1" -o tsv)
    cognitive_endpoint=$(az cognitiveservices account show --name "${cognitive_account}" --resource-group "${RESOURCE_GROUP}" --query "properties.endpoint" -o tsv)
    
    set_secret "cognitive-services-key" "${cognitive_key}" "Cognitive Services API key"
    set_secret "cognitive-services-endpoint" "${cognitive_endpoint}" "Cognitive Services endpoint"
else
    echo "⚠️  No Cognitive Services found, using placeholder values"
    set_secret "cognitive-services-key" "placeholder-cognitive-key" "Cognitive Services key (placeholder)"
    set_secret "cognitive-services-endpoint" "https://placeholder-cognitive.cognitiveservices.azure.com/" "Cognitive Services endpoint (placeholder)"
fi

echo "📊 Setting up monitoring secrets..."

# Get Application Insights connection string
app_insights=$(az monitor app-insights component list --resource-group "${RESOURCE_GROUP}" --query "[].{name:name}" -o tsv)
if [ -n "${app_insights}" ]; then
    app_insights_name=$(echo "${app_insights}" | head -n 1)
    app_insights_connection_string=$(az monitor app-insights component show --app "${app_insights_name}" --resource-group "${RESOURCE_GROUP}" --query "connectionString" -o tsv)
    
    set_secret "application-insights-connection-string" "${app_insights_connection_string}" "Application Insights connection string"
else
    echo "⚠️  No Application Insights found, using placeholder value"
    set_secret "application-insights-connection-string" "InstrumentationKey=placeholder-key" "Application Insights connection string (placeholder)"
fi

echo "✅ Key Vault secrets setup completed!"
echo ""
echo "🔍 Summary of secrets created/updated:"
az keyvault secret list --vault-name "${KEY_VAULT_NAME}" --query "[].{Name:name, Created:attributes.created}" --output table

echo ""
echo "🔧 Next steps:"
echo "1. Update Container Apps configuration to use secretRef instead of value"
echo "2. Redeploy the Container Apps to pick up the new secrets"
echo "3. Replace placeholder values with real Azure AD app registration details"
echo ""
echo "💡 To update Azure AD secrets with real values:"
echo "   export AZURE_CLIENT_ID='your-real-client-id'"
echo "   export AZURE_TENANT_ID='your-real-tenant-id'"
echo "   ./setup-keyvault-secrets.sh ${ENVIRONMENT}"