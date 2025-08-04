#!/bin/bash

# Sync Key Vault secrets to Container Apps
# This script creates secret references in Container Apps that point to Key Vault secrets

set -e

ENVIRONMENT="${1:-dev}"
RESOURCE_GROUP="rg-pcx-app-${ENVIRONMENT}"
KEY_VAULT_NAME="kv-pcx-${ENVIRONMENT}"

echo "🔐 Syncing Key Vault secrets to Container Apps for ${ENVIRONMENT} environment"
echo "Resource Group: ${RESOURCE_GROUP}"
echo "Key Vault: ${KEY_VAULT_NAME}"

# Get Key Vault URI
KEY_VAULT_URI="https://${KEY_VAULT_NAME}.vault.azure.net"
echo "Key Vault URI: ${KEY_VAULT_URI}"

# List of secrets that Container Apps need
REQUIRED_SECRETS=(
    "jwt-secret"
    "encryption-key"
    "azure-client-id"
    "azure-tenant-id"
    "cosmos-endpoint"
    "cosmos-key"
    "redis-connection-string"
    "storage-account-name"
    "cognitive-services-key"
    "cognitive-services-endpoint"
    "application-insights-connection-string"
)

# Container Apps that need the secrets
CONTAINER_APPS=(
    "ca-pcx-gateway-dev"
    "ca-pcx-azureint-dev"
    "ca-pcx-ai-dev"
    "ca-pcx-dataproc-dev"
    "ca-pcx-chat-dev"
    "ca-pcx-notify-dev"
)

# Function to update Container App secrets
update_container_app_secrets() {
    local app_name=$1
    echo "📦 Updating secrets for ${app_name}..."
    
    # Build the secrets parameter string
    local secrets_params=""
    for secret in "${REQUIRED_SECRETS[@]}"; do
        # Get the secret value from Key Vault
        secret_value=$(az keyvault secret show --vault-name "${KEY_VAULT_NAME}" --name "${secret}" --query "value" -o tsv 2>/dev/null || echo "")
        
        if [ -n "${secret_value}" ]; then
            # Add to secrets parameters (escape special characters)
            if [ -n "${secrets_params}" ]; then
                secrets_params="${secrets_params} "
            fi
            secrets_params="${secrets_params}${secret}=\"${secret_value}\""
        else
            echo "⚠️  Warning: Secret '${secret}' not found in Key Vault"
        fi
    done
    
    # Update the container app with all secrets at once
    if [ -n "${secrets_params}" ]; then
        echo "Setting secrets for ${app_name}..."
        # Use --replace-all-secrets to replace all secrets at once
        eval "az containerapp secret set \
            --name ${app_name} \
            --resource-group ${RESOURCE_GROUP} \
            --secrets ${secrets_params} \
            --output none"
        echo "✅ Secrets updated for ${app_name}"
    else
        echo "❌ No secrets found to update for ${app_name}"
    fi
}

# Update each Container App
for app in "${CONTAINER_APPS[@]}"; do
    if az containerapp show --name "${app}" --resource-group "${RESOURCE_GROUP}" &>/dev/null; then
        update_container_app_secrets "${app}"
    else
        echo "⚠️  Container App ${app} not found, skipping..."
    fi
done

echo ""
echo "✅ Key Vault secrets sync completed!"
echo ""
echo "📝 Note: Container Apps now have copies of the Key Vault secrets."
echo "   For better security, consider using managed identity to access Key Vault directly."