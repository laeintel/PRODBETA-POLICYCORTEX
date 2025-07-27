#!/bin/bash
# Script to update all Container Apps with latest images
# This should be added to your Application Pipeline after build/push steps

set -e

# Configuration
ENVIRONMENT=${1:-"dev"}
RESOURCE_GROUP="rg-policortex001-app-${ENVIRONMENT}"
CONTAINER_REGISTRY="crpolicortex001${ENVIRONMENT}.azurecr.io"

echo "🚀 Updating Container Apps with latest images for ${ENVIRONMENT} environment"
echo "Resource Group: ${RESOURCE_GROUP}"
echo "Container Registry: ${CONTAINER_REGISTRY}"

# Function to update a Container App with Key Vault environment variables
update_container_app() {
    local app_name=$1
    local image_name=$2
    local service_name=$(echo $app_name | sed "s/ca-//g" | sed "s/-${ENVIRONMENT}//g")
    
    echo "📦 Updating ${app_name} with image ${image_name}:latest and Key Vault secrets"
    
    # Create revision suffix to force new revision
    local revision_suffix="r$(date +%s)"
    
    if [[ "$service_name" == "frontend" ]]; then
        # Get dynamic FQDNs
        local api_fqdn=$(az containerapp show --name "ca-api-gateway-${ENVIRONMENT}" --resource-group "${RESOURCE_GROUP}" --query "properties.configuration.ingress.fqdn" -o tsv)
        local frontend_fqdn=$(az containerapp show --name "ca-frontend-${ENVIRONMENT}" --resource-group "${RESOURCE_GROUP}" --query "properties.configuration.ingress.fqdn" -o tsv)
        
        # Frontend with Key Vault secrets
        az containerapp update \
            --name "${app_name}" \
            --resource-group "${RESOURCE_GROUP}" \
            --image "${CONTAINER_REGISTRY}/${image_name}:latest" \
            --revision-suffix "${revision_suffix}" \
            --set-env-vars \
                "ENVIRONMENT=${ENVIRONMENT}" \
                "SERVICE_NAME=frontend" \
                "PORT=8080" \
                "LOG_LEVEL=INFO" \
                "VITE_API_BASE_URL=https://${api_fqdn}/api" \
                "VITE_WS_URL=wss://${api_fqdn}/ws" \
                "VITE_AZURE_REDIRECT_URI=https://${frontend_fqdn}" \
                "VITE_APP_VERSION=1.0.0" \
            --replace-env-vars \
                "VITE_AZURE_CLIENT_ID=secretref:azure-client-id" \
                "VITE_AZURE_TENANT_ID=secretref:azure-tenant-id" \
            --output table
    else
        # Backend services with Key Vault secrets
        local service_port
        case $service_name in
            api-gateway) service_port=8000;;
            azure-integration) service_port=8001;;
            ai-engine) service_port=8002;;
            data-processing) service_port=8003;;
            conversation) service_port=8004;;
            notification) service_port=8005;;
            *) service_port=8000;;
        esac
        
        az containerapp update \
            --name "${app_name}" \
            --resource-group "${RESOURCE_GROUP}" \
            --image "${CONTAINER_REGISTRY}/${image_name}:latest" \
            --revision-suffix "${revision_suffix}" \
            --set-env-vars \
                "ENVIRONMENT=${ENVIRONMENT}" \
                "SERVICE_NAME=${service_name}" \
                "SERVICE_PORT=${service_port}" \
                "LOG_LEVEL=INFO" \
            --replace-env-vars \
                "JWT_SECRET_KEY=secretref:jwt-secret" \
                "ENCRYPTION_KEY=secretref:encryption-key" \
                "AZURE_CLIENT_ID=secretref:azure-client-id" \
                "AZURE_TENANT_ID=secretref:azure-tenant-id" \
                "AZURE_COSMOS_ENDPOINT=secretref:cosmos-endpoint" \
                "AZURE_COSMOS_KEY=secretref:cosmos-key" \
                "REDIS_CONNECTION_STRING=secretref:redis-connection-string" \
                "AZURE_STORAGE_ACCOUNT_NAME=secretref:storage-account-name" \
                "COGNITIVE_SERVICES_KEY=secretref:cognitive-services-key" \
                "COGNITIVE_SERVICES_ENDPOINT=secretref:cognitive-services-endpoint" \
                "APPLICATION_INSIGHTS_CONNECTION_STRING=secretref:application-insights-connection-string" \
            --output table
    fi
    
    echo "✅ ${app_name} updated successfully with revision ${revision_suffix}"
    echo ""
}

# Update all Container Apps
echo "🔄 Starting Container App updates..."

# Backend Services
update_container_app "ca-api-gateway-${ENVIRONMENT}" "policortex001-api-gateway"
update_container_app "ca-azure-integration-${ENVIRONMENT}" "policortex001-azure-integration"
update_container_app "ca-ai-engine-${ENVIRONMENT}" "policortex001-ai-engine"
update_container_app "ca-data-processing-${ENVIRONMENT}" "policortex001-data-processing"
update_container_app "ca-conversation-${ENVIRONMENT}" "policortex001-conversation"
update_container_app "ca-notification-${ENVIRONMENT}" "policortex001-notification"

# Frontend
update_container_app "ca-frontend-${ENVIRONMENT}" "policortex001-frontend"

echo "🎉 All Container Apps updated with latest images!"
echo ""

# Verify revisions were created
echo "🔍 Checking new revisions..."
for app in "ca-api-gateway-${ENVIRONMENT}" "ca-frontend-${ENVIRONMENT}"; do
    echo "Latest revision for ${app}:"
    az containerapp revision list \
        --name "${app}" \
        --resource-group "${RESOURCE_GROUP}" \
        --query "[0].{Name:name, CreatedTime:properties.createdTime, Active:properties.active}" \
        --output table
    echo ""
done

echo "✅ Container Apps update completed!"