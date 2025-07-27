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

# Function to update a Container App
update_container_app() {
    local app_name=$1
    local image_name=$2
    
    echo "📦 Updating ${app_name} with image ${image_name}:latest"
    
    az containerapp update \
        --name "${app_name}" \
        --resource-group "${RESOURCE_GROUP}" \
        --image "${CONTAINER_REGISTRY}/${image_name}:latest" \
        --output table
    
    echo "✅ ${app_name} updated successfully"
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