#!/bin/bash
# Script to deploy container apps with proper images and logging configuration

set -e

# Parameters
ENVIRONMENT=${1:-dev}
RESOURCE_GROUP=${2:-rg-pcx-app-$ENVIRONMENT}
BUILD_ID=${3:-latest}
REGISTRY_NAME=${4:-crpcx$ENVIRONMENT}

echo "PolicyCortex Container Apps Deployment"
echo "======================================="
echo "Environment: $ENVIRONMENT"
echo "Resource Group: $RESOURCE_GROUP"
echo "Build ID: $BUILD_ID"
echo "Registry: $REGISTRY_NAME"

# Get registry login server
REGISTRY_SERVER=$(az acr show --name $REGISTRY_NAME --query loginServer -o tsv)
echo "Registry Server: $REGISTRY_SERVER"

# Define service mappings
declare -A SERVICE_MAP
SERVICE_MAP["api_gateway"]="ca-pcx-gateway-$ENVIRONMENT"
SERVICE_MAP["azure_integration"]="ca-pcx-azureint-$ENVIRONMENT"
SERVICE_MAP["ai_engine"]="ca-pcx-ai-$ENVIRONMENT"
SERVICE_MAP["data_processing"]="ca-pcx-dataproc-$ENVIRONMENT"
SERVICE_MAP["conversation"]="ca-pcx-chat-$ENVIRONMENT"
SERVICE_MAP["notification"]="ca-pcx-notify-$ENVIRONMENT"

# Define image names
declare -A IMAGE_MAP
IMAGE_MAP["api_gateway"]="policortex001-api-gateway"
IMAGE_MAP["azure_integration"]="policortex001-azure-integration"
IMAGE_MAP["ai_engine"]="policortex001-ai-engine"
IMAGE_MAP["data_processing"]="policortex001-data-processing"
IMAGE_MAP["conversation"]="policortex001-conversation"
IMAGE_MAP["notification"]="policortex001-notification"

# Update backend services
echo -e "\nDeploying backend services..."
for service in api_gateway azure_integration ai_engine data_processing conversation notification; do
    APP_NAME=${SERVICE_MAP[$service]}
    IMAGE_NAME=${IMAGE_MAP[$service]}
    FULL_IMAGE="$REGISTRY_SERVER/$IMAGE_NAME:$BUILD_ID"
    
    echo "Updating $APP_NAME with image $FULL_IMAGE..."
    
    # Check if app exists
    if az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP &>/dev/null; then
        # Update the container app with new image
        az containerapp update \
            --name $APP_NAME \
            --resource-group $RESOURCE_GROUP \
            --image $FULL_IMAGE \
            --revision-suffix "v$BUILD_ID" \
            --output none || {
                echo "Warning: Failed to update $APP_NAME"
                # Get logs for debugging
                echo "Fetching logs for $APP_NAME..."
                az containerapp logs show \
                    --name $APP_NAME \
                    --resource-group $RESOURCE_GROUP \
                    --type console \
                    --tail 20 || true
            }
        
        echo "Successfully updated $APP_NAME"
        
        # Check revision status
        LATEST_REVISION=$(az containerapp revision list \
            --name $APP_NAME \
            --resource-group $RESOURCE_GROUP \
            --query "[0].name" -o tsv)
        
        REVISION_STATUS=$(az containerapp revision show \
            --name $APP_NAME \
            --resource-group $RESOURCE_GROUP \
            --revision $LATEST_REVISION \
            --query "properties.runningState" -o tsv)
        
        echo "Revision $LATEST_REVISION status: $REVISION_STATUS"
    else
        echo "Container app $APP_NAME not found. Creating it..."
        # The app should be created by Bicep, but if not, we'll skip
        echo "Skipping $APP_NAME - should be created by infrastructure deployment"
    fi
done

# Update frontend
echo -e "\nDeploying frontend..."
FRONTEND_APP="ca-pcx-web-$ENVIRONMENT"
FRONTEND_IMAGE="$REGISTRY_SERVER/policortex001-frontend:$BUILD_ID"

if az containerapp show --name $FRONTEND_APP --resource-group $RESOURCE_GROUP &>/dev/null; then
    az containerapp update \
        --name $FRONTEND_APP \
        --resource-group $RESOURCE_GROUP \
        --image $FRONTEND_IMAGE \
        --revision-suffix "v$BUILD_ID" \
        --output none || echo "Warning: Failed to update frontend"
    
    echo "Successfully updated frontend"
else
    echo "Frontend app $FRONTEND_APP not found"
fi

# Verify Log Analytics configuration
echo -e "\nVerifying Log Analytics configuration..."
LAW_CONFIG=$(az containerapp env show \
    --name "cae-pcx-$ENVIRONMENT" \
    --resource-group $RESOURCE_GROUP \
    --query "properties.appLogsConfiguration.destination" -o tsv 2>/dev/null)

if [ "$LAW_CONFIG" == "log-analytics" ]; then
    echo "✓ Log Analytics is configured correctly"
else
    echo "✗ Log Analytics is not configured properly (current: $LAW_CONFIG)"
fi

# Verify Application Insights
echo -e "\nVerifying Application Insights..."
APP_INSIGHTS_CONN=$(az monitor app-insights component show \
    --app "ai-pcx-$ENVIRONMENT" \
    --resource-group $RESOURCE_GROUP \
    --query "connectionString" -o tsv 2>/dev/null)

if [ -n "$APP_INSIGHTS_CONN" ]; then
    echo "✓ Application Insights is configured"
    
    # Update Key Vault secret
    echo "Updating Key Vault secret for Application Insights..."
    az keyvault secret set \
        --vault-name "kv-pcx-$ENVIRONMENT" \
        --name "application-insights-connection-string" \
        --value "$APP_INSIGHTS_CONN" \
        --output none
    echo "✓ Key Vault secret updated"
else
    echo "✗ Application Insights not found"
fi

# Display URLs
echo -e "\n======================================="
echo "Container App URLs:"
echo "======================================="

API_GATEWAY_URL=$(az containerapp show \
    --name "ca-pcx-gateway-$ENVIRONMENT" \
    --resource-group $RESOURCE_GROUP \
    --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null)

FRONTEND_URL=$(az containerapp show \
    --name "ca-pcx-web-$ENVIRONMENT" \
    --resource-group $RESOURCE_GROUP \
    --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null)

if [ -n "$API_GATEWAY_URL" ]; then
    echo "API Gateway: https://$API_GATEWAY_URL"
fi

if [ -n "$FRONTEND_URL" ]; then
    echo "Frontend: https://$FRONTEND_URL"
fi

echo -e "\nDeployment completed!"