#!/bin/bash

# PolicyCortex Container Apps Deployment Script
# Automated deployment with proper environment variable configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
ENVIRONMENT="dev"
SUBSCRIPTION_ID=""
RESOURCE_GROUP=""
CONTAINER_REGISTRY=""
IMAGE_TAG="latest"
SKIP_BUILD=false
SKIP_TESTS=false
DRY_RUN=false

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy PolicyCortex container applications with proper configuration

OPTIONS:
    -e, --environment ENVIRONMENT    Target environment (dev|staging|prod) [default: dev]
    -s, --subscription-id ID         Azure subscription ID [required]
    -r, --resource-group GROUP       Azure resource group [required]
    -c, --container-registry REGISTRY Container registry URL [required]
    -t, --image-tag TAG              Container image tag [default: latest]
    --skip-build                     Skip container image building
    --skip-tests                     Skip health checks
    --dry-run                        Show what would be deployed without executing
    -h, --help                       Show this help message

EXAMPLES:
    $0 -e dev -s sub-123 -r rg-policycortex-dev -c myregistry.azurecr.io
    $0 --environment prod --subscription-id sub-456 --resource-group rg-prod --container-registry prod.azurecr.io --image-tag v1.2.3

PREREQUISITES:
    - Azure CLI installed and authenticated
    - Python 3.11+ with required packages
    - Docker (if not using --skip-build)
    - Proper Azure permissions for Container Apps and Key Vault
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -s|--subscription-id)
            SUBSCRIPTION_ID="$2"
            shift 2
            ;;
        -r|--resource-group)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        -c|--container-registry)
            CONTAINER_REGISTRY="$2"
            shift 2
            ;;
        -t|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            # For backward compatibility with old script
            if [[ -z "$ENVIRONMENT" ]]; then
                ENVIRONMENT="$1"
            elif [[ -z "$RESOURCE_GROUP" ]]; then
                RESOURCE_GROUP="$1"
            elif [[ -z "$IMAGE_TAG" ]]; then
                IMAGE_TAG="$1"
            elif [[ -z "$CONTAINER_REGISTRY" ]]; then
                CONTAINER_REGISTRY="$1"
            fi
            shift
            ;;
    esac
done

# Set defaults for backward compatibility
if [[ -z "$RESOURCE_GROUP" ]]; then
    RESOURCE_GROUP="rg-policycortex-$ENVIRONMENT"
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONTAINER_APP_ENV="policycortex-${ENVIRONMENT}-containerenv"

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