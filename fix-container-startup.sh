#!/bin/bash

# Fix Container Startup Issues Script
# This script rebuilds and redeploys containers with proper startup configuration

set -e

echo "🔧 Fixing container startup issues..."

# Configuration
ENVIRONMENT=${1:-dev}
REGISTRY_NAME="crpolicortex001${ENVIRONMENT}"
RESOURCE_GROUP="rg-policortex001-${ENVIRONMENT}"
LOCATION="eastus"

echo "📋 Configuration:"
echo "  Environment: $ENVIRONMENT"
echo "  Registry: $REGISTRY_NAME"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"

# Build and push all services with startup fixes
echo "🏗️  Building and pushing containers..."

# API Gateway
echo "📦 Building API Gateway..."
docker build -t $REGISTRY_NAME.azurecr.io/policortex001-api-gateway:latest \
  -f backend/services/api_gateway/Dockerfile \
  backend/

# Azure Integration
echo "📦 Building Azure Integration..."
docker build -t $REGISTRY_NAME.azurecr.io/policortex001-azure-integration:latest \
  -f backend/services/azure_integration/Dockerfile \
  backend/

# AI Engine
echo "📦 Building AI Engine..."
docker build -t $REGISTRY_NAME.azurecr.io/policortex001-ai-engine:latest \
  -f backend/services/ai_engine/Dockerfile \
  backend/

# Data Processing
echo "📦 Building Data Processing..."
docker build -t $REGISTRY_NAME.azurecr.io/policortex001-data-processing:latest \
  -f backend/services/data_processing/Dockerfile \
  backend/

# Conversation
echo "📦 Building Conversation..."
docker build -t $REGISTRY_NAME.azurecr.io/policortex001-conversation:latest \
  -f backend/services/conversation/Dockerfile \
  backend/

# Notification
echo "📦 Building Notification..."
docker build -t $REGISTRY_NAME.azurecr.io/policortex001-notification:latest \
  -f backend/services/notification/Dockerfile \
  backend/

# Frontend
echo "📦 Building Frontend..."
docker build -t $REGISTRY_NAME.azurecr.io/policortex001-frontend:latest \
  -f frontend/Dockerfile \
  frontend/

# Push all images
echo "🚀 Pushing images to registry..."
docker push $REGISTRY_NAME.azurecr.io/policortex001-api-gateway:latest
docker push $REGISTRY_NAME.azurecr.io/policortex001-azure-integration:latest
docker push $REGISTRY_NAME.azurecr.io/policortex001-ai-engine:latest
docker push $REGISTRY_NAME.azurecr.io/policortex001-data-processing:latest
docker push $REGISTRY_NAME.azurecr.io/policortex001-conversation:latest
docker push $REGISTRY_NAME.azurecr.io/policortex001-notification:latest
docker push $REGISTRY_NAME.azurecr.io/policortex001-frontend:latest

# Deploy infrastructure with fixes
echo "🚀 Deploying infrastructure with startup fixes..."
cd infrastructure/bicep

# Deploy with the updated configuration
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file main.bicep \
  --parameters environment=$ENVIRONMENT \
  --verbose

echo "✅ Container startup fixes deployed!"
echo ""
echo "📊 Check container status:"
echo "  az containerapp revision list --name ca-api-gateway-$ENVIRONMENT --resource-group $RESOURCE_GROUP"
echo ""
echo "📋 View logs:"
echo "  az containerapp logs show --name ca-api-gateway-$ENVIRONMENT --resource-group $RESOURCE_GROUP --follow" 