#!/bin/bash

# Deploy Container Apps Script
# This script builds and deploys all PolicyCortex services to Azure Container Apps

set -e

# Variables
RESOURCE_GROUP="rg-pcx-app-dev"
REGISTRY="crpcxdev"
SUBSCRIPTION_ID="205b477d-17e7-4b3b-92c1-32cf02626b78"

echo "🚀 Starting PolicyCortex deployment..."

# Login to Azure Container Registry
echo "📦 Logging in to Azure Container Registry..."
az acr login --name $REGISTRY

# Build and push images
echo "🏗️ Building Docker images..."

# API Gateway
echo "Building API Gateway..."
docker build -t $REGISTRY.azurecr.io/pcx-api-gateway:latest -f backend/services/api_gateway/Dockerfile backend/services/api_gateway/
docker push $REGISTRY.azurecr.io/pcx-api-gateway:latest

# Azure Integration
echo "Building Azure Integration..."
docker build -t $REGISTRY.azurecr.io/pcx-azure-integration:latest -f backend/services/azure_integration/Dockerfile backend/
docker push $REGISTRY.azurecr.io/pcx-azure-integration:latest

# AI Engine
echo "Building AI Engine..."
docker build -t $REGISTRY.azurecr.io/pcx-ai-engine:latest -f backend/services/ai_engine/Dockerfile backend/
docker push $REGISTRY.azurecr.io/pcx-ai-engine:latest

# Data Processing
echo "Building Data Processing..."
docker build -t $REGISTRY.azurecr.io/pcx-data-processing:latest -f backend/services/data_processing/Dockerfile backend/
docker push $REGISTRY.azurecr.io/pcx-data-processing:latest

# Conversation
echo "Building Conversation..."
docker build -t $REGISTRY.azurecr.io/pcx-conversation:latest -f backend/services/conversation/Dockerfile backend/
docker push $REGISTRY.azurecr.io/pcx-conversation:latest

# Notification
echo "Building Notification..."
docker build -t $REGISTRY.azurecr.io/pcx-notification:latest -f backend/services/notification/Dockerfile backend/
docker push $REGISTRY.azurecr.io/pcx-notification:latest

# Frontend
echo "Building Frontend..."
docker build -t $REGISTRY.azurecr.io/pcx-frontend:latest -f frontend/Dockerfile frontend/
docker push $REGISTRY.azurecr.io/pcx-frontend:latest

# Update Container Apps
echo "🔄 Updating Container Apps..."

# Update API Gateway
echo "Updating API Gateway..."
az containerapp update \
  --name ca-pcx-gateway-dev \
  --resource-group $RESOURCE_GROUP \
  --image $REGISTRY.azurecr.io/pcx-api-gateway:latest \
  --set-env-vars "AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID" \
  --output none

# Update Azure Integration
echo "Updating Azure Integration..."
az containerapp update \
  --name ca-pcx-azureint-dev \
  --resource-group $RESOURCE_GROUP \
  --image $REGISTRY.azurecr.io/pcx-azure-integration:latest \
  --set-env-vars "AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID" \
  --output none

# Update AI Engine
echo "Updating AI Engine..."
az containerapp update \
  --name ca-pcx-ai-dev \
  --resource-group $RESOURCE_GROUP \
  --image $REGISTRY.azurecr.io/pcx-ai-engine:latest \
  --output none

# Update Data Processing
echo "Updating Data Processing..."
az containerapp update \
  --name ca-pcx-dataproc-dev \
  --resource-group $RESOURCE_GROUP \
  --image $REGISTRY.azurecr.io/pcx-data-processing:latest \
  --output none

# Update Conversation
echo "Updating Conversation..."
az containerapp update \
  --name ca-pcx-chat-dev \
  --resource-group $RESOURCE_GROUP \
  --image $REGISTRY.azurecr.io/pcx-conversation:latest \
  --output none

# Update Notification
echo "Updating Notification..."
az containerapp update \
  --name ca-pcx-notify-dev \
  --resource-group $RESOURCE_GROUP \
  --image $REGISTRY.azurecr.io/pcx-notification:latest \
  --output none

# Update Frontend
echo "Updating Frontend..."
az containerapp update \
  --name ca-pcx-web-dev \
  --resource-group $RESOURCE_GROUP \
  --image $REGISTRY.azurecr.io/pcx-frontend:latest \
  --output none

echo "✅ Deployment completed successfully!"

# Check health status
echo "🏥 Checking service health..."
sleep 30

# Check revisions
echo "📊 Container App Revisions:"
az containerapp revision list --name ca-pcx-gateway-dev --resource-group $RESOURCE_GROUP --query "[0].{name:name,active:properties.active,replicas:properties.replicas,status:properties.runningState}" --output table
az containerapp revision list --name ca-pcx-azureint-dev --resource-group $RESOURCE_GROUP --query "[0].{name:name,active:properties.active,replicas:properties.replicas,status:properties.runningState}" --output table

echo "🌐 Service URLs:"
echo "Frontend: https://ca-pcx-web-dev.lemonfield-7e1ea681.eastus.azurecontainerapps.io"
echo "API Gateway: https://ca-pcx-gateway-dev.lemonfield-7e1ea681.eastus.azurecontainerapps.io"