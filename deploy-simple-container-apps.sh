#!/bin/bash

# Simple Container Apps deployment script
set -e

echo "🚀 Deploying simple Container Apps infrastructure..."

# Variables
RESOURCE_GROUP="rg-policycortex001-app-dev"
LOCATION="East US"
ACR_NAME="crpolicortex001dev"
ENVIRONMENT_NAME="cae-policortex001-dev"

# Check if resource group exists, create if not
echo "📦 Checking resource group..."
if ! az group show --name $RESOURCE_GROUP > /dev/null 2>&1; then
    echo "Creating resource group $RESOURCE_GROUP..."
    az group create --name $RESOURCE_GROUP --location "$LOCATION"
fi

# Check if Container Registry exists
echo "📋 Checking Container Registry..."
if ! az acr show --name $ACR_NAME > /dev/null 2>&1; then
    echo "Creating Container Registry $ACR_NAME..."
    az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled
fi

# Check if Container Apps Environment exists
echo "🌐 Checking Container Apps Environment..."
if ! az containerapp env show --name $ENVIRONMENT_NAME --resource-group $RESOURCE_GROUP > /dev/null 2>&1; then
    echo "Creating Container Apps Environment $ENVIRONMENT_NAME..."
    az containerapp env create \
        --name $ENVIRONMENT_NAME \
        --resource-group $RESOURCE_GROUP \
        --location "$LOCATION"
fi

# Create API Gateway Container App
echo "🔧 Creating API Gateway Container App..."
az containerapp create \
    --name ca-api-gateway-dev \
    --resource-group $RESOURCE_GROUP \
    --environment $ENVIRONMENT_NAME \
    --image mcr.microsoft.com/azuredocs/containerapps-helloworld:latest \
    --target-port 80 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 3 \
    --cpu 1.0 \
    --memory 2Gi \
    --env-vars \
        ENVIRONMENT=development \
        SERVICE_NAME=api_gateway \
        SERVICE_PORT=8000 \
        LOG_LEVEL=INFO

echo "✅ Simple Container Apps deployment completed!"
echo "🌍 API Gateway URL: https://$(az containerapp show --name ca-api-gateway-dev --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)"