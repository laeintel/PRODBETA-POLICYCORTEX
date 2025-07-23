#!/bin/bash

# Exit on error
set -e

# Variables
ENVIRONMENT="dev"
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
RESOURCE_GROUP="rg-policycortex-app-dev"
DEPLOYMENT_NAME="policycortex-${ENVIRONMENT}-fix-$(date +%Y%m%d-%H%M%S)"

echo "Starting deployment fix for Policortex ${ENVIRONMENT} environment..."
echo "Subscription: ${SUBSCRIPTION_ID}"
echo "Resource Group: ${RESOURCE_GROUP}"
echo "Deployment Name: ${DEPLOYMENT_NAME}"

# Deploy the updated Bicep templates
echo "Deploying updated Bicep templates..."
az deployment group create \
  --name "${DEPLOYMENT_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --template-file main.bicep \
  --parameters environments/${ENVIRONMENT}/parameters.json \
  --mode Incremental \
  --verbose

echo "Deployment completed successfully!"

# List deployed resources
echo "Listing deployed resources..."
az resource list --resource-group "${RESOURCE_GROUP}" --output table