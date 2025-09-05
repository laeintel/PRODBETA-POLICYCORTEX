#!/bin/bash

# Script to create Azure resources for PolicyCortex deployment
# This script must be run before the GitHub Actions deployment will work

# Configuration
RESOURCE_GROUP="policycortex-rg"
LOCATION="eastus"
APP_NAME="policycortex"
API_APP_NAME="policycortex-api"
APP_SERVICE_PLAN="policycortex-plan"
SKU="B1"  # Basic tier for testing, can be upgraded to P1V2 for production

echo "Creating Azure resources for PolicyCortex..."

# Check if logged in to Azure
az account show &>/dev/null
if [ $? -ne 0 ]; then
    echo "Please login to Azure first using: az login"
    exit 1
fi

# Create Resource Group
echo "Creating resource group: $RESOURCE_GROUP"
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create App Service Plan
echo "Creating App Service Plan: $APP_SERVICE_PLAN"
az appservice plan create \
    --name $APP_SERVICE_PLAN \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku $SKU \
    --is-linux

# Create Frontend Web App
echo "Creating Frontend Web App: $APP_NAME"
az webapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --plan $APP_SERVICE_PLAN \
    --runtime "NODE:20-lts"

# Create Backend API Web App
echo "Creating Backend API Web App: $API_APP_NAME"
az webapp create \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --plan $APP_SERVICE_PLAN \
    --runtime "NODE:20-lts"

# Configure deployment credentials (for GitHub Actions)
echo "Configuring deployment credentials..."

# Enable system-assigned managed identity for both apps
az webapp identity assign \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP

az webapp identity assign \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP

# Configure CORS for API
echo "Configuring CORS for API..."
az webapp cors add \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --allowed-origins "https://$APP_NAME.azurewebsites.net" "http://localhost:3000" "http://localhost:3001"

# Output deployment credentials
echo ""
echo "Resources created successfully!"
echo ""
echo "Next steps:"
echo "1. Create a service principal for GitHub Actions:"
echo "   az ad sp create-for-rbac --name 'policycortex-github' --role contributor --scopes /subscriptions/{subscription-id}/resourceGroups/$RESOURCE_GROUP --sdk-auth"
echo ""
echo "2. Add the output as a secret named AZURE_CREDENTIALS in your GitHub repository"
echo ""
echo "3. Add your Azure Subscription ID as a secret named AZURE_SUBSCRIPTION_ID"
echo ""
echo "4. The GitHub Actions workflow should now be able to deploy to these resources"
echo ""
echo "Frontend URL: https://$APP_NAME.azurewebsites.net"
echo "API URL: https://$API_APP_NAME.azurewebsites.net"