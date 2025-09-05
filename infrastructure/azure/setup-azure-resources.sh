#!/bin/bash

# Azure Resource Setup Script for PolicyCortex
# This script provisions all required Azure resources for the application

set -e

# Configuration
RESOURCE_GROUP="policycortex-rg"
LOCATION="eastus"
APP_NAME="policycortex"
ENVIRONMENT="${1:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Setting up Azure resources for PolicyCortex${NC}"
echo -e "${YELLOW}Environment: ${ENVIRONMENT}${NC}"

# Check if logged in to Azure
echo "Checking Azure login status..."
if ! az account show &>/dev/null; then
    echo -e "${RED}Not logged in to Azure. Please run 'az login' first.${NC}"
    exit 1
fi

SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo -e "${GREEN}✓ Using subscription: ${SUBSCRIPTION_ID}${NC}"

# Create Resource Group
echo "Creating resource group..."
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION \
    --tags Environment=$ENVIRONMENT Application=PolicyCortex

# Create App Service Plan
echo "Creating App Service Plan..."
if [ "$ENVIRONMENT" == "production" ]; then
    SKU="P2v3"
else
    SKU="B2"
fi

az appservice plan create \
    --name ${APP_NAME}-plan \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku $SKU \
    --is-linux

# Create Frontend Web App
echo "Creating Frontend Web App..."
az webapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --plan ${APP_NAME}-plan \
    --runtime "NODE:20-lts"

# Create Backend API Web App
echo "Creating Backend API Web App..."
az webapp create \
    --name ${APP_NAME}-api \
    --resource-group $RESOURCE_GROUP \
    --plan ${APP_NAME}-plan \
    --runtime "NODE:20-lts"

# Configure Frontend App Settings
echo "Configuring Frontend App Settings..."
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        NEXT_PUBLIC_DEMO_MODE=false \
        USE_REAL_DATA=true \
        NEXT_PUBLIC_API_URL=https://${APP_NAME}.azurewebsites.net \
        NEXT_PUBLIC_REAL_API_BASE=https://${APP_NAME}-api.azurewebsites.net \
        NODE_ENV=production \
        WEBSITE_NODE_DEFAULT_VERSION=20-lts

# Configure Backend API App Settings
echo "Configuring Backend API App Settings..."
az webapp config appsettings set \
    --name ${APP_NAME}-api \
    --resource-group $RESOURCE_GROUP \
    --settings \
        PORT=8080 \
        AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID \
        NODE_ENV=production \
        WEBSITE_NODE_DEFAULT_VERSION=20-lts

# Create Storage Account for static assets
echo "Creating Storage Account..."
STORAGE_NAME="${APP_NAME}storage${RANDOM}"
az storage account create \
    --name $STORAGE_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard_LRS \
    --kind StorageV2

# Create Container Registry (for Docker images)
echo "Creating Container Registry..."
ACR_NAME="${APP_NAME}acr"
az acr create \
    --name $ACR_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Basic \
    --admin-enabled true

# Create PostgreSQL Database (Production only)
if [ "$ENVIRONMENT" == "production" ]; then
    echo "Creating PostgreSQL Database..."
    DB_NAME="${APP_NAME}-db"
    DB_ADMIN="pcadmin"
    DB_PASSWORD=$(openssl rand -base64 32)
    
    az postgres flexible-server create \
        --name $DB_NAME \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --admin-user $DB_ADMIN \
        --admin-password $DB_PASSWORD \
        --sku-name Standard_B2s \
        --tier Burstable \
        --version 15 \
        --storage-size 32 \
        --public-access 0.0.0.0
    
    # Save database connection string
    echo -e "${YELLOW}Database Password: ${DB_PASSWORD}${NC}"
    echo -e "${YELLOW}Save this password securely!${NC}"
    
    # Configure database firewall
    az postgres flexible-server firewall-rule create \
        --name $DB_NAME \
        --resource-group $RESOURCE_GROUP \
        --rule-name AllowAllAzureServices \
        --start-ip-address 0.0.0.0 \
        --end-ip-address 0.0.0.0
fi

# Create Redis Cache (for session management)
echo "Creating Redis Cache..."
az redis create \
    --name ${APP_NAME}-cache \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Basic \
    --vm-size c0

# Create Application Insights
echo "Creating Application Insights..."
az monitor app-insights component create \
    --app ${APP_NAME}-insights \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --kind web

INSTRUMENTATION_KEY=$(az monitor app-insights component show \
    --app ${APP_NAME}-insights \
    --resource-group $RESOURCE_GROUP \
    --query instrumentationKey -o tsv)

# Update app settings with Application Insights
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings APPINSIGHTS_INSTRUMENTATIONKEY=$INSTRUMENTATION_KEY

az webapp config appsettings set \
    --name ${APP_NAME}-api \
    --resource-group $RESOURCE_GROUP \
    --settings APPINSIGHTS_INSTRUMENTATIONKEY=$INSTRUMENTATION_KEY

# Create Key Vault for secrets
echo "Creating Key Vault..."
KEYVAULT_NAME="${APP_NAME}-kv-${RANDOM}"
az keyvault create \
    --name $KEYVAULT_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

# Enable managed identity for Web Apps
echo "Enabling Managed Identity..."
az webapp identity assign \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP

az webapp identity assign \
    --name ${APP_NAME}-api \
    --resource-group $RESOURCE_GROUP

# Get identity principals
FRONTEND_IDENTITY=$(az webapp identity show \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query principalId -o tsv)

API_IDENTITY=$(az webapp identity show \
    --name ${APP_NAME}-api \
    --resource-group $RESOURCE_GROUP \
    --query principalId -o tsv)

# Grant Key Vault access
echo "Granting Key Vault access..."
az keyvault set-policy \
    --name $KEYVAULT_NAME \
    --object-id $FRONTEND_IDENTITY \
    --secret-permissions get list

az keyvault set-policy \
    --name $KEYVAULT_NAME \
    --object-id $API_IDENTITY \
    --secret-permissions get list

# Create CDN Profile and Endpoint
echo "Creating CDN Profile..."
az cdn profile create \
    --name ${APP_NAME}-cdn \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard_Microsoft

az cdn endpoint create \
    --name ${APP_NAME}-endpoint \
    --profile-name ${APP_NAME}-cdn \
    --resource-group $RESOURCE_GROUP \
    --origin ${APP_NAME}.azurewebsites.net \
    --origin-host-header ${APP_NAME}.azurewebsites.net

# Configure CORS for API
echo "Configuring CORS..."
az webapp cors add \
    --name ${APP_NAME}-api \
    --resource-group $RESOURCE_GROUP \
    --allowed-origins "https://${APP_NAME}.azurewebsites.net" "http://localhost:3000"

# Enable Always On
echo "Enabling Always On..."
az webapp config set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --always-on true

az webapp config set \
    --name ${APP_NAME}-api \
    --resource-group $RESOURCE_GROUP \
    --always-on true

# Configure deployment slots (staging)
if [ "$ENVIRONMENT" == "production" ]; then
    echo "Creating deployment slots..."
    az webapp deployment slot create \
        --name $APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --slot staging
    
    az webapp deployment slot create \
        --name ${APP_NAME}-api \
        --resource-group $RESOURCE_GROUP \
        --slot staging
fi

# Output summary
echo -e "\n${GREEN}✅ Azure resources created successfully!${NC}\n"
echo "Resource Summary:"
echo "=================="
echo -e "Resource Group: ${YELLOW}${RESOURCE_GROUP}${NC}"
echo -e "Frontend URL: ${YELLOW}https://${APP_NAME}.azurewebsites.net${NC}"
echo -e "API URL: ${YELLOW}https://${APP_NAME}-api.azurewebsites.net${NC}"
echo -e "Container Registry: ${YELLOW}${ACR_NAME}.azurecr.io${NC}"
echo -e "Key Vault: ${YELLOW}${KEYVAULT_NAME}${NC}"
echo -e "CDN Endpoint: ${YELLOW}https://${APP_NAME}-endpoint.azureedge.net${NC}"

if [ "$ENVIRONMENT" == "production" ]; then
    echo -e "Database Server: ${YELLOW}${DB_NAME}.postgres.database.azure.com${NC}"
    echo -e "Database Admin: ${YELLOW}${DB_ADMIN}${NC}"
fi

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "1. Configure GitHub Actions secrets:"
echo "   - AZURE_CREDENTIALS (service principal)"
echo "   - AZURE_SUBSCRIPTION_ID"
echo "   - DB_PASSWORD (if production)"
echo "2. Run database migrations"
echo "3. Deploy application code"
echo "4. Configure custom domain (optional)"

echo -e "\n${GREEN}Setup complete! 🎉${NC}"