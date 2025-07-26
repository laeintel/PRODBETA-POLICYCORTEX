# PolicyCortex Deployment Guide

This guide provides step-by-step instructions for deploying PolicyCortex to Azure.

## Prerequisites

1. Azure CLI installed and logged in
2. Docker installed and running
3. PowerShell 7+ (for build scripts)
4. Appropriate Azure permissions (Contributor or Owner on subscription)

## Step 1: Infrastructure Deployment (Without Container Apps)

First, deploy the infrastructure without Container Apps to avoid image pull issues:

```bash
# Navigate to bicep directory
cd infrastructure/bicep

# Deploy infrastructure only (Container Apps disabled)
az deployment sub create \
  --location "East US" \
  --template-file main.bicep \
  --parameters environments/dev.bicepparam \
  --parameters deployContainerApps=false \
  --name "policortex001-infra-$(date +%Y%m%d-%H%M%S)"
```

This will create:
- Resource Groups
- Virtual Network
- Azure Container Registry (ACR)
- Key Vault
- Storage Account
- Data Services (Cosmos DB, Redis)
- AI Services
- Application Insights
- Container Apps Environment

## Step 2: Build and Push Container Images

After infrastructure is deployed, build and push the container images:

```powershell
# From the root directory
./build-and-push-images.ps1 -Environment dev -RegistryName crpolicortex001dev -SubscriptionId "your-subscription-id"
```

This script will:
- Login to Azure and ACR
- Build all service Docker images
- Push images to ACR with proper tags

## Step 3: Deploy Container Apps with Real Images

Once images are available in ACR, you have two options:

### Option A: Update existing deployment
Replace the current `container-apps.bicep` with the real images version and redeploy:

```bash
# Copy the real images version
cp modules/container-apps-with-real-images.bicep modules/container-apps.bicep

# Redeploy with Container Apps enabled
az deployment sub create \
  --location "East US" \
  --template-file main.bicep \
  --parameters environments/dev.bicepparam \
  --parameters deployContainerApps=true \
  --name "policortex001-apps-$(date +%Y%m%d-%H%M%S)"
```

### Option B: Manual Container Apps Deployment
Deploy Container Apps separately:

```bash
# Get resource group and environment details
RESOURCE_GROUP="rg-policortex001-app-dev"
ENVIRONMENT_ID="/subscriptions/your-sub-id/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.App/managedEnvironments/cae-policortex001-dev"

# Deploy Container Apps
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file modules/container-apps-with-real-images.bicep \
  --parameters environment=dev \
  --parameters containerAppsEnvironmentId=$ENVIRONMENT_ID \
  --parameters containerRegistryLoginServer=crpolicortex001dev.azurecr.io \
  --parameters userAssignedIdentityId="/subscriptions/your-sub-id/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id-policortex001-dev" \
  --parameters keyVaultName=kvpolicortex001dev \
  --parameters containerAppsEnvironmentDefaultDomain="your-environment-domain" \
  --name "container-apps-$(date +%Y%m%d-%H%M%S)"
```

## Step 4: Verify Deployment

Check that all services are running:

```bash
# List Container Apps
az containerapp list --resource-group rg-policortex001-app-dev --output table

# Check individual app status
az containerapp show --name ca-api-gateway-dev --resource-group rg-policortex001-app-dev --query "properties.provisioningState"

# Get app URLs
az containerapp show --name ca-frontend-dev --resource-group rg-policortex001-app-dev --query "properties.configuration.ingress.fqdn"
```

## Step 5: Access the Application

1. **Frontend**: Access via the frontend Container App URL
2. **API Gateway**: Access via the API gateway Container App URL
3. **Health Checks**: Test `/health` endpoints on each service

## Troubleshooting

### Common Issues

1. **ACR Authentication Errors**
   - Ensure the managed identity has `AcrPull` role on ACR
   - Verify the identity is correctly assigned to Container Apps

2. **Key Vault Access Issues**
   - Check managed identity has proper access policies on Key Vault
   - Verify secret names match what's expected in the app

3. **Container Apps Won't Start**
   - Check container logs: `az containerapp logs show --name ca-api-gateway-dev --resource-group rg-policortex001-app-dev`
   - Verify environment variables are correctly set

4. **Images Not Found**
   - Ensure images were built and pushed successfully
   - Check ACR repository list: `az acr repository list --name crpolicortex001dev`

### Useful Commands

```bash
# View Container App logs
az containerapp logs show --name ca-api-gateway-dev --resource-group rg-policortex001-app-dev --follow

# Scale Container App
az containerapp update --name ca-api-gateway-dev --resource-group rg-policortex001-app-dev --min-replicas 1 --max-replicas 5

# Update Container App image
az containerapp update --name ca-api-gateway-dev --resource-group rg-policortex001-app-dev --image crpolicortex001dev.azurecr.io/policortex001-api-gateway:latest

# Restart Container App
az containerapp revision restart --name ca-api-gateway-dev --resource-group rg-policortex001-app-dev
```

## Environment Variables

The Container Apps are configured with the following key environment variables:

- `ENVIRONMENT`: dev/staging/prod
- `SERVICE_NAME`: Name of the service
- `SERVICE_PORT`: Port the service runs on
- `JWT_SECRET_KEY`: From Key Vault
- `AZURE_CLIENT_ID`: Managed identity client ID
- `AZURE_COSMOS_ENDPOINT`: Cosmos DB endpoint
- `REDIS_CONNECTION_STRING`: Redis connection string

All secrets are sourced from Azure Key Vault using the managed identity.

## Next Steps

1. Set up monitoring and alerting
2. Configure custom domains and SSL certificates
3. Set up CI/CD pipelines for automated deployments
4. Configure backup and disaster recovery
5. Implement auto-scaling policies