#\!/bin/bash

echo "🚀 Deploying simplified Container Apps Environment..."

# Variables
RG="rg-policortex001-app-dev"
LOCATION="eastus"
ENV_NAME="cae-policortex001-dev"
LOG_WORKSPACE="law-policortex001-dev"

# Create Log Analytics Workspace first
echo "📊 Creating Log Analytics Workspace..."
az monitor log-analytics workspace create   --resource-group "$RG"   --workspace-name "$LOG_WORKSPACE"   --location "$LOCATION"   --output none || echo "Log Analytics workspace already exists"

# Get workspace details
WORKSPACE_ID=$(az monitor log-analytics workspace show   --resource-group "$RG"   --workspace-name "$LOG_WORKSPACE"   --query customerId -o tsv)

WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys   --resource-group "$RG"   --workspace-name "$LOG_WORKSPACE"   --query primarySharedKey -o tsv)

# Create Container Apps Environment with consumption plan
echo "🌐 Creating Container Apps Environment (consumption-based)..."
az containerapp env create   --name "$ENV_NAME"   --resource-group "$RG"   --location "$LOCATION"   --logs-workspace-id "$WORKSPACE_ID"   --logs-workspace-key "$WORKSPACE_KEY"   --output none

echo "✅ Container Apps Environment created successfully\!"

# Create other essential resources
echo "🗝️ Creating Key Vault..."
az keyvault create   --name "kv-pcx001-dev02"   --resource-group "$RG"   --location "$LOCATION"   --enable-soft-delete true   --retention-days 7   --output none || echo "Key Vault already exists"

echo "📦 Creating Container Registry..."
az acr create   --name "crpolicortex001dev"   --resource-group "$RG"   --location "$LOCATION"   --sku Basic   --admin-enabled true   --output none || echo "Container Registry already exists"

echo "🆔 Creating User Identity..."
az identity create   --name "id-policortex001-dev"   --resource-group "$RG"   --location "$LOCATION"   --output none || echo "Identity already exists"

echo "🎉 Core infrastructure deployed successfully\!"
echo ""
echo "Resources created:"
echo "- Container Apps Environment: $ENV_NAME (consumption-based)"
echo "- Key Vault: kv-pcx001-dev02"
echo "- Container Registry: crpolicortex001dev"
echo "- User Identity: id-policortex001-dev"
echo ""
echo "Next steps: Run the GitHub Actions pipeline to deploy the full infrastructure"
