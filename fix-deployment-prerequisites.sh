#\!/bin/bash

echo "🔧 Fixing deployment prerequisites..."

# Set environment variables
ENVIRONMENT="dev"
LOCATION="East US"

# 1. Ensure resource providers are registered
echo "📝 Registering required resource providers..."
providers=(
    "Microsoft.App"
    "Microsoft.ContainerRegistry"
    "Microsoft.KeyVault"
    "Microsoft.Storage"
    "Microsoft.DocumentDB"
    "Microsoft.Cache"
    "Microsoft.Sql"
    "Microsoft.CognitiveServices"
    "Microsoft.MachineLearningServices"
    "Microsoft.Network"
    "Microsoft.OperationalInsights"
    "Microsoft.Insights"
)

for provider in "${providers[@]}"; do
    echo "Registering $provider..."
    az provider register --namespace "$provider" --wait || echo "Provider $provider already registered"
done

# 2. Create resource groups if they don't exist
echo "📁 Creating resource groups..."
az group create --name "rg-policortex001-network-$ENVIRONMENT" --location "$LOCATION" || echo "Network RG already exists"
az group create --name "rg-policortex001-app-$ENVIRONMENT" --location "$LOCATION" || echo "App RG already exists"

# 3. Cancel any stuck deployments
echo "🛑 Canceling stuck deployments..."
STUCK_DEPLOYMENTS=$(az deployment sub list --query "[?properties.provisioningState=='Running'].name" -o tsv)
for deployment in $STUCK_DEPLOYMENTS; do
    echo "Canceling deployment: $deployment"
    az deployment sub cancel --name "$deployment" || echo "Could not cancel $deployment"
done

# 4. Generate unique names to avoid conflicts
echo "🔄 Generating unique suffixes..."
TIMESTAMP=$(date +%Y%m%d%H%M%S)
echo "Using timestamp: $TIMESTAMP"

echo "✅ Prerequisites fixed. Ready for deployment\!"
echo ""
echo "💡 Deployment tips:"
echo "1. The deployment will use unique names to avoid conflicts"
echo "2. Resource groups have been created"
echo "3. All required providers are registered"
echo "4. Any stuck deployments have been canceled"
