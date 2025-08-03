#\!/bin/bash

echo "🚀 Starting infrastructure deployment with fixed approach..."

ENVIRONMENT="dev"
LOCATION="eastus"

# Create a parameters file to avoid command line issues
echo "📝 Creating parameters file..."
cat > deployment-params.json << PARAMS
{
  "\": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "environment": { "value": "$ENVIRONMENT" },
    "location": { "value": "East US" },
    "owner": { "value": "AeoliTech" },
    "allowedIps": { "value": [] },
    "createTerraformAccessPolicy": { "value": false },
    "deployContainerApps": { "value": false },
    "deploySqlServer": { "value": false },
    "deployMLWorkspace": { "value": false },
    "deployOpenAI": { "value": false },
    "jwtSecretKey": { "value": "development-secret-key-change-in-production" }
  }
}
PARAMS

# Deploy with parameters file
echo "📦 Deploying infrastructure..."
DEPLOYMENT_NAME="policortex001-dev-$(date +%Y%m%d-%H%M%S)"

az deployment sub create   --location "$LOCATION"   --template-file infrastructure/bicep/main.bicep   --parameters @deployment-params.json   --name "$DEPLOYMENT_NAME"   --no-wait

echo "⏳ Deployment started: $DEPLOYMENT_NAME"
echo "Waiting for deployment to complete..."

# Check deployment status
sleep 10
az deployment sub show --name "$DEPLOYMENT_NAME" --query "properties.provisioningState" -o tsv

echo "🔍 Checking deployment status..."
az deployment sub wait --name "$DEPLOYMENT_NAME" --created

DEPLOYMENT_STATE=$(az deployment sub show --name "$DEPLOYMENT_NAME" --query "properties.provisioningState" -o tsv)

if [ "$DEPLOYMENT_STATE" == "Succeeded" ]; then
    echo "✅ Deployment succeeded\!"
    
    # Get outputs
    echo "📊 Deployment outputs:"
    az deployment sub show --name "$DEPLOYMENT_NAME" --query "properties.outputs" -o json
else
    echo "❌ Deployment failed or is still running. State: $DEPLOYMENT_STATE"
    
    # Get error details
    echo "📋 Error details:"
    az deployment sub show --name "$DEPLOYMENT_NAME" --query "properties.error" -o json
fi

# Cleanup
rm -f deployment-params.json
