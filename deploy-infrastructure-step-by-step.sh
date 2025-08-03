#\!/bin/bash

echo "🚀 Starting step-by-step infrastructure deployment..."

ENVIRONMENT="dev"
LOCATION="eastus"

# Test with a minimal deployment first
echo "📦 Step 1: Deploying core infrastructure (storage, identity, key vault)..."

az deployment sub create   --location "$LOCATION"   --template-file infrastructure/bicep/main.bicep   --parameters environment=$ENVIRONMENT   --parameters location="East US"   --parameters owner="AeoliTech"   --parameters allowedIps=[]   --parameters createTerraformAccessPolicy=false   --parameters deployContainerApps=false   --parameters deploySqlServer=false   --parameters deployMLWorkspace=false   --parameters deployOpenAI=false   --parameters jwtSecretKey="development-secret-key-change-in-production"   --name "policortex001-core-$(date +%Y%m%d-%H%M%S)"   --output json

if [ $? -eq 0 ]; then
    echo "✅ Core infrastructure deployed successfully\!"
else
    echo "❌ Core infrastructure deployment failed. Check the error above."
    exit 1
fi

echo "🎉 Deployment completed successfully\!"
