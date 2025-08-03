#\!/bin/bash

echo "🚀 Deploying infrastructure with minimal configuration..."

# Deploy with minimal parameters to isolate the issue
az deployment sub create   --location "eastus"   --template-file infrastructure/bicep/main.bicep   --parameters     environment=dev     location="East US"     owner=AeoliTech     allowedIps=[]     createTerraformAccessPolicy=false     deployContainerApps=false     deploySqlServer=false     deployMLWorkspace=false     deployOpenAI=false     jwtSecretKey="dev-secret-key-123"   --name "pcx-dev-$(date +%Y%m%d%H%M%S)"   --only-show-errors

echo "✅ Deployment command executed"
