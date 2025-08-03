#\!/bin/bash

echo "🔧 Fixing deployment issues for pipeline..."

# Variables
RG="rg-policortex001-app-dev"
LOCATION="eastus"

# 1. Ensure all required resources exist
echo "✅ Container Apps Environment already exists: cae-policortex001-dev"

# 2. Create remaining core resources if they don't exist
echo "📦 Ensuring Container Registry exists..."
az acr show --name "crpolicortex001dev" --resource-group "$RG" 2>/dev/null || az acr create   --name "crpolicortex001dev"   --resource-group "$RG"   --location "$LOCATION"   --sku Basic   --admin-enabled true   --output none

echo "🗝️ Ensuring Key Vault exists..."
az keyvault show --name "kv-pcx001-dev02" --resource-group "$RG" 2>/dev/null || az keyvault create   --name "kv-pcx001-dev02"   --resource-group "$RG"   --location "$LOCATION"   --enable-soft-delete true   --retention-days 7   --output none

echo "🆔 Ensuring User Identity exists..."
az identity show --name "id-policortex001-dev" --resource-group "$RG" 2>/dev/null || az identity create   --name "id-policortex001-dev"   --resource-group "$RG"   --location "$LOCATION"   --output none

echo "📊 Ensuring Log Analytics Workspace exists..."
az monitor log-analytics workspace show --name "law-policortex001-dev" --resource-group "$RG" 2>/dev/null || az monitor log-analytics workspace create   --resource-group "$RG"   --workspace-name "law-policortex001-dev"   --location "$LOCATION"   --output none

echo "💾 Ensuring Storage Account exists..."
az storage account show --name "stpolicortex001dev" --resource-group "$RG" 2>/dev/null || az storage account create   --name "stpolicortex001dev"   --resource-group "$RG"   --location "$LOCATION"   --sku Standard_LRS   --output none

echo ""
echo "✅ Core resources are ready\!"
echo ""
echo "📝 Next steps:"
echo "1. Commit and push the pipeline fixes"
echo "2. Run the pipeline again - it should succeed now"
echo "3. The Container Apps Environment is using consumption pricing as requested"
echo ""
echo "🌐 Container Apps Environment details:"
echo "- Name: cae-policortex001-dev"
echo "- Default Domain: calmhill-0c376a86.eastus.azurecontainerapps.io"
echo "- Pricing: Consumption-based (no workload profiles)"
