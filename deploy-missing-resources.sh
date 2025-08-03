#!/bin/bash

echo "🚀 Deploying missing resources for PolicyCortex..."

RG="rg-policortex001-app-dev"
LOCATION="eastus"

# 1. Create Storage Account (with unique name)
echo "📦 Creating Storage Account..."
STORAGE_NAME="stpcx001dev$(date +%s | tail -c 5)"
az storage account create \
  --name "$STORAGE_NAME" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --output none || echo "Storage account creation failed"

# 2. Create Container Registry (with unique name)
echo "🐳 Creating Container Registry..."
ACR_NAME="crpcx001dev$(date +%s | tail -c 5)"
az acr create \
  --name "$ACR_NAME" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --sku Basic \
  --admin-enabled true \
  --output none || echo "ACR creation failed"

# 3. Create Key Vault (already exists as kv-pcx001-dev02)
echo "🔑 Key Vault already exists: kv-pcx001-dev02"

# 4. Create Application Insights
echo "📊 Creating Application Insights..."
az monitor app-insights component create \
  --app "ai-policortex001-dev" \
  --location "$LOCATION" \
  --resource-group "$RG" \
  --workspace "law-policortex001-dev" \
  --output none || echo "App Insights already exists"

# 5. Create Cosmos DB (with simpler config)
echo "🌐 Creating Cosmos DB..."
az cosmosdb create \
  --name "policortex001-cosmos-dev" \
  --resource-group "$RG" \
  --locations regionName="$LOCATION" failoverPriority=0 \
  --default-consistency-level "Session" \
  --output none || echo "Cosmos DB creation failed"

# 6. Create Redis Cache (with unique name)
echo "💾 Creating Redis Cache..."
REDIS_NAME="pcx001redis$(date +%s | tail -c 6)"
az redis create \
  --name "$REDIS_NAME" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --sku "Basic" \
  --vm-size "C0" \
  --output none || echo "Redis creation failed"

echo "✅ Resource deployment complete!"
echo ""
echo "Created resources:"
echo "- Storage: $STORAGE_NAME"
echo "- ACR: $ACR_NAME"
echo "- App Insights: ai-policortex001-dev"
echo "- Cosmos DB: policortex001-cosmos-dev"
echo "- Redis: $REDIS_NAME"