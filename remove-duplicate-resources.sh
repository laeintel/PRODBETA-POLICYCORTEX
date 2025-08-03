#\!/bin/bash

echo "🧹 Removing duplicate and unnecessary resources..."

RG="rg-policortex001-app-dev"

# DUPLICATES TO REMOVE:

# 1. Duplicate Key Vault (keep kv-pcx001-dev02, remove kv-pcx001-dev)
echo "❌ Removing duplicate Key Vault: kv-pcx001-dev"
az keyvault delete --name "kv-pcx001-dev" --resource-group "$RG" || echo "Already deleted"

# 2. Duplicate Redis (keep pcx001-redis-dev-wd42, remove policortex001-redis-dev)
echo "❌ Removing duplicate Redis: policortex001-redis-dev"
az redis delete --name "policortex001-redis-dev" --resource-group "$RG" --yes || echo "Already deleted"

# 3. Remove old private endpoints for the duplicate Redis
echo "❌ Removing old Redis private endpoint"
az network private-endpoint delete --name "policortex001-redis-pe-dev" --resource-group "$RG" --yes || echo "Already deleted"

# RESOURCES NOT IN BICEP CONFIG (should be removed):

# 4. Event Grid Topic (not in Bicep)
echo "❌ Removing Event Grid Topic: pc001-ml-events-inc-dev"
az eventgrid topic delete --name "pc001-ml-events-inc-dev" --resource-group "$RG" --yes || echo "Already deleted"

# 5. Remove associated private endpoints for Event Grid
echo "❌ Removing Event Grid private endpoint"
az network private-endpoint delete --name "pc001-eg-pe-inc-dev" --resource-group "$RG" --yes || echo "Already deleted"

# 6. Application Insights Smart Detection action group (auto-created, not needed)
echo "❌ Removing auto-created action group"
az monitor action-group delete --name "Application Insights Smart Detection" --resource-group "$RG" --yes || echo "Already deleted"

echo ""
echo "✅ Cleanup complete\!"
echo ""
echo "📋 Resources to KEEP (as per Bicep config):"
echo "- kv-pcx001-dev02 (Key Vault)"
echo "- pcx001-redis-dev-wd42 (Redis Cache with unique suffix)"
echo "- cae-policortex001-dev (Container Apps Environment)"
echo "- crpolicortex001dev (Container Registry)"
echo "- stpolicortex001dev (Storage Account)"
echo "- law-policortex001-dev (Log Analytics)"
echo "- ai-policortex001-dev (Application Insights)"
echo "- id-policortex001-dev (Managed Identity)"
echo "- policortex001-cosmos-dev (Cosmos DB)"
echo "- policortex001-cog-inc-dev (Cognitive Services)"
echo "- Action groups for alerts"
echo ""
echo "📝 Note: Private endpoints and NICs will be recreated by Bicep if needed"
