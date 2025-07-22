#!/bin/bash

# Script to clean up failed resources before redeployment
# Exit on error
set -e

RESOURCE_GROUP="rg-policycortex-app-dev"
ENVIRONMENT="dev"

echo "Starting cleanup of failed resources..."

# Delete failed Cosmos DB account
echo "Checking for failed Cosmos DB account..."
if az cosmosdb show --name "policycortex-cosmos-${ENVIRONMENT}" --resource-group "${RESOURCE_GROUP}" &>/dev/null; then
    echo "Deleting failed Cosmos DB account: policycortex-cosmos-${ENVIRONMENT}"
    az cosmosdb delete --name "policycortex-cosmos-${ENVIRONMENT}" --resource-group "${RESOURCE_GROUP}" --yes
    echo "Cosmos DB account deleted successfully"
else
    echo "Cosmos DB account not found or already deleted"
fi

# Check for other potentially failed resources
echo "Checking for other failed resources..."

# Redis Cache
if az redis show --name "policycortex-redis-${ENVIRONMENT}" --resource-group "${RESOURCE_GROUP}" &>/dev/null; then
    REDIS_STATE=$(az redis show --name "policycortex-redis-${ENVIRONMENT}" --resource-group "${RESOURCE_GROUP}" --query "provisioningState" -o tsv)
    if [ "$REDIS_STATE" == "Failed" ]; then
        echo "Deleting failed Redis cache: policycortex-redis-${ENVIRONMENT}"
        az redis delete --name "policycortex-redis-${ENVIRONMENT}" --resource-group "${RESOURCE_GROUP}" --yes
    fi
fi

# Cognitive Services
if az cognitiveservices account show --name "policycortex-cognitive-${ENVIRONMENT}" --resource-group "${RESOURCE_GROUP}" &>/dev/null; then
    COGNITIVE_STATE=$(az cognitiveservices account show --name "policycortex-cognitive-${ENVIRONMENT}" --resource-group "${RESOURCE_GROUP}" --query "properties.provisioningState" -o tsv)
    if [ "$COGNITIVE_STATE" == "Failed" ]; then
        echo "Deleting failed Cognitive Services account: policycortex-cognitive-${ENVIRONMENT}"
        az cognitiveservices account delete --name "policycortex-cognitive-${ENVIRONMENT}" --resource-group "${RESOURCE_GROUP}"
    fi
fi

echo "Cleanup completed!"
echo ""
echo "You can now run the deployment again:"
echo "cd infrastructure/bicep && ./deploy-fix.sh"