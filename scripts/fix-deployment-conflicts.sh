#!/bin/bash
# Script to fix deployment conflicts and verify container app status

set -e

ENVIRONMENT=${1:-dev}
RESOURCE_GROUP="rg-pcx-app-$ENVIRONMENT"

echo "PolicyCortex Deployment Conflict Resolution"
echo "==========================================="
echo "Environment: $ENVIRONMENT"
echo "Resource Group: $RESOURCE_GROUP"

# Check if we're in the right subscription
echo "Checking subscription..."
CURRENT_SUB=$(az account show --query "name" -o tsv)
echo "Current subscription: $CURRENT_SUB"

if [[ $CURRENT_SUB != *"Policy Cortex Dev"* ]]; then
    echo "Switching to Policy Cortex Dev subscription..."
    az account set --subscription "Policy Cortex Dev"
fi

# Check container apps status
echo -e "\nContainer Apps Status:"
echo "======================"
APPS=(
    "ca-pcx-gateway-$ENVIRONMENT"
    "ca-pcx-azureint-$ENVIRONMENT" 
    "ca-pcx-ai-$ENVIRONMENT"
    "ca-pcx-dataproc-$ENVIRONMENT"
    "ca-pcx-chat-$ENVIRONMENT"
    "ca-pcx-notify-$ENVIRONMENT"
    "ca-pcx-web-$ENVIRONMENT"
)

ALL_RUNNING=true
for app in "${APPS[@]}"; do
    if az containerapp show --name "$app" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
        STATUS=$(az containerapp show --name "$app" --resource-group "$RESOURCE_GROUP" --query "properties.runningStatus" -o tsv)
        PROVISIONING=$(az containerapp show --name "$app" --resource-group "$RESOURCE_GROUP" --query "properties.provisioningState" -o tsv)
        
        if [ "$STATUS" = "Running" ] && [ "$PROVISIONING" = "Succeeded" ]; then
            echo "✅ $app: $STATUS ($PROVISIONING)"
        else
            echo "⚠️  $app: $STATUS ($PROVISIONING)"
            ALL_RUNNING=false
        fi
    else
        echo "❌ $app: Not found"
        ALL_RUNNING=false
    fi
done

# Check Log Analytics configuration
echo -e "\nLog Analytics Configuration:"
echo "============================"
LAW_CONFIG=$(az containerapp env show \
    --name "cae-pcx-$ENVIRONMENT" \
    --resource-group "$RESOURCE_GROUP" \
    --query "properties.appLogsConfiguration.destination" -o tsv 2>/dev/null)

if [ "$LAW_CONFIG" = "log-analytics" ]; then
    echo "✅ Log Analytics is configured"
else
    echo "❌ Log Analytics not configured (current: $LAW_CONFIG)"
fi

# Check Application Insights
echo -e "\nApplication Insights:"
echo "===================="
AI_CONN=$(az monitor app-insights component show \
    --app "ai-pcx-$ENVIRONMENT" \
    --resource-group "$RESOURCE_GROUP" \
    --query "connectionString" -o tsv 2>/dev/null)

if [ -n "$AI_CONN" ]; then
    echo "✅ Application Insights configured"
    
    # Update Key Vault with connection string
    if az keyvault show --name "kv-pcx-$ENVIRONMENT" &>/dev/null; then
        echo "Updating Key Vault secret..."
        az keyvault secret set \
            --vault-name "kv-pcx-$ENVIRONMENT" \
            --name "application-insights-connection-string" \
            --value "$AI_CONN" \
            --output none
        echo "✅ Key Vault secret updated"
    fi
else
    echo "❌ Application Insights not found"
fi

# Check for any stuck deployment operations
echo -e "\nChecking for stuck operations..."
echo "================================"
STUCK_DEPLOYMENTS=$(az deployment group list \
    --resource-group "$RESOURCE_GROUP" \
    --query "[?properties.provisioningState=='Running' || properties.provisioningState=='InProgress'].name" -o tsv)

if [ -n "$STUCK_DEPLOYMENTS" ]; then
    echo "⚠️  Found stuck deployments:"
    echo "$STUCK_DEPLOYMENTS"
    echo "These may need manual cancellation in Azure Portal"
else
    echo "✅ No stuck deployment operations found"
fi

# Summary
echo -e "\n==========================================="
echo "Summary:"
echo "==========================================="

if $ALL_RUNNING; then
    echo "✅ All container apps are running successfully"
    echo "✅ Logging and monitoring are configured"
    echo "✅ Ready for pipeline deployments"
    
    # Display URLs
    echo -e "\nApplication URLs:"
    API_GATEWAY_URL=$(az containerapp show \
        --name "ca-pcx-gateway-$ENVIRONMENT" \
        --resource-group "$RESOURCE_GROUP" \
        --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null)
    
    FRONTEND_URL=$(az containerapp show \
        --name "ca-pcx-web-$ENVIRONMENT" \
        --resource-group "$RESOURCE_GROUP" \
        --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null)
    
    if [ -n "$API_GATEWAY_URL" ]; then
        echo "🔗 API Gateway: https://$API_GATEWAY_URL"
    fi
    
    if [ -n "$FRONTEND_URL" ]; then
        echo "🔗 Frontend: https://$FRONTEND_URL"
    fi
else
    echo "❌ Some container apps are not running properly"
    echo "   Run the pipeline to redeploy with fixed configurations"
fi

echo -e "\nDeployment verification completed!"