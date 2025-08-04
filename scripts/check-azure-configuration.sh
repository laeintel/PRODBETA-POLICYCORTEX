#!/bin/bash

# Check Azure Configuration Script
# This script verifies the configuration of PolicyCortex services in Azure

set -e

# Variables
RESOURCE_GROUP="rg-pcx-app-dev"
MANAGED_IDENTITY_CLIENT_ID="a353fcad-3e64-43b5-9bb1-9f898ac22643"
SUBSCRIPTION_ID="205b477d-17e7-4b3b-92c1-32cf02626b78"

echo "🔍 Checking PolicyCortex Azure Configuration..."

# Check Container Apps
echo -e "\n📦 Container Apps Status:"
az containerapp list --resource-group $RESOURCE_GROUP --output table

# Check Managed Identity Permissions
echo -e "\n🔐 Managed Identity Permissions:"
az role assignment list --assignee $MANAGED_IDENTITY_CLIENT_ID --all --output table

# Check Key Vault Secrets
echo -e "\n🔑 Key Vault Secrets:"
az keyvault secret list --vault-name kv-pcx-dev --query "[].{name:name,enabled:attributes.enabled}" --output table

# Check Container App Environment Variables
echo -e "\n🌍 Environment Variables Configuration:"

# API Gateway
echo -e "\n--- API Gateway (ca-pcx-gateway-dev) ---"
az containerapp show --name ca-pcx-gateway-dev --resource-group $RESOURCE_GROUP --query "properties.template.containers[0].env[?name=='AZURE_SUBSCRIPTION_ID' || name=='AZURE_CLIENT_ID' || name=='AZURE_INTEGRATION_URL'].{name:name,value:value,secretRef:secretRef}" --output table

# Azure Integration
echo -e "\n--- Azure Integration (ca-pcx-azureint-dev) ---"
az containerapp show --name ca-pcx-azureint-dev --resource-group $RESOURCE_GROUP --query "properties.template.containers[0].env[?name=='AZURE_SUBSCRIPTION_ID' || name=='AZURE_CLIENT_ID' || name=='SERVICE_NAME'].{name:name,value:value,secretRef:secretRef}" --output table

# Check Container App Revisions
echo -e "\n📊 Container App Revisions:"
for app in ca-pcx-gateway-dev ca-pcx-azureint-dev ca-pcx-ai-dev ca-pcx-dataproc-dev ca-pcx-chat-dev ca-pcx-notify-dev ca-pcx-web-dev; do
    echo -e "\n$app:"
    az containerapp revision list --name $app --resource-group $RESOURCE_GROUP --query "[0].{name:name,active:properties.active,replicas:properties.replicas,status:properties.runningState,created:properties.createdTime}" --output table
done

# Check if services are healthy
echo -e "\n🏥 Service Health Checks:"

# API Gateway Health
echo -n "API Gateway: "
if curl -s -o /dev/null -w "%{http_code}" https://ca-pcx-gateway-dev.lemonfield-7e1ea681.eastus.azurecontainerapps.io/health | grep -q "200"; then
    echo "✅ Healthy"
else
    echo "❌ Unhealthy"
fi

# Frontend Health
echo -n "Frontend: "
if curl -s -o /dev/null -w "%{http_code}" https://ca-pcx-web-dev.lemonfield-7e1ea681.eastus.azurecontainerapps.io/health.html | grep -q "200"; then
    echo "✅ Healthy"
else
    echo "❌ Unhealthy"
fi

# Check for required permissions
echo -e "\n🔒 Required Azure Permissions for Live Data:"
echo "The managed identity needs the following roles:"
echo "- Reader (on subscription) - for reading Azure resources"
echo "- Policy Insights Data Writer - for reading policy compliance data"
echo "- Security Reader - for reading security recommendations"
echo "- Cost Management Reader - for reading cost data"

# Current permissions
echo -e "\nCurrent permissions for managed identity $MANAGED_IDENTITY_CLIENT_ID:"
az role assignment list --assignee $MANAGED_IDENTITY_CLIENT_ID --all --query "[].{role:roleDefinitionName,scope:scope}" --output table

# Recommendations
echo -e "\n💡 Recommendations:"
echo "1. Grant Reader role to managed identity: az role assignment create --assignee $MANAGED_IDENTITY_CLIENT_ID --role \"Reader\" --scope \"/subscriptions/$SUBSCRIPTION_ID\""
echo "2. Check container logs if services are unhealthy: az containerapp logs show --name <app-name> --resource-group $RESOURCE_GROUP --tail 50"
echo "3. Ensure all services have AZURE_SUBSCRIPTION_ID environment variable set"