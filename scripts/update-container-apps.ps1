# PowerShell script to update all Container Apps with latest images
# This should be added to your Application Pipeline after build/push steps

param(
    [Parameter(Mandatory=$false)]
    [string]$Environment = "dev"
)

$ErrorActionPreference = "Stop"

# Configuration
$ResourceGroup = "rg-policortex001-app-$Environment"
$ContainerRegistry = "crpolicortex001$Environment.azurecr.io"

Write-Host "🚀 Updating Container Apps with latest images for $Environment environment" -ForegroundColor Green
Write-Host "Resource Group: $ResourceGroup"
Write-Host "Container Registry: $ContainerRegistry"

# Function to update a Container App with Key Vault environment variables
function Update-ContainerApp {
    param($AppName, $ImageName)
    
    $ServiceName = $AppName -replace "ca-", "" -replace "-$Environment", ""
    Write-Host "📦 Updating $AppName with image ${ImageName}:latest and Key Vault secrets" -ForegroundColor Yellow
    
    # Create revision suffix to force new revision
    $RevisionSuffix = "r$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
    
    if ($ServiceName -eq "frontend") {
        # Get dynamic FQDNs
        $ApiFqdn = az containerapp show --name "ca-api-gateway-$Environment" --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" -o tsv
        $FrontendFqdn = az containerapp show --name "ca-frontend-$Environment" --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" -o tsv
        
        # Frontend with Key Vault secrets
        az containerapp update `
            --name $AppName `
            --resource-group $ResourceGroup `
            --image "$ContainerRegistry/${ImageName}:latest" `
            --revision-suffix $RevisionSuffix `
            --replace-env-vars `
                "ENVIRONMENT=$Environment" `
                "SERVICE_NAME=frontend" `
                "PORT=8080" `
                "LOG_LEVEL=INFO" `
                "VITE_API_BASE_URL=https://$ApiFqdn/api" `
                "VITE_WS_URL=wss://$ApiFqdn/ws" `
                "VITE_AZURE_REDIRECT_URI=https://$FrontendFqdn" `
                "VITE_APP_VERSION=1.0.0" `
                "VITE_AZURE_CLIENT_ID=secretref:azure-client-id" `
                "VITE_AZURE_TENANT_ID=secretref:azure-tenant-id" `
            --output table
    } else {
        # Backend services with Key Vault secrets
        $ServicePort = switch ($ServiceName) {
            "api-gateway" { 8000 }
            "azure-integration" { 8001 }
            "ai-engine" { 8002 }
            "data-processing" { 8003 }
            "conversation" { 8004 }
            "notification" { 8005 }
            default { 8000 }
        }
        
        az containerapp update `
            --name $AppName `
            --resource-group $ResourceGroup `
            --image "$ContainerRegistry/${ImageName}:latest" `
            --revision-suffix $RevisionSuffix `
            --replace-env-vars `
                "ENVIRONMENT=$Environment" `
                "SERVICE_NAME=$ServiceName" `
                "SERVICE_PORT=$ServicePort" `
                "LOG_LEVEL=INFO" `
                "API_GATEWAY_URL=http://ca-api-gateway-$Environment" `
                "AZURE_INTEGRATION_URL=http://ca-azure-integration-$Environment" `
                "AI_ENGINE_URL=http://ca-ai-engine-$Environment" `
                "DATA_PROCESSING_URL=http://ca-data-processing-$Environment" `
                "CONVERSATION_URL=http://ca-conversation-$Environment" `
                "NOTIFICATION_URL=http://ca-notification-$Environment" `
                "JWT_SECRET_KEY=secretref:jwt-secret" `
                "ENCRYPTION_KEY=secretref:encryption-key" `
                "AZURE_CLIENT_ID=secretref:azure-client-id" `
                "AZURE_TENANT_ID=secretref:azure-tenant-id" `
                "AZURE_COSMOS_ENDPOINT=secretref:cosmos-endpoint" `
                "AZURE_COSMOS_KEY=secretref:cosmos-key" `
                "REDIS_CONNECTION_STRING=secretref:redis-connection-string" `
                "AZURE_STORAGE_ACCOUNT_NAME=secretref:storage-account-name" `
                "COGNITIVE_SERVICES_KEY=secretref:cognitive-services-key" `
                "COGNITIVE_SERVICES_ENDPOINT=secretref:cognitive-services-endpoint" `
                "APPLICATION_INSIGHTS_CONNECTION_STRING=secretref:application-insights-connection-string" `
            --output table
    }
    
    Write-Host "✅ $AppName updated successfully with revision $RevisionSuffix" -ForegroundColor Green
    Write-Host ""
}

# Update all Container Apps
Write-Host "🔄 Starting Container App updates..." -ForegroundColor Cyan

# Backend Services
Update-ContainerApp "ca-api-gateway-$Environment" "policortex001-api-gateway"
Update-ContainerApp "ca-azure-integration-$Environment" "policortex001-azure-integration"
Update-ContainerApp "ca-ai-engine-$Environment" "policortex001-ai-engine"
Update-ContainerApp "ca-data-processing-$Environment" "policortex001-data-processing"
Update-ContainerApp "ca-conversation-$Environment" "policortex001-conversation"
Update-ContainerApp "ca-notification-$Environment" "policortex001-notification"

# Frontend
Update-ContainerApp "ca-frontend-$Environment" "policortex001-frontend"

Write-Host "🎉 All Container Apps updated with latest images!" -ForegroundColor Green
Write-Host ""

# Verify revisions were created
Write-Host "🔍 Checking new revisions..." -ForegroundColor Cyan
$apps = @("ca-api-gateway-$Environment", "ca-frontend-$Environment")

foreach ($app in $apps) {
    Write-Host "Latest revision for ${app}:" -ForegroundColor Yellow
    az containerapp revision list `
        --name $app `
        --resource-group $ResourceGroup `
        --query "[0].{Name:name, CreatedTime:properties.createdTime, Active:properties.active}" `
        --output table
    Write-Host ""
}

Write-Host "✅ Container Apps update completed!" -ForegroundColor Green