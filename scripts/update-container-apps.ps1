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

# Function to update a Container App
function Update-ContainerApp {
    param($AppName, $ImageName)
    
    Write-Host "📦 Updating $AppName with image ${ImageName}:latest" -ForegroundColor Yellow
    
    az containerapp update `
        --name $AppName `
        --resource-group $ResourceGroup `
        --image "$ContainerRegistry/${ImageName}:latest" `
        --output table
    
    Write-Host "✅ $AppName updated successfully" -ForegroundColor Green
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