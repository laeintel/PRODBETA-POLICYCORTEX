#!/usr/bin/env pwsh
<#
.SYNOPSIS
Build and push Docker images to Azure Container Registry for PolicyCortex

.DESCRIPTION
This script builds all Docker images for PolicyCortex services and pushes them to ACR.
It should be run before deploying Container Apps.

.PARAMETER Environment
The environment (dev, staging, prod)

.PARAMETER RegistryName
The name of the Azure Container Registry

.PARAMETER SubscriptionId
The Azure subscription ID

.EXAMPLE
./build-and-push-images.ps1 -Environment dev -RegistryName crpolicortex001dev -SubscriptionId "your-subscription-id"
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("dev", "staging", "prod")]
    [string]$Environment,
    
    [Parameter(Mandatory=$true)]
    [string]$RegistryName,
    
    [Parameter(Mandatory=$true)]
    [string]$SubscriptionId
)

# Set error action
$ErrorActionPreference = "Stop"

Write-Host "🚀 Building and pushing PolicyCortex images to ACR" -ForegroundColor Green
Write-Host "Environment: $Environment" -ForegroundColor Yellow
Write-Host "Registry: $RegistryName" -ForegroundColor Yellow

# Login to Azure and ACR
Write-Host "🔐 Logging into Azure and ACR..." -ForegroundColor Blue
az account set --subscription $SubscriptionId
az acr login --name $RegistryName

# Get ACR login server
$loginServer = az acr show --name $RegistryName --query loginServer --output tsv

# Define services to build
$services = @(
    @{
        Name = "api-gateway"
        Path = "./backend"
        Dockerfile = "services/api_gateway/Dockerfile"
        ImageName = "policortex001-api-gateway"
    },
    @{
        Name = "azure-integration"
        Path = "./backend"
        Dockerfile = "services/azure_integration/Dockerfile"
        ImageName = "policortex001-azure-integration"
    },
    @{
        Name = "ai-engine"
        Path = "./backend"
        Dockerfile = "services/ai_engine/Dockerfile"
        ImageName = "policortex001-ai-engine"
    },
    @{
        Name = "data-processing"
        Path = "./backend"
        Dockerfile = "services/data_processing/Dockerfile"
        ImageName = "policortex001-data-processing"
    },
    @{
        Name = "conversation"
        Path = "./backend"
        Dockerfile = "services/conversation/Dockerfile"
        ImageName = "policortex001-conversation"
    },
    @{
        Name = "notification"
        Path = "./backend"
        Dockerfile = "services/notification/Dockerfile"
        ImageName = "policortex001-notification"
    },
    @{
        Name = "frontend"
        Path = "./frontend"
        Dockerfile = "Dockerfile"
        ImageName = "policortex001-frontend"
    }
)

# Build and push each service
foreach ($service in $services) {
    Write-Host "🏗️ Building $($service.Name)..." -ForegroundColor Cyan
    
    $imageName = "$loginServer/$($service.ImageName):latest"
    $taggedImageName = "$loginServer/$($service.ImageName):$Environment-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    
    try {
        # Build the image
        docker build -t $imageName -t $taggedImageName -f "$($service.Path)/$($service.Dockerfile)" $service.Path
        
        if ($LASTEXITCODE -ne 0) {
            throw "Docker build failed for $($service.Name)"
        }
        
        Write-Host "✅ Built $($service.Name) successfully" -ForegroundColor Green
        
        # Push the images
        Write-Host "📤 Pushing $($service.Name) to ACR..." -ForegroundColor Cyan
        docker push $imageName
        docker push $taggedImageName
        
        if ($LASTEXITCODE -ne 0) {
            throw "Docker push failed for $($service.Name)"
        }
        
        Write-Host "✅ Pushed $($service.Name) successfully" -ForegroundColor Green
        
    }
    catch {
        Write-Host "❌ Failed to build/push $($service.Name): $_" -ForegroundColor Red
        exit 1
    }
}

Write-Host "🎉 All images built and pushed successfully!" -ForegroundColor Green
Write-Host "You can now deploy the Container Apps with the real images." -ForegroundColor Yellow

Write-Host "`n📋 Next steps:" -ForegroundColor Blue
Write-Host "1. Update the container-apps.bicep to use the real images:" -ForegroundColor White
Write-Host "   image: '$loginServer/policortex001-\${service.name}:latest'" -ForegroundColor Gray
Write-Host "2. Re-run your Bicep deployment" -ForegroundColor White