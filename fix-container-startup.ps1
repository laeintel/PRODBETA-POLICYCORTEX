# Fix Container Startup Issues Script
# This script rebuilds and redeploys containers with proper startup configuration

param(
    [string]$Environment = "dev"
)

Write-Host "🔧 Fixing container startup issues..." -ForegroundColor Green

# Configuration
$RegistryName = "crpolicortex001$Environment"
$ResourceGroup = "rg-policortex001-$Environment"
$Location = "eastus"

Write-Host "📋 Configuration:" -ForegroundColor Yellow
Write-Host "  Environment: $Environment"
Write-Host "  Registry: $RegistryName"
Write-Host "  Resource Group: $ResourceGroup"
Write-Host "  Location: $Location"

# Build and push all services with startup fixes
Write-Host "🏗️  Building and pushing containers..." -ForegroundColor Green

# API Gateway
Write-Host "📦 Building API Gateway..." -ForegroundColor Cyan
docker build -t $RegistryName.azurecr.io/policortex001-api-gateway:latest `
  -f backend/services/api_gateway/Dockerfile `
  backend/

# Azure Integration
Write-Host "📦 Building Azure Integration..." -ForegroundColor Cyan
docker build -t $RegistryName.azurecr.io/policortex001-azure-integration:latest `
  -f backend/services/azure_integration/Dockerfile `
  backend/

# AI Engine
Write-Host "📦 Building AI Engine..." -ForegroundColor Cyan
docker build -t $RegistryName.azurecr.io/policortex001-ai-engine:latest `
  -f backend/services/ai_engine/Dockerfile `
  backend/

# Data Processing
Write-Host "📦 Building Data Processing..." -ForegroundColor Cyan
docker build -t $RegistryName.azurecr.io/policortex001-data-processing:latest `
  -f backend/services/data_processing/Dockerfile `
  backend/

# Conversation
Write-Host "📦 Building Conversation..." -ForegroundColor Cyan
docker build -t $RegistryName.azurecr.io/policortex001-conversation:latest `
  -f backend/services/conversation/Dockerfile `
  backend/

# Notification
Write-Host "📦 Building Notification..." -ForegroundColor Cyan
docker build -t $RegistryName.azurecr.io/policortex001-notification:latest `
  -f backend/services/notification/Dockerfile `
  backend/

# Frontend
Write-Host "📦 Building Frontend..." -ForegroundColor Cyan
docker build -t $RegistryName.azurecr.io/policortex001-frontend:latest `
  -f frontend/Dockerfile `
  frontend/

# Push all images
Write-Host "🚀 Pushing images to registry..." -ForegroundColor Green
docker push $RegistryName.azurecr.io/policortex001-api-gateway:latest
docker push $RegistryName.azurecr.io/policortex001-azure-integration:latest
docker push $RegistryName.azurecr.io/policortex001-ai-engine:latest
docker push $RegistryName.azurecr.io/policortex001-data-processing:latest
docker push $RegistryName.azurecr.io/policortex001-conversation:latest
docker push $RegistryName.azurecr.io/policortex001-notification:latest
docker push $RegistryName.azurecr.io/policortex001-frontend:latest

# Deploy infrastructure with fixes
Write-Host "🚀 Deploying infrastructure with startup fixes..." -ForegroundColor Green
Set-Location infrastructure/bicep

# Deploy with the updated configuration
az deployment group create `
  --resource-group $ResourceGroup `
  --template-file main.bicep `
  --parameters environment=$Environment `
  --verbose

Write-Host "✅ Container startup fixes deployed!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Check container status:" -ForegroundColor Yellow
Write-Host "  az containerapp revision list --name ca-api-gateway-$Environment --resource-group $ResourceGroup"
Write-Host ""
Write-Host "📋 View logs:" -ForegroundColor Yellow
Write-Host "  az containerapp logs show --name ca-api-gateway-$Environment --resource-group $ResourceGroup --follow" 