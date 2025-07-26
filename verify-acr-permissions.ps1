#!/usr/bin/env pwsh
<#
.SYNOPSIS
Verify ACR permissions for PolicyCortex managed identity

.DESCRIPTION
This script verifies that the managed identity has proper permissions on ACR and can authenticate correctly.

.PARAMETER Environment
The environment (dev, staging, prod)

.PARAMETER SubscriptionId
The Azure subscription ID

.EXAMPLE
./verify-acr-permissions.ps1 -Environment dev -SubscriptionId "your-subscription-id"
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("dev", "staging", "prod")]
    [string]$Environment,
    
    [Parameter(Mandatory=$true)]
    [string]$SubscriptionId
)

# Set error action
$ErrorActionPreference = "Stop"

Write-Host "🔍 Verifying ACR permissions for PolicyCortex" -ForegroundColor Green
Write-Host "Environment: $Environment" -ForegroundColor Yellow

# Set subscription
az account set --subscription $SubscriptionId

# Define resource names
$resourceGroupName = "rg-policortex001-app-$Environment"
$registryName = "crpolicortex001$Environment"
$identityName = "id-policortex001-$Environment"

Write-Host "📋 Resource Details:" -ForegroundColor Blue
Write-Host "  Resource Group: $resourceGroupName" -ForegroundColor White
Write-Host "  ACR Name: $registryName" -ForegroundColor White
Write-Host "  Identity Name: $identityName" -ForegroundColor White

# Get managed identity details
Write-Host "`n🔐 Getting managed identity details..." -ForegroundColor Blue
$identity = az identity show --name $identityName --resource-group $resourceGroupName --query "{clientId:clientId,principalId:principalId}" --output json | ConvertFrom-Json

if (!$identity) {
    Write-Host "❌ Managed identity not found!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Managed Identity found:" -ForegroundColor Green
Write-Host "  Client ID: $($identity.clientId)" -ForegroundColor White
Write-Host "  Principal ID: $($identity.principalId)" -ForegroundColor White

# Get ACR details
Write-Host "`n🏗️ Getting ACR details..." -ForegroundColor Blue
$acr = az acr show --name $registryName --query "{id:id,loginServer:loginServer}" --output json | ConvertFrom-Json

if (!$acr) {
    Write-Host "❌ ACR not found!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ ACR found:" -ForegroundColor Green
Write-Host "  ID: $($acr.id)" -ForegroundColor White
Write-Host "  Login Server: $($acr.loginServer)" -ForegroundColor White

# Check role assignments
Write-Host "`n🔑 Checking role assignments..." -ForegroundColor Blue

# Check AcrPull permission
$acrPullAssignment = az role assignment list --assignee $identity.principalId --scope $acr.id --role "AcrPull" --query "[0].roleDefinitionName" --output tsv

if ($acrPullAssignment -eq "AcrPull") {
    Write-Host "✅ AcrPull permission: GRANTED" -ForegroundColor Green
} else {
    Write-Host "❌ AcrPull permission: MISSING" -ForegroundColor Red
    Write-Host "   Run: az role assignment create --assignee $($identity.principalId) --scope $($acr.id) --role AcrPull" -ForegroundColor Yellow
}

# Check AcrPush permission
$acrPushAssignment = az role assignment list --assignee $identity.principalId --scope $acr.id --role "AcrPush" --query "[0].roleDefinitionName" --output tsv

if ($acrPushAssignment -eq "AcrPush") {
    Write-Host "✅ AcrPush permission: GRANTED" -ForegroundColor Green
} else {
    Write-Host "❌ AcrPush permission: MISSING" -ForegroundColor Red
    Write-Host "   Run: az role assignment create --assignee $($identity.principalId) --scope $($acr.id) --role AcrPush" -ForegroundColor Yellow
}

# Test authentication
Write-Host "`n🧪 Testing ACR authentication..." -ForegroundColor Blue

try {
    # Login with managed identity (this simulates what Container Apps will do)
    az acr login --name $registryName --expose-token --output none
    Write-Host "✅ ACR authentication: SUCCESS" -ForegroundColor Green
} catch {
    Write-Host "❌ ACR authentication: FAILED" -ForegroundColor Red
    Write-Host "   Error: $_" -ForegroundColor Red
}

# List repositories
Write-Host "`n📦 ACR Repository List:" -ForegroundColor Blue
$repositories = az acr repository list --name $registryName --output json | ConvertFrom-Json

if ($repositories -and $repositories.Count -gt 0) {
    foreach ($repo in $repositories) {
        Write-Host "  📁 $repo" -ForegroundColor White
        
        # Get tags for each repository
        $tags = az acr repository show-tags --name $registryName --repository $repo --output json | ConvertFrom-Json
        if ($tags -and $tags.Count -gt 0) {
            foreach ($tag in $tags) {
                Write-Host "    🏷️  $tag" -ForegroundColor Gray
            }
        } else {
            Write-Host "    🏷️  No tags found" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "  📁 No repositories found (images need to be built and pushed)" -ForegroundColor Yellow
}

# Summary
Write-Host "`n📊 Summary:" -ForegroundColor Blue
Write-Host "  Managed Identity: ✅ Found" -ForegroundColor White
Write-Host "  ACR Registry: ✅ Found" -ForegroundColor White
Write-Host "  AcrPull Permission: $(if ($acrPullAssignment -eq 'AcrPull') { '✅ Granted' } else { '❌ Missing' })" -ForegroundColor White
Write-Host "  AcrPush Permission: $(if ($acrPushAssignment -eq 'AcrPush') { '✅ Granted' } else { '❌ Missing' })" -ForegroundColor White

if ($acrPullAssignment -eq "AcrPull" -and $acrPushAssignment -eq "AcrPush") {
    Write-Host "`n🎉 All ACR permissions are correctly configured!" -ForegroundColor Green
    Write-Host "Container Apps should be able to pull images without authentication issues." -ForegroundColor Green
} else {
    Write-Host "`n⚠️  Some permissions are missing. Container Apps may fail to pull images." -ForegroundColor Yellow
    Write-Host "Please redeploy the infrastructure to apply the missing role assignments." -ForegroundColor Yellow
}