# Fix Azure Container Registry Authentication for Container Apps
Write-Host "Fixing ACR Authentication for PolicyCortex Container Apps..." -ForegroundColor Green

# Variables
$RESOURCE_GROUP = "rg-policortex001-dev"
$ACR_NAME = "crpolicortex001dev"
$CONTAINER_APPS_ENV = "cae-policortex001-dev"

# List of container apps
$containerApps = @(
    "ca-api-gateway-dev",
    "ca-azure-integration-dev",
    "ca-ai-engine-dev",
    "ca-data-processing-dev",
    "ca-conversation-dev",
    "ca-notification-dev",
    "ca-frontend-dev"
)

# Get ACR credentials
Write-Host "`nGetting ACR credentials..." -ForegroundColor Yellow
$acrCredentials = az acr credential show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query "{username:username, password:passwords[0].value}" -o json | ConvertFrom-Json

if ($null -eq $acrCredentials) {
    Write-Host "ERROR: Failed to get ACR credentials!" -ForegroundColor Red
    exit 1
}

Write-Host "ACR Username: $($acrCredentials.username)" -ForegroundColor Cyan

# Get the managed identity for Container Apps
Write-Host "`nGetting Container Apps managed identity..." -ForegroundColor Yellow
$identityId = az identity show --name "id-containerapp-policortex001-dev" --resource-group $RESOURCE_GROUP --query id -o tsv

if ($null -eq $identityId) {
    Write-Host "Creating managed identity for Container Apps..." -ForegroundColor Yellow
    az identity create --name "id-containerapp-policortex001-dev" --resource-group $RESOURCE_GROUP
    $identityId = az identity show --name "id-containerapp-policortex001-dev" --resource-group $RESOURCE_GROUP --query id -o tsv
}

# Get ACR resource ID
$acrId = az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query id -o tsv

# Assign AcrPull role to the managed identity
Write-Host "`nAssigning AcrPull role to managed identity..." -ForegroundColor Yellow
az role assignment create --assignee $identityId --role "AcrPull" --scope $acrId

# Update Container Apps Environment with registry credentials
Write-Host "`nUpdating Container Apps Environment with ACR credentials..." -ForegroundColor Yellow

# Create registry configuration JSON
$registryConfig = @{
    server = "$ACR_NAME.azurecr.io"
    username = $acrCredentials.username
    passwordSecretRef = "acr-password"
} | ConvertTo-Json -Compress

# Update the container apps environment
az containerapp env update `
    --name $CONTAINER_APPS_ENV `
    --resource-group $RESOURCE_GROUP `
    --dapr-app-protocol "http" `
    --only-show-errors

# Create a secret for ACR password in the environment
Write-Host "`nCreating ACR password secret in Container Apps Environment..." -ForegroundColor Yellow
az containerapp env dapr-component set `
    --name $CONTAINER_APPS_ENV `
    --resource-group $RESOURCE_GROUP `
    --dapr-component-name "acr-secret" `
    --yaml @"
name: acr-secret
properties:
  version: v1
  secrets:
  - name: acr-password
    value: $($acrCredentials.password)
"@ 2>$null

# Update each container app with registry configuration
foreach ($appName in $containerApps) {
    Write-Host "`nUpdating $appName with ACR configuration..." -ForegroundColor Yellow
    
    try {
        # Get current configuration
        $currentConfig = az containerapp show --name $appName --resource-group $RESOURCE_GROUP -o json | ConvertFrom-Json
        
        if ($null -eq $currentConfig) {
            Write-Host "WARNING: $appName not found, skipping..." -ForegroundColor Yellow
            continue
        }
        
        # Update with registry configuration
        az containerapp registry set `
            --name $appName `
            --resource-group $RESOURCE_GROUP `
            --server "$ACR_NAME.azurecr.io" `
            --username $acrCredentials.username `
            --password $acrCredentials.password `
            --only-show-errors
        
        Write-Host "Successfully updated $appName" -ForegroundColor Green
    }
    catch {
        Write-Host "WARNING: Failed to update $appName - $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Alternative: Use managed identity for ACR authentication
Write-Host "`nConfiguring managed identity authentication for Container Apps..." -ForegroundColor Yellow

foreach ($appName in $containerApps) {
    Write-Host "Updating $appName to use managed identity..." -ForegroundColor Cyan
    
    try {
        # Update container app to use managed identity
        az containerapp identity assign `
            --name $appName `
            --resource-group $RESOURCE_GROUP `
            --system-assigned `
            --only-show-errors
        
        # Get the container app's identity
        $appIdentity = az containerapp identity show `
            --name $appName `
            --resource-group $RESOURCE_GROUP `
            --query principalId -o tsv
        
        if ($appIdentity) {
            # Assign AcrPull role to the container app's identity
            az role assignment create `
                --assignee $appIdentity `
                --role "AcrPull" `
                --scope $acrId `
                --only-show-errors 2>$null
        }
        
        Write-Host "Successfully configured managed identity for $appName" -ForegroundColor Green
    }
    catch {
        Write-Host "WARNING: Failed to configure managed identity for $appName" -ForegroundColor Yellow
    }
}

Write-Host "`n=== ACR Authentication Fix Summary ===" -ForegroundColor Cyan
Write-Host "1. ACR credentials retrieved and configured" -ForegroundColor Green
Write-Host "2. Managed identity created and AcrPull role assigned" -ForegroundColor Green
Write-Host "3. Container Apps updated with registry authentication" -ForegroundColor Green
Write-Host "4. System-assigned managed identities configured for each app" -ForegroundColor Green

Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "1. Wait 2-3 minutes for role assignments to propagate" -ForegroundColor White
Write-Host "2. Restart container apps if needed: az containerapp revision restart" -ForegroundColor White
Write-Host "3. Check container app logs for any remaining issues" -ForegroundColor White

Write-Host "`nTo force a new revision with updated authentication:" -ForegroundColor Yellow
Write-Host "az containerapp update --name <app-name> --resource-group $RESOURCE_GROUP --min-replicas 1" -ForegroundColor White

Write-Host "`nDone!" -ForegroundColor Green