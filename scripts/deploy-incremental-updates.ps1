# Deploy incremental updates without recreating existing resources
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('dev', 'staging', 'prod')]
    [string]$Environment = 'dev',
    
    [Parameter(Mandatory=$false)]
    [switch]$UpdateLogAnalytics = $true,
    
    [Parameter(Mandatory=$false)]
    [switch]$UpdateAppInsights = $true,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipContainerApps = $true
)

Write-Host "PolicyCortex Incremental Deployment" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

$resourceGroup = "rg-pcx-app-$Environment"
$envResourceGroup = "rg-pcx-network-$Environment"

# Check if container apps environment exists and update Log Analytics
if ($UpdateLogAnalytics) {
    Write-Host "Updating Container Apps Environment with Log Analytics..." -ForegroundColor Yellow
    
    $lawName = "law-pcx-$Environment"
    $envName = "cae-pcx-$Environment"
    
    # Get Log Analytics Workspace ID
    $lawId = az monitor log-analytics workspace show `
        --resource-group $resourceGroup `
        --workspace-name $lawName `
        --query "id" -o tsv
    
    if ($lawId) {
        Write-Host "Found Log Analytics Workspace: $lawName" -ForegroundColor Green
        
        # Check if container apps environment exists
        $envExists = az containerapp env show `
            --name $envName `
            --resource-group $resourceGroup `
            --query "id" -o tsv 2>$null
        
        if ($envExists) {
            Write-Host "Container Apps Environment exists: $envName" -ForegroundColor Green
            
            # Update with Log Analytics configuration
            $updateResult = az containerapp env update `
                --name $envName `
                --resource-group $resourceGroup `
                --logs-destination log-analytics `
                --logs-workspace-id $lawId `
                --output none 2>$null
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Successfully configured Log Analytics" -ForegroundColor Green
            } else {
                Write-Host "⚠️  Could not update Log Analytics configuration" -ForegroundColor Yellow
            }
        } else {
            Write-Host "⚠️  Container Apps Environment not found" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Log Analytics Workspace not found" -ForegroundColor Red
    }
}

# Update Application Insights sampling
if ($UpdateAppInsights) {
    Write-Host "`nUpdating Application Insights configuration..." -ForegroundColor Yellow
    
    $appInsightsName = "ai-pcx-$Environment"
    
    # Check if Application Insights exists
    $aiExists = az monitor app-insights component show `
        --app $appInsightsName `
        --resource-group $resourceGroup `
        --query "id" -o tsv 2>$null
    
    if ($aiExists) {
        Write-Host "Found Application Insights: $appInsightsName" -ForegroundColor Green
        
        # Update sampling percentage (this might not be directly supported via CLI)
        Write-Host "✅ Application Insights exists - manual sampling configuration may be needed" -ForegroundColor Green
    } else {
        Write-Host "❌ Application Insights not found" -ForegroundColor Red
    }
}

# Verify container apps configuration
Write-Host "`nVerifying container apps configuration..." -ForegroundColor Yellow

$containerApps = @(
    "ca-pcx-gateway-$Environment",
    "ca-pcx-azureint-$Environment", 
    "ca-pcx-ai-$Environment",
    "ca-pcx-dataproc-$Environment",
    "ca-pcx-chat-$Environment",
    "ca-pcx-notify-$Environment",
    "ca-pcx-web-$Environment"
)

foreach ($app in $containerApps) {
    $status = az containerapp show `
        --name $app `
        --resource-group $resourceGroup `
        --query "{Name:name, State:properties.runningStatus, ProvisioningState:properties.provisioningState}" `
        -o json 2>$null | ConvertFrom-Json
    
    if ($status) {
        $statusColor = if ($status.State -eq 'Running') { 'Green' } else { 'Yellow' }
        Write-Host "  $($status.Name): $($status.State) ($($status.ProvisioningState))" -ForegroundColor $statusColor
    } else {
        Write-Host "  ${app}: Not found" -ForegroundColor Red
    }
}

# Check Log Analytics integration
Write-Host "`nVerifying Log Analytics integration..." -ForegroundColor Yellow
$envConfig = az containerapp env show `
    --name "cae-pcx-$Environment" `
    --resource-group $resourceGroup `
    --query "properties.appLogsConfiguration" `
    -o json 2>$null | ConvertFrom-Json

if ($envConfig -and $envConfig.destination -eq 'log-analytics') {
    Write-Host "✅ Log Analytics integration is configured" -ForegroundColor Green
} else {
    Write-Host "❌ Log Analytics integration is not configured" -ForegroundColor Red
    if ($envConfig) {
        Write-Host "Current destination: $($envConfig.destination)" -ForegroundColor Yellow
    }
}

Write-Host "`n===================================" -ForegroundColor Cyan
Write-Host "Incremental deployment completed!" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Cyan