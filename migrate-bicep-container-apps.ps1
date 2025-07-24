# PowerShell script to migrate Container Apps deployed via Bicep to dedicated workload profiles

param(
    [string]$Environment = "dev",
    [switch]$WhatIf = $false
)

# Configuration
$ResourceGroup = "rg-policycortex-$Environment"
$ContainerAppsEnvironment = "cae-policycortex-$Environment"

Write-Host "Starting migration of Bicep-deployed Container Apps to dedicated workload profiles" -ForegroundColor Green
Write-Host "Environment: $Environment"
Write-Host "Resource Group: $ResourceGroup"
Write-Host "Container Apps Environment: $ContainerAppsEnvironment"
Write-Host ""

if ($WhatIf) {
    Write-Host "Running in WHAT-IF mode - no changes will be made" -ForegroundColor Yellow
    Write-Host ""
}

# Check if logged in to Azure
try {
    $null = az account show 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Not logged in"
    }
} catch {
    Write-Host "Not logged into Azure CLI. Please run 'az login' first." -ForegroundColor Red
    exit 1
}

# Verify the Container Apps Environment exists
Write-Host "Verifying Container Apps Environment..." -ForegroundColor Yellow
try {
    $null = az containerapp env show --name $ContainerAppsEnvironment --resource-group $ResourceGroup 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Environment not found"
    }
} catch {
    Write-Host "Container Apps Environment '$ContainerAppsEnvironment' not found in resource group '$ResourceGroup'" -ForegroundColor Red
    exit 1
}

# List all container apps in the environment
Write-Host "Discovering Container Apps..." -ForegroundColor Yellow
$ContainerApps = az containerapp list --resource-group $ResourceGroup --query "[?properties.managedEnvironmentId contains '$ContainerAppsEnvironment'].name" -o tsv

if (-not $ContainerApps) {
    Write-Host "No Container Apps found in environment '$ContainerAppsEnvironment'" -ForegroundColor Red
    exit 1
}

Write-Host "Found Container Apps:"
foreach ($app in $ContainerApps) {
    # Check current workload profile
    $currentProfile = az containerapp show --name $app --resource-group $ResourceGroup --query "properties.template.workloadProfileName" -o tsv 2>$null
    if (-not $currentProfile) {
        $currentProfile = "Consumption (default)"
    }
    Write-Host "  - $app (currently: $currentProfile)"
}
Write-Host ""

# Define workload profile mappings for Bicep deployments
$WorkloadProfileMappings = @{
    'ca-api-gateway-*' = 'GeneralPurpose'
    'ca-azure-integration-*' = 'GeneralPurpose'
    'ca-ai-engine-*' = 'HighPerformance'
    'ca-data-processing-*' = 'GeneralPurpose'
    'ca-conversation-*' = 'GeneralPurpose'
    'ca-notification-*' = 'GeneralPurpose'
    'ca-frontend-*' = 'GeneralPurpose'
}

if ($WhatIf) {
    Write-Host "WHAT-IF: Would migrate apps to dedicated workload profiles:" -ForegroundColor Cyan
    foreach ($app in $ContainerApps) {
        $targetProfile = $null
        foreach ($pattern in $WorkloadProfileMappings.Keys) {
            if ($app -like $pattern) {
                $targetProfile = $WorkloadProfileMappings[$pattern]
                break
            }
        }
        if ($targetProfile) {
            Write-Host "   $app -> $targetProfile" -ForegroundColor Yellow
        } else {
            Write-Host "   $app -> GeneralPurpose (default)" -ForegroundColor Yellow
        }
    }
    Write-Host ""
    Write-Host "WHAT-IF mode completed. Run without -WhatIf to perform actual migration." -ForegroundColor Cyan
    exit 0
}

# Migrate each app
$SuccessfulApps = @()
$FailedApps = @()

foreach ($app in $ContainerApps) {
    Write-Host "Updating $app to use dedicated workload profile..." -ForegroundColor Yellow
    
    # Determine target workload profile
    $targetProfile = $null
    foreach ($pattern in $WorkloadProfileMappings.Keys) {
        if ($app -like $pattern) {
            $targetProfile = $WorkloadProfileMappings[$pattern]
            break
        }
    }
    
    # Default to GeneralPurpose if no specific mapping found
    if (-not $targetProfile) {
        $targetProfile = 'GeneralPurpose'
    }
    
    try {
        az containerapp update `
            --name $app `
            --resource-group $ResourceGroup `
            --workload-profile-name $targetProfile `
            --output none 2>$null
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   Successfully updated $app to $targetProfile" -ForegroundColor Green
            $SuccessfulApps += @{ Name = $app; Profile = $targetProfile }
        } else {
            Write-Host "   Failed to update $app" -ForegroundColor Red
            $FailedApps += $app
        }
    } catch {
        Write-Host "   Error updating $app`: $_" -ForegroundColor Red
        $FailedApps += $app
    }
    Write-Host ""
}

# Summary
Write-Host "Migration Summary" -ForegroundColor Cyan
Write-Host "==================="
Write-Host "Successfully migrated: $($SuccessfulApps.Count) apps"
foreach ($app in $SuccessfulApps) {
    Write-Host "  $($app.Name) -> $($app.Profile)" -ForegroundColor Green
}

if ($FailedApps.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed to migrate: $($FailedApps.Count) apps" -ForegroundColor Red
    foreach ($app in $FailedApps) {
        Write-Host "  $app" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Please check the failed apps manually and retry if needed." -ForegroundColor Yellow
    exit 1
} else {
    Write-Host ""
    Write-Host "All Container Apps successfully migrated to dedicated workload profiles!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Verification: Checking workload profile assignments..." -ForegroundColor Yellow
foreach ($app in $SuccessfulApps) {
    $profile = az containerapp show --name $app.Name --resource-group $ResourceGroup --query "properties.template.workloadProfileName" -o tsv
    Write-Host "  $($app.Name): $profile"
}

Write-Host ""
Write-Host "Migration completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "   1. Verify apps are running properly on dedicated profiles"
Write-Host "   2. Monitor performance and resource utilization"
Write-Host "   3. Apply updated Bicep templates to make this persistent"
Write-Host "   4. Update your CI/CD pipeline to use the new Bicep configuration"
Write-Host ""
Write-Host "To verify workload profile assignments:"
Write-Host "   az containerapp list --resource-group $ResourceGroup --query `"[].{Name:name,WorkloadProfile:properties.template.workloadProfileName}`" -o table"
Write-Host ""
Write-Host "Bicep Workload Profile Mappings:" -ForegroundColor Cyan
Write-Host "   GeneralPurpose (D4): API Gateway, Azure Integration, Data Processing, Conversation, Notification, Frontend"
Write-Host "   HighPerformance (D8): AI Engine"
