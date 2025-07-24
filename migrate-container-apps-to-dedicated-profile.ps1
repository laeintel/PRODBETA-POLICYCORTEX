# PowerShell script to migrate Container Apps to dedicated workload profile

param(
    [string]$Environment = "dev"
)

# Configuration
$ResourceGroup = "rg-policycortex-$Environment"
$ContainerAppsEnvironment = "cae-policycortex-$Environment"
$DedicatedProfile = "Dedicated-D4"

Write-Host "🚀 Starting migration of Container Apps to dedicated workload profile" -ForegroundColor Green
Write-Host "Environment: $Environment"
Write-Host "Resource Group: $ResourceGroup"
Write-Host "Container Apps Environment: $ContainerAppsEnvironment"
Write-Host "Target Workload Profile: $DedicatedProfile"
Write-Host ""

# Check if logged in to Azure
try {
    $null = az account show 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Not logged in"
    }
} catch {
    Write-Host "❌ Not logged into Azure CLI. Please run 'az login' first." -ForegroundColor Red
    exit 1
}

# Verify the Container Apps Environment exists
Write-Host "🔍 Verifying Container Apps Environment..." -ForegroundColor Yellow
try {
    $null = az containerapp env show --name $ContainerAppsEnvironment --resource-group $ResourceGroup 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Environment not found"
    }
} catch {
    Write-Host "❌ Container Apps Environment '$ContainerAppsEnvironment' not found in resource group '$ResourceGroup'" -ForegroundColor Red
    exit 1
}

# List all container apps in the environment
Write-Host "📋 Discovering Container Apps..." -ForegroundColor Yellow
$ContainerApps = az containerapp list --resource-group $ResourceGroup --query "[?properties.managedEnvironmentId contains '$ContainerAppsEnvironment'].name" -o tsv

if (-not $ContainerApps) {
    Write-Host "❌ No Container Apps found in environment '$ContainerAppsEnvironment'" -ForegroundColor Red
    exit 1
}

Write-Host "Found Container Apps:"
foreach ($app in $ContainerApps) {
    Write-Host "  - $app"
}
Write-Host ""

# Function to update a container app
function Update-ContainerApp {
    param($AppName)
    
    Write-Host "🔧 Updating $AppName to use workload profile '$DedicatedProfile'..." -ForegroundColor Yellow
    
    try {
        az containerapp update `
            --name $AppName `
            --resource-group $ResourceGroup `
            --workload-profile-name $DedicatedProfile `
            --output table
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Successfully updated $AppName" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ Failed to update $AppName" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "❌ Error updating $AppName`: $_" -ForegroundColor Red
        return $false
    }
}

# Update each container app
Write-Host "🔄 Starting migration process..." -ForegroundColor Yellow
Write-Host ""

$SuccessfulApps = @()
$FailedApps = @()

foreach ($app in $ContainerApps) {
    if (Update-ContainerApp -AppName $app) {
        $SuccessfulApps += $app
    } else {
        $FailedApps += $app
    }
    Write-Host ""
}

# Summary
Write-Host "📊 Migration Summary" -ForegroundColor Cyan
Write-Host "==================="
Write-Host "Successfully migrated: $($SuccessfulApps.Count) apps"
foreach ($app in $SuccessfulApps) {
    Write-Host "  ✅ $app" -ForegroundColor Green
}

if ($FailedApps.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed to migrate: $($FailedApps.Count) apps" -ForegroundColor Red
    foreach ($app in $FailedApps) {
        Write-Host "  ❌ $app" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Please check the failed apps manually and retry if needed." -ForegroundColor Yellow
    exit 1
} else {
    Write-Host ""
    Write-Host "🎉 All Container Apps successfully migrated to dedicated workload profile!" -ForegroundColor Green
}

Write-Host ""
Write-Host "🔍 Verification: Checking workload profile assignments..." -ForegroundColor Yellow
foreach ($app in $SuccessfulApps) {
    $profile = az containerapp show --name $app --resource-group $ResourceGroup --query "properties.template.workloadProfileName" -o tsv
    Write-Host "  $app`: $profile"
}

Write-Host ""
Write-Host "✨ Migration completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "💡 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Update your infrastructure code (Terraform/Bicep) to use dedicated profiles by default"
Write-Host "   2. Apply the same migration to staging and prod environments"
Write-Host "   3. Monitor app performance on the dedicated profile" 