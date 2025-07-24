# PowerShell script to migrate all environments to dedicated workload profiles

param(
    [switch]$WhatIf = $false
)

$Environments = @("dev", "staging", "prod")

Write-Host "🚀 Starting migration of Container Apps to dedicated workload profiles across all environments" -ForegroundColor Green
Write-Host ""

if ($WhatIf) {
    Write-Host "⚠️  Running in WHAT-IF mode - no changes will be made" -ForegroundColor Yellow
    Write-Host ""
}

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

$CurrentSubscription = az account show --query "name" -o tsv
Write-Host "📋 Current Azure Subscription: $CurrentSubscription" -ForegroundColor Cyan
Write-Host ""

foreach ($Environment in $Environments) {
    $ResourceGroup = "rg-policycortex-$Environment"
    $ContainerAppsEnvironment = "cae-policycortex-$Environment"
    $DedicatedProfile = "Dedicated-D4"
    
    Write-Host "🔄 Processing Environment: $Environment" -ForegroundColor Yellow
    Write-Host "   Resource Group: $ResourceGroup"
    Write-Host "   Container Apps Environment: $ContainerAppsEnvironment"
    Write-Host "   Target Workload Profile: $DedicatedProfile"
    Write-Host ""
    
    # Check if the Container Apps Environment exists
    try {
        $null = az containerapp env show --name $ContainerAppsEnvironment --resource-group $ResourceGroup 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "⚠️  Container Apps Environment '$ContainerAppsEnvironment' not found in '$ResourceGroup' - skipping" -ForegroundColor Yellow
            Write-Host ""
            continue
        }
    } catch {
        Write-Host "⚠️  Container Apps Environment '$ContainerAppsEnvironment' not found in '$ResourceGroup' - skipping" -ForegroundColor Yellow
        Write-Host ""
        continue
    }
    
    # List all container apps in the environment
    $ContainerApps = az containerapp list --resource-group $ResourceGroup --query "[?properties.managedEnvironmentId contains '$ContainerAppsEnvironment'].name" -o tsv
    
    if (-not $ContainerApps) {
        Write-Host "ℹ️  No Container Apps found in environment '$ContainerAppsEnvironment' - skipping" -ForegroundColor Blue
        Write-Host ""
        continue
    }
    
    Write-Host "📦 Found Container Apps in $Environment":"
    foreach ($app in $ContainerApps) {
        # Check current workload profile
        $currentProfile = az containerapp show --name $app --resource-group $ResourceGroup --query "properties.template.workloadProfileName" -o tsv 2>$null
        if (-not $currentProfile) {
            $currentProfile = "Consumption (default)"
        }
        Write-Host "   - $app (currently: $currentProfile)"
    }
    Write-Host ""
    
    if ($WhatIf) {
        Write-Host "🔍 WHAT-IF: Would migrate the above apps to '$DedicatedProfile' profile" -ForegroundColor Cyan
        Write-Host ""
        continue
    }
    
    # Migrate each app
    $SuccessfulApps = @()
    $FailedApps = @()
    
    foreach ($app in $ContainerApps) {
        Write-Host "🔧 Updating $app to use workload profile '$DedicatedProfile'..." -ForegroundColor Yellow
        
        try {
            # Special handling for AI Engine - use Dedicated-D8 for better performance
            $targetProfile = if ($app -like "*ai-engine*") { "Dedicated-D8" } else { $DedicatedProfile }
            
            az containerapp update `
                --name $app `
                --resource-group $ResourceGroup `
                --workload-profile-name $targetProfile `
                --output none 2>$null
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "   ✅ Successfully updated $app to $targetProfile" -ForegroundColor Green
                $SuccessfulApps += @{ Name = $app; Profile = $targetProfile }
            } else {
                Write-Host "   ❌ Failed to update $app" -ForegroundColor Red
                $FailedApps += $app
            }
        } catch {
            Write-Host "   ❌ Error updating $app`: $_" -ForegroundColor Red
            $FailedApps += $app
        }
    }
    
    # Environment summary
    Write-Host ""
    Write-Host "📊 $Environment Migration Summary:" -ForegroundColor Cyan
    Write-Host "   Successful: $($SuccessfulApps.Count) apps"
    foreach ($app in $SuccessfulApps) {
        Write-Host "     ✅ $($app.Name) → $($app.Profile)" -ForegroundColor Green
    }
    
    if ($FailedApps.Count -gt 0) {
        Write-Host "   Failed: $($FailedApps.Count) apps" -ForegroundColor Red
        foreach ($app in $FailedApps) {
            Write-Host "     ❌ $app" -ForegroundColor Red
        }
    }
    Write-Host ""
}

if ($WhatIf) {
    Write-Host "🔍 WHAT-IF mode completed. Run without -WhatIf to perform actual migration." -ForegroundColor Cyan
} else {
    Write-Host "✨ Migration completed for all environments!" -ForegroundColor Green
}

Write-Host ""
Write-Host "💡 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Verify apps are running properly on dedicated profiles"
Write-Host "   2. Monitor performance and resource utilization"
Write-Host "   3. Update your infrastructure code to always deploy to dedicated profiles"
Write-Host "   4. Apply infrastructure changes with Terraform/Bicep to make this persistent"
Write-Host ""
Write-Host "🔗 To verify workload profile assignments:"
Write-Host "   az containerapp list --resource-group rg-policycortex-[ENV] --query \"[].{Name:name,WorkloadProfile:properties.template.workloadProfileName}\" -o table" 