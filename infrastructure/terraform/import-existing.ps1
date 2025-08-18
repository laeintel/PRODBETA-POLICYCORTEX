# PowerShell script to import existing Azure resources into Terraform state
# This script checks for existing resources and imports them before applying

param(
    [string]$Environment = "dev",
    [string]$SubscriptionId = "205b477d-17e7-4b3b-92c1-32cf02626b78"
)

Write-Host "Checking for existing Azure resources to import..." -ForegroundColor Yellow

# Function to check if resource exists and import it
function Import-ResourceIfExists {
    param(
        [string]$ResourceId,
        [string]$TerraformAddress,
        [string]$ResourceType
    )
    
    Write-Host "Checking $ResourceType..." -NoNewline
    
    # Check if resource exists using Azure CLI
    $exists = $false
    try {
        $result = az resource show --ids $ResourceId 2>$null
        if ($LASTEXITCODE -eq 0) {
            $exists = $true
        }
    } catch {
        # Resource doesn't exist
    }
    
    if ($exists) {
        Write-Host " Found! Importing..." -ForegroundColor Green
        terraform import $TerraformAddress $ResourceId
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  Warning: Import failed or already in state" -ForegroundColor Yellow
        }
    } else {
        Write-Host " Not found (will be created)" -ForegroundColor Gray
    }
}

# Ensure we're in the right directory
Set-Location -Path $PSScriptRoot

# Initialize Terraform if needed
if (-not (Test-Path ".terraform")) {
    Write-Host "Initializing Terraform..." -ForegroundColor Yellow
    terraform init
}

# Define resources to check and import
$resourcesToImport = @(
    @{
        Type = "Resource Group"
        Address = "azurerm_resource_group.main"
        Id = "/subscriptions/$SubscriptionId/resourceGroups/rg-cortex-$Environment"
    },
    @{
        Type = "Container Apps Environment"
        Address = "azurerm_container_app_environment.main"
        Id = "/subscriptions/$SubscriptionId/resourceGroups/rg-cortex-$Environment/providers/Microsoft.App/managedEnvironments/cae-cortex-$Environment"
    },
    @{
        Type = "Core Container App"
        Address = "azurerm_container_app.core"
        Id = "/subscriptions/$SubscriptionId/resourceGroups/rg-cortex-$Environment/providers/Microsoft.App/containerApps/ca-cortex-core-$Environment"
    },
    @{
        Type = "Frontend Container App"
        Address = "azurerm_container_app.frontend"
        Id = "/subscriptions/$SubscriptionId/resourceGroups/rg-cortex-$Environment/providers/Microsoft.App/containerApps/ca-cortex-frontend-$Environment"
    },
    @{
        Type = "Log Analytics Workspace"
        Address = "azurerm_log_analytics_workspace.main"
        Id = "/subscriptions/$SubscriptionId/resourceGroups/rg-cortex-$Environment/providers/Microsoft.OperationalInsights/workspaces/log-cortex-$Environment"
    },
    @{
        Type = "Application Insights"
        Address = "azurerm_application_insights.main"
        Id = "/subscriptions/$SubscriptionId/resourceGroups/rg-cortex-$Environment/providers/Microsoft.Insights/components/appi-cortex-$Environment"
    }
)

# Import each resource if it exists
foreach ($resource in $resourcesToImport) {
    Import-ResourceIfExists -ResourceId $resource.Id -TerraformAddress $resource.Address -ResourceType $resource.Type
}

Write-Host "`nImport check complete!" -ForegroundColor Green
Write-Host "You can now run 'terraform plan' to see what changes will be made." -ForegroundColor Cyan