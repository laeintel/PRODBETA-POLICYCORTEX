# PolicyCortex Terraform Deployment Script for Windows
# Deploys infrastructure using the new naming convention: <resource>-cortex-<env>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("dev", "prod")]
    [string]$Environment = "dev",
    
    [Parameter(Mandatory=$false)]
    [switch]$Apply = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Destroy = $false
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "PolicyCortex Infrastructure Deployment" -ForegroundColor Cyan
Write-Host "Environment: $Environment" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

# Set working directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Generate a stable hash from repository name for uniqueness
$repoName = "laeintel/policycortex"
$hashBytes = [System.Text.Encoding]::UTF8.GetBytes($repoName)
$sha1 = [System.Security.Cryptography.SHA1]::Create()
$hash = $sha1.ComputeHash($hashBytes)
$hashString = [BitConverter]::ToString($hash).Replace("-", "").ToLower().Substring(0, 6)

# Backend configuration
$backendRG = "rg-tfstate-cortex-$Environment"
$backendSA = "sttfcortex$Environment$hashString"
$backendContainer = "tfstate"
$backendKey = "$Environment.tfstate"

Write-Host "`nBackend Configuration:" -ForegroundColor Green
Write-Host "  Resource Group: $backendRG"
Write-Host "  Storage Account: $backendSA"
Write-Host "  Container: $backendContainer"
Write-Host "  State Key: $backendKey"

# Check if logged in to Azure
Write-Host "`nChecking Azure login..." -ForegroundColor Yellow
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "Not logged in to Azure. Please login..." -ForegroundColor Red
    az login
    $account = az account show | ConvertFrom-Json
}

Write-Host "Logged in as: $($account.user.name)" -ForegroundColor Green
Write-Host "Subscription: $($account.name) ($($account.id))" -ForegroundColor Green

# Create backend storage if it doesn't exist
Write-Host "`nProvisioning Terraform backend storage..." -ForegroundColor Yellow

# Create resource group
az group create --name $backendRG --location eastus --output none 2>$null
Write-Host "✓ Resource group: $backendRG" -ForegroundColor Green

# Create storage account
$saExists = az storage account show --name $backendSA --resource-group $backendRG 2>$null
if (-not $saExists) {
    az storage account create `
        --name $backendSA `
        --resource-group $backendRG `
        --location eastus `
        --sku Standard_LRS `
        --encryption-services blob `
        --output none
    Write-Host "✓ Storage account created: $backendSA" -ForegroundColor Green
} else {
    Write-Host "✓ Storage account exists: $backendSA" -ForegroundColor Green
}

# Create container
az storage container create `
    --name $backendContainer `
    --account-name $backendSA `
    --auth-mode login `
    --output none 2>$null
Write-Host "✓ Container: $backendContainer" -ForegroundColor Green

# Initialize Terraform
Write-Host "`nInitializing Terraform..." -ForegroundColor Yellow
terraform init `
    -backend-config="resource_group_name=$backendRG" `
    -backend-config="storage_account_name=$backendSA" `
    -backend-config="container_name=$backendContainer" `
    -backend-config="key=$backendKey" `
    -backend-config="use_azuread_auth=true" `
    -reconfigure

if ($LASTEXITCODE -ne 0) {
    Write-Host "Terraform init failed!" -ForegroundColor Red
    exit 1
}

# Select workspace (optional - for multi-environment)
# terraform workspace select $Environment 2>$null || terraform workspace new $Environment

# Validate configuration
Write-Host "`nValidating Terraform configuration..." -ForegroundColor Yellow
terraform validate

if ($LASTEXITCODE -ne 0) {
    Write-Host "Terraform validation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Configuration is valid" -ForegroundColor Green

# ==============================================
# Import-on-exist: Ensure state owns existing AZ resources
# ==============================================

function Import-If-Exists {
    param(
        [Parameter(Mandatory=$true)][string]$Address,
        [Parameter(Mandatory=$true)][string]$ResourceId
    )
    try {
        $inState = terraform state list 2>$null | Select-String -SimpleMatch $Address
    } catch { $inState = $null }

    if ($inState) {
        Write-Host "✓ In state: $Address" -ForegroundColor Green
        return
    }

    $res = az resource show --ids $ResourceId 2>$null
    if ($LASTEXITCODE -eq 0 -and $res) {
        Write-Host "↪ Importing existing Azure resource into state: $Address" -ForegroundColor Yellow
        terraform import $Address $ResourceId | Out-Host
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Imported: $Address" -ForegroundColor Green
        } else {
            Write-Host "⚠ Import failed for $Address (you can retry manually): terraform import `"$Address`" `"$ResourceId`"" -ForegroundColor Red
        }
    } else {
        Write-Host "⧗ Not found in Azure (will be created on apply): $Address" -ForegroundColor DarkGray
    }
}

# Compute deterministic resource IDs from naming convention
$subId = az account show --query id -o tsv 2>$null
$rgName = "rg-cortex-$Environment"
$envName = "cae-cortex-$Environment"
$coreApp = "ca-cortex-core-$Environment"
$feApp = "ca-cortex-frontend-$Environment"

if ($subId) {
    $rgId   = "/subscriptions/$subId/resourceGroups/$rgName"
    $envId  = "/subscriptions/$subId/resourceGroups/$rgName/providers/Microsoft.App/managedEnvironments/$envName"
    $coreId = "/subscriptions/$subId/resourceGroups/$rgName/providers/Microsoft.App/containerApps/$coreApp"
    $feId   = "/subscriptions/$subId/resourceGroups/$rgName/providers/Microsoft.App/containerApps/$feApp"

    Write-Host "`nReconciling Terraform state with existing Azure resources..." -ForegroundColor Yellow
    Import-If-Exists -Address "azurerm_resource_group.main" -ResourceId $rgId
    Import-If-Exists -Address "azurerm_container_app_environment.main" -ResourceId $envId
    Import-If-Exists -Address "azurerm_container_app.core" -ResourceId $coreId
    Import-If-Exists -Address "azurerm_container_app.frontend" -ResourceId $feId
} else {
    Write-Host "Could not resolve subscription id; skipping import-on-exist step" -ForegroundColor Yellow
}

# Plan or Apply based on parameters
$tfvarsFile = "environments/$Environment/terraform.tfvars"

if ($Destroy) {
    Write-Host "`n⚠️  DESTROY MODE - This will delete all resources!" -ForegroundColor Red
    $confirm = Read-Host "Are you sure you want to destroy the $Environment environment? (yes/no)"
    if ($confirm -eq "yes") {
        terraform destroy -var-file=$tfvarsFile -auto-approve
    } else {
        Write-Host "Destroy cancelled" -ForegroundColor Yellow
    }
} elseif ($Apply) {
    Write-Host "`nApplying Terraform configuration..." -ForegroundColor Yellow
    terraform apply -var-file=$tfvarsFile -auto-approve
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Deployment successful!" -ForegroundColor Green
        Write-Host "`nResource Summary:" -ForegroundColor Cyan
        Write-Host "=================" -ForegroundColor Cyan
        
        # Show outputs
        $outputs = terraform output -json | ConvertFrom-Json
        foreach ($output in $outputs.PSObject.Properties) {
            if ($output.Value.sensitive -ne $true) {
                Write-Host "$($output.Name): $($output.Value.value)" -ForegroundColor White
            }
        }
        
        Write-Host "`nNaming Convention:" -ForegroundColor Yellow
        Write-Host "  Resource Group: rg-cortex-$Environment"
        Write-Host "  PostgreSQL: psql-cortex-$Environment"
        Write-Host "  Cosmos DB: cosmos-cortex-$Environment"
        Write-Host "  Storage: stcortex$Environment*"
        Write-Host "  Key Vault: kv-cortex-$Environment-*"
        Write-Host "  VM: vm-cortex-$Environment (dev only)"
        Write-Host "  Network: vnet-cortex-$Environment"
        Write-Host "  App Insights: appi-cortex-$Environment"
        Write-Host "  Container Registry: crcortex$Environment*"
        if ($Environment -eq "prod") {
            Write-Host "  Service Bus: sb-cortex-$Environment"
        }
    } else {
        Write-Host "Deployment failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`nPlanning Terraform changes..." -ForegroundColor Yellow
    terraform plan -var-file=$tfvarsFile -out="$Environment.tfplan"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Plan created successfully" -ForegroundColor Green
        Write-Host "To apply these changes, run:" -ForegroundColor Yellow
        Write-Host "  .\deploy.ps1 -Environment $Environment -Apply" -ForegroundColor Cyan
    } else {
        Write-Host "Planning failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Deployment script completed" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan