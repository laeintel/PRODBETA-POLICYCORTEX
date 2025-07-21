# Quick fix script to import existing resource groups into Terraform state
# Run this from the infrastructure/terraform directory

param(
    [Parameter(Mandatory=$false)]
    [string]$Environment = "dev"
)

Write-Host "Quick Import Fix for PolicyCortex" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Get subscription ID
$subscriptionId = az account show --query id -o tsv
Write-Host "Subscription ID: $subscriptionId" -ForegroundColor Yellow

# Change to terraform directory
Set-Location infrastructure/terraform -ErrorAction SilentlyContinue

# Initialize Terraform if needed
Write-Host "Initializing Terraform..." -ForegroundColor Green
terraform init

# Import the resource groups
Write-Host ""
Write-Host "Importing resource groups..." -ForegroundColor Green

# Network Resource Group
$networkRgName = "rg-policycortex-network-$Environment"
$networkRgId = "/subscriptions/$subscriptionId/resourceGroups/$networkRgName"
Write-Host "Importing $networkRgName..." -ForegroundColor Yellow
terraform import azurerm_resource_group.network $networkRgId

# Application Resource Group
$appRgName = "rg-policycortex-app-$Environment"
$appRgId = "/subscriptions/$subscriptionId/resourceGroups/$appRgName"
Write-Host "Importing $appRgName..." -ForegroundColor Yellow
terraform import azurerm_resource_group.app $appRgId

Write-Host ""
Write-Host "Import completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Now you can run:" -ForegroundColor Yellow
Write-Host "  terraform plan -var-file=`"environments/$Environment/terraform.tfvars`"" -ForegroundColor Cyan
Write-Host "  terraform apply -var-file=`"environments/$Environment/terraform.tfvars`"" -ForegroundColor Cyan 