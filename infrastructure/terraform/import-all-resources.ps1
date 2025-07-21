# Comprehensive Terraform Import Script for PolicyCortex
# This script imports all existing Azure resources into Terraform state

Write-Host "=== PolicyCortex Comprehensive Resource Import Script ===" -ForegroundColor Green
Write-Host "This script will import ALL existing Azure resources into Terraform state" -ForegroundColor Yellow
Write-Host "WARNING: This may take several minutes to complete" -ForegroundColor Red
Write-Host ""

# Check if we're in the correct directory
if (!(Test-Path "main.tf")) {
    Write-Host "Error: Please run this script from the infrastructure/terraform directory" -ForegroundColor Red
    exit 1
}

# Set Azure CLI environment
$env:ARM_USE_CLI = "true"

# Define subscription and resource group info
$subscriptionId = "9f16cc88-89ce-49ba-a96d-308ed3169595"
$appRg = "rg-policycortex-app-dev"
$networkRg = "rg-policycortex-network-dev"

Write-Host "=== Importing Main Resources ===" -ForegroundColor Cyan

# Import Key Vault access policy
Write-Host "Importing Key Vault access policy..." -ForegroundColor White
try {
    terraform import azurerm_key_vault_access_policy.current_client "/subscriptions/$subscriptionId/resourceGroups/$appRg/providers/Microsoft.KeyVault/vaults/kvpolicycortexdevv2/objectId/178e2973-bb20-49da-ab80-0d1ddc7b0649" 2>$null
    Write-Host "✓ Key Vault access policy imported" -ForegroundColor Green
} catch {
    Write-Host "! Key Vault access policy import failed or already exists" -ForegroundColor Yellow
}

# Import role assignments (we'll handle these separately as they need IDs)
Write-Host "Importing role assignments..." -ForegroundColor White
Write-Host "! Role assignments need to be handled separately with their actual IDs" -ForegroundColor Yellow

# Import Log Analytics Workspace
Write-Host "Importing Log Analytics Workspace..." -ForegroundColor White
try {
    terraform import azurerm_log_analytics_workspace.main "/subscriptions/$subscriptionId/resourceGroups/$appRg/providers/Microsoft.OperationalInsights/workspaces/law-policycortex-dev" 2>$null
    Write-Host "✓ Log Analytics Workspace imported" -ForegroundColor Green
} catch {
    Write-Host "! Log Analytics Workspace import failed or already exists" -ForegroundColor Yellow
}

# Import User Assigned Identity
Write-Host "Importing User Assigned Identity..." -ForegroundColor White
try {
    terraform import azurerm_user_assigned_identity.container_apps "/subscriptions/$subscriptionId/resourceGroups/$appRg/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id-policycortex-dev" 2>$null
    Write-Host "✓ User Assigned Identity imported" -ForegroundColor Green
} catch {
    Write-Host "! User Assigned Identity import failed or already exists" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Importing Networking Resources ===" -ForegroundColor Cyan

# Import Virtual Network
Write-Host "Importing Virtual Network..." -ForegroundColor White
try {
    terraform import "module.networking.azurerm_virtual_network.main" "/subscriptions/$subscriptionId/resourceGroups/$networkRg/providers/Microsoft.Network/virtualNetworks/policycortex-dev-vnet" 2>$null
    Write-Host "✓ Virtual Network imported" -ForegroundColor Green
} catch {
    Write-Host "! Virtual Network import failed or already exists" -ForegroundColor Yellow
}

# Import Network Security Groups
$nsgs = @(
    @{Name="data_services"; Resource="policycortex-dev-nsg-data_services"},
    @{Name="private_endpoints"; Resource="policycortex-dev-nsg-private_endpoints"},
    @{Name="container_apps"; Resource="policycortex-dev-nsg-container_apps"},
    @{Name="ai_services"; Resource="policycortex-dev-nsg-ai_services"},
    @{Name="app_gateway"; Resource="policycortex-dev-nsg-app_gateway"}
)

foreach ($nsg in $nsgs) {
    Write-Host "Importing NSG: $($nsg.Resource)..." -ForegroundColor White
    try {
        terraform import "module.networking.azurerm_network_security_group.subnet_nsgs[`"$($nsg.Name)`"]" "/subscriptions/$subscriptionId/resourceGroups/$networkRg/providers/Microsoft.Network/networkSecurityGroups/$($nsg.Resource)" 2>$null
        Write-Host "✓ NSG $($nsg.Resource) imported" -ForegroundColor Green
    } catch {
        Write-Host "! NSG $($nsg.Resource) import failed or already exists" -ForegroundColor Yellow
    }
}

# Import Route Table
Write-Host "Importing Route Table..." -ForegroundColor White
try {
    terraform import "module.networking.azurerm_route_table.main" "/subscriptions/$subscriptionId/resourceGroups/$networkRg/providers/Microsoft.Network/routeTables/policycortex-dev-rt" 2>$null
    Write-Host "✓ Route Table imported" -ForegroundColor Green
} catch {
    Write-Host "! Route Table import failed or already exists" -ForegroundColor Yellow
}

# Import Network Watcher
Write-Host "Importing Network Watcher..." -ForegroundColor White
try {
    terraform import "module.networking.azurerm_network_watcher.main[0]" "/subscriptions/$subscriptionId/resourceGroups/$networkRg/providers/Microsoft.Network/networkWatchers/policycortex-dev-nw" 2>$null
    Write-Host "✓ Network Watcher imported" -ForegroundColor Green
} catch {
    Write-Host "! Network Watcher import failed or already exists" -ForegroundColor Yellow
}

# Import Private DNS Zones
$dnsZones = @(
    @{Name="internal"; Zone="policycortex.internal"},
    @{Name="sql"; Zone="privatelink.database.windows.net"},
    @{Name="cosmos"; Zone="privatelink.documents.azure.com"},
    @{Name="redis"; Zone="privatelink.redis.cache.windows.net"},
    @{Name="cognitive"; Zone="privatelink.cognitiveservices.azure.com"},
    @{Name="ml"; Zone="privatelink.api.azureml.ms"},
    @{Name="openai"; Zone="privatelink.openai.azure.com"}
)

foreach ($zone in $dnsZones) {
    Write-Host "Importing DNS Zone: $($zone.Zone)..." -ForegroundColor White
    try {
        terraform import "module.networking.azurerm_private_dns_zone.$($zone.Name)" "/subscriptions/$subscriptionId/resourceGroups/$networkRg/providers/Microsoft.Network/privateDnsZones/$($zone.Zone)" 2>$null
        Write-Host "✓ DNS Zone $($zone.Zone) imported" -ForegroundColor Green
    } catch {
        Write-Host "! DNS Zone $($zone.Zone) import failed or already exists" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== Import Phase 1 Complete ===" -ForegroundColor Green
Write-Host "Now we need to handle role assignments and other complex resources..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run: terraform plan -var-file=environments/dev/terraform.tfvars" -ForegroundColor White
Write-Host "2. Look for any remaining 'already exists' errors" -ForegroundColor White
Write-Host "3. Use 'az role assignment list' to get role assignment IDs for import" -ForegroundColor White
Write-Host ""

Write-Host "=== Running Terraform Plan to Check Status ===" -ForegroundColor Cyan
terraform plan -var-file=environments/dev/terraform.tfvars