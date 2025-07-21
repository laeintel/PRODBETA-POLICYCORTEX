# Fix Terraform State Issues Script
# This script resolves the Azure resource conflicts preventing Terraform deployment

Write-Host "=== PolicyCortex Terraform State Fix Script ===" -ForegroundColor Green
Write-Host "This script will fix the resource conflicts in the Terraform state" -ForegroundColor Yellow
Write-Host ""

# Check if we're in the correct directory
if (!(Test-Path "main.tf")) {
    Write-Host "Error: Please run this script from the infrastructure/terraform directory" -ForegroundColor Red
    exit 1
}

# Set Azure CLI environment to avoid path issues
$env:ARM_USE_CLI = "true"

Write-Host "Step 1: Importing existing Key Vault access policy..." -ForegroundColor Cyan
try {
    terraform import azurerm_key_vault_access_policy.current_client "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-app-dev/providers/Microsoft.KeyVault/vaults/kvpolicycortexdevv2/objectId/178e2973-bb20-49da-ab80-0d1ddc7b0649"
    Write-Host "✓ Key Vault access policy imported successfully" -ForegroundColor Green
} catch {
    Write-Host "! Key Vault access policy import failed or already imported: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Step 2: Importing existing role assignment..." -ForegroundColor Cyan
try {
    # Get the role assignment ID first
    $roleAssignmentId = az role assignment list --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-app-dev/providers/Microsoft.KeyVault/vaults/kvpolicycortexdevv2" --query "[?roleDefinitionName=='Key Vault Secrets User' && principalId=='8b5f65a9-0033-4e50-8dd8-ac8f1630cf39'].id" --output tsv
    
    if ($roleAssignmentId) {
        terraform import azurerm_role_assignment.key_vault_secrets_user_container_apps $roleAssignmentId
        Write-Host "✓ Role assignment imported successfully" -ForegroundColor Green
    } else {
        Write-Host "! Role assignment not found or not yet created" -ForegroundColor Yellow
    }
} catch {
    Write-Host "! Role assignment import failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Step 3: Importing existing SQL admin password secret..." -ForegroundColor Cyan
try {
    terraform import "module.data_services.azurerm_key_vault_secret.sql_admin_password" "https://kvpolicycortexdevv2.vault.azure.net/secrets/sql-admin-password/58a1f2872a3d480f8ec6b4d8c3ae2283"
    Write-Host "✓ SQL admin password secret imported successfully" -ForegroundColor Green
} catch {
    Write-Host "! SQL admin password secret import failed or already imported: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Step 4: Running Terraform plan to check for remaining issues..." -ForegroundColor Cyan
terraform plan -var-file=environments/dev/terraform.tfvars

Write-Host ""
Write-Host "Step 5: If no errors above, run Terraform apply..." -ForegroundColor Cyan
$confirmation = Read-Host "Do you want to proceed with terraform apply? (y/N)"
if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
    terraform apply -var-file=environments/dev/terraform.tfvars -auto-approve
    Write-Host "✓ Terraform apply completed" -ForegroundColor Green
} else {
    Write-Host "Skipped terraform apply. Run manually when ready:" -ForegroundColor Yellow
    Write-Host "terraform apply -var-file=environments/dev/terraform.tfvars" -ForegroundColor White
}

Write-Host ""
Write-Host "=== Fix Script Completed ===" -ForegroundColor Green
Write-Host "If you still see issues, check the Azure portal for any soft-deleted resources" -ForegroundColor Yellow