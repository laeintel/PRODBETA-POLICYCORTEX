#!/bin/bash
# Fix Terraform State Issues Script
# This script resolves the Azure resource conflicts preventing Terraform deployment

echo "=== PolicyCortex Terraform State Fix Script ==="
echo "This script will fix the resource conflicts in the Terraform state"
echo ""

# Check if we're in the correct directory
if [ ! -f "main.tf" ]; then
    echo "Error: Please run this script from the infrastructure/terraform directory"
    exit 1
fi

# Set Azure CLI environment to avoid path issues
export ARM_USE_CLI=true

echo "Step 1: Importing existing Key Vault access policy..."
if terraform import azurerm_key_vault_access_policy.current_client "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-app-dev/providers/Microsoft.KeyVault/vaults/kvpolicycortexdevv2/objectId/178e2973-bb20-49da-ab80-0d1ddc7b0649" 2>/dev/null; then
    echo "✓ Key Vault access policy imported successfully"
else
    echo "! Key Vault access policy import failed or already imported"
fi

echo ""
echo "Step 2: Importing existing role assignment..."
# Get the role assignment ID first
ROLE_ASSIGNMENT_ID=$(az role assignment list --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-app-dev/providers/Microsoft.KeyVault/vaults/kvpolicycortexdevv2" --query "[?roleDefinitionName=='Key Vault Secrets User' && principalId=='8b5f65a9-0033-4e50-8dd8-ac8f1630cf39'].id" --output tsv 2>/dev/null)

if [ -n "$ROLE_ASSIGNMENT_ID" ]; then
    if terraform import azurerm_role_assignment.key_vault_secrets_user_container_apps "$ROLE_ASSIGNMENT_ID" 2>/dev/null; then
        echo "✓ Role assignment imported successfully"
    else
        echo "! Role assignment import failed"
    fi
else
    echo "! Role assignment not found or not yet created"
fi

echo ""
echo "Step 3: Importing existing SQL admin password secret..."
if terraform import "module.data_services.azurerm_key_vault_secret.sql_admin_password" "https://kvpolicycortexdevv2.vault.azure.net/secrets/sql-admin-password/58a1f2872a3d480f8ec6b4d8c3ae2283" 2>/dev/null; then
    echo "✓ SQL admin password secret imported successfully"
else
    echo "! SQL admin password secret import failed or already imported"
fi

echo ""
echo "Step 4: Running Terraform plan to check for remaining issues..."
terraform plan -var-file=environments/dev/terraform.tfvars

echo ""
echo "Step 5: If no errors above, run Terraform apply..."
read -p "Do you want to proceed with terraform apply? (y/N): " confirmation
if [[ "$confirmation" =~ ^[Yy]$ ]]; then
    terraform apply -var-file=environments/dev/terraform.tfvars -auto-approve
    echo "✓ Terraform apply completed"
else
    echo "Skipped terraform apply. Run manually when ready:"
    echo "terraform apply -var-file=environments/dev/terraform.tfvars"
fi

echo ""
echo "=== Fix Script Completed ==="
echo "If you still see issues, check the Azure portal for any soft-deleted resources"