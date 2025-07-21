#!/bin/bash
# Import Role Assignments Script
# This script handles the complex role assignment imports that need dynamic IDs

echo "=== PolicyCortex Role Assignments Import Script ==="
echo "This script imports role assignments with their actual IDs"
echo ""

# Check if we're in the correct directory
if [ ! -f "main.tf" ]; then
    echo "Error: Please run this script from the infrastructure/terraform directory"
    exit 1
fi

# Set Azure CLI environment
export ARM_USE_CLI=true

echo "=== Getting Role Assignment IDs from Azure ==="

# Get Key Vault admin role assignment ID
echo "Finding Key Vault Admin role assignment..."
KEYVAULT_ADMIN_ID=$(az role assignment list \
    --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-app-dev/providers/Microsoft.KeyVault/vaults/kvpolicycortexdevv2" \
    --query "[?roleDefinitionName=='Key Vault Administrator' && principalId=='178e2973-bb20-49da-ab80-0d1ddc7b0649'].id" \
    --output tsv 2>/dev/null)

if [ -n "$KEYVAULT_ADMIN_ID" ]; then
    echo "Found Key Vault Admin role assignment: $KEYVAULT_ADMIN_ID"
    if terraform import azurerm_role_assignment.key_vault_admin_current_client "$KEYVAULT_ADMIN_ID" 2>/dev/null; then
        echo "✓ Key Vault Admin role assignment imported"
    else
        echo "! Key Vault Admin role assignment import failed"
    fi
else
    echo "! Key Vault Admin role assignment not found"
fi

# Get Container Apps related role assignments
echo ""
echo "Finding Container Apps role assignments..."

# Get user assigned identity principal ID
IDENTITY_PRINCIPAL_ID=$(az identity show \
    --name "id-policycortex-dev" \
    --resource-group "rg-policycortex-app-dev" \
    --query principalId \
    --output tsv 2>/dev/null)

if [ -n "$IDENTITY_PRINCIPAL_ID" ]; then
    echo "Container Apps identity principal ID: $IDENTITY_PRINCIPAL_ID"
    
    # Find and import various role assignments for the container apps identity
    echo "Finding Key Vault Secrets User role assignment..."
    KV_SECRETS_ROLE_ID=$(az role assignment list \
        --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-app-dev/providers/Microsoft.KeyVault/vaults/kvpolicycortexdevv2" \
        --query "[?roleDefinitionName=='Key Vault Secrets User' && principalId=='$IDENTITY_PRINCIPAL_ID'].id" \
        --output tsv 2>/dev/null)
    
    if [ -n "$KV_SECRETS_ROLE_ID" ]; then
        echo "Found Key Vault Secrets User role: $KV_SECRETS_ROLE_ID"
        if terraform import azurerm_role_assignment.key_vault_secrets_user_container_apps "$KV_SECRETS_ROLE_ID" 2>/dev/null; then
            echo "✓ Key Vault Secrets User role assignment imported"
        else
            echo "! Key Vault Secrets User role assignment import failed"
        fi
    fi
    
    # Find Resource Group Reader role
    echo "Finding Resource Group Reader role assignment..."
    RG_READER_ROLE_ID=$(az role assignment list \
        --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-app-dev" \
        --query "[?roleDefinitionName=='Reader' && principalId=='$IDENTITY_PRINCIPAL_ID'].id" \
        --output tsv 2>/dev/null)
    
    if [ -n "$RG_READER_ROLE_ID" ]; then
        echo "Found Resource Group Reader role: $RG_READER_ROLE_ID"
        if terraform import azurerm_role_assignment.container_apps_rg_reader "$RG_READER_ROLE_ID" 2>/dev/null; then
            echo "✓ Resource Group Reader role assignment imported"
        else
            echo "! Resource Group Reader role assignment import failed"
        fi
    fi
    
    # Find Log Analytics Contributor role
    echo "Finding Log Analytics Contributor role assignment..."
    LA_CONTRIBUTOR_ROLE_ID=$(az role assignment list \
        --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-app-dev/providers/Microsoft.OperationalInsights/workspaces/law-policycortex-dev" \
        --query "[?roleDefinitionName=='Log Analytics Contributor' && principalId=='$IDENTITY_PRINCIPAL_ID'].id" \
        --output tsv 2>/dev/null)
    
    if [ -n "$LA_CONTRIBUTOR_ROLE_ID" ]; then
        echo "Found Log Analytics Contributor role: $LA_CONTRIBUTOR_ROLE_ID"
        if terraform import azurerm_role_assignment.container_apps_log_analytics "$LA_CONTRIBUTOR_ROLE_ID" 2>/dev/null; then
            echo "✓ Log Analytics Contributor role assignment imported"
        else
            echo "! Log Analytics Contributor role assignment import failed"
        fi
    fi
    
else
    echo "! Could not find Container Apps identity principal ID"
fi

echo ""
echo "=== Role Assignment Import Complete ==="
echo "Running terraform plan to check remaining issues..."
echo ""

terraform plan -var-file=environments/dev/terraform.tfvars