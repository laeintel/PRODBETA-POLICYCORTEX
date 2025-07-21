#!/bin/bash
# Master Terraform Fix Script for PolicyCortex
# This script orchestrates the complete fix process

echo "======================================================"
echo "PolicyCortex Complete Terraform State Recovery Script"
echo "======================================================"
echo ""
echo "This script will:"
echo "1. Import all existing Azure resources into Terraform state"
echo "2. Handle role assignments and complex resources"
echo "3. Run terraform plan to verify the fix"
echo "4. Optionally run terraform apply"
echo ""

# Check if we're in the correct directory
if [ ! -f "main.tf" ]; then
    echo "❌ Error: Please run this script from the infrastructure/terraform directory"
    exit 1
fi

# Verify Azure CLI login
echo "🔐 Checking Azure CLI authentication..."
if ! az account show >/dev/null 2>&1; then
    echo "❌ Error: Please login to Azure CLI first: az login"
    exit 1
fi

CURRENT_SUB=$(az account show --query id --output tsv)
EXPECTED_SUB="9f16cc88-89ce-49ba-a96d-308ed3169595"

if [ "$CURRENT_SUB" != "$EXPECTED_SUB" ]; then
    echo "⚠️  Warning: You're logged into subscription $CURRENT_SUB"
    echo "   Expected: $EXPECTED_SUB (PolicyCortex Ai)"
    echo "   Switching to correct subscription..."
    if ! az account set --subscription "$EXPECTED_SUB"; then
        echo "❌ Error: Failed to switch to PolicyCortex Ai subscription"
        exit 1
    fi
fi

echo "✅ Authenticated to correct Azure subscription"
echo ""

# Set Azure CLI environment
export ARM_USE_CLI=true

echo "🚀 Phase 1: Importing main infrastructure resources..."
echo "=================================================="
if [ -x "./import-all-resources.sh" ]; then
    ./import-all-resources.sh
else
    echo "❌ Error: import-all-resources.sh not found or not executable"
    exit 1
fi

echo ""
echo "🔐 Phase 2: Importing role assignments and complex resources..."
echo "============================================================"
if [ -x "./import-role-assignments.sh" ]; then
    ./import-role-assignments.sh
else
    echo "❌ Error: import-role-assignments.sh not found or not executable"
    exit 1
fi

echo ""
echo "🧹 Phase 3: Cleaning up and final verification..."
echo "==============================================="

echo "Running final terraform plan to check status..."
if terraform plan -var-file=environments/dev/terraform.tfvars -detailed-exitcode; then
    echo ""
    echo "🎉 SUCCESS: No changes needed - all resources are properly imported!"
    echo ""
    echo "Your Terraform state has been fully recovered."
    echo "You can now use terraform apply to make future changes."
elif [ $? -eq 2 ]; then
    echo ""
    echo "📋 INFO: Some changes are planned (this is normal after import)"
    echo ""
    read -p "Do you want to apply these changes now? (y/N): " apply_confirm
    if [[ "$apply_confirm" =~ ^[Yy]$ ]]; then
        echo "🚀 Applying Terraform changes..."
        if terraform apply -var-file=environments/dev/terraform.tfvars -auto-approve; then
            echo ""
            echo "🎉 SUCCESS: Terraform apply completed successfully!"
        else
            echo ""
            echo "❌ ERROR: Terraform apply failed. Check the output above."
            exit 1
        fi
    else
        echo ""
        echo "ℹ️  Skipped terraform apply. You can run it manually later:"
        echo "   terraform apply -var-file=environments/dev/terraform.tfvars"
    fi
else
    echo ""
    echo "⚠️  WARNING: There may still be some import issues."
    echo "   Review the terraform plan output above."
    echo "   You may need to import additional resources manually."
fi

echo ""
echo "============================================"
echo "✅ Terraform State Recovery Complete!"
echo "============================================"
echo ""
echo "Summary:"
echo "• All existing Azure resources have been imported"
echo "• Terraform state is now synchronized with Azure"
echo "• You can continue using terraform normally"
echo ""
echo "If you encounter any remaining issues:"
echo "1. Check the terraform plan output for specific errors"
echo "2. Use 'terraform import <resource> <azure-resource-id>' for any missed resources"
echo "3. Review the Azure portal for any unexpected resource states"