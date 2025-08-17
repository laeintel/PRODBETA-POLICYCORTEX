#!/bin/bash
# Script to handle existing resources before terraform apply

set -e

echo "Handling existing resources..."

# Import tfstate resource group if it exists
RG_TFSTATE="rg-tfstate-cortex-dev"
if az group show --name "$RG_TFSTATE" 2>/dev/null; then
    echo "Importing existing tfstate resource group..."
    terraform import azurerm_resource_group.tfstate "/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${RG_TFSTATE}" || true
fi

# Import tfstate storage account if it exists
SA_TFSTATE="sttfcortexdev3p0bata"
if az storage account show --name "$SA_TFSTATE" --resource-group "$RG_TFSTATE" 2>/dev/null; then
    echo "Importing existing tfstate storage account..."
    terraform import azurerm_storage_account.tfstate "/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${RG_TFSTATE}/providers/Microsoft.Storage/storageAccounts/${SA_TFSTATE}" || true
fi

# Import tfstate container if it exists
if az storage container show --name "tfstate" --account-name "$SA_TFSTATE" 2>/dev/null; then
    echo "Importing existing tfstate container..."
    terraform import azurerm_storage_container.tfstate "https://${SA_TFSTATE}.blob.core.windows.net/tfstate" || true
fi

# Purge soft-deleted Cognitive Services if needed
COGNITIVE_NAME="cogao-cortex-dev"
RG_NAME="rg-cortex-dev"
echo "Checking for soft-deleted Cognitive Services..."

# First check if the resource exists in the resource group
if az cognitiveservices account show --name "$COGNITIVE_NAME" --resource-group "$RG_NAME" 2>/dev/null; then
    echo "Cognitive Services account exists, skipping purge check"
else
    # Check for soft-deleted accounts
    DELETED_ACCOUNTS=$(az cognitiveservices account list-deleted --query "[?name=='${COGNITIVE_NAME}']" -o json 2>/dev/null || echo "[]")
    if [ "$DELETED_ACCOUNTS" != "[]" ] && [ -n "$DELETED_ACCOUNTS" ]; then
        echo "Found soft-deleted Cognitive Services account: $COGNITIVE_NAME"
        echo "Purging soft-deleted account..."
        # Try to purge without specifying resource group (for soft-deleted resources)
        az cognitiveservices account purge --name "$COGNITIVE_NAME" --location "eastus" 2>/dev/null || \
        az cognitiveservices account purge --name "$COGNITIVE_NAME" --resource-group "$RG_NAME" --location "eastus" 2>/dev/null || \
        echo "Warning: Could not purge soft-deleted account, may need manual intervention"
    fi
fi

echo "Resource handling complete."