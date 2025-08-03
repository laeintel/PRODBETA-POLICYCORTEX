#\!/bin/bash

echo "🔍 Checking for stuck resources..."

# Check for any existing deployments that might be stuck
echo "Checking for running deployments..."
RUNNING_DEPLOYMENTS=$(az deployment sub list --query "[?properties.provisioningState=='Running'].name" -o tsv)

if [ \! -z "$RUNNING_DEPLOYMENTS" ]; then
    echo "Found running deployments:"
    echo "$RUNNING_DEPLOYMENTS"
    echo "These might be blocking new deployments."
fi

# Check for resource groups that might exist
echo "Checking for existing resource groups..."
az group list --query "[?contains(name, 'policortex')]" --output table

# Check for any resources in soft-deleted state
echo "Checking for soft-deleted Key Vaults..."
az keyvault list-deleted --query "[?properties.deletionDate \!= null && contains(name, 'pcx001')]" --output table

echo "✅ Resource check complete"
