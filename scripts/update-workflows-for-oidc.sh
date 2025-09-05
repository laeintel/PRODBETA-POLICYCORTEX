#!/bin/bash
# Update all GitHub workflows to use new OIDC authentication

echo "Updating GitHub workflows for OIDC authentication..."

# Update application.yml
sed -i 's/AZURE_SUBSCRIPTION_ID_DEV || '\''6dc7cfa2-0332-4740-98b6-bac9f1a23de9'\''/AZURE_SUBSCRIPTION_ID_DEV/g' .github/workflows/application.yml
sed -i 's/AZURE_SUBSCRIPTION_ID_PROD || '\''6dc7cfa2-0332-4740-98b6-bac9f1a23de9'\''/AZURE_SUBSCRIPTION_ID_PROD/g' .github/workflows/application.yml
sed -i 's/fromJSON(secrets.AZURE_CREDENTIALS).clientId/secrets.AZURE_CLIENT_ID_DEV/g' .github/workflows/application.yml
sed -i 's/fromJSON(secrets.AZURE_CREDENTIALS).tenantId/secrets.AZURE_TENANT_ID/g' .github/workflows/application.yml
sed -i 's/fromJSON(secrets.AZURE_CREDENTIALS).clientSecret//g' .github/workflows/application.yml
sed -i 's/AZURE_CLIENT_SECRET: ${{ }}/# AZURE_CLIENT_SECRET not needed with OIDC/g' .github/workflows/application.yml

# Update azure-infra.yml  
sed -i 's/AZURE_CLIENT_ID/AZURE_CLIENT_ID_DEV/g' .github/workflows/azure-infra.yml
sed -i 's/AZURE_SUBSCRIPTION_ID/AZURE_SUBSCRIPTION_ID_DEV/g' .github/workflows/azure-infra.yml

# Update monorepo-ci-entry.yml if it exists
if [ -f ".github/workflows/monorepo-ci-entry.yml" ]; then
  sed -i 's/AZURE_CLIENT_ID/AZURE_CLIENT_ID_DEV/g' .github/workflows/monorepo-ci-entry.yml
  sed -i 's/AZURE_SUBSCRIPTION_ID/AZURE_SUBSCRIPTION_ID_DEV/g' .github/workflows/monorepo-ci-entry.yml
fi

echo "Workflow updates complete!"