@echo off
REM Setup GitHub Secrets for PolicyCortex Dev and Prod Environments
REM This script configures all necessary secrets for AKS deployment

echo Setting up GitHub secrets for PolicyCortex...
echo.

REM Common secrets (same for both environments)
echo Setting common secrets...
gh secret set AZURE_TENANT_ID --body "9ef5b184-d371-462a-bc75-5024ce8baff7" -R laeintel/policycortex

REM Dev Environment Secrets
echo.
echo Setting DEV environment secrets...
gh secret set AZURE_CLIENT_ID_DEV --body "1ecc95d1-e5bb-43e2-9324-30a17cb6b01c" -R laeintel/policycortex
gh secret set AZURE_SUBSCRIPTION_ID_DEV --body "205b477d-17e7-4b3b-92c1-32cf02626b78" -R laeintel/policycortex

REM Prod Environment Secrets  
echo.
echo Setting PROD environment secrets...
gh secret set AZURE_CLIENT_ID_PROD --body "8f0208b4-82b1-47cd-b02a-75e2f7afddb5" -R laeintel/policycortex
gh secret set AZURE_SUBSCRIPTION_ID_PROD --body "9f16cc88-89ce-49ba-a96d-308ed3169595" -R laeintel/policycortex

REM AKS Related Secrets
echo.
echo Setting AKS related secrets...
gh secret set ACR_NAME_DEV --body "crpcxdev" -R laeintel/policycortex
gh secret set ACR_NAME_PROD --body "crcortexprodvb9v2h" -R laeintel/policycortex
gh secret set AKS_RESOURCE_GROUP_DEV --body "pcx42178531-rg" -R laeintel/policycortex
gh secret set AKS_CLUSTER_NAME_DEV --body "pcx42178531-aks" -R laeintel/policycortex

REM Clean up old/deprecated secrets
echo.
echo Removing deprecated secrets...
echo Please manually remove these deprecated secrets from GitHub:
echo - AZURE_CLIENT_ID (replaced by AZURE_CLIENT_ID_DEV/PROD)
echo - AZURE_SUBSCRIPTION_ID (replaced by AZURE_SUBSCRIPTION_ID_DEV/PROD)
echo - AZURE_CREDENTIALS (replaced by OIDC authentication)
echo - AZURE_CLIENT_SECRET (using OIDC instead)

echo.
echo Next steps:
echo 1. Add AZURE_CLIENT_SECRET_DEV if using service principal auth (or setup OIDC)
echo 2. Add AZURE_SUBSCRIPTION_ID_PROD when you have the prod subscription
echo 3. Add AZURE_CLIENT_SECRET_PROD if needed
echo 4. Configure federated credentials for OIDC authentication
echo.
echo Done!