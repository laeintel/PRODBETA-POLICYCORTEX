@echo off
REM Simple batch script to populate Key Vault with essential secrets
REM Run this script after infrastructure deployment

set ENVIRONMENT=dev
set RESOURCE_GROUP=rg-policycortex001-app-%ENVIRONMENT%
set KEY_VAULT_NAME=kvpolicycortex001%ENVIRONMENT%

echo Setting up Key Vault secrets for PolicyCortex %ENVIRONMENT%
echo Resource Group: %RESOURCE_GROUP%
echo Key Vault: %KEY_VAULT_NAME%

REM Check if Key Vault exists
az keyvault show --name %KEY_VAULT_NAME% --resource-group %RESOURCE_GROUP% >nul 2>&1
if errorlevel 1 (
    echo ERROR: Key Vault %KEY_VAULT_NAME% not found
    echo Please ensure infrastructure is deployed first
    exit /b 1
)

echo Key Vault found, setting up secrets...

REM Generate random secrets for JWT and encryption
for /f %%i in ('powershell -command "[Convert]::ToBase64String([System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes(32))"') do set JWT_SECRET=%%i
for /f %%i in ('powershell -command "[Convert]::ToBase64String([System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes(32))"') do set ENCRYPTION_KEY=%%i

REM Set authentication secrets
echo Setting authentication secrets...
az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "jwt-secret" --value "%JWT_SECRET%" --output none
az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "encryption-key" --value "%ENCRYPTION_KEY%" --output none

REM Set Azure AD secrets (use environment variables if set, otherwise defaults)
if not defined AZURE_CLIENT_ID set AZURE_CLIENT_ID=e8c5b8a0-123e-4567-8901-234567890123
if not defined AZURE_TENANT_ID set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7

echo Setting Azure AD secrets...
az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "azure-client-id" --value "%AZURE_CLIENT_ID%" --output none
az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "azure-tenant-id" --value "%AZURE_TENANT_ID%" --output none

REM Try to get real Azure resource values
echo Retrieving Azure resource details...

REM Cosmos DB
for /f "tokens=*" %%i in ('az cosmosdb list --resource-group %RESOURCE_GROUP% --query "[0].name" -o tsv 2^>nul') do set COSMOS_ACCOUNT=%%i
if defined COSMOS_ACCOUNT (
    echo Found Cosmos DB: %COSMOS_ACCOUNT%
    for /f "tokens=*" %%i in ('az cosmosdb keys list --name %COSMOS_ACCOUNT% --resource-group %RESOURCE_GROUP% --query "primaryMasterKey" -o tsv') do set COSMOS_KEY=%%i
    set COSMOS_ENDPOINT=https://%COSMOS_ACCOUNT%.documents.azure.com:443/
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "cosmos-endpoint" --value "!COSMOS_ENDPOINT!" --output none
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "cosmos-key" --value "%COSMOS_KEY%" --output none
) else (
    echo No Cosmos DB found, using placeholders
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "cosmos-endpoint" --value "https://placeholder-cosmos.documents.azure.com:443/" --output none
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "cosmos-key" --value "placeholder-cosmos-key" --output none
)

REM Redis
for /f "tokens=*" %%i in ('az redis list --resource-group %RESOURCE_GROUP% --query "[0].name" -o tsv 2^>nul') do set REDIS_NAME=%%i
if defined REDIS_NAME (
    echo Found Redis: %REDIS_NAME%
    for /f "tokens=*" %%i in ('az redis list-keys --name %REDIS_NAME% --resource-group %RESOURCE_GROUP% --query "primaryKey" -o tsv') do set REDIS_KEY=%%i
    for /f "tokens=*" %%i in ('az redis show --name %REDIS_NAME% --resource-group %RESOURCE_GROUP% --query "hostName" -o tsv') do set REDIS_HOST=%%i
    for /f "tokens=*" %%i in ('az redis show --name %REDIS_NAME% --resource-group %RESOURCE_GROUP% --query "sslPort" -o tsv') do set REDIS_PORT=%%i
    set REDIS_CONNECTION_STRING=%REDIS_HOST%:%REDIS_PORT%,password=%REDIS_KEY%,ssl=True,abortConnect=False
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "redis-connection-string" --value "!REDIS_CONNECTION_STRING!" --output none
) else (
    echo No Redis found, using placeholder
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "redis-connection-string" --value "localhost:6379,password=placeholder-redis-key" --output none
)

REM Storage Account
for /f "tokens=*" %%i in ('az storage account list --resource-group %RESOURCE_GROUP% --query "[0].name" -o tsv 2^>nul') do set STORAGE_ACCOUNT=%%i
if defined STORAGE_ACCOUNT (
    echo Found Storage Account: %STORAGE_ACCOUNT%
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "storage-account-name" --value "%STORAGE_ACCOUNT%" --output none
) else (
    echo No Storage Account found, using placeholder
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "storage-account-name" --value "placeholder-storage" --output none
)

REM Cognitive Services
for /f "tokens=*" %%i in ('az cognitiveservices account list --resource-group %RESOURCE_GROUP% --query "[0].name" -o tsv 2^>nul') do set COGNITIVE_ACCOUNT=%%i
if defined COGNITIVE_ACCOUNT (
    echo Found Cognitive Services: %COGNITIVE_ACCOUNT%
    for /f "tokens=*" %%i in ('az cognitiveservices account keys list --name %COGNITIVE_ACCOUNT% --resource-group %RESOURCE_GROUP% --query "key1" -o tsv') do set COGNITIVE_KEY=%%i
    for /f "tokens=*" %%i in ('az cognitiveservices account show --name %COGNITIVE_ACCOUNT% --resource-group %RESOURCE_GROUP% --query "properties.endpoint" -o tsv') do set COGNITIVE_ENDPOINT=%%i
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "cognitive-services-key" --value "%COGNITIVE_KEY%" --output none
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "cognitive-services-endpoint" --value "%COGNITIVE_ENDPOINT%" --output none
) else (
    echo No Cognitive Services found, using placeholders
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "cognitive-services-key" --value "placeholder-cognitive-key" --output none
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "cognitive-services-endpoint" --value "https://placeholder-cognitive.cognitiveservices.azure.com/" --output none
)

REM Application Insights
for /f "tokens=*" %%i in ('az monitor app-insights component list --resource-group %RESOURCE_GROUP% --query "[0].name" -o tsv 2^>nul') do set APP_INSIGHTS=%%i
if defined APP_INSIGHTS (
    echo Found Application Insights: %APP_INSIGHTS%
    for /f "tokens=*" %%i in ('az monitor app-insights component show --app %APP_INSIGHTS% --resource-group %RESOURCE_GROUP% --query "connectionString" -o tsv') do set APP_INSIGHTS_CONNECTION=%%i
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "application-insights-connection-string" --value "!APP_INSIGHTS_CONNECTION!" --output none
) else (
    echo No Application Insights found, using placeholder
    az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "application-insights-connection-string" --value "InstrumentationKey=placeholder-key" --output none
)

echo.
echo ✓ Key Vault secrets setup completed!
echo.
echo Summary of secrets:
az keyvault secret list --vault-name %KEY_VAULT_NAME% --query "[].name" -o tsv

echo.
echo Next steps:
echo 1. Redeploy Container Apps to pick up the new secrets
echo 2. Update Azure AD secrets with real values if needed:
echo    set AZURE_CLIENT_ID=your-real-client-id
echo    set AZURE_TENANT_ID=your-real-tenant-id
echo    setup-keyvault-simple.bat

pause