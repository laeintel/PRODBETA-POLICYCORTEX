@echo off
REM Verify environment configuration for PolicyCortex

echo ======================================
echo Verifying Environment Configuration
echo ======================================
echo.

echo [1] Checking Docker Compose files...
if exist docker-compose.yml (
    echo ✓ docker-compose.yml found
) else (
    echo ✗ docker-compose.yml missing
)

if exist docker-compose.dev.yml (
    echo ✓ docker-compose.dev.yml found
) else (
    echo ✗ docker-compose.dev.yml missing
)

if exist docker-compose.prod.yml (
    echo ✓ docker-compose.prod.yml found
) else (
    echo ✗ docker-compose.prod.yml missing
)
echo.

echo [2] Checking environment files...
if exist .env.dev (
    echo ✓ .env.dev found
    echo   Key variables:
    findstr /C:"AZURE_CLIENT_ID" .env.dev
    findstr /C:"AZURE_SUBSCRIPTION_ID" .env.dev
    findstr /C:"ACR_NAME" .env.dev
    findstr /C:"AKS_CLUSTER_NAME" .env.dev
) else (
    echo ✗ .env.dev missing
)
echo.

if exist .env.prod (
    echo ✓ .env.prod found
    echo   Key variables:
    findstr /C:"AZURE_CLIENT_ID" .env.prod
    findstr /C:"AZURE_SUBSCRIPTION_ID" .env.prod
    findstr /C:"ACR_NAME" .env.prod
    findstr /C:"AKS_CLUSTER_NAME" .env.prod
    findstr /C:"NEXT_PUBLIC_APP_URL" .env.prod
) else (
    echo ✗ .env.prod missing
)
echo.

echo [3] Checking GitHub workflow configurations...
echo.
echo DEV Environment Secrets (should be configured in GitHub):
echo - AZURE_CLIENT_ID_DEV: 1ecc95d1-e5bb-43e2-9324-30a17cb6b01c
echo - AZURE_SUBSCRIPTION_ID_DEV: 205b477d-17e7-4b3b-92c1-32cf02626b78
echo - ACR_NAME_DEV: crpcxdev
echo - AKS_CLUSTER_NAME_DEV: pcx42178531-aks
echo - AKS_RESOURCE_GROUP_DEV: pcx42178531-rg
echo.

echo PROD Environment Secrets (should be configured in GitHub):
echo - AZURE_CLIENT_ID_PROD: 8f0208b4-82b1-47cd-b02a-75e2f7afddb5
echo - AZURE_SUBSCRIPTION_ID_PROD: 9f16cc88-89ce-49ba-a96d-308ed3169595
echo - ACR_NAME_PROD: crcortexprodvb9v2h
echo - AKS_CLUSTER_NAME_PROD: policycortex-prod-aks
echo - AKS_RESOURCE_GROUP_PROD: policycortex-prod-rg
echo.

echo Common Secrets:
echo - AZURE_TENANT_ID: 9ef5b184-d371-462a-bc75-5024ce8baff7
echo.

echo [4] Checking production domain configuration...
findstr /C:"policycortex.com" .env.prod >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Production domain configured as policycortex.com
) else (
    echo ✗ Production domain not configured
)
echo.

echo [5] Checking Next.js configuration...
findstr /C:"policycortex.com" frontend\next.config.js >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Next.js configured for policycortex.com
) else (
    echo ✗ Next.js not configured for production domain
)
echo.

echo ======================================
echo Verification Complete
echo ======================================
echo.
echo IMPORTANT: Ensure all GitHub secrets listed above are configured
echo Run: scripts\setup-github-secrets.bat to configure them
echo.