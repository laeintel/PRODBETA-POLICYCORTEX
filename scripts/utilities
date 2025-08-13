@echo off
echo ===============================================
echo PolicyCortex - Connect to Real Azure Data
echo ===============================================
echo.

REM Check if Azure CLI is logged in
az account show >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Not logged into Azure CLI
    echo Please run: az login
    exit /b 1
)

echo [OK] Azure CLI connected
az account show --query "{Subscription:name, ID:id}" -o table

REM Set environment variables for Azure connection
set AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
set USE_REAL_AZURE=true

echo.
echo [OK] Environment configured for real Azure data
echo.
echo Starting API with Azure connection...
cd backend\services\api_gateway
python main.py