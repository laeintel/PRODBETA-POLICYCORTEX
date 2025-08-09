@echo off
REM Script to import existing Azure resources into Terraform state (Windows version)
REM This prevents "already exists" errors when resources were created outside Terraform

setlocal enabledelayedexpansion

REM Get environment from argument or default to dev
set ENV=%1
if "%ENV%"=="" set ENV=dev

set SUBSCRIPTION_ID=%AZURE_SUBSCRIPTION_ID%
if "%SUBSCRIPTION_ID%"=="" set SUBSCRIPTION_ID=%2

if "%SUBSCRIPTION_ID%"=="" (
    echo Error: AZURE_SUBSCRIPTION_ID not set
    exit /b 1
)

echo Starting resource import for environment: %ENV%
echo Subscription ID: %SUBSCRIPTION_ID%

REM Ensure correct subscription context
az account set --subscription "%SUBSCRIPTION_ID%" >nul 2>&1

REM Container App - Core
set CA_CORE_NAME=ca-cortex-core-%ENV%
set CA_CORE_ID=/subscriptions/%SUBSCRIPTION_ID%/resourceGroups/rg-cortex-%ENV%/providers/Microsoft.App/containerApps/%CA_CORE_NAME%

echo.
echo Checking Container App Core: %CA_CORE_NAME%...

REM Check if resource exists in state
terraform state show azurerm_container_app.core >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   Already in state
) else (
    REM Check if the resource exists in Azure
    az resource show --ids "%CA_CORE_ID%" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo   Resource exists in Azure, importing...
        terraform import azurerm_container_app.core "%CA_CORE_ID%"
        if !ERRORLEVEL! EQU 0 (
            echo   Imported successfully
        ) else (
            echo   Import failed - manual intervention may be required
        )
    ) else (
        echo   Resource not found in Azure
    )
)

REM Container App - Frontend
set CA_FRONTEND_NAME=ca-cortex-frontend-%ENV%
set CA_FRONTEND_ID=/subscriptions/%SUBSCRIPTION_ID%/resourceGroups/rg-cortex-%ENV%/providers/Microsoft.App/containerApps/%CA_FRONTEND_NAME%

echo.
echo Checking Container App Frontend: %CA_FRONTEND_NAME%...

REM Check if resource exists in state
terraform state show azurerm_container_app.frontend >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   Already in state
) else (
    REM Check if the resource exists in Azure
    az resource show --ids "%CA_FRONTEND_ID%" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo   Resource exists in Azure, importing...
        terraform import azurerm_container_app.frontend "%CA_FRONTEND_ID%"
        if !ERRORLEVEL! EQU 0 (
            echo   Imported successfully
        ) else (
            echo   Import failed - manual intervention may be required
        )
    ) else (
        echo   Resource not found in Azure
    )
)

REM Container Apps Environment
set CAE_NAME=cae-cortex-%ENV%
set CAE_ID=/subscriptions/%SUBSCRIPTION_ID%/resourceGroups/rg-cortex-%ENV%/providers/Microsoft.App/managedEnvironments/%CAE_NAME%

echo.
echo Checking Container Apps Environment: %CAE_NAME%...

REM Check if resource exists in state
terraform state show azurerm_container_app_environment.main >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   Already in state
) else (
    REM Check if the resource exists in Azure
    az resource show --ids "%CAE_ID%" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo   Resource exists in Azure, importing...
        terraform import azurerm_container_app_environment.main "%CAE_ID%"
        if !ERRORLEVEL! EQU 0 (
            echo   Imported successfully
        ) else (
            echo   Import failed - manual intervention may be required
        )
    ) else (
        echo   Resource not found in Azure
    )
)

echo.
echo Import scan complete for Container Apps!
echo Run 'terraform plan' to verify the state.

endlocal