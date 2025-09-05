@echo off
REM Script to completely destroy ALL resources in the environment
REM WARNING: This will delete EVERYTHING!

setlocal enabledelayedexpansion

REM Configuration
set ENVIRONMENT=%1
if "%ENVIRONMENT%"=="" set ENVIRONMENT=dev

set SUBSCRIPTION_ID=%2
if "%SUBSCRIPTION_ID%"=="" set SUBSCRIPTION_ID=6dc7cfa2-0332-4740-98b6-bac9f1a23de9

echo ========================================
echo WARNING: COMPLETE RESOURCE DESTRUCTION
echo ========================================
echo Environment: %ENVIRONMENT%
echo Subscription: %SUBSCRIPTION_ID%
echo.

REM Confirm destruction
set /p confirm="Are you ABSOLUTELY SURE you want to destroy ALL resources? Type 'DESTROY' to confirm: "
if not "%confirm%"=="DESTROY" (
    echo Destruction cancelled.
    exit /b 0
)

REM Set subscription
echo Setting Azure subscription...
call az account set --subscription "%SUBSCRIPTION_ID%"

REM Delete resource groups
echo.
echo Deleting resource groups...

REM Main resource group
echo Deleting rg-cortex-%ENVIRONMENT%...
call az group delete -n "rg-cortex-%ENVIRONMENT%" --yes --no-wait 2>nul

REM Terraform state resource group
echo Deleting rg-tfstate-cortex-%ENVIRONMENT%...
call az group delete -n "rg-tfstate-cortex-%ENVIRONMENT%" --yes --no-wait 2>nul

REM AI Foundry resource group
echo Deleting policycortex-gpt4o-resource...
call az group delete -n "policycortex-gpt4o-resource" --yes --no-wait 2>nul

REM Other resource groups
echo Deleting rg-datafactory-demo...
call az group delete -n "rg-datafactory-demo" --yes --no-wait 2>nul

echo Deleting rg-hklaw-datasource-prod...
call az group delete -n "rg-hklaw-datasource-prod" --yes --no-wait 2>nul

echo.
echo Waiting for deletions to complete...

:wait_loop
timeout /t 10 /nobreak >nul
az group exists -n "rg-cortex-%ENVIRONMENT%" >nul 2>&1
if %errorlevel%==0 goto wait_loop

echo.
echo ========================================
echo DESTRUCTION COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Run the Terraform pipeline to recreate everything
echo 2. Or use: terraform apply -var="environment=%ENVIRONMENT%"
echo.

endlocal