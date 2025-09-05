@echo off
REM Script to completely recreate the environment from scratch
REM This proves Infrastructure as Code is working correctly

setlocal enabledelayedexpansion

REM Configuration
set ENVIRONMENT=%1
if "%ENVIRONMENT%"=="" set ENVIRONMENT=dev

set SUBSCRIPTION_ID=%2
if "%SUBSCRIPTION_ID%"=="" set SUBSCRIPTION_ID=6dc7cfa2-0332-4740-98b6-bac9f1a23de9

set DESTROY_FIRST=%3
if "%DESTROY_FIRST%"=="" set DESTROY_FIRST=false

echo ========================================
echo PolicyCortex Environment Recreation
echo ========================================
echo Environment: %ENVIRONMENT%
echo Subscription: %SUBSCRIPTION_ID%
echo Destroy First: %DESTROY_FIRST%
echo.

REM Set subscription
echo Setting Azure subscription...
call az account set --subscription "%SUBSCRIPTION_ID%"

REM Get resource names using consistent pattern
set PROJECT=cortex
set RG=rg-%PROJECT%-%ENVIRONMENT%

if "%DESTROY_FIRST%"=="true" (
    echo ========================================
    echo DESTROYING EXISTING ENVIRONMENT
    echo ========================================
    
    set /p confirm="Are you SURE you want to destroy the %ENVIRONMENT% environment? (yes/no): "
    if not "!confirm!"=="yes" (
        echo Destruction cancelled.
        exit /b 0
    )
    
    REM Delete resource group
    echo Deleting resource group %RG%...
    az group exists -n "%RG%" >nul 2>&1
    if %errorlevel%==0 (
        call az group delete -n "%RG%" --yes --no-wait
        echo Resource group deletion initiated.
        
        REM Wait for deletion
        echo Waiting for deletion to complete...
        :wait_delete
        az group exists -n "%RG%" >nul 2>&1
        if %errorlevel%==0 (
            timeout /t 10 /nobreak >nul
            goto wait_delete
        )
        echo Resource group deleted successfully.
    ) else (
        echo Resource group does not exist.
    )
    
    REM Clean Terraform state
    echo Cleaning Terraform state...
    cd infrastructure\terraform
    if exist .terraform rmdir /s /q .terraform
    if exist terraform.tfstate del terraform.tfstate*
    if exist .terraform.lock.hcl del .terraform.lock.hcl
    echo Terraform state cleaned.
)

echo ========================================
echo CREATING ENVIRONMENT FROM SCRATCH
echo ========================================

REM Navigate to Terraform directory
cd infrastructure\terraform

REM Create terraform.tfvars
echo Creating terraform.tfvars...
(
echo environment = "%ENVIRONMENT%"
echo location    = "eastus"
echo project_name = "%PROJECT%"
) > terraform.tfvars

REM Initialize Terraform
echo Initializing Terraform...
call terraform init

REM Plan Terraform
echo Planning Terraform deployment...
call terraform plan -out=tfplan

REM Apply Terraform
echo Applying Terraform configuration...
call terraform apply tfplan

REM Get outputs
echo Getting Terraform outputs...
call terraform output -json > outputs.json

echo ========================================
echo ENVIRONMENT CREATED SUCCESSFULLY!
echo ========================================
echo.

REM Trigger CI/CD pipeline
echo Triggering CI/CD pipeline to deploy applications...
call gh workflow run entry.yml ^
    --ref main ^
    -f full_run=true ^
    -f target_env="%ENVIRONMENT%"

echo Pipeline triggered. Check GitHub Actions for progress.
echo.
echo ========================================
echo RECREATION COMPLETE!
echo ========================================

endlocal