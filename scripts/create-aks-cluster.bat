@echo off
REM Create AKS cluster for PolicyCortex

echo ======================================
echo Creating AKS Cluster for Development
echo ======================================
echo.

REM Set variables
set SUBSCRIPTION_ID=6dc7cfa2-0332-4740-98b6-bac9f1a23de9
set RESOURCE_GROUP=rg-cortex-dev
set AKS_NAME=cortex-dev-aks
set ACR_NAME=crcortexdev3p0bata
set LOCATION=eastus

echo Using:
echo - Subscription: %SUBSCRIPTION_ID%
echo - Resource Group: %RESOURCE_GROUP%
echo - AKS Name: %AKS_NAME%
echo - ACR: %ACR_NAME%
echo - Location: %LOCATION%
echo.

REM Set subscription
echo Setting subscription...
az account set --subscription %SUBSCRIPTION_ID%

REM Create AKS cluster
echo Creating AKS cluster...
az aks create ^
  --resource-group %RESOURCE_GROUP% ^
  --name %AKS_NAME% ^
  --node-count 2 ^
  --node-vm-size Standard_B2s ^
  --enable-managed-identity ^
  --generate-ssh-keys ^
  --attach-acr %ACR_NAME% ^
  --location %LOCATION% ^
  --kubernetes-version 1.29

if %errorlevel% neq 0 (
    echo ERROR: Failed to create AKS cluster
    exit /b 1
)

echo.
echo ======================================
echo AKS Cluster Created Successfully!
echo ======================================
echo.
echo To connect to the cluster:
echo az aks get-credentials --resource-group %RESOURCE_GROUP% --name %AKS_NAME%
echo.