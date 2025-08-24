@echo off
REM PolicyCortex Azure Resources Setup Script
REM This script creates and configures Azure resources for PolicyCortex
REM Requires Azure CLI to be installed and authenticated

setlocal enabledelayedexpansion

echo =============================================================
echo PolicyCortex Azure Resources Setup Script v2.0
echo Creating and configuring Azure resources...
echo =============================================================
echo.

REM Set variables
set SUBSCRIPTION_ID_DEV=205b477d-17e7-4b3b-92c1-32cf02626b78
set SUBSCRIPTION_ID_PROD=9f16cc88-89ce-49ba-a96d-308ed3169595
set TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set LOCATION=eastus
set ERRORS_OCCURRED=0

REM Development resource names
set RG_DEV=pcx42178531-rg
set ACR_DEV=crpcxdev
set AKS_DEV=pcx42178531-aks
set APP_DEV=PolicyCortex-Dev
set CLIENT_ID_DEV=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c

REM Production resource names (will be created)
set RG_PROD=policycortex-prod-rg
set ACR_PROD=crcortexprodvb9v2h
set AKS_PROD=policycortex-prod-aks
set APP_PROD=PolicyCortex-PROD
set CLIENT_ID_PROD=8f0208b4-82b1-47cd-b02a-75e2f7afddb5

REM Function to log messages
:log_info
echo [INFO] %~1
goto :eof

:log_error
echo [ERROR] %~1
set ERRORS_OCCURRED=1
goto :eof

:log_success
echo [SUCCESS] %~1
goto :eof

:check_prerequisites
echo.
call :log_info "Checking prerequisites..."

REM Check Azure CLI
az version >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "Azure CLI is not installed"
    call :log_info "Please install Azure CLI from https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    goto :error_exit
)

REM Check authentication
az account show >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "Not authenticated with Azure"
    call :log_info "Please run: az login"
    goto :error_exit
)

call :log_success "Prerequisites check passed"
goto :eof

:setup_development_resources
echo.
call :log_info "Setting up development environment resources..."

REM Switch to development subscription
call :log_info "Switching to development subscription..."
az account set --subscription %SUBSCRIPTION_ID_DEV%
if %errorLevel% neq 0 (
    call :log_error "Failed to switch to development subscription"
    goto :eof
)

REM Check if development resource group exists
call :log_info "Checking development resource group..."
az group show --name %RG_DEV% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating development resource group..."
    az group create --name %RG_DEV% --location %LOCATION%
    if %errorLevel% equ 0 (
        call :log_success "Development resource group created"
    ) else (
        call :log_error "Failed to create development resource group"
        goto :eof
    )
) else (
    call :log_info "Development resource group already exists"
)

REM Check if development ACR exists
call :log_info "Checking development container registry..."
az acr show --name %ACR_DEV% --resource-group %RG_DEV% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating development container registry..."
    az acr create --name %ACR_DEV% --resource-group %RG_DEV% --sku Standard --admin-enabled false
    if %errorLevel% equ 0 (
        call :log_success "Development container registry created"
    ) else (
        call :log_error "Failed to create development container registry"
        goto :eof
    )
) else (
    call :log_info "Development container registry already exists"
)

REM Check if development AKS exists
call :log_info "Checking development AKS cluster..."
az aks show --name %AKS_DEV% --resource-group %RG_DEV% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating development AKS cluster..."
    az aks create --name %AKS_DEV% --resource-group %RG_DEV% --node-count 2 --node-vm-size Standard_D2s_v3 --enable-managed-identity --attach-acr %ACR_DEV% --generate-ssh-keys
    if %errorLevel% equ 0 (
        call :log_success "Development AKS cluster created"
    ) else (
        call :log_error "Failed to create development AKS cluster"
        goto :eof
    )
) else (
    call :log_info "Development AKS cluster already exists"
)

call :log_success "Development resources setup completed"
goto :eof

:setup_production_resources
echo.
call :log_info "Setting up production environment resources..."

REM Switch to production subscription
call :log_info "Switching to production subscription..."
az account set --subscription %SUBSCRIPTION_ID_PROD%
if %errorLevel% neq 0 (
    call :log_error "Failed to switch to production subscription"
    goto :eof
)

REM Check if production resource group exists
call :log_info "Checking production resource group..."
az group show --name %RG_PROD% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating production resource group..."
    az group create --name %RG_PROD% --location %LOCATION%
    if %errorLevel% equ 0 (
        call :log_success "Production resource group created"
    ) else (
        call :log_error "Failed to create production resource group"
        goto :eof
    )
) else (
    call :log_info "Production resource group already exists"
)

REM Check if production ACR exists
call :log_info "Checking production container registry..."
az acr show --name %ACR_PROD% --resource-group %RG_PROD% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating production container registry..."
    az acr create --name %ACR_PROD% --resource-group %RG_PROD% --sku Premium --admin-enabled false --public-network-enabled false
    if %errorLevel% equ 0 (
        call :log_success "Production container registry created"
    ) else (
        call :log_error "Failed to create production container registry"
        goto :eof
    )
) else (
    call :log_info "Production container registry already exists"
)

REM Check if production AKS exists
call :log_info "Checking production AKS cluster..."
az aks show --name %AKS_PROD% --resource-group %RG_PROD% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating production AKS cluster..."
    az aks create --name %AKS_PROD% --resource-group %RG_PROD% --node-count 3 --node-vm-size Standard_D4s_v3 --enable-managed-identity --attach-acr %ACR_PROD% --generate-ssh-keys --enable-cluster-autoscaler --min-count 3 --max-count 10 --network-plugin azure
    if %errorLevel% equ 0 (
        call :log_success "Production AKS cluster created"
    ) else (
        call :log_error "Failed to create production AKS cluster"
        goto :eof
    )
) else (
    call :log_info "Production AKS cluster already exists"
)

call :log_success "Production resources setup completed"
goto :eof

:setup_azure_ad_applications
echo.
call :log_info "Setting up Azure AD applications..."

REM Check if development app exists
call :log_info "Checking development Azure AD application..."
az ad app show --id %CLIENT_ID_DEV% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating development Azure AD application..."
    az ad app create --display-name "%APP_DEV%" --sign-in-audience AzureADMyOrg --web-redirect-uris "http://localhost:3000" "http://localhost:3005" --web-home-page-url "http://localhost:3005" --identifier-uris "api://policycortex-dev"
    if %errorLevel% equ 0 (
        call :log_success "Development Azure AD application created"
    ) else (
        call :log_error "Failed to create development Azure AD application"
        goto :eof
    )
) else (
    call :log_info "Development Azure AD application already exists"
)

REM Check if production app exists
call :log_info "Checking production Azure AD application..."
az ad app show --id %CLIENT_ID_PROD% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating production Azure AD application..."
    az ad app create --display-name "%APP_PROD%" --sign-in-audience AzureADMyOrg --web-redirect-uris "https://policycortex.com" "https://www.policycortex.com" --web-home-page-url "https://policycortex.com" --identifier-uris "api://policycortex-prod"
    if %errorLevel% equ 0 (
        call :log_success "Production Azure AD application created"
    ) else (
        call :log_error "Failed to create production Azure AD application"
        goto :eof
    )
) else (
    call :log_info "Production Azure AD application already exists"
)

call :log_success "Azure AD applications setup completed"
goto :eof

:setup_service_principals
echo.
call :log_info "Setting up service principals for CI/CD..."

REM Development service principal
call :log_info "Setting up development service principal..."
az ad sp show --id %CLIENT_ID_DEV% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating development service principal..."
    az ad sp create --id %CLIENT_ID_DEV%
    if %errorLevel% equ 0 (
        call :log_success "Development service principal created"
    ) else (
        call :log_error "Failed to create development service principal"
        goto :eof
    )
) else (
    call :log_info "Development service principal already exists"
)

REM Production service principal
call :log_info "Setting up production service principal..."
az ad sp show --id %CLIENT_ID_PROD% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating production service principal..."
    az ad sp create --id %CLIENT_ID_PROD%
    if %errorLevel% equ 0 (
        call :log_success "Production service principal created"
    ) else (
        call :log_error "Failed to create production service principal"
        goto :eof
    )
) else (
    call :log_info "Production service principal already exists"
)

call :log_success "Service principals setup completed"
goto :eof

:setup_role_assignments
echo.
call :log_info "Setting up role assignments..."

REM Development role assignments
call :log_info "Setting development role assignments..."
az account set --subscription %SUBSCRIPTION_ID_DEV%
az role assignment create --assignee %CLIENT_ID_DEV% --role "Contributor" --scope "/subscriptions/%SUBSCRIPTION_ID_DEV%"
az role assignment create --assignee %CLIENT_ID_DEV% --role "AcrPush" --scope "/subscriptions/%SUBSCRIPTION_ID_DEV%/resourceGroups/%RG_DEV%/providers/Microsoft.ContainerRegistry/registries/%ACR_DEV%"
az role assignment create --assignee %CLIENT_ID_DEV% --role "Azure Kubernetes Service Cluster Admin Role" --scope "/subscriptions/%SUBSCRIPTION_ID_DEV%/resourceGroups/%RG_DEV%/providers/Microsoft.ContainerService/managedClusters/%AKS_DEV%"

REM Production role assignments
call :log_info "Setting production role assignments..."
az account set --subscription %SUBSCRIPTION_ID_PROD%
az role assignment create --assignee %CLIENT_ID_PROD% --role "Contributor" --scope "/subscriptions/%SUBSCRIPTION_ID_PROD%"
az role assignment create --assignee %CLIENT_ID_PROD% --role "AcrPush" --scope "/subscriptions/%SUBSCRIPTION_ID_PROD%/resourceGroups/%RG_PROD%/providers/Microsoft.ContainerRegistry/registries/%ACR_PROD%"
az role assignment create --assignee %CLIENT_ID_PROD% --role "Azure Kubernetes Service Cluster Admin Role" --scope "/subscriptions/%SUBSCRIPTION_ID_PROD%/resourceGroups/%RG_PROD%/providers/Microsoft.ContainerService/managedClusters/%AKS_PROD%"

call :log_success "Role assignments completed"
goto :eof

:setup_oidc_federation
echo.
call :log_info "Setting up OIDC federation for GitHub Actions..."

REM Development OIDC federation
call :log_info "Setting development OIDC federation..."
az account set --subscription %SUBSCRIPTION_ID_DEV%
az ad app federated-credential create --id %CLIENT_ID_DEV% --parameters '{
    "name": "github-actions-dev",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:laeintel/policycortex:environment:development",
    "audiences": ["api://AzureADTokenExchange"],
    "description": "GitHub Actions OIDC for development environment"
}' >nul 2>&1

REM Production OIDC federation
call :log_info "Setting production OIDC federation..."
az account set --subscription %SUBSCRIPTION_ID_PROD%
az ad app federated-credential create --id %CLIENT_ID_PROD% --parameters '{
    "name": "github-actions-prod",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:laeintel/policycortex:environment:production",
    "audiences": ["api://AzureADTokenExchange"],
    "description": "GitHub Actions OIDC for production environment"
}' >nul 2>&1

REM Main branch federation for CI
az ad app federated-credential create --id %CLIENT_ID_DEV% --parameters '{
    "name": "github-actions-main",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:laeintel/policycortex:ref:refs/heads/main",
    "audiences": ["api://AzureADTokenExchange"],
    "description": "GitHub Actions OIDC for main branch CI"
}' >nul 2>&1

call :log_success "OIDC federation setup completed"
goto :eof

:setup_azure_openai
echo.
call :log_info "Setting up Azure OpenAI resources..."

REM Switch to development subscription
az account set --subscription %SUBSCRIPTION_ID_DEV%

REM Check if Azure OpenAI exists
call :log_info "Checking Azure OpenAI service..."
az cognitiveservices account show --name "policycortex-openai" --resource-group %RG_DEV% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating Azure OpenAI service..."
    az cognitiveservices account create --name "policycortex-openai" --resource-group %RG_DEV% --kind OpenAI --sku S0 --location eastus --yes
    if %errorLevel% equ 0 (
        call :log_success "Azure OpenAI service created"
        
        REM Deploy GPT-4 Turbo model
        call :log_info "Deploying GPT-4 Turbo model..."
        az cognitiveservices account deployment create --name "policycortex-openai" --resource-group %RG_DEV% --deployment-name "gpt-4-turbo" --model-name "gpt-4" --model-version "turbo-2024-04-09" --model-format OpenAI --scale-type Standard
        
        REM Deploy text-embedding-ada-002 model
        call :log_info "Deploying text-embedding-ada-002 model..."
        az cognitiveservices account deployment create --name "policycortex-openai" --resource-group %RG_DEV% --deployment-name "text-embedding-ada-002" --model-name "text-embedding-ada-002" --model-version "2" --model-format OpenAI --scale-type Standard
    ) else (
        call :log_error "Failed to create Azure OpenAI service"
    )
) else (
    call :log_info "Azure OpenAI service already exists"
)

call :log_success "Azure OpenAI setup completed"
goto :eof

:setup_key_vault
echo.
call :log_info "Setting up Azure Key Vault..."

REM Development Key Vault
call :log_info "Setting up development Key Vault..."
az account set --subscription %SUBSCRIPTION_ID_DEV%
az keyvault show --name "policycortex-kv-dev" --resource-group %RG_DEV% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating development Key Vault..."
    az keyvault create --name "policycortex-kv-dev" --resource-group %RG_DEV% --location %LOCATION% --enable-rbac-authorization true
    if %errorLevel% equ 0 (
        call :log_success "Development Key Vault created"
    ) else (
        call :log_error "Failed to create development Key Vault"
    )
) else (
    call :log_info "Development Key Vault already exists"
)

REM Production Key Vault
call :log_info "Setting up production Key Vault..."
az account set --subscription %SUBSCRIPTION_ID_PROD%
az keyvault show --name "policycortex-kv-prod" --resource-group %RG_PROD% >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Creating production Key Vault..."
    az keyvault create --name "policycortex-kv-prod" --resource-group %RG_PROD% --location %LOCATION% --enable-rbac-authorization true --public-network-access Disabled
    if %errorLevel% equ 0 (
        call :log_success "Production Key Vault created"
    ) else (
        call :log_error "Failed to create production Key Vault"
    )
) else (
    call :log_info "Production Key Vault already exists"
)

call :log_success "Key Vault setup completed"
goto :eof

:display_summary
echo.
echo =============================================================
call :log_success "Azure Resources Setup Summary"
echo =============================================================
echo.
echo Development Environment:
echo - Subscription: %SUBSCRIPTION_ID_DEV%
echo - Resource Group: %RG_DEV%
echo - Container Registry: %ACR_DEV%.azurecr.io
echo - AKS Cluster: %AKS_DEV%
echo - Azure AD App: %APP_DEV% (%CLIENT_ID_DEV%)
echo - Key Vault: policycortex-kv-dev
echo.
echo Production Environment:
echo - Subscription: %SUBSCRIPTION_ID_PROD%
echo - Resource Group: %RG_PROD%
echo - Container Registry: %ACR_PROD%.azurecr.io
echo - AKS Cluster: %AKS_PROD%
echo - Azure AD App: %APP_PROD% (%CLIENT_ID_PROD%)
echo - Key Vault: policycortex-kv-prod
echo.
echo Additional Services:
echo - Azure OpenAI: policycortex-openai
echo - OIDC Federation: Configured for GitHub Actions
echo.
echo Next steps:
echo 1. Configure API permissions for Azure AD applications
echo 2. Update GitHub secrets with actual Azure OpenAI keys
echo 3. Test Azure authentication from the application
echo 4. Configure monitoring and alerts
echo.
goto :eof

:error_exit
echo.
call :log_error "Azure setup failed. Please check the errors above and try again."
exit /b 1

:main
REM Main execution flow
call :check_prerequisites
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_development_resources
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_production_resources
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_azure_ad_applications
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_service_principals
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_role_assignments
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_oidc_federation
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_azure_openai
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_key_vault
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :display_summary

call :log_success "Azure resources setup completed successfully!"
goto :eof

REM Execute main function
call :main