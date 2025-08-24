@echo off
REM PolicyCortex GitHub Repository Setup Script
REM This script configures GitHub secrets, variables, and environments
REM Requires GitHub CLI (gh) to be installed and authenticated

setlocal enabledelayedexpansion

echo =============================================================
echo PolicyCortex GitHub Repository Setup Script v2.0
echo Configuring GitHub secrets, variables, and environments...
echo =============================================================
echo.

REM Set variables
set REPO=laeintel/policycortex
set ERRORS_OCCURRED=0

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

REM Check GitHub CLI
gh version >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "GitHub CLI is not installed"
    call :log_info "Please install GitHub CLI from https://cli.github.com/"
    goto :error_exit
)

REM Check authentication
gh auth status >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "Not authenticated with GitHub"
    call :log_info "Please run: gh auth login"
    goto :error_exit
)

call :log_success "Prerequisites check passed"
goto :eof

:setup_environments
echo.
call :log_info "Setting up GitHub environments..."

REM Create development environment
call :log_info "Creating development environment..."
gh api repos/%REPO%/environments/development -X PUT --field deployment_branch_policy=null >nul 2>&1
if %errorLevel% equ 0 (
    call :log_success "Development environment created"
) else (
    call :log_info "Development environment already exists or failed to create"
)

REM Create production environment
call :log_info "Creating production environment..."
gh api repos/%REPO%/environments/production -X PUT --field deployment_branch_policy="{\"protected_branches\":true,\"custom_branch_policies\":false}" >nul 2>&1
if %errorLevel% equ 0 (
    call :log_success "Production environment created"
) else (
    call :log_info "Production environment already exists or failed to create"
)

REM Create staging environment
call :log_info "Creating staging environment..."
gh api repos/%REPO%/environments/staging -X PUT --field deployment_branch_policy=null >nul 2>&1
if %errorLevel% equ 0 (
    call :log_success "Staging environment created"
) else (
    call :log_info "Staging environment already exists or failed to create"
)

goto :eof

:setup_repository_secrets
echo.
call :log_info "Setting up repository secrets..."

REM Azure Authentication - Base Configuration
call :log_info "Setting Azure authentication secrets..."
gh secret set AZURE_TENANT_ID --body "9ef5b184-d371-462a-bc75-5024ce8baff7" --repo %REPO%
gh secret set AZURE_CLIENT_ID_DEV --body "1ecc95d1-e5bb-43e2-9324-30a17cb6b01c" --repo %REPO%
gh secret set AZURE_CLIENT_ID_PROD --body "8f0208b4-82b1-47cd-b02a-75e2f7afddb5" --repo %REPO%
gh secret set AZURE_SUBSCRIPTION_ID_DEV --body "205b477d-17e7-4b3b-92c1-32cf02626b78" --repo %REPO%
gh secret set AZURE_SUBSCRIPTION_ID_PROD --body "9f16cc88-89ce-49ba-a96d-308ed3169595" --repo %REPO%

REM Container Registry & AKS
call :log_info "Setting container registry and AKS secrets..."
gh secret set ACR_NAME_DEV --body "crpcxdev" --repo %REPO%
gh secret set ACR_NAME_PROD --body "crcortexprodvb9v2h" --repo %REPO%
gh secret set AKS_CLUSTER_NAME_DEV --body "pcx42178531-aks" --repo %REPO%
gh secret set AKS_RESOURCE_GROUP_DEV --body "pcx42178531-rg" --repo %REPO%

REM Generate and set JWT secrets
call :log_info "Generating JWT secrets..."
for /f "delims=" %%i in ('powershell -Command "[System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes([System.Guid]::NewGuid().ToString() + [System.Guid]::NewGuid().ToString()))"') do set JWT_SECRET_DEV=%%i
for /f "delims=" %%i in ('powershell -Command "[System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes([System.Guid]::NewGuid().ToString() + [System.Guid]::NewGuid().ToString()))"') do set JWT_SECRET_PROD=%%i

gh secret set JWT_SECRET_KEY_DEV --body "!JWT_SECRET_DEV!" --repo %REPO%
gh secret set JWT_SECRET_KEY_PROD --body "!JWT_SECRET_PROD!" --repo %REPO%

REM Azure OpenAI (placeholders - update with actual values)
call :log_info "Setting Azure OpenAI secrets (placeholders)..."
gh secret set AOAI_API_KEY --body "your-azure-openai-api-key-here" --repo %REPO%
gh secret set AOAI_API_VERSION --body "2024-02-15-preview" --repo %REPO%
gh secret set AOAI_CHAT_DEPLOYMENT --body "gpt-4-turbo" --repo %REPO%
gh secret set AOAI_ENDPOINT --body "https://your-openai-resource.openai.azure.com/" --repo %REPO%

REM Security & Tools
call :log_info "Setting security and tools secrets..."
gh secret set GITLEAKS_LICENSE --body "your-gitleaks-license-here" --repo %REPO%

REM Database secrets
call :log_info "Setting database secrets..."
for /f "delims=" %%i in ('powershell -Command "[System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes([System.Guid]::NewGuid().ToString()))"') do set DB_PASSWORD_DEV=%%i
for /f "delims=" %%i in ('powershell -Command "[System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes([System.Guid]::NewGuid().ToString()))"') do set DB_PASSWORD_PROD=%%i

gh secret set DATABASE_PASSWORD_DEV --body "!DB_PASSWORD_DEV!" --repo %REPO%
gh secret set DATABASE_PASSWORD_PROD --body "!DB_PASSWORD_PROD!" --repo %REPO%

REM Terraform Backend
call :log_info "Setting Terraform backend secrets..."
gh secret set TERRAFORM_BACKEND_CONTAINER --body "tfstate" --repo %REPO%
gh secret set TERRAFORM_BACKEND_RESOURCE_GROUP --body "terraform-state-rg" --repo %REPO%
gh secret set TERRAFORM_BACKEND_STORAGE_ACCOUNT --body "terraformstatestorageacct" --repo %REPO%

call :log_success "Repository secrets configured"
goto :eof

:setup_environment_secrets
echo.
call :log_info "Setting up environment-specific secrets..."

REM Development environment secrets
call :log_info "Setting development environment secrets..."
gh secret set AZURE_CLIENT_ID --body "1ecc95d1-e5bb-43e2-9324-30a17cb6b01c" --env development --repo %REPO%
gh secret set AZURE_SUBSCRIPTION_ID --body "205b477d-17e7-4b3b-92c1-32cf02626b78" --env development --repo %REPO%
gh secret set ACR_NAME --body "crpcxdev" --env development --repo %REPO%
gh secret set AKS_CLUSTER_NAME --body "pcx42178531-aks" --env development --repo %REPO%
gh secret set AKS_RESOURCE_GROUP --body "pcx42178531-rg" --env development --repo %REPO%
gh secret set JWT_SECRET --body "!JWT_SECRET_DEV!" --env development --repo %REPO%
gh secret set DATABASE_PASSWORD --body "!DB_PASSWORD_DEV!" --env development --repo %REPO%

REM Production environment secrets
call :log_info "Setting production environment secrets..."
gh secret set AZURE_CLIENT_ID --body "8f0208b4-82b1-47cd-b02a-75e2f7afddb5" --env production --repo %REPO%
gh secret set AZURE_SUBSCRIPTION_ID --body "9f16cc88-89ce-49ba-a96d-308ed3169595" --env production --repo %REPO%
gh secret set ACR_NAME --body "crcortexprodvb9v2h" --env production --repo %REPO%
gh secret set AKS_CLUSTER_NAME --body "policycortex-prod-aks" --env production --repo %REPO%
gh secret set AKS_RESOURCE_GROUP --body "policycortex-prod-rg" --env production --repo %REPO%
gh secret set JWT_SECRET --body "!JWT_SECRET_PROD!" --env production --repo %REPO%
gh secret set DATABASE_PASSWORD --body "!DB_PASSWORD_PROD!" --env production --repo %REPO%

REM Staging environment secrets
call :log_info "Setting staging environment secrets..."
gh secret set AZURE_CLIENT_ID --body "1ecc95d1-e5bb-43e2-9324-30a17cb6b01c" --env staging --repo %REPO%
gh secret set AZURE_SUBSCRIPTION_ID --body "205b477d-17e7-4b3b-92c1-32cf02626b78" --env staging --repo %REPO%
gh secret set ACR_NAME --body "crpcxdev" --env staging --repo %REPO%
gh secret set AKS_CLUSTER_NAME --body "pcx42178531-aks" --env staging --repo %REPO%
gh secret set AKS_RESOURCE_GROUP --body "pcx42178531-rg" --env staging --repo %REPO%
gh secret set JWT_SECRET --body "!JWT_SECRET_DEV!" --env staging --repo %REPO%
gh secret set DATABASE_PASSWORD --body "!DB_PASSWORD_DEV!" --env staging --repo %REPO%

call :log_success "Environment secrets configured"
goto :eof

:setup_repository_variables
echo.
call :log_info "Setting up repository variables..."

REM Project configuration
gh variable set PROJECT_NAME --body "PolicyCortex" --repo %REPO%
gh variable set PROJECT_DESCRIPTION --body "Enterprise AI-powered Azure governance platform" --repo %REPO%
gh variable set PROJECT_VERSION --body "2.0.0" --repo %REPO%

REM Build configuration
gh variable set DOCKER_BUILDKIT --body "1" --repo %REPO%
gh variable set COMPOSE_DOCKER_CLI_BUILD --body "1" --repo %REPO%
gh variable set NODE_VERSION --body "20" --repo %REPO%
gh variable set RUST_VERSION --body "1.75" --repo %REPO%
gh variable set PYTHON_VERSION --body "3.11" --repo %REPO%

REM Feature flags
gh variable set ENABLE_LIVE_DATA --body "true" --repo %REPO%
gh variable set ENABLE_ML_FEATURES --body "true" --repo %REPO%
gh variable set ENABLE_WEBSOCKET --body "true" --repo %REPO%
gh variable set ENABLE_MONITORING --body "true" --repo %REPO%

REM URLs and endpoints
gh variable set FRONTEND_URL_DEV --body "http://localhost:3005" --repo %REPO%
gh variable set BACKEND_URL_DEV --body "http://localhost:8085" --repo %REPO%
gh variable set GRAPHQL_URL_DEV --body "http://localhost:4001/graphql" --repo %REPO%

gh variable set FRONTEND_URL_PROD --body "https://policycortex.com" --repo %REPO%
gh variable set BACKEND_URL_PROD --body "https://api.policycortex.com" --repo %REPO%
gh variable set GRAPHQL_URL_PROD --body "https://graphql.policycortex.com/graphql" --repo %REPO%

REM Database configuration
gh variable set DATABASE_NAME --body "policycortex" --repo %REPO%
gh variable set DATABASE_USER --body "postgres" --repo %REPO%
gh variable set REDIS_PORT --body "6379" --repo %REPO%
gh variable set EVENTSTORE_PORT --body "2113" --repo %REPO%

call :log_success "Repository variables configured"
goto :eof

:setup_branch_protection
echo.
call :log_info "Setting up branch protection rules..."

REM Main branch protection
call :log_info "Setting up main branch protection..."
gh api repos/%REPO%/branches/main/protection -X PUT --field required_status_checks='{"strict":true,"checks":[{"context":"ci/frontend"},{"context":"ci/core"},{"context":"ci/security"}]}' --field enforce_admins=true --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true,"require_code_owner_reviews":true}' --field restrictions=null >nul 2>&1
if %errorLevel% equ 0 (
    call :log_success "Main branch protection configured"
) else (
    call :log_info "Main branch protection may already exist or failed to configure"
)

goto :eof

:setup_webhooks
echo.
call :log_info "Setting up webhooks (optional)..."

REM Example webhook for external integrations
REM gh api repos/%REPO%/hooks -X POST --field name=web --field config='{"url":"https://your-webhook-url.com/github","content_type":"json"}' --field events='["push","pull_request"]' >nul 2>&1

call :log_info "Webhooks setup completed (or skipped)"
goto :eof

:display_summary
echo.
echo =============================================================
call :log_success "GitHub Repository Setup Summary"
echo =============================================================
echo.
echo Secrets configured:
echo - Azure authentication (OIDC-based)
echo - Container registries and AKS clusters
echo - JWT secrets (generated)
echo - Azure OpenAI configuration
echo - Database passwords (generated)
echo - Terraform backend configuration
echo.
echo Variables configured:
echo - Project metadata
echo - Build configuration
echo - Feature flags
echo - Environment URLs
echo - Database settings
echo.
echo Environments created:
echo - development
echo - production
echo - staging
echo.
echo Next steps:
echo 1. Update Azure OpenAI secrets with actual values
echo 2. Update Gitleaks license if you have one
echo 3. Verify branch protection rules
echo 4. Test CI/CD pipeline
echo.
call :log_info "You can view all secrets and variables in the repository settings"
call :log_info "GitHub Repository: https://github.com/%REPO%/settings"
echo.
goto :eof

:error_exit
echo.
call :log_error "GitHub setup failed. Please check the errors above and try again."
exit /b 1

:main
REM Main execution flow
call :check_prerequisites
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_environments
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_repository_secrets
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_environment_secrets
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_repository_variables
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_branch_protection
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_webhooks
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :display_summary

call :log_success "GitHub repository setup completed successfully!"
goto :eof

REM Execute main function
call :main