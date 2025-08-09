# Setup federated identity credentials for GitHub Actions OIDC authentication
# This configures the Azure Service Principal to trust GitHub Actions

param(
    [string]$AzureClientId = "1ecc95d1-e5bb-43e2-9324-30a17cb6b01c",
    [string]$GitHubOrg = "laeintel",
    [string]$GitHubRepo = "policycortex"
)

Write-Host "Setting up GitHub Actions OIDC authentication for Azure..." -ForegroundColor Green
Write-Host "=================================================="
Write-Host "Service Principal ID: $AzureClientId"
Write-Host "GitHub Repository: $GitHubOrg/$GitHubRepo"
Write-Host ""

# Check if logged in to Azure
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "Please login to Azure first:" -ForegroundColor Yellow
    az login
    $account = az account show | ConvertFrom-Json
}

# Get the Service Principal Object ID
$spObjectId = az ad sp show --id $AzureClientId --query id -o tsv
Write-Host "Service Principal Object ID: $spObjectId"

# Configure federated identity credential for main branch
Write-Host "Creating federated identity credential for main branch..." -ForegroundColor Cyan
$mainCredential = @{
    name = "GitHub-main"
    issuer = "https://token.actions.githubusercontent.com"
    subject = "repo:${GitHubOrg}/${GitHubRepo}:ref:refs/heads/main"
    description = "GitHub Actions main branch"
    audiences = @("api://AzureADTokenExchange")
} | ConvertTo-Json -Compress

try {
    az ad app federated-credential create --id $AzureClientId --parameters $mainCredential 2>$null
    Write-Host "✓ Main branch credential created" -ForegroundColor Green
} catch {
    Write-Host "Main branch credential already exists" -ForegroundColor Yellow
}

# Configure federated identity credential for pull requests
Write-Host "Creating federated identity credential for pull requests..." -ForegroundColor Cyan
$prCredential = @{
    name = "GitHub-PR"
    issuer = "https://token.actions.githubusercontent.com"
    subject = "repo:${GitHubOrg}/${GitHubRepo}:pull_request"
    description = "GitHub Actions pull requests"
    audiences = @("api://AzureADTokenExchange")
} | ConvertTo-Json -Compress

try {
    az ad app federated-credential create --id $AzureClientId --parameters $prCredential 2>$null
    Write-Host "✓ Pull request credential created" -ForegroundColor Green
} catch {
    Write-Host "Pull request credential already exists" -ForegroundColor Yellow
}

# Configure federated identity credential for environment: dev
Write-Host "Creating federated identity credential for dev environment..." -ForegroundColor Cyan
$devCredential = @{
    name = "GitHub-env-dev"
    issuer = "https://token.actions.githubusercontent.com"
    subject = "repo:${GitHubOrg}/${GitHubRepo}:environment:dev"
    description = "GitHub Actions dev environment"
    audiences = @("api://AzureADTokenExchange")
} | ConvertTo-Json -Compress

try {
    az ad app federated-credential create --id $AzureClientId --parameters $devCredential 2>$null
    Write-Host "✓ Dev environment credential created" -ForegroundColor Green
} catch {
    Write-Host "Dev environment credential already exists" -ForegroundColor Yellow
}

# Configure federated identity credential for environment: prod
Write-Host "Creating federated identity credential for prod environment..." -ForegroundColor Cyan
$prodCredential = @{
    name = "GitHub-env-prod"
    issuer = "https://token.actions.githubusercontent.com"
    subject = "repo:${GitHubOrg}/${GitHubRepo}:environment:prod"
    description = "GitHub Actions prod environment"
    audiences = @("api://AzureADTokenExchange")
} | ConvertTo-Json -Compress

try {
    az ad app federated-credential create --id $AzureClientId --parameters $prodCredential 2>$null
    Write-Host "✓ Prod environment credential created" -ForegroundColor Green
} catch {
    Write-Host "Prod environment credential already exists" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ GitHub Actions OIDC setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Required GitHub Secrets:" -ForegroundColor Yellow
Write-Host "========================"
Write-Host "AZURE_CLIENT_ID=$AzureClientId"
Write-Host "AZURE_TENANT_ID=$($account.tenantId)"
Write-Host "AZURE_SUBSCRIPTION_ID=$($account.id)"
Write-Host ""
Write-Host "Please ensure these secrets are configured in your GitHub repository settings." -ForegroundColor Yellow