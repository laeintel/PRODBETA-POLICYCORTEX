# Setup GitHub Environments for PolicyCortex
# This script helps you create the required GitHub environments for the workflow

Write-Host "GitHub Environments Setup for PolicyCortex" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This workflow requires the following GitHub environments to be created:" -ForegroundColor Yellow
Write-Host "1. dev" -ForegroundColor Green
Write-Host "2. staging" -ForegroundColor Green  
Write-Host "3. prod" -ForegroundColor Green
Write-Host ""

Write-Host "To create these environments:" -ForegroundColor Yellow
Write-Host "1. Go to your GitHub repository: https://github.com/YOUR_ORG/YOUR_REPO/settings/environments"
Write-Host "2. Click 'New environment' for each environment (dev, staging, prod)"
Write-Host "3. Configure protection rules as needed:"
Write-Host "   - For 'staging': Consider adding required reviewers"
Write-Host "   - For 'prod': Add required reviewers and deployment branches (only from main)"
Write-Host ""

Write-Host "Environment-specific secrets that may be needed:" -ForegroundColor Yellow
Write-Host "- AZURE_CREDENTIALS (if different per environment)"
Write-Host "- TERRAFORM_BACKEND_STORAGE_ACCOUNT"
Write-Host "- TERRAFORM_BACKEND_CONTAINER" 
Write-Host "- TERRAFORM_BACKEND_RESOURCE_GROUP"
Write-Host ""

Write-Host "Press any key to open GitHub settings in your browser..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Try to get repo info from git
try {
    $remoteUrl = git config --get remote.origin.url
    if ($remoteUrl -match "github.com[:/](.+?)/(.*?)(\.git)?$") {
        $owner = $matches[1]
        $repo = $matches[2]
        $settingsUrl = "https://github.com/$owner/$repo/settings/environments"
        Start-Process $settingsUrl
        Write-Host "Opening: $settingsUrl" -ForegroundColor Green
    } else {
        Write-Host "Could not determine GitHub repository URL" -ForegroundColor Red
        Write-Host "Please manually navigate to your repository settings > Environments" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Could not determine GitHub repository URL" -ForegroundColor Red
    Write-Host "Please manually navigate to your repository settings > Environments" -ForegroundColor Yellow
} 