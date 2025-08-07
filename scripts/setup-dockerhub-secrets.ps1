#!/usr/bin/env powershell

<#
.SYNOPSIS
    Setup Docker Hub authentication for GitHub Actions
.DESCRIPTION
    Creates GitHub repository secrets for Docker Hub authentication to avoid rate limiting
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$DockerHubUsername,
    
    [Parameter(Mandatory=$true)]
    [string]$DockerHubToken
)

Write-Host "🐳 Setting up Docker Hub authentication for GitHub Actions..." -ForegroundColor Cyan

# Check if GitHub CLI is available
$ghExists = Get-Command "gh" -ErrorAction SilentlyContinue
if (-not $ghExists) {
    Write-Host "❌ GitHub CLI (gh) is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install GitHub CLI: https://cli.github.com/" -ForegroundColor Yellow
    exit 1
}

# Check if we're in a git repository
$gitRoot = git rev-parse --show-toplevel 2>$null
if (-not $gitRoot) {
    Write-Host "❌ Not in a git repository" -ForegroundColor Red
    exit 1
}

# Check GitHub CLI authentication
$authStatus = gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ GitHub CLI is not authenticated" -ForegroundColor Red
    Write-Host "Please run: gh auth login" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ GitHub CLI is authenticated" -ForegroundColor Green

try {
    # Set Docker Hub username secret
    Write-Host "Setting DOCKERHUB_USERNAME secret..." -ForegroundColor Blue
    gh secret set DOCKERHUB_USERNAME --body $DockerHubUsername
    
    # Set Docker Hub token secret
    Write-Host "Setting DOCKERHUB_TOKEN secret..." -ForegroundColor Blue
    gh secret set DOCKERHUB_TOKEN --body $DockerHubToken
    
    Write-Host "✅ Docker Hub secrets configured successfully!" -ForegroundColor Green
    Write-Host "" -ForegroundColor White
    Write-Host "GitHub Actions will now use authenticated Docker Hub pulls to avoid rate limiting." -ForegroundColor Green
    Write-Host "" -ForegroundColor White
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Commit and push the updated workflow file" -ForegroundColor White  
    Write-Host "2. The next pipeline run will use authenticated Docker Hub access" -ForegroundColor White
    
} catch {
    Write-Host "❌ Failed to set GitHub secrets: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}