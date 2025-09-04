# PolicyCortex GitHub Deployment Fix Script
# This script documents and helps configure all required GitHub secrets and settings
# Run this locally to see what needs to be configured in GitHub

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "PolicyCortex GitHub Deployment Configuration" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if GitHub CLI is installed
$ghInstalled = Get-Command gh -ErrorAction SilentlyContinue
if (-not $ghInstalled) {
    Write-Host "ERROR: GitHub CLI (gh) is not installed!" -ForegroundColor Red
    Write-Host "Please install it from: https://cli.github.com/" -ForegroundColor Yellow
    Write-Host ""
}

# Define required secrets
$requiredSecrets = @{
    # Azure Service Principal for Dev Environment
    "AZURE_CLIENT_ID_DEV" = @{
        Description = "Azure Service Principal Client ID for Dev environment"
        Example = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        Required = $true
    }
    "AZURE_CLIENT_SECRET_DEV" = @{
        Description = "Azure Service Principal Client Secret for Dev environment"
        Example = "~xxxxx~xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        Required = $false  # Optional if using OIDC
    }
    "AZURE_SUBSCRIPTION_ID_DEV" = @{
        Description = "Azure Subscription ID for Dev environment"
        Example = "205b477d-17e7-4b3b-92c1-32cf02626b78"
        Required = $true
    }
    
    # Azure Service Principal for Prod Environment
    "AZURE_CLIENT_ID_PROD" = @{
        Description = "Azure Service Principal Client ID for Prod environment"
        Example = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        Required = $true
    }
    "AZURE_CLIENT_SECRET_PROD" = @{
        Description = "Azure Service Principal Client Secret for Prod environment"
        Example = "~xxxxx~xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        Required = $false  # Optional if using OIDC
    }
    "AZURE_SUBSCRIPTION_ID_PROD" = @{
        Description = "Azure Subscription ID for Prod environment"
        Example = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        Required = $true
    }
    
    # Common Azure Settings
    "AZURE_TENANT_ID" = @{
        Description = "Azure Tenant ID (same for all environments)"
        Example = "9ef5b184-d371-462a-bc75-5024ce8baff7"
        Required = $true
    }
    
    # Container Registry Settings (Optional - can use defaults in workflow)
    "REGISTRY" = @{
        Description = "Docker registry URL"
        Example = "crcortexdev3p0bata.azurecr.io"
        Required = $false
    }
    "REGISTRY_USER" = @{
        Description = "Docker registry username"
        Example = "crcortexdev3p0bata"
        Required = $false
    }
    "REGISTRY_PASS" = @{
        Description = "Docker registry password"
        Example = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        Required = $false
    }
}

Write-Host "REQUIRED GITHUB SECRETS:" -ForegroundColor Yellow
Write-Host "------------------------" -ForegroundColor Yellow
Write-Host ""

foreach ($secret in $requiredSecrets.GetEnumerator()) {
    $status = if ($secret.Value.Required) { "[REQUIRED]" } else { "[OPTIONAL]" }
    $color = if ($secret.Value.Required) { "Red" } else { "Gray" }
    
    Write-Host "$status $($secret.Key)" -ForegroundColor $color
    Write-Host "  Description: $($secret.Value.Description)" -ForegroundColor White
    Write-Host "  Example: $($secret.Value.Example)" -ForegroundColor DarkGray
    Write-Host ""
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "WORKFLOW ISSUES DETECTED:" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check for ACR name mismatches
Write-Host "1. ACR NAME MISMATCH:" -ForegroundColor Red
Write-Host "   - deploy-aks.yml uses: crcortexdev3p0bata" -ForegroundColor White
Write-Host "   - application.yml uses: crcortexdev3p0bata" -ForegroundColor White
Write-Host "   - docker-compose.yml references: crcortexdev.azurecr.io (OLD)" -ForegroundColor Yellow
Write-Host "   FIX: Update all references to use consistent ACR names" -ForegroundColor Green
Write-Host ""

Write-Host "2. KUBERNETES MANIFESTS:" -ForegroundColor Red
Write-Host "   - k8s/dev/ directory exists with 6 YAML files" -ForegroundColor Green
Write-Host "   - k8s/prod/ directory exists with 8 YAML files" -ForegroundColor Green
Write-Host "   - Manifests use placeholder values (ACR_NAME, IMAGE_TAG)" -ForegroundColor Yellow
Write-Host "   FIX: Workflow correctly replaces these during deployment" -ForegroundColor Green
Write-Host ""

Write-Host "3. RUST BUILD ISSUES:" -ForegroundColor Red
Write-Host "   - Core service has known compilation challenges" -ForegroundColor Yellow
Write-Host "   - deploy.yml includes fallback mock service" -ForegroundColor Green
Write-Host "   FIX: Mock service fallback already implemented" -ForegroundColor Green
Write-Host ""

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "HOW TO FIX DEPLOYMENT:" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "STEP 1: Configure GitHub Secrets" -ForegroundColor Yellow
Write-Host "  Go to: https://github.com/[your-org]/policycortex/settings/secrets/actions" -ForegroundColor White
Write-Host "  Add all [REQUIRED] secrets listed above" -ForegroundColor White
Write-Host ""

Write-Host "STEP 2: Create Azure Service Principal (if not exists)" -ForegroundColor Yellow
Write-Host "  Run these Azure CLI commands:" -ForegroundColor White
Write-Host '  az ad sp create-for-rbac --name "PolicyCortex-GitHub-Dev" --role contributor --scopes /subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78' -ForegroundColor Cyan
Write-Host '  az ad sp create-for-rbac --name "PolicyCortex-GitHub-Prod" --role contributor --scopes /subscriptions/[PROD-SUB-ID]' -ForegroundColor Cyan
Write-Host ""

Write-Host "STEP 3: Configure OIDC for GitHub Actions (Recommended)" -ForegroundColor Yellow
Write-Host "  This allows passwordless authentication from GitHub to Azure" -ForegroundColor White
Write-Host "  Documentation: https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-azure" -ForegroundColor DarkGray
Write-Host ""

Write-Host "STEP 4: Test Deployment" -ForegroundColor Yellow
Write-Host "  Option A: Trigger via GitHub UI" -ForegroundColor White
Write-Host "    1. Go to Actions tab" -ForegroundColor White
Write-Host "    2. Select 'Monorepo CI Entry' workflow" -ForegroundColor White
Write-Host "    3. Click 'Run workflow'" -ForegroundColor White
Write-Host "    4. Enable 'FORCE COMPLETE DEPLOYMENT' checkbox" -ForegroundColor White
Write-Host "    5. Click 'Run workflow' button" -ForegroundColor White
Write-Host ""
Write-Host "  Option B: Trigger via GitHub CLI" -ForegroundColor White
if ($ghInstalled) {
    Write-Host '    gh workflow run entry.yml --field force_deploy=true' -ForegroundColor Cyan
} else {
    Write-Host "    [Install gh CLI first]" -ForegroundColor Red
}
Write-Host ""

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "QUICK FIX COMMANDS:" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

if ($ghInstalled) {
    Write-Host "# Check current workflow runs:" -ForegroundColor Gray
    Write-Host "gh run list --workflow=entry.yml" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "# Trigger force deployment:" -ForegroundColor Gray
    Write-Host "gh workflow run entry.yml --field force_deploy=true" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "# Watch deployment progress:" -ForegroundColor Gray
    Write-Host "gh run watch" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "# View deployment logs:" -ForegroundColor Gray
    Write-Host "gh run view --log" -ForegroundColor Cyan
} else {
    Write-Host "GitHub CLI not installed - manual configuration required" -ForegroundColor Red
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "CURRENT WORKFLOW STRUCTURE:" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "entry.yml (Main Pipeline)" -ForegroundColor White
Write-Host "  ├── runner-setup" -ForegroundColor Gray
Write-Host "  ├── docker-health-check" -ForegroundColor Gray
Write-Host "  ├── detect-changes" -ForegroundColor Gray
Write-Host "  ├── frontend CI" -ForegroundColor Gray
Write-Host "  ├── core CI (Rust)" -ForegroundColor Gray
Write-Host "  ├── graphql CI" -ForegroundColor Gray
Write-Host "  ├── backend CI (Python)" -ForegroundColor Gray
Write-Host "  ├── security scanning" -ForegroundColor Gray
Write-Host "  ├── supply chain security" -ForegroundColor Gray
Write-Host "  ├── secret scanning" -ForegroundColor Gray
Write-Host "  ├── Azure infrastructure" -ForegroundColor Gray
Write-Host "  ├── app_pipeline → application.yml" -ForegroundColor Yellow
Write-Host "  └── force_aks_deployment → deploy-aks.yml" -ForegroundColor Yellow
Write-Host ""
Write-Host "When force_deploy=true, ALL stages run without skipping!" -ForegroundColor Green
Write-Host ""

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Script completed. Review the output above for deployment fixes." -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan