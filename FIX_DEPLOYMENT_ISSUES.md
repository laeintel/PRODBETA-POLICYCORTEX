# PolicyCortex Deployment Issues - Root Cause and Fix

## Root Cause Analysis

The deployment was failing and resources were being destroyed due to several issues:

### 1. **State Management Conflicts**
- The workflow was trying to manipulate Terraform state by removing and re-importing resources
- This caused Terraform to see resources as "orphaned" and destroy them
- The `deploy_container_apps` variable defaults to `false`, causing conditional resource creation

### 2. **Environment Variable Confusion**
- Staging and prod deployments were incorrectly using `dev` environment variables
- Resource names were mismatched (looking for `policycortex-cosmos-dev` in staging/prod)

### 3. **Missing GitHub Environments**
- The workflow references GitHub environments (dev, staging, prod) that don't exist
- This causes linter errors and potential deployment failures

### 4. **Terraform Init Missing**
- The apply-dev job was missing a Terraform init step before importing resources
- This caused state management operations to fail

## Fixes Applied

### 1. **Fixed Environment-Specific Deployments**
- Updated staging and prod jobs to use correct environment-specific variables
- Fixed resource names to match the target environment
- Ensured each environment uses its own terraform.tfvars file

### 2. **Simplified Container App Deployment**
- Removed problematic state manipulation (terraform state rm commands)
- Added proper resource existence checks before waiting
- Created a clean deployment approach using variable overrides

### 3. **Added Missing Terraform Init**
- Added terraform init step in apply-dev job before import operations
- Ensures backend is properly configured before state operations

### 4. **Created Setup Script**
- Added `setup-github-environments.ps1` to help create required GitHub environments

### 5. **Automatic Resource Group Import**
- Added automatic import of existing resource groups before Terraform operations
- This prevents the "resource already exists" error
- Checks if resources exist in Azure and imports them if not in state

## How to Deploy Successfully

### Step 1: Create GitHub Environments
```powershell
.\setup-github-environments.ps1
```
Create these environments in GitHub:
- `dev`
- `staging` 
- `prod`

### Step 2: Deploy Infrastructure First
The deployment is designed to work in two phases:

1. **Infrastructure Deployment** (deploy_container_apps = false)
   - Creates resource groups, networking, storage, Key Vault, etc.
   - Creates Container Registry and Container Apps Environment
   - Deploys data services (Cosmos DB, Redis, etc.)

2. **Container Apps Deployment** (deploy_container_apps = true)
   - Deploys the actual Container Apps
   - Uses the infrastructure created in phase 1

### Step 3: Run the Workflow
1. Go to Actions → Infrastructure Deployment
2. Click "Run workflow"
3. Select:
   - Environment: `dev`
   - Terraform action: `apply`
4. The workflow will:
   - Deploy infrastructure
   - Wait for resources to be ready
   - Automatically enable container apps and deploy them

### Step 4: Verify Deployment
After successful deployment:
1. Check the workflow summary for resource details and URLs
2. Verify in Azure Portal that all resources are created
3. Access the application URLs provided in the summary

## Preventing Future Issues

### 1. **Always Deploy in Order**
- Deploy to dev first, then staging, then prod
- Don't skip environments

### 2. **Don't Manually Delete Resources**
- If you need to clean up, use `terraform destroy` or the destroy workflow
- Manual deletions cause state inconsistencies

### 3. **Monitor Resource Status**
- The workflow now checks resource readiness before proceeding
- Look for "NotFound" vs "Succeeded" status messages

### 4. **State Management Best Practices**
- Don't manually manipulate terraform state unless absolutely necessary
- Use terraform import for existing resources, not state rm/import cycles
- Keep terraform state in sync with actual Azure resources

## Troubleshooting

### If Resources Are Still Being Destroyed:
1. Check terraform state: `terraform state list`
2. Verify no manual changes were made in Azure Portal
3. Ensure the backend state storage is accessible
4. Run `terraform refresh` to update state from Azure

### If Container Apps Fail to Deploy:
1. Verify all prerequisite resources are "Succeeded" status
2. Check that placeholder images were pushed to ACR
3. Ensure managed identity has proper permissions
4. Review container app logs in Azure Portal

### If State Is Corrupted:
1. Back up current state file
2. Use `terraform import` to re-import resources
3. Or start fresh with new state (careful - this can cause duplicates)

## Summary

The main issue was trying to manage Terraform state in ways that confused Terraform about resource ownership. The fix simplifies the deployment process, removes problematic state manipulation, and ensures each environment is properly isolated with its own configuration. 