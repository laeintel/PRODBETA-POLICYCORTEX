# Grant Service Principal access to subscription
# Run this with an account that has Owner or User Access Administrator role

param(
    [string]$ServicePrincipalId = "1ecc95d1-e5bb-43e2-9324-30a17cb6b01c",
    [string]$SubscriptionId = "205b477d-17e7-4b3b-92c1-32cf02626b78",
    [string]$Role = "Contributor"
)

Write-Host "Granting Service Principal access to subscription..." -ForegroundColor Cyan

# Check if logged in
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "Please login to Azure first:" -ForegroundColor Yellow
    az login
}

# Set the subscription
az account set --subscription $SubscriptionId

# Get current user's role
$currentUser = az account show --query user.name -o tsv
Write-Host "Current user: $currentUser" -ForegroundColor Green

# Check if user has permission to assign roles
$userRoles = az role assignment list --assignee $currentUser --scope "/subscriptions/$SubscriptionId" --query "[].roleDefinitionName" -o tsv
Write-Host "Your roles: $userRoles" -ForegroundColor Yellow

if ($userRoles -notcontains "Owner" -and $userRoles -notcontains "User Access Administrator") {
    Write-Host "WARNING: You may not have permission to assign roles. You need Owner or User Access Administrator role." -ForegroundColor Red
}

# Grant the Service Principal access
Write-Host "Assigning $Role role to Service Principal..." -ForegroundColor Cyan
try {
    az role assignment create `
        --assignee $ServicePrincipalId `
        --role $Role `
        --scope "/subscriptions/$SubscriptionId" `
        --output table
    
    Write-Host "✅ Successfully granted $Role access to Service Principal!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to grant access. Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual steps to grant access:" -ForegroundColor Yellow
    Write-Host "1. Go to Azure Portal (portal.azure.com)"
    Write-Host "2. Navigate to Subscriptions → 'Policy Cortex Dev'"
    Write-Host "3. Click 'Access control (IAM)'"
    Write-Host "4. Click '+ Add' → 'Add role assignment'"
    Write-Host "5. Select 'Contributor' role"
    Write-Host "6. Search for: PolicyCortex Dev"
    Write-Host "7. Select it and click 'Review + assign'"
}

# Verify the assignment
Write-Host ""
Write-Host "Verifying role assignment..." -ForegroundColor Cyan
$assignments = az role assignment list --assignee $ServicePrincipalId --scope "/subscriptions/$SubscriptionId" --query "[].{Role:roleDefinitionName, Scope:scope}" -o table
if ($assignments) {
    Write-Host "Current role assignments:" -ForegroundColor Green
    Write-Host $assignments
} else {
    Write-Host "No role assignments found for the Service Principal" -ForegroundColor Red
}