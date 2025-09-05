# Update Azure Configuration Script
# New Azure Configuration Values
$oldTenantId = "e1f3e196-aa55-4709-9c55-0e334c0b444f"
$newTenantId = "e1f3e196-aa55-4709-9c55-0e334c0b444f"

$oldSubscriptionId = "6dc7cfa2-0332-4740-98b6-bac9f1a23de9"
$newSubscriptionId = "6dc7cfa2-0332-4740-98b6-bac9f1a23de9"

$oldClientId = "232c44f7-d0cf-4825-a9b5-beba9f587ffb"
$newClientId = "232c44f7-d0cf-4825-a9b5-beba9f587ffb"

Write-Host "Updating Azure Configuration Files..." -ForegroundColor Yellow

# Get all files that need updating
$files = Get-ChildItem -Path "." -Recurse -File | Where-Object {
    $_.Extension -in @('.md', '.yml', '.yaml', '.json', '.ts', '.js', '.py', '.rs', '.bat', '.ps1', '.sh', '.sql', '.env', '.example')
}

$updatedFiles = @()

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $updated = $false
    
    # Replace tenant ID
    if ($content -match $oldTenantId) {
        $content = $content -replace $oldTenantId, $newTenantId
        $updated = $true
    }
    
    # Replace subscription ID
    if ($content -match $oldSubscriptionId) {
        $content = $content -replace $oldSubscriptionId, $newSubscriptionId
        $updated = $true
    }
    
    # Replace client ID
    if ($content -match $oldClientId) {
        $content = $content -replace $oldClientId, $newClientId
        $updated = $true
    }
    
    if ($updated) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        $updatedFiles += $file.FullName
        Write-Host "Updated: $($file.FullName)" -ForegroundColor Green
    }
}

Write-Host "`nTotal files updated: $($updatedFiles.Count)" -ForegroundColor Cyan
Write-Host "`nConfiguration update complete!" -ForegroundColor Green

# Display new configuration
Write-Host "`nNew Azure Configuration:" -ForegroundColor Yellow
Write-Host "Tenant ID: $newTenantId" -ForegroundColor Cyan
Write-Host "Subscription ID: $newSubscriptionId" -ForegroundColor Cyan
Write-Host "Client ID: $newClientId" -ForegroundColor Cyan
Write-Host "Client Secret: [Set in GitHub Secrets]" -ForegroundColor Cyan