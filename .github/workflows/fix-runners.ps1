# PowerShell script to fix all workflow files to use ubuntu-latest
Get-ChildItem -Path ".github/workflows" -Filter "*.yml" | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    $newContent = $content -replace 'runs-on:\s*self-hosted', 'runs-on: ubuntu-latest'
    if ($content -ne $newContent) {
        Set-Content -Path $_.FullName -Value $newContent
        Write-Host "Fixed: $($_.Name)" -ForegroundColor Green
    } else {
        Write-Host "No changes needed: $($_.Name)" -ForegroundColor Gray
    }
}