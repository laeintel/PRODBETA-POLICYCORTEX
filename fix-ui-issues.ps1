# Fix UI Issues in PolicyCortex Frontend
Write-Host "Fixing PolicyCortex UI Issues..." -ForegroundColor Cyan

# List of files to fix
$files = @(
    "frontend\app\settings\page.tsx",
    "frontend\app\tactical\ai\page.tsx",
    "frontend\app\tactical\governance\page.tsx",
    "frontend\app\tactical\operations\page.tsx",
    "frontend\app\tactical\security\page.tsx"
)

Write-Host "`n1. Removing empty chart placeholders (border-dashed elements)..." -ForegroundColor Yellow

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "  - Fixing: $file" -ForegroundColor Gray
        
        # Read file content
        $content = Get-Content $file -Raw
        
        # Remove the border-dashed div lines (empty chart placeholders)
        $content = $content -replace '<div className="w-8 border-2 border-dashed border-gray-400 absolute"[^>]*?/>', ''
        $content = $content -replace '<div className="w-8 border-2 border-dashed border-gray-400 absolute"[^>]*?></div>', ''
        
        # Fix the chart height calculation to ensure minimum height
        $content = $content -replace '\$\{([^}]*?)/ 50 \* 200\}', '${Math.max($1 / 50 * 200, 10)}'
        
        # Save the fixed content
        Set-Content -Path $file -Value $content -NoNewline
    }
}

Write-Host "`n2. Font sizes have been fixed in globals.css" -ForegroundColor Yellow
Write-Host "   - Changed from fluid typography (clamp) to fixed sizes" -ForegroundColor Gray
Write-Host "   - Body text: 14px (0.875rem)" -ForegroundColor Gray
Write-Host "   - Titles: 18px (1.125rem)" -ForegroundColor Gray

Write-Host "`n3. Summary of fixes:" -ForegroundColor Green
Write-Host "   ✅ Removed empty chart placeholders (dashed borders)" -ForegroundColor Green
Write-Host "   ✅ Fixed oversized text (removed fluid typography)" -ForegroundColor Green
Write-Host "   ✅ Improved chart rendering with minimum heights" -ForegroundColor Green

Write-Host "`n4. Next steps:" -ForegroundColor Cyan
Write-Host "   - Refresh your browser (Ctrl+F5) to see changes" -ForegroundColor White
Write-Host "   - The text should now be normal sized" -ForegroundColor White
Write-Host "   - Empty chart placeholders should be gone" -ForegroundColor White
Write-Host "   - Access the app at: http://localhost:3001" -ForegroundColor Yellow

Write-Host "`nUI fixes complete!" -ForegroundColor Green