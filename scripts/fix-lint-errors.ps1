# PowerShell script to fix all linting errors in the codebase

Write-Host "PolicyCortex Linting Fix Script" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent (Get-Location).Path

# Install required tools
Write-Host "`nInstalling Python linting tools..." -ForegroundColor Yellow
pip install flake8 black isort autopep8 --quiet

# Fix Python files
Write-Host "`nFixing Python linting errors..." -ForegroundColor Yellow

# Run autopep8 to fix basic issues (whitespace, etc.)
Write-Host "  Running autopep8 to fix whitespace issues..." -ForegroundColor Cyan
Get-ChildItem -Path $ProjectRoot -Include "*.py" -Recurse | ForEach-Object {
    $file = $_.FullName
    if ($file -notmatch "venv|node_modules|\.git|__pycache__|migrations") {
        autopep8 --in-place --aggressive --aggressive $file 2>$null
    }
}

# Run black for code formatting
Write-Host "  Running black for code formatting..." -ForegroundColor Cyan
$backendPath = Join-Path $ProjectRoot "backend"
if (Test-Path $backendPath) {
    black $backendPath --line-length 100 --quiet 2>$null
}

# Run isort for import sorting
Write-Host "  Running isort for import sorting..." -ForegroundColor Cyan
if (Test-Path $backendPath) {
    isort $backendPath --profile black --line-length 100 --quiet 2>$null
}

# Check remaining issues with flake8
Write-Host "`nChecking remaining issues with flake8..." -ForegroundColor Yellow
$flakeOutput = flake8 $backendPath --max-line-length=100 --exclude=venv,__pycache__,migrations 2>&1

if ($flakeOutput) {
    Write-Host "  Remaining issues found. Fixing specific problems..." -ForegroundColor Yellow
    
    # Fix specific files mentioned in the error
    $filesToFix = @(
        "backend\services\ai_engine\services\sentiment_analyzer.py"
    )
    
    foreach ($relPath in $filesToFix) {
        $filePath = Join-Path $ProjectRoot $relPath
        if (Test-Path $filePath) {
            Write-Host "  Fixing $relPath..." -ForegroundColor Gray
            
            # Read file content
            $content = Get-Content $filePath -Raw
            
            # Remove trailing whitespace from lines
            $content = $content -replace '[ \t]+$', ''
            
            # Ensure file ends with newline
            if (-not $content.EndsWith("`n")) {
                $content += "`n"
            }
            
            # Remove multiple blank lines
            $content = $content -replace '(\r?\n){3,}', "`n`n"
            
            # Save fixed content
            $content | Set-Content $filePath -NoNewline
        }
    }
}

# Fix TypeScript/JavaScript files
Write-Host "`nFixing TypeScript/JavaScript linting errors..." -ForegroundColor Yellow

$frontendPath = Join-Path $ProjectRoot "frontend"
if (Test-Path $frontendPath) {
    Set-Location $frontendPath
    
    # Install ESLint and Prettier if needed
    if (-not (Test-Path "node_modules")) {
        Write-Host "  Installing frontend dependencies..." -ForegroundColor Cyan
        npm install --silent
    }
    
    # Run ESLint fix
    Write-Host "  Running ESLint fix..." -ForegroundColor Cyan
    npx eslint . --fix --ext .ts,.tsx,.js,.jsx 2>$null
    
    # Run Prettier
    Write-Host "  Running Prettier..." -ForegroundColor Cyan
    npx prettier --write "src/**/*.{ts,tsx,js,jsx,css,scss,json}" 2>$null
}

# Final verification
Write-Host "`nRunning final verification..." -ForegroundColor Yellow
Set-Location $ProjectRoot

# Check Python files
$pythonIssues = flake8 backend --max-line-length=100 --exclude=venv,__pycache__,migrations --count 2>&1
if ($pythonIssues -match "^\d+$" -and [int]$pythonIssues -eq 0) {
    Write-Host "  ✓ Python: No linting errors found" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Python: $pythonIssues issues remaining" -ForegroundColor Yellow
}

# Check TypeScript files
if (Test-Path $frontendPath) {
    Set-Location $frontendPath
    $tsIssues = npx eslint . --ext .ts,.tsx --format unix 2>&1 | Measure-Object -Line
    if ($tsIssues.Lines -eq 0) {
        Write-Host "  ✓ TypeScript: No linting errors found" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ TypeScript: $($tsIssues.Lines) issues remaining" -ForegroundColor Yellow
    }
}

Write-Host "`nLinting fixes completed!" -ForegroundColor Green
Write-Host "Run 'git diff' to review changes before committing." -ForegroundColor Cyan

Set-Location $ProjectRoot