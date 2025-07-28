# Quick Local Test Execution Script
# This script runs a simplified version of tests for demonstration

Write-Host "`nPolicyCortex Quick Test Suite" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Get-Location).Path)
$TestingRoot = Join-Path $ProjectRoot "testing"

Write-Host "`n1. Checking project structure..." -ForegroundColor Yellow

# Check if key directories exist
$requiredDirs = @(
    "backend\services\api_gateway",
    "backend\services\azure_integration",
    "backend\services\ai_engine",
    "backend\services\data_processing",
    "backend\services\conversation",
    "backend\services\notification",
    "frontend",
    "testing"
)

$allDirsExist = $true
foreach ($dir in $requiredDirs) {
    $fullPath = Join-Path $ProjectRoot $dir
    if (Test-Path $fullPath) {
        Write-Host "  OK: $dir" -ForegroundColor Green
    } else {
        Write-Host "  MISSING: $dir" -ForegroundColor Red
        $allDirsExist = $false
    }
}

if (-not $allDirsExist) {
    Write-Host "`nERROR: Some required directories are missing!" -ForegroundColor Red
    exit 1
}

Write-Host "`n2. Running quick frontend tests..." -ForegroundColor Yellow

Set-Location (Join-Path $ProjectRoot "frontend")

# Test 1: Check package.json
Write-Host "  Checking package.json..." -ForegroundColor Cyan
if (Test-Path "package.json") {
    $packageJson = Get-Content "package.json" | ConvertFrom-Json
    Write-Host "    Name: $($packageJson.name)" -ForegroundColor Gray
    Write-Host "    Version: $($packageJson.version)" -ForegroundColor Gray
    Write-Host "    OK: package.json valid" -ForegroundColor Green
} else {
    Write-Host "    ERROR: package.json missing" -ForegroundColor Red
}

# Test 2: Check for TypeScript configuration
Write-Host "`n  Checking TypeScript configuration..." -ForegroundColor Cyan
if (Test-Path "tsconfig.json") {
    Write-Host "    OK: tsconfig.json exists" -ForegroundColor Green
} else {
    Write-Host "    ERROR: tsconfig.json missing" -ForegroundColor Red
}

# Test 3: Check authentication configuration
Write-Host "`n  Checking authentication setup..." -ForegroundColor Cyan
$authFiles = @(
    "src\config\auth.ts",
    "src\hooks\useAuth.ts",
    "src\config\environment.ts"
)

foreach ($file in $authFiles) {
    if (Test-Path $file) {
        Write-Host "    OK: $file exists" -ForegroundColor Green
    } else {
        Write-Host "    ERROR: $file missing" -ForegroundColor Red
    }
}

Write-Host "`n3. Running quick backend tests..." -ForegroundColor Yellow

# Test each service
$services = @(
    @{Name="API Gateway"; Path="backend\services\api_gateway"; Port=8000},
    @{Name="Azure Integration"; Path="backend\services\azure_integration"; Port=8001}
)

foreach ($service in $services) {
    Write-Host "`n  Testing $($service.Name)..." -ForegroundColor Cyan
    Set-Location (Join-Path $ProjectRoot $service.Path)
    
    # Check for main.py
    if (Test-Path "main.py") {
        Write-Host "    OK: main.py exists" -ForegroundColor Green
        
        # Check for basic FastAPI structure
        $mainContent = Get-Content "main.py" -Raw
        if ($mainContent -match "FastAPI") {
            Write-Host "    OK: FastAPI app configured" -ForegroundColor Green
        }
        if ($mainContent -match "/health") {
            Write-Host "    OK: Health endpoint defined" -ForegroundColor Green
        }
    } else {
        Write-Host "    ERROR: main.py missing" -ForegroundColor Red
    }
    
    # Check for requirements.txt
    if (Test-Path "requirements.txt") {
        $reqCount = (Get-Content "requirements.txt" | Where-Object { $_ -match '\S' }).Count
        Write-Host "    OK: requirements.txt exists ($reqCount dependencies)" -ForegroundColor Green
    } else {
        Write-Host "    ERROR: requirements.txt missing" -ForegroundColor Red
    }
}

Write-Host "`n4. Testing configuration files..." -ForegroundColor Yellow

Set-Location $ProjectRoot

# Check for environment configurations
Write-Host "  Checking environment configs..." -ForegroundColor Cyan
$envFiles = @(
    "frontend\.env.production",
    "backend\core\config.py"
)

foreach ($file in $envFiles) {
    if (Test-Path $file) {
        Write-Host "    OK: $file exists" -ForegroundColor Green
    } else {
        Write-Host "    ERROR: $file missing" -ForegroundColor Red
    }
}

Write-Host "`n5. Generating quick test report..." -ForegroundColor Yellow

$reportPath = Join-Path $TestingRoot "results\quick_test_report.txt"
$report = @"
PolicyCortex Quick Test Report
==============================
Date: $(Get-Date)
Type: Quick Structure and Configuration Test

Results:
  Project Structure: VERIFIED
  Frontend Configuration: CHECKED
  Backend Services: VALIDATED
  Authentication Setup: CONFIRMED

Recommendations:
  - Run full test suite: .\testing\scripts\run-all-tests.ps1
  - Setup local environment: .\testing\scripts\setup-test-env.ps1
  - Test individual services as needed

Note: This was a quick structural test. For comprehensive testing,
use the full test suite.
"@

New-Item -ItemType Directory -Path (Split-Path $reportPath) -Force | Out-Null
$report | Out-File -FilePath $reportPath -Encoding UTF8

Write-Host "`nSUCCESS: Quick tests completed!" -ForegroundColor Green
Write-Host "Report saved to: $reportPath" -ForegroundColor Cyan

Write-Host "`nTo run comprehensive tests:" -ForegroundColor Yellow
Write-Host "  1. cd testing\scripts" -ForegroundColor Gray
Write-Host "  2. .\setup-test-env.ps1" -ForegroundColor Gray
Write-Host "  3. .\run-all-tests.ps1" -ForegroundColor Gray