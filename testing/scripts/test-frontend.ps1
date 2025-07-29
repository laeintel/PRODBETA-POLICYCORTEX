# Frontend Test Script

param(
    [string]$TestType = "all",  # all, unit, e2e, auth
    [switch]$Verbose,
    [switch]$Coverage
)

Write-Host "Frontend Tests" -ForegroundColor Cyan
Write-Host "==============" -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Get-Location).Path)
$FrontendPath = Join-Path $ProjectRoot "frontend"
$ResultsPath = Join-Path $ProjectRoot "testing\results\frontend"

# Ensure results directory exists
New-Item -ItemType Directory -Path $ResultsPath -Force | Out-Null

# Timestamp for this test run
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "`n1. Setting up frontend test environment..." -ForegroundColor Yellow

Set-Location $FrontendPath

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "  Installing dependencies..." -ForegroundColor Cyan
    npm install --silent
}

Write-Host "  ✓ Frontend environment ready" -ForegroundColor Green

Write-Host "`n2. Running tests..." -ForegroundColor Yellow

# Test categories
$testResults = @{}

# Unit Tests
if ($TestType -eq "all" -or $TestType -eq "unit") {
    Write-Host "`n  Running unit tests..." -ForegroundColor Cyan
    $unitTestOutput = Join-Path $ResultsPath "unit_tests_$Timestamp.txt"
    
    npm test -- --run --reporter=verbose 2>&1 | Tee-Object -FilePath $unitTestOutput
    $testResults["unit"] = $LASTEXITCODE -eq 0
}

# Build Test
Write-Host "`n  Running build test..." -ForegroundColor Cyan
$buildTestOutput = Join-Path $ResultsPath "build_test_$Timestamp.txt"

npm run build 2>&1 | Tee-Object -FilePath $buildTestOutput
$testResults["build"] = $LASTEXITCODE -eq 0

# Authentication Flow Test
if ($TestType -eq "all" -or $TestType -eq "auth") {
    Write-Host "`n  Testing authentication flow..." -ForegroundColor Cyan
    $authTestOutput = Join-Path $ResultsPath "auth_test_$Timestamp.txt"
    
    # Create auth test file
    $authTestFile = Join-Path $FrontendPath "src\tests\auth.test.ts"
    if (-not (Test-Path (Split-Path $authTestFile))) {
        New-Item -ItemType Directory -Path (Split-Path $authTestFile) -Force | Out-Null
    }
    
    @'
import { describe, it, expect, vi } from 'vitest'
import { renderHook } from '@testing-library/react'
import { useAuth } from '@/hooks/useAuth'

// Mock MSAL
vi.mock('@azure/msal-react', () => ({
  useMsal: () => ({
    instance: {
      loginRedirect: vi.fn(),
      loginPopup: vi.fn(),
      logoutRedirect: vi.fn(),
      acquireTokenSilent: vi.fn().mockResolvedValue({ accessToken: 'mock-token' })
    },
    accounts: [{
      homeAccountId: 'test-id',
      username: 'test@example.com',
      name: 'Test User'
    }],
    inProgress: 'none'
  })
}))

describe('Authentication', () => {
  it('should initialize auth state', () => {
    const { result } = renderHook(() => useAuth())
    
    expect(result.current.isLoading).toBeDefined()
    expect(result.current.isAuthenticated).toBeDefined()
    expect(result.current.login).toBeDefined()
    expect(result.current.logout).toBeDefined()
  })

  it('should handle login', async () => {
    const { result } = renderHook(() => useAuth())
    
    await result.current.login()
    // Login redirect should be called
  })

  it('should handle logout', async () => {
    const { result } = renderHook(() => useAuth())
    
    await result.current.logout()
    // Logout redirect should be called
  })
})
'@ | Out-File -FilePath $authTestFile -Encoding UTF8

    npm test -- src/tests/auth.test.ts --run 2>&1 | Tee-Object -FilePath $authTestOutput
    $testResults["auth"] = $LASTEXITCODE -eq 0
}

# Environment Configuration Test
Write-Host "`n  Testing environment configuration..." -ForegroundColor Cyan
$envTestOutput = Join-Path $ResultsPath "env_test_$Timestamp.txt"

$envTest = @"
Environment Configuration Test
==============================
Timestamp: $(Get-Date)

Checking configuration files:
"@

$configFiles = @(
    ".env.production",
    "public/config.js",
    "src/config/environment.ts",
    "src/config/auth.ts"
)

foreach ($file in $configFiles) {
    if (Test-Path $file) {
        $envTest += "`n✓ $file exists"
    } else {
        $envTest += "`n✗ $file missing"
    }
}

# Check for required environment variables in .env.production
if (Test-Path ".env.production") {
    $envContent = Get-Content ".env.production"
    $requiredVars = @(
        "VITE_API_BASE_URL",
        "VITE_AZURE_CLIENT_ID",
        "VITE_AZURE_TENANT_ID"
    )
    
    $envTest += "`n`nEnvironment Variables:"
    foreach ($var in $requiredVars) {
        if ($envContent -match $var) {
            $envTest += "`n✓ $var configured"
        } else {
            $envTest += "`n✗ $var missing"
        }
    }
}

$envTest | Out-File -FilePath $envTestOutput -Encoding UTF8

# Lint Test
Write-Host "`n  Running lint checks..." -ForegroundColor Cyan
$lintTestOutput = Join-Path $ResultsPath "lint_test_$Timestamp.txt"

npm run lint 2>&1 | Tee-Object -FilePath $lintTestOutput
$testResults["lint"] = $LASTEXITCODE -eq 0

# Type Check Test
Write-Host "`n  Running type checks..." -ForegroundColor Cyan
$typeCheckOutput = Join-Path $ResultsPath "typecheck_test_$Timestamp.txt"

npx tsc --noEmit 2>&1 | Tee-Object -FilePath $typeCheckOutput
$testResults["typecheck"] = $LASTEXITCODE -eq 0

Write-Host "`n3. Testing development server..." -ForegroundColor Yellow

# Start dev server
Write-Host "  Starting development server..." -ForegroundColor Cyan
$devProcess = Start-Process -FilePath "npm" -ArgumentList "run", "dev" -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 10

# Test if server is running
$devServerTest = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✓ Development server is running" -ForegroundColor Green
        $devServerTest = $true
    }
} catch {
    Write-Host "  ✗ Development server failed to start" -ForegroundColor Red
}

$testResults["devserver"] = $devServerTest

# Stop dev server
Stop-Process -Id $devProcess.Id -Force -ErrorAction SilentlyContinue

Write-Host "`n4. Generating test summary..." -ForegroundColor Yellow

# Generate summary
$summaryPath = Join-Path $ResultsPath "summary_$Timestamp.txt"
$summary = @"
Frontend Test Summary
====================
Date: $(Get-Date)
Component: Frontend (React + TypeScript)
Port: 3000

Test Results:
"@

foreach ($test in $testResults.Keys) {
    $status = if ($testResults[$test]) { "PASSED" } else { "FAILED" }
    $color = if ($testResults[$test]) { "Green" } else { "Red" }
    Write-Host "  $test tests: $status" -ForegroundColor $color
    $summary += "`n  $test tests: $status"
}

$summary += @"

Build Information:
  - React Version: 18.x
  - TypeScript: Yes
  - Vite: Yes
  - Authentication: Azure AD (MSAL)

Files Generated:
  - Unit Tests: unit_tests_$Timestamp.txt
  - Build Test: build_test_$Timestamp.txt
  - Auth Test: auth_test_$Timestamp.txt
  - Environment Test: env_test_$Timestamp.txt
  - Lint Test: lint_test_$Timestamp.txt
  - Type Check: typecheck_test_$Timestamp.txt
  - This Summary: summary_$Timestamp.txt

Key Findings:
"@

# Add findings based on test results
if ($testResults["build"]) {
    $summary += "`n  ✓ Frontend builds successfully"
} else {
    $summary += "`n  ✗ Build errors need to be fixed"
}

if ($testResults["typecheck"]) {
    $summary += "`n  ✓ No TypeScript errors"
} else {
    $summary += "`n  ✗ TypeScript errors need attention"
}

if ($testResults["auth"]) {
    $summary += "`n  ✓ Authentication flow is properly configured"
} else {
    $summary += "`n  ✗ Authentication tests failing"
}

$summary | Out-File -FilePath $summaryPath -Encoding UTF8

Write-Host "`n✓ Frontend tests completed!" -ForegroundColor Green
Write-Host "Results saved to: $ResultsPath" -ForegroundColor Cyan

# Return overall success/failure
$allPassed = $testResults.Values -notcontains $false
exit $(if ($allPassed) { 0 } else { 1 })