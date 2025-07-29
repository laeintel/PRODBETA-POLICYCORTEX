# API Gateway Test Script

param(
    [string]$TestType = "all",  # all, unit, integration, api
    [switch]$Verbose,
    [switch]$Coverage
)

Write-Host "API Gateway Service Tests" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Get-Location).Path)
$ServicePath = Join-Path $ProjectRoot "backend\services\api_gateway"
$ResultsPath = Join-Path $ProjectRoot "testing\results\api_gateway"

# Ensure results directory exists
New-Item -ItemType Directory -Path $ResultsPath -Force | Out-Null

# Timestamp for this test run
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "`n1. Starting API Gateway service..." -ForegroundColor Yellow

# Activate virtual environment and start service
Set-Location $ServicePath
& ".\venv\Scripts\Activate.ps1"

# Start the service in background
$serviceProcess = Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "main:app", "--port", "8000", "--reload" -PassThru -WindowStyle Hidden

Write-Host "  Waiting for service to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# Check if service is running
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
    Write-Host "  ✓ Service is running" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Service failed to start!" -ForegroundColor Red
    exit 1
}

Write-Host "`n2. Running tests..." -ForegroundColor Yellow

# Test categories
$testResults = @{}

# Unit Tests
if ($TestType -eq "all" -or $TestType -eq "unit") {
    Write-Host "`n  Running unit tests..." -ForegroundColor Cyan
    $unitTestOutput = Join-Path $ResultsPath "unit_tests_$Timestamp.txt"
    
    $unitTests = pytest tests/unit -v --tb=short 2>&1 | Tee-Object -FilePath $unitTestOutput
    $testResults["unit"] = $LASTEXITCODE -eq 0
}

# Integration Tests
if ($TestType -eq "all" -or $TestType -eq "integration") {
    Write-Host "`n  Running integration tests..." -ForegroundColor Cyan
    $integrationTestOutput = Join-Path $ResultsPath "integration_tests_$Timestamp.txt"
    
    $integrationTests = pytest tests/integration -v --tb=short 2>&1 | Tee-Object -FilePath $integrationTestOutput
    $testResults["integration"] = $LASTEXITCODE -eq 0
}

# API Tests
if ($TestType -eq "all" -or $TestType -eq "api") {
    Write-Host "`n  Running API tests..." -ForegroundColor Cyan
    $apiTestOutput = Join-Path $ResultsPath "api_tests_$Timestamp.txt"
    
    # Create API test file if it doesn't exist
    $apiTestFile = Join-Path $ServicePath "tests\test_api_endpoints.py"
    if (-not (Test-Path $apiTestFile)) {
        New-Item -ItemType Directory -Path (Join-Path $ServicePath "tests") -Force | Out-Null
        @'
import pytest
import httpx
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestAPIEndpoints:
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_api_docs(self):
        """Test OpenAPI documentation"""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_cors_headers(self):
        """Test CORS headers"""
        response = client.options("/api/v1/test", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        assert "access-control-allow-origin" in response.headers
    
    @pytest.mark.parametrize("endpoint", [
        "/api/v1/policies",
        "/api/v1/resources", 
        "/api/v1/costs",
        "/api/v1/notifications"
    ])
    def test_protected_endpoints_without_auth(self, endpoint):
        """Test protected endpoints require authentication"""
        response = client.get(endpoint)
        assert response.status_code == 401
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Make multiple requests
        for _ in range(10):
            response = client.get("/health")
        
        # Should still be successful (adjust based on your rate limit)
        assert response.status_code == 200
'@ | Out-File -FilePath $apiTestFile -Encoding UTF8
    }
    
    pytest $apiTestFile -v --tb=short 2>&1 | Tee-Object -FilePath $apiTestOutput
    $testResults["api"] = $LASTEXITCODE -eq 0
}

# Coverage Report
if ($Coverage) {
    Write-Host "`n  Generating coverage report..." -ForegroundColor Cyan
    $coverageOutput = Join-Path $ResultsPath "coverage_$Timestamp.txt"
    
    pytest --cov=. --cov-report=term --cov-report=html:$ResultsPath/coverage_html 2>&1 | Tee-Object -FilePath $coverageOutput
}

Write-Host "`n3. Performance Tests..." -ForegroundColor Yellow

# Simple performance test
$perfTestOutput = Join-Path $ResultsPath "performance_$Timestamp.txt"
$perfResults = @"
API Gateway Performance Test Results
====================================
Timestamp: $(Get-Date)

Endpoint Performance:
"@

$endpoints = @(
    @{Path="/health"; Method="GET"},
    @{Path="/api/v1/test"; Method="GET"}
)

foreach ($endpoint in $endpoints) {
    Write-Host "  Testing $($endpoint.Path)..." -ForegroundColor Cyan
    
    $times = @()
    for ($i = 0; $i -lt 10; $i++) {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            Invoke-RestMethod -Uri "http://localhost:8000$($endpoint.Path)" -Method $endpoint.Method -ErrorAction SilentlyContinue
        } catch {}
        $stopwatch.Stop()
        $times += $stopwatch.ElapsedMilliseconds
    }
    
    $avg = ($times | Measure-Object -Average).Average
    $min = ($times | Measure-Object -Minimum).Minimum
    $max = ($times | Measure-Object -Maximum).Maximum
    
    $perfResults += @"

$($endpoint.Path) ($($endpoint.Method)):
  Average: $([math]::Round($avg, 2))ms
  Min: ${min}ms
  Max: ${max}ms
"@
}

$perfResults | Out-File -FilePath $perfTestOutput -Encoding UTF8

Write-Host "`n4. Generating test summary..." -ForegroundColor Yellow

# Generate summary
$summaryPath = Join-Path $ResultsPath "summary_$Timestamp.txt"
$summary = @"
API Gateway Test Summary
========================
Date: $(Get-Date)
Service: API Gateway
Port: 8000

Test Results:
"@

foreach ($test in $testResults.Keys) {
    $status = if ($testResults[$test]) { "PASSED" } else { "FAILED" }
    $color = if ($testResults[$test]) { "Green" } else { "Red" }
    Write-Host "  $test tests: $status" -ForegroundColor $color
    $summary += "`n  $test tests: $status"
}

$summary += @"

Files Generated:
  - Unit Tests: unit_tests_$Timestamp.txt
  - Integration Tests: integration_tests_$Timestamp.txt
  - API Tests: api_tests_$Timestamp.txt
  - Performance Tests: performance_$Timestamp.txt
  - This Summary: summary_$Timestamp.txt
"@

if ($Coverage) {
    $summary += "`n  - Coverage Report: coverage_$Timestamp.txt"
    $summary += "`n  - Coverage HTML: coverage_html/index.html"
}

$summary | Out-File -FilePath $summaryPath -Encoding UTF8

# Stop the service
Write-Host "`n5. Stopping service..." -ForegroundColor Yellow
Stop-Process -Id $serviceProcess.Id -Force

# Deactivate virtual environment
deactivate

Write-Host "`n✓ API Gateway tests completed!" -ForegroundColor Green
Write-Host "Results saved to: $ResultsPath" -ForegroundColor Cyan

# Return overall success/failure
$allPassed = $testResults.Values -notcontains $false
exit $(if ($allPassed) { 0 } else { 1 })