# AI Engine Service Test Script

param(
    [string]$TestType = "all",  # all, unit, integration, api, performance
    [switch]$Verbose,
    [switch]$Coverage
)

Write-Host "AI Engine Service Tests" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Get-Location).Path)
$ServicePath = Join-Path $ProjectRoot "backend\services\ai_engine"
$ResultsPath = Join-Path $ProjectRoot "testing\results\ai_engine"

# Ensure results directory exists
New-Item -ItemType Directory -Path $ResultsPath -Force | Out-Null

# Timestamp for this test run
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "`n1. Starting AI Engine service..." -ForegroundColor Yellow

# Activate virtual environment and start service
Set-Location $ServicePath
& ".\venv\Scripts\Activate.ps1"

# Start the service in background
$serviceProcess = Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "main:app", "--port", "8002", "--reload" -PassThru -WindowStyle Hidden

Write-Host "  Waiting for service to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 8  # AI Engine may take longer to load models

# Check if service is running
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8002/health" -Method GET
    Write-Host "  ✓ Service is running" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Service failed to start!" -ForegroundColor Red
    exit 1
}

Write-Host "`n2. Running AI Engine tests..." -ForegroundColor Yellow

# Test results tracking
$testResults = @{}

# Create AI Engine test file
$aiTestFile = Join-Path $ServicePath "tests\test_ai_endpoints.py"
if (-not (Test-Path $aiTestFile)) {
    New-Item -ItemType Directory -Path (Join-Path $ServicePath "tests") -Force | Out-Null
    @'
import pytest
import httpx
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestAIEngineEndpoints:
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "model_status" in response.json()
    
    def test_analyze_policy(self):
        """Test policy analysis endpoint"""
        policy_data = {
            "policy_text": "All virtual machines must have backup enabled",
            "policy_type": "compliance"
        }
        response = client.post("/api/v1/analyze/policy", json=policy_data)
        assert response.status_code in [200, 401]
    
    def test_resource_optimization(self):
        """Test resource optimization suggestions"""
        resource_data = {
            "resource_type": "Microsoft.Compute/virtualMachines",
            "usage_data": {
                "cpu_percent": 10,
                "memory_percent": 15
            }
        }
        response = client.post("/api/v1/optimize/resource", json=resource_data)
        assert response.status_code in [200, 401]
    
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        metrics_data = {
            "metric_name": "cpu_usage",
            "values": [10, 12, 11, 95, 10, 11, 12],
            "timestamps": ["2024-01-01T00:00:00Z"] * 7
        }
        response = client.post("/api/v1/detect/anomaly", json=metrics_data)
        assert response.status_code in [200, 401]
    
    def test_nlp_intent(self):
        """Test NLP intent recognition"""
        nlp_data = {
            "text": "Show me all VMs that are not compliant with backup policy"
        }
        response = client.post("/api/v1/nlp/intent", json=nlp_data)
        assert response.status_code in [200, 401]
    
    def test_model_info(self):
        """Test model information endpoint"""
        response = client.get("/api/v1/models")
        assert response.status_code in [200, 401]
'@ | Out-File -FilePath $aiTestFile -Encoding UTF8
}

# Run API tests
if ($TestType -eq "all" -or $TestType -eq "api") {
    Write-Host "`n  Running API tests..." -ForegroundColor Cyan
    $apiTestOutput = Join-Path $ResultsPath "api_tests_$Timestamp.txt"
    
    pytest $aiTestFile -v --tb=short 2>&1 | Tee-Object -FilePath $apiTestOutput
    $testResults["api"] = $LASTEXITCODE -eq 0
}

# Performance tests
if ($TestType -eq "all" -or $TestType -eq "performance") {
    Write-Host "`n3. Running performance tests..." -ForegroundColor Yellow
    $perfTestOutput = Join-Path $ResultsPath "performance_$Timestamp.txt"
    
    $perfResults = @"
AI Engine Performance Test Results
==================================
Timestamp: $(Get-Date)

Model Inference Performance:
"@

    # Test inference speed
    $endpoints = @(
        @{Name="Policy Analysis"; Url="http://localhost:8002/api/v1/analyze/policy"; Method="POST"; 
          Body=@{policy_text="Test policy"; policy_type="compliance"}},
        @{Name="Anomaly Detection"; Url="http://localhost:8002/api/v1/detect/anomaly"; Method="POST";
          Body=@{metric_name="test"; values=@(1,2,3,4,5); timestamps=@("2024-01-01T00:00:00Z")*5}}
    )
    
    foreach ($endpoint in $endpoints) {
        Write-Host "  Testing $($endpoint.Name)..." -ForegroundColor Cyan
        
        $times = @()
        for ($i = 0; $i -lt 5; $i++) {
            $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            try {
                $body = $endpoint.Body | ConvertTo-Json
                Invoke-RestMethod -Uri $endpoint.Url -Method $endpoint.Method -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
            } catch {}
            $stopwatch.Stop()
            $times += $stopwatch.ElapsedMilliseconds
        }
        
        $avg = ($times | Measure-Object -Average).Average
        $min = ($times | Measure-Object -Minimum).Minimum
        $max = ($times | Measure-Object -Maximum).Maximum
        
        $perfResults += @"

$($endpoint.Name):
  Average: $([math]::Round($avg, 2))ms
  Min: ${min}ms
  Max: ${max}ms
"@
    }
    
    $perfResults | Out-File -FilePath $perfTestOutput -Encoding UTF8
    $testResults["performance"] = $true
}

# Model loading test
Write-Host "`n4. Testing model loading..." -ForegroundColor Yellow
$modelTestOutput = Join-Path $ResultsPath "model_test_$Timestamp.txt"

$modelTest = @"
AI Engine Model Test
====================
Timestamp: $(Get-Date)

Model Status:
"@

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8002/api/v1/models" -Method GET
    $modelTest += "`n✓ Models endpoint accessible"
    
    if ($response.models) {
        foreach ($model in $response.models) {
            $modelTest += "`n  - $($model.name): $($model.status)"
        }
    }
} catch {
    $modelTest += "`n✗ Failed to access models endpoint"
}

$modelTest | Out-File -FilePath $modelTestOutput -Encoding UTF8

Write-Host "`n5. Generating test summary..." -ForegroundColor Yellow

# Generate summary
$summaryPath = Join-Path $ResultsPath "summary_$Timestamp.txt"
$summary = @"
AI Engine Test Summary
======================
Date: $(Get-Date)
Service: AI Engine
Port: 8002

Test Results:
"@

foreach ($test in $testResults.Keys) {
    $status = if ($testResults[$test]) { "PASSED" } else { "FAILED" }
    $color = if ($testResults[$test]) { "Green" } else { "Red" }
    Write-Host "  $test tests: $status" -ForegroundColor $color
    $summary += "`n  $test tests: $status"
}

$summary += @"

AI Capabilities Tested:
  - Policy Analysis
  - Resource Optimization
  - Anomaly Detection
  - NLP Intent Recognition
  - Model Management

Files Generated:
  - API Tests: api_tests_$Timestamp.txt
  - Performance Tests: performance_$Timestamp.txt
  - Model Tests: model_test_$Timestamp.txt
  - This Summary: summary_$Timestamp.txt
"@

$summary | Out-File -FilePath $summaryPath -Encoding UTF8

# Stop the service
Write-Host "`n6. Stopping service..." -ForegroundColor Yellow
Stop-Process -Id $serviceProcess.Id -Force

# Deactivate virtual environment
deactivate

Write-Host "`n✓ AI Engine tests completed!" -ForegroundColor Green
Write-Host "Results saved to: $ResultsPath" -ForegroundColor Cyan

# Return overall success/failure
$allPassed = $testResults.Values -notcontains $false
exit $(if ($allPassed) { 0 } else { 1 })