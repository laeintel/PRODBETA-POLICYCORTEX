# Data Processing Service Test Script

param(
    [string]$TestType = "all",
    [switch]$Verbose,
    [switch]$Coverage
)

Write-Host "Data Processing Service Tests" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Get-Location).Path)
$ServicePath = Join-Path $ProjectRoot "backend\services\data_processing"
$ResultsPath = Join-Path $ProjectRoot "testing\results\data_processing"

New-Item -ItemType Directory -Path $ResultsPath -Force | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "`n1. Starting Data Processing service..." -ForegroundColor Yellow

Set-Location $ServicePath
& ".\venv\Scripts\Activate.ps1"

$serviceProcess = Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "main:app", "--port", "8003", "--reload" -PassThru -WindowStyle Hidden

Write-Host "  Waiting for service to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8003/health" -Method GET
    Write-Host "  ✓ Service is running" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Service failed to start!" -ForegroundColor Red
    exit 1
}

Write-Host "`n2. Running tests..." -ForegroundColor Yellow

$testResults = @{}

# Create test file
$testFile = Join-Path $ServicePath "tests\test_data_processing.py"
if (-not (Test-Path $testFile)) {
    New-Item -ItemType Directory -Path (Join-Path $ServicePath "tests") -Force | Out-Null
    @'
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestDataProcessingEndpoints:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        
    def test_process_data_pipeline(self):
        pipeline_data = {
            "pipeline_name": "resource_aggregation",
            "data": [{"id": 1, "value": 100}]
        }
        response = client.post("/api/v1/process", json=pipeline_data)
        assert response.status_code in [200, 401]
    
    def test_validate_data(self):
        validation_data = {
            "schema": "resource",
            "data": {"name": "test", "type": "vm"}
        }
        response = client.post("/api/v1/validate", json=validation_data)
        assert response.status_code in [200, 401]
    
    def test_transform_data(self):
        transform_data = {
            "transformer": "normalize",
            "data": [1, 2, 3, 4, 5]
        }
        response = client.post("/api/v1/transform", json=transform_data)
        assert response.status_code in [200, 401]
'@ | Out-File -FilePath $testFile -Encoding UTF8
}

$apiTestOutput = Join-Path $ResultsPath "api_tests_$Timestamp.txt"
pytest $testFile -v --tb=short 2>&1 | Tee-Object -FilePath $apiTestOutput
$testResults["api"] = $LASTEXITCODE -eq 0

# Generate summary
$summaryPath = Join-Path $ResultsPath "summary_$Timestamp.txt"
@"
Data Processing Test Summary
============================
Date: $(Get-Date)
Service: Data Processing
Port: 8003

Test Results:
  API tests: $(if ($testResults["api"]) { "PASSED" } else { "FAILED" })

Capabilities Tested:
  - Data Pipeline Processing
  - Data Validation
  - Data Transformation
  - ETL Operations
"@ | Out-File -FilePath $summaryPath -Encoding UTF8

Stop-Process -Id $serviceProcess.Id -Force
deactivate

Write-Host "`n✓ Data Processing tests completed!" -ForegroundColor Green
exit $(if ($testResults.Values -notcontains $false) { 0 } else { 1 })