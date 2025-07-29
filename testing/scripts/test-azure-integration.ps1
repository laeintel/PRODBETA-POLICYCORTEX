# Azure Integration Service Test Script

param(
    [string]$TestType = "all",  # all, unit, integration, api
    [switch]$Verbose,
    [switch]$Coverage
)

Write-Host "Azure Integration Service Tests" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Get-Location).Path)
$ServicePath = Join-Path $ProjectRoot "backend\services\azure_integration"
$ResultsPath = Join-Path $ProjectRoot "testing\results\azure_integration"

# Ensure results directory exists
New-Item -ItemType Directory -Path $ResultsPath -Force | Out-Null

# Timestamp for this test run
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "`n1. Starting Azure Integration service..." -ForegroundColor Yellow

# Activate virtual environment and start service
Set-Location $ServicePath
& ".\venv\Scripts\Activate.ps1"

# Set test environment variables
$env:AZURE_TENANT_ID = "test-tenant-id"
$env:AZURE_CLIENT_ID = "test-client-id"
$env:AZURE_CLIENT_SECRET = "test-client-secret"
$env:AZURE_SUBSCRIPTION_ID = "test-subscription-id"
$env:USE_MOCK_AZURE = "true"

# Start the service in background
$serviceProcess = Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "main:app", "--port", "8001", "--reload" -PassThru -WindowStyle Hidden

Write-Host "  Waiting for service to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# Check if service is running
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method GET
    Write-Host "  ✓ Service is running" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Service failed to start!" -ForegroundColor Red
    exit 1
}

Write-Host "`n2. Running tests..." -ForegroundColor Yellow

# Create test file for Azure Integration specific tests
$azureTestFile = Join-Path $ServicePath "tests\test_azure_endpoints.py"
if (-not (Test-Path $azureTestFile)) {
    New-Item -ItemType Directory -Path (Join-Path $ServicePath "tests") -Force | Out-Null
    @'
import pytest
import httpx
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from main import app

client = TestClient(app)

class TestAzureEndpoints:
    @pytest.fixture
    def mock_azure_client(self):
        """Mock Azure clients"""
        with patch('services.azure_service.ResourceManagementClient') as mock:
            yield mock
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "azure_connection" in response.json()
    
    def test_list_subscriptions(self, mock_azure_client):
        """Test listing Azure subscriptions"""
        # Mock subscription data
        mock_sub = Mock()
        mock_sub.subscription_id = "test-sub-1"
        mock_sub.display_name = "Test Subscription"
        mock_azure_client.return_value.subscriptions.list.return_value = [mock_sub]
        
        response = client.get("/api/v1/subscriptions")
        assert response.status_code in [200, 401]  # 401 if auth required
    
    def test_list_resource_groups(self, mock_azure_client):
        """Test listing resource groups"""
        response = client.get("/api/v1/resource-groups?subscription_id=test-sub")
        assert response.status_code in [200, 401]
    
    def test_list_resources(self, mock_azure_client):
        """Test listing Azure resources"""
        response = client.get("/api/v1/resources?subscription_id=test-sub")
        assert response.status_code in [200, 401]
    
    @pytest.mark.parametrize("resource_type", [
        "Microsoft.Compute/virtualMachines",
        "Microsoft.Storage/storageAccounts",
        "Microsoft.Network/virtualNetworks"
    ])
    def test_list_resources_by_type(self, resource_type, mock_azure_client):
        """Test listing resources by type"""
        response = client.get(f"/api/v1/resources?subscription_id=test-sub&resource_type={resource_type}")
        assert response.status_code in [200, 401]
    
    def test_get_resource_details(self, mock_azure_client):
        """Test getting resource details"""
        resource_id = "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm"
        response = client.get(f"/api/v1/resources/{resource_id}")
        assert response.status_code in [200, 401, 404]
    
    def test_cost_management_current_month(self, mock_azure_client):
        """Test cost management API - current month"""
        response = client.get("/api/v1/costs/current-month?subscription_id=test-sub")
        assert response.status_code in [200, 401]
    
    def test_cost_management_by_service(self, mock_azure_client):
        """Test cost breakdown by service"""
        response = client.get("/api/v1/costs/by-service?subscription_id=test-sub")
        assert response.status_code in [200, 401]
    
    def test_service_bus_send_message(self):
        """Test Service Bus message sending"""
        message_data = {
            "topic": "test-topic",
            "message": {
                "action": "resource_created",
                "resource_id": "test-resource"
            }
        }
        response = client.post("/api/v1/servicebus/send", json=message_data)
        assert response.status_code in [200, 201, 401]
'@ | Out-File -FilePath $azureTestFile -Encoding UTF8
}

# Test categories
$testResults = @{}

# Unit Tests
if ($TestType -eq "all" -or $TestType -eq "unit") {
    Write-Host "`n  Running unit tests..." -ForegroundColor Cyan
    $unitTestOutput = Join-Path $ResultsPath "unit_tests_$Timestamp.txt"
    
    pytest tests/unit -v --tb=short 2>&1 | Tee-Object -FilePath $unitTestOutput
    $testResults["unit"] = $LASTEXITCODE -eq 0
}

# Integration Tests
if ($TestType -eq "all" -or $TestType -eq "integration") {
    Write-Host "`n  Running integration tests..." -ForegroundColor Cyan
    $integrationTestOutput = Join-Path $ResultsPath "integration_tests_$Timestamp.txt"
    
    pytest tests/integration -v --tb=short 2>&1 | Tee-Object -FilePath $integrationTestOutput
    $testResults["integration"] = $LASTEXITCODE -eq 0
}

# API Tests
if ($TestType -eq "all" -or $TestType -eq "api") {
    Write-Host "`n  Running API tests..." -ForegroundColor Cyan
    $apiTestOutput = Join-Path $ResultsPath "api_tests_$Timestamp.txt"
    
    pytest $azureTestFile -v --tb=short 2>&1 | Tee-Object -FilePath $apiTestOutput
    $testResults["api"] = $LASTEXITCODE -eq 0
}

Write-Host "`n3. Azure Connection Tests..." -ForegroundColor Yellow

$connectionTestOutput = Join-Path $ResultsPath "connection_tests_$Timestamp.txt"
$connectionResults = @"
Azure Integration Connection Tests
==================================
Timestamp: $(Get-Date)

Connection Tests:
"@

# Test various Azure connections
$azureEndpoints = @(
    @{Name="Subscriptions"; Url="http://localhost:8001/api/v1/subscriptions"},
    @{Name="Resource Groups"; Url="http://localhost:8001/api/v1/resource-groups?subscription_id=test"},
    @{Name="Resources"; Url="http://localhost:8001/api/v1/resources?subscription_id=test"},
    @{Name="Costs"; Url="http://localhost:8001/api/v1/costs/current-month?subscription_id=test"}
)

foreach ($endpoint in $azureEndpoints) {
    Write-Host "  Testing $($endpoint.Name)..." -ForegroundColor Cyan
    
    try {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-RestMethod -Uri $endpoint.Url -Method GET -ErrorAction Stop
        $stopwatch.Stop()
        
        $connectionResults += @"

$($endpoint.Name):
  Status: SUCCESS
  Response Time: $($stopwatch.ElapsedMilliseconds)ms
  Status Code: 200
"@
        Write-Host "    ✓ Success" -ForegroundColor Green
    } catch {
        $connectionResults += @"

$($endpoint.Name):
  Status: FAILED
  Error: $($_.Exception.Message)
"@
        Write-Host "    ✗ Failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

$connectionResults | Out-File -FilePath $connectionTestOutput -Encoding UTF8

Write-Host "`n4. Mock Azure Operations Tests..." -ForegroundColor Yellow

# Test mock Azure operations
$mockOpsOutput = Join-Path $ResultsPath "mock_operations_$Timestamp.txt"
$mockResults = @"
Mock Azure Operations Tests
===========================
Timestamp: $(Get-Date)

Operations:
"@

# Test create resource (mock)
Write-Host "  Testing resource creation (mock)..." -ForegroundColor Cyan
$createResourceBody = @{
    name = "test-vm-$(Get-Random)"
    location = "eastus"
    resource_type = "Microsoft.Compute/virtualMachines"
    properties = @{
        vmSize = "Standard_B2s"
    }
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8001/api/v1/resources" -Method POST -Body $createResourceBody -ContentType "application/json" -ErrorAction Stop
    $mockResults += "`n✓ Resource Creation: SUCCESS"
} catch {
    $mockResults += "`n✗ Resource Creation: FAILED - $($_.Exception.Message)"
}

$mockResults | Out-File -FilePath $mockOpsOutput -Encoding UTF8

# Coverage Report
if ($Coverage) {
    Write-Host "`n5. Generating coverage report..." -ForegroundColor Cyan
    $coverageOutput = Join-Path $ResultsPath "coverage_$Timestamp.txt"
    
    pytest --cov=. --cov-report=term --cov-report=html:$ResultsPath/coverage_html 2>&1 | Tee-Object -FilePath $coverageOutput
}

Write-Host "`n6. Generating test summary..." -ForegroundColor Yellow

# Generate summary
$summaryPath = Join-Path $ResultsPath "summary_$Timestamp.txt"
$summary = @"
Azure Integration Service Test Summary
======================================
Date: $(Get-Date)
Service: Azure Integration
Port: 8001
Mock Mode: Enabled

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
  - Connection Tests: connection_tests_$Timestamp.txt
  - Mock Operations: mock_operations_$Timestamp.txt
  - This Summary: summary_$Timestamp.txt
"@

if ($Coverage) {
    $summary += "`n  - Coverage Report: coverage_$Timestamp.txt"
    $summary += "`n  - Coverage HTML: coverage_html/index.html"
}

$summary | Out-File -FilePath $summaryPath -Encoding UTF8

# Stop the service
Write-Host "`n7. Stopping service..." -ForegroundColor Yellow
Stop-Process -Id $serviceProcess.Id -Force

# Deactivate virtual environment
deactivate

Write-Host "`n✓ Azure Integration tests completed!" -ForegroundColor Green
Write-Host "Results saved to: $ResultsPath" -ForegroundColor Cyan

# Return overall success/failure
$allPassed = $testResults.Values -notcontains $false
exit $(if ($allPassed) { 0 } else { 1 })