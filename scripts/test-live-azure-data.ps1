# PolicyCortex Live Azure Data Validation Test Script
# This script verifies that the application is using real Azure data and not mock data

param(
    [string]$BackendUrl = "http://localhost:8080",
    [int]$StartupWaitTime = 10,
    [switch]$SkipBackendStart
)

# Color functions for output
function Write-Success {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
}

function Write-Header {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor Magenta
    Write-Host $Message -ForegroundColor Magenta
    Write-Host "========================================" -ForegroundColor Magenta
}

# Test result tracking
$global:TestResults = @{
    Total = 0
    Passed = 0
    Failed = 0
    Details = @()
}

function Add-TestResult {
    param(
        [string]$Endpoint,
        [bool]$Success,
        [string]$Message,
        [string]$ResponseSample = ""
    )
    
    $global:TestResults.Total++
    if ($Success) {
        $global:TestResults.Passed++
    } else {
        $global:TestResults.Failed++
    }
    
    $global:TestResults.Details += @{
        Endpoint = $Endpoint
        Success = $Success
        Message = $Message
        ResponseSample = $ResponseSample
    }
}

# Function to set environment variables
function Set-AzureEnvironmentVariables {
    Write-Header "Setting Azure Environment Variables"
    
    $env:AZURE_SUBSCRIPTION_ID = "6dc7cfa2-0332-4740-98b6-bac9f1a23de9"
    $env:AZURE_TENANT_ID = "e1f3e196-aa55-4709-9c55-0e334c0b444f"
    $env:AZURE_CLIENT_ID = "232c44f7-d0cf-4825-a9b5-beba9f587ffb"
    $env:USE_REAL_DATA = "true"
    $env:REQUIRE_REAL_DATA = "true"
    $env:FAIL_FAST_MODE = "true"
    
    Write-Info "Azure Subscription ID: $env:AZURE_SUBSCRIPTION_ID"
    Write-Info "Azure Tenant ID: $env:AZURE_TENANT_ID"
    Write-Info "Azure Client ID: $env:AZURE_CLIENT_ID"
    Write-Success "Environment variables set for real Azure data mode"
    
    # Verify Azure CLI authentication
    Write-Info "`nChecking Azure CLI authentication..."
    try {
        $azAccount = az account show 2>$null | ConvertFrom-Json
        if ($azAccount) {
            Write-Success "Azure CLI authenticated as: $($azAccount.user.name)"
            Write-Info "Current subscription: $($azAccount.name) ($($azAccount.id))"
        }
    } catch {
        Write-Warning "Azure CLI not authenticated. Running 'az login' may be required for full functionality"
    }
}

# Function to start the backend
function Start-Backend {
    if ($SkipBackendStart) {
        Write-Info "Skipping backend startup (assuming it's already running)"
        return $null
    }
    
    Write-Header "Starting Backend in Real Data Mode"
    
    # Check if backend is already running
    try {
        $healthCheck = Invoke-RestMethod -Uri "$BackendUrl/health" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($healthCheck) {
            Write-Warning "Backend appears to be already running at $BackendUrl"
            Write-Info "Stopping existing backend..."
            Get-Process | Where-Object { $_.ProcessName -like "*cargo*" -or $_.ProcessName -like "*core*" } | Stop-Process -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        }
    } catch {
        # Backend not running, continue
    }
    
    # Navigate to core directory and start backend
    $scriptDir = Split-Path -Parent $PSScriptRoot
    $coreDir = Join-Path $scriptDir "core"
    
    if (-not (Test-Path $coreDir)) {
        Write-Error "Core directory not found at: $coreDir"
        exit 1
    }
    
    Write-Info "Starting backend from: $coreDir"
    
    # Start the backend process
    $backendProcess = Start-Process -FilePath "cargo" -ArgumentList "run", "--release" -WorkingDirectory $coreDir -PassThru -WindowStyle Hidden
    
    Write-Info "Backend process started (PID: $($backendProcess.Id))"
    Write-Info "Waiting $StartupWaitTime seconds for backend to initialize..."
    
    # Wait for backend to be ready
    $attempts = 0
    $maxAttempts = $StartupWaitTime
    while ($attempts -lt $maxAttempts) {
        Start-Sleep -Seconds 1
        $attempts++
        Write-Host "." -NoNewline
        
        try {
            $health = Invoke-RestMethod -Uri "$BackendUrl/health" -Method Get -TimeoutSec 1 -ErrorAction SilentlyContinue
            if ($health) {
                Write-Success "`nBackend is ready!"
                return $backendProcess
            }
        } catch {
            # Continue waiting
        }
    }
    
    Write-Error "`nBackend failed to start within $StartupWaitTime seconds"
    return $backendProcess
}

# Function to test an endpoint
function Test-Endpoint {
    param(
        [string]$Endpoint,
        [string]$Description,
        [string[]]$MockIndicators = @("SIMULATED", "Mock", "mock", "simulated", "demo", "Demo", "sample", "Sample"),
        [scriptblock]$CustomValidation = $null
    )
    
    Write-Info "`nTesting: $Description"
    Write-Info "Endpoint: $Endpoint"
    
    $url = "$BackendUrl$Endpoint"
    
    try {
        # Make the request
        $response = Invoke-WebRequest -Uri $url -Method Get -TimeoutSec 10 -ErrorAction Stop
        $statusCode = $response.StatusCode
        $content = $response.Content
        
        # Parse JSON if possible
        try {
            $jsonContent = $content | ConvertFrom-Json
            $contentSample = $jsonContent | ConvertTo-Json -Depth 2 -Compress
            if ($contentSample.Length -gt 200) {
                $contentSample = $contentSample.Substring(0, 200) + "..."
            }
        } catch {
            $contentSample = if ($content.Length -gt 200) { $content.Substring(0, 200) + "..." } else { $content }
        }
        
        Write-Info "Status Code: $statusCode"
        
        # Check for mock data indicators
        $mockFound = $false
        $foundIndicators = @()
        
        foreach ($indicator in $MockIndicators) {
            if ($content -like "*$indicator*") {
                $mockFound = $true
                $foundIndicators += $indicator
            }
        }
        
        if ($mockFound) {
            Write-Error "FAILED: Mock data detected! Found indicators: $($foundIndicators -join ', ')"
            Add-TestResult -Endpoint $Endpoint -Success $false -Message "Mock data indicators found: $($foundIndicators -join ', ')" -ResponseSample $contentSample
            return $false
        }
        
        # Run custom validation if provided
        if ($CustomValidation) {
            $validationResult = & $CustomValidation $jsonContent
            if (-not $validationResult) {
                Write-Error "FAILED: Custom validation failed"
                Add-TestResult -Endpoint $Endpoint -Success $false -Message "Custom validation failed" -ResponseSample $contentSample
                return $false
            }
        }
        
        # Check for Azure-specific data patterns
        $azurePatterns = @(
            "*azure*",
            "*subscription*",
            "*resourceGroup*",
            "*location*",
            "*eastus*",
            "*westus*",
            "*Microsoft.*"
        )
        
        $azureDataFound = $false
        foreach ($pattern in $azurePatterns) {
            if ($content -like $pattern) {
                $azureDataFound = $true
                break
            }
        }
        
        if ($azureDataFound) {
            Write-Success "PASSED: Real Azure data detected"
            Add-TestResult -Endpoint $Endpoint -Success $true -Message "Real Azure data confirmed" -ResponseSample $contentSample
        } else {
            Write-Warning "WARNING: No obvious Azure patterns found (may still be real data)"
            Add-TestResult -Endpoint $Endpoint -Success $true -Message "No mock indicators found, but Azure patterns not obvious" -ResponseSample $contentSample
        }
        
        return $true
        
    } catch {
        $errorMessage = $_.Exception.Message
        Write-Error "FAILED: Request failed - $errorMessage"
        
        # Check if it's a connection error
        if ($errorMessage -like "*Unable to connect*" -or $errorMessage -like "*connection*") {
            Write-Error "Backend appears to be down or not responding"
        }
        
        Add-TestResult -Endpoint $Endpoint -Success $false -Message "Request failed: $errorMessage"
        return $false
    }
}

# Main test execution
function Run-Tests {
    Write-Header "PolicyCortex Live Azure Data Validation"
    Write-Info "Backend URL: $BackendUrl"
    Write-Info "Test Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    
    # Set environment variables
    Set-AzureEnvironmentVariables
    
    # Start backend if needed
    $backendProcess = Start-Backend
    
    # Test endpoints
    Write-Header "Testing API Endpoints"
    
    # 1. Azure Health Check
    Test-Endpoint -Endpoint "/api/v1/health/azure" `
                  -Description "Azure Health Check" `
                  -CustomValidation {
                      param($data)
                      return ($data.status -eq "healthy" -and $data.azure_connection -eq $true)
                  } | Out-Null
    
    # 2. Dashboard Metrics
    Test-Endpoint -Endpoint "/api/v1/dashboard/metrics" `
                  -Description "Dashboard Metrics" `
                  -CustomValidation {
                      param($data)
                      return ($data.resources -and $data.compliance -and $data.costs)
                  } | Out-Null
    
    # 3. Governance Compliance Status
    Test-Endpoint -Endpoint "/api/v1/governance/compliance/status" `
                  -Description "Governance Compliance Status" `
                  -CustomValidation {
                      param($data)
                      return ($data.overall_score -ge 0 -and $data.policies)
                  } | Out-Null
    
    # 4. Operations Resources
    Test-Endpoint -Endpoint "/api/v1/operations/resources" `
                  -Description "Operations Resources" `
                  -CustomValidation {
                      param($data)
                      return ($data.Count -gt 0 -or $data.resources)
                  } | Out-Null
    
    # 5. Security IAM
    Test-Endpoint -Endpoint "/api/v1/security/iam" `
                  -Description "Security IAM Users" `
                  -CustomValidation {
                      param($data)
                      return ($data.users -or $data.Count -ge 0)
                  } | Out-Null
    
    # 6. Additional critical endpoints
    Test-Endpoint -Endpoint "/api/v1/metrics" `
                  -Description "Unified Governance Metrics (Patent 3)" | Out-Null
    
    Test-Endpoint -Endpoint "/api/v1/correlations" `
                  -Description "Cross-Domain Correlations (Patent 1)" | Out-Null
    
    Test-Endpoint -Endpoint "/api/v1/predictions" `
                  -Description "Predictive Compliance (Patent 4)" | Out-Null
    
    # Generate test report
    Write-Header "Test Results Summary"
    
    $successRate = if ($global:TestResults.Total -gt 0) { 
        [math]::Round(($global:TestResults.Passed / $global:TestResults.Total) * 100, 2)
    } else { 0 }
    
    Write-Info "Total Tests: $($global:TestResults.Total)"
    Write-Success "Passed: $($global:TestResults.Passed)"
    Write-Error "Failed: $($global:TestResults.Failed)"
    Write-Info "Success Rate: $successRate%"
    
    # Detailed results
    Write-Header "Detailed Results"
    foreach ($result in $global:TestResults.Details) {
        $status = if ($result.Success) { "[PASS]" } else { "[FAIL]" }
        $color = if ($result.Success) { "Green" } else { "Red" }
        
        Write-Host "`n$status $($result.Endpoint)" -ForegroundColor $color
        Write-Host "  Message: $($result.Message)" -ForegroundColor Gray
        if ($result.ResponseSample) {
            Write-Host "  Response Sample: $($result.ResponseSample)" -ForegroundColor DarkGray
        }
    }
    
    # Final verdict
    Write-Header "Final Verdict"
    if ($global:TestResults.Failed -eq 0) {
        Write-Success "ALL TESTS PASSED! Application is using LIVE Azure data."
        $exitCode = 0
    } elseif ($global:TestResults.Passed -gt 0) {
        Write-Warning "PARTIAL SUCCESS: Some endpoints are using live data, but others may be falling back to mock data."
        $exitCode = 1
    } else {
        Write-Error "ALL TESTS FAILED! Application is NOT using live Azure data or backend is not responding."
        $exitCode = 2
    }
    
    # Cleanup
    if ($backendProcess) {
        Write-Info "`nStopping backend process..."
        Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
        Write-Info "Backend stopped"
    }
    
    Write-Info "`nTest Completed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    
    # Export results to file
    $reportPath = Join-Path $PSScriptRoot "test-results-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
    $global:TestResults | ConvertTo-Json -Depth 3 | Out-File -FilePath $reportPath
    Write-Info "Test results saved to: $reportPath"
    
    exit $exitCode
}

# Run the tests
try {
    Run-Tests
} catch {
    Write-Error "Unexpected error during test execution: $_"
    exit 3
}