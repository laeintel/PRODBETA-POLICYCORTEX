# Simplified Test Runner for PolicyCortex
# This runs tests without full environment setup

Write-Host "`nPolicyCortex Test Suite - Simplified Runner" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Get-Location).Path)
$TestingRoot = Join-Path $ProjectRoot "testing"
$RunTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunResultsPath = Join-Path $TestingRoot "results\full_run_$RunTimestamp"

# Create results directory
New-Item -ItemType Directory -Path $RunResultsPath -Force | Out-Null

# Initialize test results
$allTestResults = @{}
$testStartTime = Get-Date

Write-Host "`nTest Run Started: $testStartTime" -ForegroundColor Yellow

# 1. Frontend Build Test
Write-Host "`n[1/8] Testing Frontend Build..." -ForegroundColor Yellow
$frontendPath = Join-Path $ProjectRoot "frontend"
$frontendResult = @{
    Service = "Frontend"
    StartTime = Get-Date
    Tests = @{}
}

Set-Location $frontendPath

# Test TypeScript compilation
Write-Host "  - TypeScript compilation check..." -NoNewline
try {
    $tscOutput = npx tsc --noEmit 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " PASSED" -ForegroundColor Green
        $frontendResult.Tests["TypeScript"] = "PASSED"
    } else {
        Write-Host " FAILED" -ForegroundColor Red
        $frontendResult.Tests["TypeScript"] = "FAILED"
    }
} catch {
    Write-Host " SKIPPED (tsc not available)" -ForegroundColor Yellow
    $frontendResult.Tests["TypeScript"] = "SKIPPED"
}

# Test build
Write-Host "  - Build test..." -NoNewline
$buildLog = Join-Path $RunResultsPath "frontend_build.log"
npm run build > $buildLog 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host " PASSED" -ForegroundColor Green
    $frontendResult.Tests["Build"] = "PASSED"
} else {
    Write-Host " FAILED (see $buildLog)" -ForegroundColor Red
    $frontendResult.Tests["Build"] = "FAILED"
}

$frontendResult.EndTime = Get-Date
$frontendResult.Duration = ($frontendResult.EndTime - $frontendResult.StartTime).TotalSeconds
$allTestResults["Frontend"] = $frontendResult

# 2. Backend Service Structure Tests
$services = @(
    @{Name="API Gateway"; Path="backend\services\api_gateway"; Port=8000},
    @{Name="Azure Integration"; Path="backend\services\azure_integration"; Port=8001},
    @{Name="AI Engine"; Path="backend\services\ai_engine"; Port=8002},
    @{Name="Data Processing"; Path="backend\services\data_processing"; Port=8003},
    @{Name="Conversation"; Path="backend\services\conversation"; Port=8004},
    @{Name="Notification"; Path="backend\services\notification"; Port=8005}
)

$serviceIndex = 2
foreach ($service in $services) {
    Write-Host "`n[$serviceIndex/8] Testing $($service.Name)..." -ForegroundColor Yellow
    $serviceResult = @{
        Service = $service.Name
        Port = $service.Port
        StartTime = Get-Date
        Tests = @{}
    }
    
    $servicePath = Join-Path $ProjectRoot $service.Path
    
    # Check service structure
    Write-Host "  - Service structure..." -NoNewline
    if (Test-Path $servicePath) {
        $hasMain = Test-Path (Join-Path $servicePath "main.py")
        $hasRequirements = Test-Path (Join-Path $servicePath "requirements.txt")
        
        if ($hasMain -and $hasRequirements) {
            Write-Host " PASSED" -ForegroundColor Green
            $serviceResult.Tests["Structure"] = "PASSED"
        } else {
            Write-Host " FAILED (missing files)" -ForegroundColor Red
            $serviceResult.Tests["Structure"] = "FAILED"
        }
    } else {
        Write-Host " FAILED (directory not found)" -ForegroundColor Red
        $serviceResult.Tests["Structure"] = "FAILED"
    }
    
    # Check FastAPI configuration
    if ($hasMain) {
        Write-Host "  - FastAPI configuration..." -NoNewline
        Set-Location $servicePath
        $mainContent = Get-Content "main.py" -Raw
        
        $checks = @{
            "FastAPI import" = ($mainContent -match "from fastapi import|import fastapi")
            "App creation" = ($mainContent -match "app\s*=\s*FastAPI")
            "Health endpoint" = ($mainContent -match "@app\.(get|post|put|delete).*health")
        }
        
        $allPassed = $true
        foreach ($check in $checks.GetEnumerator()) {
            if (-not $check.Value) {
                $allPassed = $false
                break
            }
        }
        
        if ($allPassed) {
            Write-Host " PASSED" -ForegroundColor Green
            $serviceResult.Tests["FastAPI"] = "PASSED"
        } else {
            Write-Host " FAILED" -ForegroundColor Red
            $serviceResult.Tests["FastAPI"] = "FAILED"
        }
    }
    
    # Simulated endpoint test
    Write-Host "  - Endpoint simulation..." -NoNewline
    $endpoints = @()
    
    switch ($service.Name) {
        "API Gateway" {
            $endpoints = @("/health", "/api/v1/auth/login", "/api/v1/policies")
        }
        "Azure Integration" {
            $endpoints = @("/health", "/api/v1/subscriptions", "/api/v1/resources")
        }
        "AI Engine" {
            $endpoints = @("/health", "/api/v1/analyze/policy", "/api/v1/models")
        }
        "Data Processing" {
            $endpoints = @("/health", "/api/v1/process", "/api/v1/validate")
        }
        "Conversation" {
            $endpoints = @("/health", "/api/v1/conversations", "/api/v1/messages")
        }
        "Notification" {
            $endpoints = @("/health", "/api/v1/notifications/send", "/api/v1/notifications")
        }
    }
    
    Write-Host " SIMULATED (would test $($endpoints.Count) endpoints)" -ForegroundColor Yellow
    $serviceResult.Tests["Endpoints"] = "SIMULATED"
    
    $serviceResult.EndTime = Get-Date
    $serviceResult.Duration = ($serviceResult.EndTime - $serviceResult.StartTime).TotalSeconds
    $allTestResults[$service.Name] = $serviceResult
    
    $serviceIndex++
}

# 3. Integration Tests (Simulated)
Write-Host "`n[8/8] Integration Tests..." -ForegroundColor Yellow
$integrationResult = @{
    Service = "Integration"
    StartTime = Get-Date
    Tests = @{}
}

Write-Host "  - Inter-service communication..." -NoNewline
Write-Host " SIMULATED" -ForegroundColor Yellow
$integrationResult.Tests["InterService"] = "SIMULATED"

Write-Host "  - End-to-end workflows..." -NoNewline
Write-Host " SIMULATED" -ForegroundColor Yellow
$integrationResult.Tests["E2E"] = "SIMULATED"

$integrationResult.EndTime = Get-Date
$integrationResult.Duration = ($integrationResult.EndTime - $integrationResult.StartTime).TotalSeconds
$allTestResults["Integration"] = $integrationResult

# Generate comprehensive report
Write-Host "`nGenerating test report..." -ForegroundColor Yellow

$testEndTime = Get-Date
$totalDuration = ($testEndTime - $testStartTime).TotalSeconds

# Calculate statistics
$totalTests = 0
$passedTests = 0
$failedTests = 0
$skippedTests = 0

foreach ($result in $allTestResults.Values) {
    foreach ($test in $result.Tests.Values) {
        $totalTests++
        switch ($test) {
            "PASSED" { $passedTests++ }
            "FAILED" { $failedTests++ }
            "SKIPPED" { $skippedTests++ }
            "SIMULATED" { $skippedTests++ }
        }
    }
}

$passRate = if ($totalTests -gt 0) { [math]::Round(($passedTests / $totalTests) * 100, 2) } else { 0 }

# Create HTML report
$htmlReport = @"
<!DOCTYPE html>
<html>
<head>
    <title>PolicyCortex Test Report - $RunTimestamp</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #333; }
        .summary { background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .passed { color: #4caf50; font-weight: bold; }
        .failed { color: #f44336; font-weight: bold; }
        .skipped { color: #ff9800; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #2196f3; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>PolicyCortex Test Report</h1>
        <p>Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")</p>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metric">
                <div class="metric-value">$totalTests</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value passed">$passedTests</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric">
                <div class="metric-value failed">$failedTests</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value skipped">$skippedTests</div>
                <div class="metric-label">Skipped/Simulated</div>
            </div>
            <div class="metric">
                <div class="metric-value">$passRate%</div>
                <div class="metric-label">Pass Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">$([math]::Round($totalDuration, 2))s</div>
                <div class="metric-label">Duration</div>
            </div>
        </div>
        
        <h2>Test Results by Service</h2>
        <table>
            <tr>
                <th>Service</th>
                <th>Tests Run</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Skipped</th>
                <th>Duration</th>
            </tr>
"@

foreach ($serviceName in $allTestResults.Keys | Sort-Object) {
    $result = $allTestResults[$serviceName]
    $servicePassed = ($result.Tests.Values | Where-Object { $_ -eq "PASSED" }).Count
    $serviceFailed = ($result.Tests.Values | Where-Object { $_ -eq "FAILED" }).Count
    $serviceSkipped = ($result.Tests.Values | Where-Object { $_ -in @("SKIPPED", "SIMULATED") }).Count
    
    $htmlReport += @"
            <tr>
                <td><strong>$serviceName</strong></td>
                <td>$($result.Tests.Count)</td>
                <td class="passed">$servicePassed</td>
                <td class="failed">$serviceFailed</td>
                <td class="skipped">$serviceSkipped</td>
                <td>$([math]::Round($result.Duration, 2))s</td>
            </tr>
"@
}

$htmlReport += @"
        </table>
        
        <h2>Detailed Results</h2>
"@

foreach ($serviceName in $allTestResults.Keys | Sort-Object) {
    $result = $allTestResults[$serviceName]
    $htmlReport += "<h3>$serviceName</h3><ul>"
    
    foreach ($testName in $result.Tests.Keys) {
        $testResult = $result.Tests[$testName]
        $cssClass = switch ($testResult) {
            "PASSED" { "passed" }
            "FAILED" { "failed" }
            default { "skipped" }
        }
        $htmlReport += "<li>$testName`: <span class='$cssClass'>$testResult</span></li>"
    }
    
    $htmlReport += "</ul>"
}

$htmlReport += @"
        <h2>Recommendations</h2>
        <ul>
"@

if ($failedTests -gt 0) {
    $htmlReport += "<li>Fix failing tests before deployment</li>"
}

if ($skippedTests -gt 0) {
    $htmlReport += "<li>Run full test suite with proper environment setup for complete coverage</li>"
}

$htmlReport += @"
            <li>Review test logs in: $RunResultsPath</li>
            <li>Consider setting up automated testing in CI/CD pipeline</li>
        </ul>
    </div>
</body>
</html>
"@

# Save reports
$htmlReportPath = Join-Path $RunResultsPath "test_report.html"
$htmlReport | Out-File -FilePath $htmlReportPath -Encoding UTF8

# Create markdown summary
$markdownReport = @"
# PolicyCortex Test Report

**Date:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")  
**Duration:** $([math]::Round($totalDuration, 2)) seconds

## Summary

- **Total Tests:** $totalTests
- **Passed:** $passedTests
- **Failed:** $failedTests
- **Skipped/Simulated:** $skippedTests
- **Pass Rate:** $passRate%

## Results by Service

| Service | Tests | Passed | Failed | Skipped |
|---------|-------|--------|--------|---------|
"@

foreach ($serviceName in $allTestResults.Keys | Sort-Object) {
    $result = $allTestResults[$serviceName]
    $servicePassed = ($result.Tests.Values | Where-Object { $_ -eq "PASSED" }).Count
    $serviceFailed = ($result.Tests.Values | Where-Object { $_ -eq "FAILED" }).Count
    $serviceSkipped = ($result.Tests.Values | Where-Object { $_ -in @("SKIPPED", "SIMULATED") }).Count
    
    $markdownReport += "`n| $serviceName | $($result.Tests.Count) | $servicePassed | $serviceFailed | $serviceSkipped |"
}

$markdownReportPath = Join-Path $RunResultsPath "test_report.md"
$markdownReport | Out-File -FilePath $markdownReportPath -Encoding UTF8

# Display summary
Write-Host "`n" -NoNewline
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "           TEST EXECUTION SUMMARY              " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

Write-Host "`nTest Statistics:" -ForegroundColor Yellow
Write-Host "  Total Tests: $totalTests"
Write-Host "  Passed: $passedTests" -ForegroundColor Green
Write-Host "  Failed: $failedTests" -ForegroundColor $(if ($failedTests -gt 0) { "Red" } else { "Green" })
Write-Host "  Skipped: $skippedTests" -ForegroundColor Yellow
Write-Host "  Pass Rate: $passRate%" -ForegroundColor $(if ($passRate -ge 80) { "Green" } elseif ($passRate -ge 60) { "Yellow" } else { "Red" })
Write-Host "  Duration: $([math]::Round($totalDuration, 2)) seconds"

Write-Host "`nReports Generated:" -ForegroundColor Yellow
Write-Host "  HTML Report: $htmlReportPath"
Write-Host "  Markdown Report: $markdownReportPath"
Write-Host "  Test Logs: $RunResultsPath"

if ($failedTests -gt 0) {
    Write-Host "`nWARNING: $failedTests tests failed!" -ForegroundColor Red
    Write-Host "Review the detailed report for failure information." -ForegroundColor Red
}

Write-Host "`n===============================================" -ForegroundColor Cyan

# Open HTML report
Start-Process $htmlReportPath

# Return to original directory
Set-Location $ProjectRoot