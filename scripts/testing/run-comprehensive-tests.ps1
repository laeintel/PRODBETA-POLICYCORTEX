# PolicyCortex Comprehensive Testing Script
# Orchestrates all testing strategies

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("all", "unit", "integration", "e2e", "performance", "security", "patents")]
    [string]$TestType = "all",
    
    [Parameter(Mandatory=$false)]
    [switch]$UseContainers,
    
    [Parameter(Mandatory=$false)]
    [switch]$Parallel,
    
    [Parameter(Mandatory=$false)]
    [switch]$Coverage,
    
    [Parameter(Mandatory=$false)]
    [string]$ReportPath = "./test-reports"
)

# Colors for output
$colors = @{
    Reset = "`e[0m"
    Red = "`e[31m"
    Green = "`e[32m"
    Yellow = "`e[33m"
    Blue = "`e[34m"
    Magenta = "`e[35m"
    Cyan = "`e[36m"
}

# Configuration
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportDir = Join-Path $ReportPath $timestamp
$exitCode = 0
$testResults = @{}

# Create report directory
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "Reset"
    )
    Write-Host "$($colors[$Color])$Message$($colors.Reset)"
}

function Start-TestContainers {
    Write-ColorOutput "🐳 Starting test containers..." "Cyan"
    
    $dockerComposePath = Join-Path $PSScriptRoot "..\..\tests\containers\docker-compose.test.yml"
    
    docker-compose -f $dockerComposePath up -d
    
    # Wait for services to be healthy
    Write-ColorOutput "⏳ Waiting for services to be healthy..." "Yellow"
    Start-Sleep -Seconds 10
    
    # Check health status
    $services = @("test-postgres", "test-cache", "test-eventstore", "test-localstack")
    foreach ($service in $services) {
        $health = docker inspect --format='{{.State.Health.Status}}' "policycortex-$service" 2>$null
        if ($health -eq "healthy") {
            Write-ColorOutput "✅ $service is healthy" "Green"
        } else {
            Write-ColorOutput "❌ $service is not healthy" "Red"
            return $false
        }
    }
    
    return $true
}

function Stop-TestContainers {
    Write-ColorOutput "🛑 Stopping test containers..." "Cyan"
    
    $dockerComposePath = Join-Path $PSScriptRoot "..\..\tests\containers\docker-compose.test.yml"
    docker-compose -f $dockerComposePath down -v
}

function Run-UnitTests {
    Write-ColorOutput "`n📦 Running Unit Tests" "Blue"
    Write-ColorOutput "===================" "Blue"
    
    $results = @{
        Frontend = $null
        Backend = $null
        Core = $null
    }
    
    # Frontend unit tests
    Write-ColorOutput "`nFrontend Unit Tests:" "Cyan"
    Push-Location frontend
    
    if ($Coverage) {
        $npmCommand = "npm run test -- --coverage --coverageDirectory=$reportDir/frontend-coverage"
    } else {
        $npmCommand = "npm run test"
    }
    
    $result = Invoke-Expression $npmCommand
    $results.Frontend = $LASTEXITCODE -eq 0
    
    Pop-Location
    
    # Backend Python tests
    Write-ColorOutput "`nBackend Python Tests:" "Cyan"
    Push-Location backend/services/api_gateway
    
    if ($Coverage) {
        $pytestCommand = "python -m pytest tests/ --cov=. --cov-report=html:$reportDir/python-coverage --cov-report=json:$reportDir/python-coverage.json"
    } else {
        $pytestCommand = "python -m pytest tests/ -v"
    }
    
    $result = Invoke-Expression $pytestCommand
    $results.Backend = $LASTEXITCODE -eq 0
    
    Pop-Location
    
    # Rust Core tests
    Write-ColorOutput "`nRust Core Tests:" "Cyan"
    Push-Location core
    
    $cargoCommand = "cargo test"
    if ($Coverage) {
        # Requires cargo-tarpaulin
        $cargoCommand = "cargo tarpaulin --out Html --output-dir $reportDir/rust-coverage"
    }
    
    $result = Invoke-Expression $cargoCommand
    $results.Core = $LASTEXITCODE -eq 0
    
    Pop-Location
    
    return $results
}

function Run-IntegrationTests {
    Write-ColorOutput "`n🔗 Running Integration Tests" "Blue"
    Write-ColorOutput "============================" "Blue"
    
    # Start test containers if requested
    if ($UseContainers) {
        if (-not (Start-TestContainers)) {
            Write-ColorOutput "Failed to start test containers" "Red"
            return @{ Success = $false }
        }
    }
    
    # Run integration tests
    Push-Location backend/services/api_gateway
    
    $env:TEST_MODE = "integration"
    $pytestCommand = "python -m pytest tests/test_integration.py -v --junit-xml=$reportDir/integration-results.xml"
    
    $result = Invoke-Expression $pytestCommand
    $success = $LASTEXITCODE -eq 0
    
    Pop-Location
    
    # Stop containers if we started them
    if ($UseContainers) {
        Stop-TestContainers
    }
    
    return @{ Success = $success }
}

function Run-E2ETests {
    Write-ColorOutput "`n🎭 Running End-to-End Tests" "Blue"
    Write-ColorOutput "===========================" "Blue"
    
    Push-Location frontend
    
    # Install Playwright browsers if needed
    Write-ColorOutput "Installing Playwright browsers..." "Yellow"
    npx playwright install --with-deps
    
    # Run E2E tests
    $playwrightCommand = "npx playwright test"
    if ($Parallel) {
        $playwrightCommand += " --workers=4"
    }
    $playwrightCommand += " --reporter=html --reporter=json"
    $playwrightCommand += " --output=$reportDir/e2e-results"
    
    $result = Invoke-Expression $playwrightCommand
    $success = $LASTEXITCODE -eq 0
    
    Pop-Location
    
    return @{ Success = $success }
}

function Run-PerformanceTests {
    Write-ColorOutput "`n⚡ Running Performance Tests" "Blue"
    Write-ColorOutput "============================" "Blue"
    
    Push-Location scripts/testing
    
    # Check if services are running
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -ne 200) {
            Write-ColorOutput "⚠️  Services not running. Please start them first." "Yellow"
            return @{ Success = $false }
        }
    } catch {
        Write-ColorOutput "⚠️  Services not running. Please start them first." "Yellow"
        return @{ Success = $false }
    }
    
    # Run performance tests
    node performance-tests.js > "$reportDir/performance-results.json"
    $performanceSuccess = $LASTEXITCODE -eq 0
    
    Pop-Location
    
    return @{ Success = $performanceSuccess }
}

function Run-SecurityTests {
    Write-ColorOutput "`n🔒 Running Security Tests" "Blue"
    Write-ColorOutput "=========================" "Blue"
    
    $securityScript = Join-Path $PSScriptRoot "run-security-tests.sh"
    
    if (Get-Command bash -ErrorAction SilentlyContinue) {
        bash $securityScript
        $success = $LASTEXITCODE -eq 0
    } else {
        # Windows fallback - run individual security tools
        $results = @{}
        
        # npm audit
        Write-ColorOutput "Running npm audit..." "Cyan"
        Push-Location frontend
        npm audit --json > "$reportDir/npm-audit.json"
        $results.NpmAudit = $LASTEXITCODE -eq 0
        Pop-Location
        
        # Run other security checks available on Windows
        # ...
        
        $success = $results.Values -notcontains $false
    }
    
    return @{ Success = $success }
}

function Run-PatentBenchmarks {
    Write-ColorOutput "`n🚀 Running Patent Performance Benchmarks" "Blue"
    Write-ColorOutput "========================================" "Blue"
    
    Push-Location scripts/testing
    
    # Check if services are running
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing -TimeoutSec 5
    } catch {
        Write-ColorOutput "⚠️  Services not running. Starting mock services..." "Yellow"
        # Start mock services for benchmarking
    }
    
    # Run patent benchmarks
    node patent-benchmarks.js > "$reportDir/patent-benchmarks.json"
    $success = $LASTEXITCODE -eq 0
    
    Pop-Location
    
    return @{ Success = $success }
}

function Generate-TestReport {
    param(
        [hashtable]$Results
    )
    
    Write-ColorOutput "`n📊 Generating Test Report" "Blue"
    Write-ColorOutput "========================" "Blue"
    
    $html = @"
<!DOCTYPE html>
<html>
<head>
    <title>PolicyCortex Test Report - $timestamp</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .summary { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .pass { color: #28a745; font-weight: bold; }
        .fail { color: #dc3545; font-weight: bold; }
        .warning { color: #ffc107; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; background: white; }
        th { background: #007acc; color: white; padding: 12px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f8f9fa; }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <h1>📋 PolicyCortex Comprehensive Test Report</h1>
    <p>Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")</p>
    
    <div class="summary">
        <h2>Executive Summary</h2>
"@
    
    $totalTests = $Results.Count
    $passedTests = ($Results.Values | Where-Object { $_.Success -eq $true }).Count
    $failedTests = $totalTests - $passedTests
    $passRate = if ($totalTests -gt 0) { [math]::Round(($passedTests / $totalTests) * 100, 2) } else { 0 }
    
    $html += @"
        <div class="metric">
            <div class="metric-value">$totalTests</div>
            <div class="metric-label">Total Test Suites</div>
        </div>
        <div class="metric">
            <div class="metric-value pass">$passedTests</div>
            <div class="metric-label">Passed</div>
        </div>
        <div class="metric">
            <div class="metric-value fail">$failedTests</div>
            <div class="metric-label">Failed</div>
        </div>
        <div class="metric">
            <div class="metric-value">$passRate%</div>
            <div class="metric-label">Pass Rate</div>
        </div>
    </div>
    
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Test Suite</th>
            <th>Status</th>
            <th>Details</th>
        </tr>
"@
    
    foreach ($suite in $Results.Keys) {
        $status = if ($Results[$suite].Success) { "<span class='pass'>✅ PASSED</span>" } else { "<span class='fail'>❌ FAILED</span>" }
        $details = $Results[$suite].Details ?? "View detailed report in $reportDir"
        
        $html += @"
        <tr>
            <td>$suite</td>
            <td>$status</td>
            <td>$details</td>
        </tr>
"@
    }
    
    $html += @"
    </table>
    
    <h2>Recommendations</h2>
    <ul>
        <li>Review all failed tests and create remediation tasks</li>
        <li>Update dependencies with security vulnerabilities</li>
        <li>Improve test coverage in areas below 80%</li>
        <li>Monitor performance metrics for regression</li>
        <li>Schedule regular security scans</li>
    </ul>
    
    <h2>Artifacts</h2>
    <ul>
        <li><a href="./coverage/index.html">Code Coverage Report</a></li>
        <li><a href="./e2e-results/index.html">E2E Test Report</a></li>
        <li><a href="./security-report.html">Security Scan Report</a></li>
        <li><a href="./performance-results.json">Performance Metrics</a></li>
        <li><a href="./patent-benchmarks.json">Patent Benchmarks</a></li>
    </ul>
</body>
</html>
"@
    
    $html | Out-File -FilePath "$reportDir/test-report.html" -Encoding UTF8
    Write-ColorOutput "✅ Report generated: $reportDir/test-report.html" "Green"
}

# Main execution
Write-ColorOutput "🚀 PolicyCortex Comprehensive Testing Suite" "Magenta"
Write-ColorOutput "==========================================" "Magenta"
Write-ColorOutput "Test Type: $TestType" "Cyan"
Write-ColorOutput "Use Containers: $UseContainers" "Cyan"
Write-ColorOutput "Parallel: $Parallel" "Cyan"
Write-ColorOutput "Coverage: $Coverage" "Cyan"
Write-ColorOutput "Report Path: $reportDir" "Cyan"

# Run tests based on type
switch ($TestType) {
    "all" {
        $testResults["Unit"] = Run-UnitTests
        $testResults["Integration"] = Run-IntegrationTests
        $testResults["E2E"] = Run-E2ETests
        $testResults["Performance"] = Run-PerformanceTests
        $testResults["Security"] = Run-SecurityTests
        $testResults["Patents"] = Run-PatentBenchmarks
    }
    "unit" {
        $testResults["Unit"] = Run-UnitTests
    }
    "integration" {
        $testResults["Integration"] = Run-IntegrationTests
    }
    "e2e" {
        $testResults["E2E"] = Run-E2ETests
    }
    "performance" {
        $testResults["Performance"] = Run-PerformanceTests
    }
    "security" {
        $testResults["Security"] = Run-SecurityTests
    }
    "patents" {
        $testResults["Patents"] = Run-PatentBenchmarks
    }
}

# Generate report
Generate-TestReport -Results $testResults

# Summary
Write-ColorOutput "`n========================================" "Blue"
Write-ColorOutput "           TEST EXECUTION SUMMARY" "Blue"
Write-ColorOutput "========================================" "Blue"

$failedSuites = $testResults.Keys | Where-Object { -not $testResults[$_].Success }

if ($failedSuites.Count -eq 0) {
    Write-ColorOutput "✅ All test suites passed successfully!" "Green"
    $exitCode = 0
} else {
    Write-ColorOutput "❌ Failed test suites:" "Red"
    foreach ($suite in $failedSuites) {
        Write-ColorOutput "  - $suite" "Red"
    }
    $exitCode = 1
}

Write-ColorOutput "`n📄 Full report available at: $reportDir/test-report.html" "Cyan"

# Open report in browser
if (-not $env:CI) {
    Start-Process "$reportDir/test-report.html"
}

exit $exitCode