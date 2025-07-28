# PolicyCortex Test Environment Setup Script for Windows

Write-Host "PolicyCortex Test Environment Setup" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

# Set error action preference
$ErrorActionPreference = "Stop"

# Get project root
$ProjectRoot = (Get-Location).Path
if ($ProjectRoot -notmatch "policycortex$") {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent (Get-Location).Path)
}

Write-Host "`n1. Checking Prerequisites..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  OK: Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found!" -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "  OK: Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Node.js not found!" -ForegroundColor Red
    exit 1
}

# Check Docker
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "  OK: Docker: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: Docker not found - some tests may fail" -ForegroundColor Yellow
}

Write-Host "`n2. Setting up Python virtual environments..." -ForegroundColor Yellow

# Services to setup
$services = @(
    "api_gateway",
    "azure_integration", 
    "ai_engine",
    "data_processing",
    "conversation",
    "notification"
)

foreach ($service in $services) {
    $servicePath = Join-Path $ProjectRoot "backend\services\$service"
    Write-Host "  Setting up $service..." -ForegroundColor Cyan
    
    if (Test-Path $servicePath) {
        # Create virtual environment
        Set-Location $servicePath
        if (-not (Test-Path "venv")) {
            python -m venv venv
            Write-Host "    OK: Created virtual environment" -ForegroundColor Green
        }
        
        # Check if requirements.txt exists
        if (Test-Path "requirements.txt") {
            Write-Host "    Installing dependencies..." -ForegroundColor Gray
            & ".\venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
            & ".\venv\Scripts\pip.exe" install -r requirements.txt --quiet
            & ".\venv\Scripts\pip.exe" install pytest pytest-asyncio pytest-cov httpx --quiet
            Write-Host "    OK: Installed dependencies" -ForegroundColor Green
        } else {
            Write-Host "    WARNING: requirements.txt not found" -ForegroundColor Yellow
        }
    } else {
        Write-Host "    WARNING: Service directory not found" -ForegroundColor Yellow
    }
}

Write-Host "`n3. Setting up Frontend..." -ForegroundColor Yellow
$frontendPath = Join-Path $ProjectRoot "frontend"
if (Test-Path $frontendPath) {
    Set-Location $frontendPath
    if (-not (Test-Path "node_modules")) {
        Write-Host "  Installing frontend dependencies..." -ForegroundColor Cyan
        npm install --silent
        Write-Host "  OK: Installed frontend dependencies" -ForegroundColor Green
    } else {
        Write-Host "  OK: Frontend dependencies already installed" -ForegroundColor Green
    }
} else {
    Write-Host "  ERROR: Frontend directory not found" -ForegroundColor Red
}

Write-Host "`n4. Creating test directories..." -ForegroundColor Yellow
$testDirs = @(
    "testing\results\api_gateway",
    "testing\results\azure_integration",
    "testing\results\ai_engine",
    "testing\results\data_processing",
    "testing\results\conversation",
    "testing\results\notification",
    "testing\results\frontend",
    "testing\results\integration"
)

foreach ($dir in $testDirs) {
    $fullPath = Join-Path $ProjectRoot $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
}
Write-Host "  OK: Created test directories" -ForegroundColor Green

Write-Host "`n5. Generating test configuration..." -ForegroundColor Yellow

# Create pytest.ini for backend services
$pytestConfig = @"
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests
    api: API tests
"@

foreach ($service in $services) {
    $servicePath = Join-Path $ProjectRoot "backend\services\$service"
    if (Test-Path $servicePath) {
        $configPath = Join-Path $servicePath "pytest.ini"
        $pytestConfig | Out-File -FilePath $configPath -Encoding UTF8
    }
}

Write-Host "  OK: Generated pytest configurations" -ForegroundColor Green

# Return to project root
Set-Location $ProjectRoot

Write-Host "`n===============================================" -ForegroundColor Green
Write-Host " Test environment setup complete!" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Run individual service tests: .\testing\scripts\test-<service>.ps1"
Write-Host "  2. Run all tests: .\testing\scripts\run-all-tests.ps1"
Write-Host "  3. Generate report: .\testing\scripts\generate-report.ps1"

Write-Host "`nNOTE: Docker containers for PostgreSQL and Redis were not started." -ForegroundColor Yellow
Write-Host "      Start them manually if needed for integration tests." -ForegroundColor Yellow