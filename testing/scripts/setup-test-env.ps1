# PolicyCortex Test Environment Setup Script for Windows

Write-Host "PolicyCortex Test Environment Setup" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

# Set error action preference
$ErrorActionPreference = "Stop"

# Get project root
$ProjectRoot = (Get-Location).Path
if ($ProjectRoot -notmatch "policycortex$") {
    $ProjectRoot = Join-Path $ProjectRoot "policycortex"
}

# Load test environment variables
$TestEnvFile = Join-Path $ProjectRoot "testing\configs\test.env"
if (Test-Path $TestEnvFile) {
    Get-Content $TestEnvFile | ForEach-Object {
        if ($_ -match "^([^#][^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

Write-Host "`n1. Checking Prerequisites..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found!" -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "  ✓ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Node.js not found!" -ForegroundColor Red
    exit 1
}

# Check Docker
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "  ✓ Docker: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker not found!" -ForegroundColor Red
    exit 1
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
    
    # Create virtual environment
    Set-Location $servicePath
    if (-not (Test-Path "venv")) {
        python -m venv venv
        Write-Host "    ✓ Created virtual environment" -ForegroundColor Green
    }
    
    # Install dependencies
    & ".\venv\Scripts\Activate.ps1"
    pip install -r requirements.txt --quiet
    pip install pytest pytest-asyncio pytest-cov httpx --quiet
    Write-Host "    ✓ Installed dependencies" -ForegroundColor Green
    deactivate
}

Write-Host "`n3. Setting up Frontend..." -ForegroundColor Yellow
Set-Location (Join-Path $ProjectRoot "frontend")
if (-not (Test-Path "node_modules")) {
    npm install --silent
    Write-Host "  ✓ Installed frontend dependencies" -ForegroundColor Green
}

Write-Host "`n4. Starting test infrastructure..." -ForegroundColor Yellow

# Create docker network if not exists
docker network create policycortex-test 2>$null

# Start PostgreSQL
Write-Host "  Starting PostgreSQL..." -ForegroundColor Cyan
docker run -d --name policycortex-postgres-test `
    --network policycortex-test `
    -e POSTGRES_USER=test_user `
    -e POSTGRES_PASSWORD=test_password `
    -e POSTGRES_DB=policycortex_test `
    -p 5432:5432 `
    postgres:14-alpine 2>$null

# Start Redis
Write-Host "  Starting Redis..." -ForegroundColor Cyan
docker run -d --name policycortex-redis-test `
    --network policycortex-test `
    -p 6379:6379 `
    redis:7-alpine 2>$null

# Wait for services to be ready
Write-Host "  Waiting for services to be ready..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

Write-Host "`n5. Creating test directories..." -ForegroundColor Yellow
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
Write-Host "  ✓ Created test directories" -ForegroundColor Green

Write-Host "`n6. Generating test configuration..." -ForegroundColor Yellow

# Create pytest.ini for backend services
$pytestConfig = @"
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers --cov=. --cov-report=html
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests
    api: API tests
"@

foreach ($service in $services) {
    $configPath = Join-Path $ProjectRoot "backend\services\$service\pytest.ini"
    $pytestConfig | Out-File -FilePath $configPath -Encoding UTF8
}

Write-Host "  ✓ Generated pytest configurations" -ForegroundColor Green

Write-Host "`n7. Creating test utilities..." -ForegroundColor Yellow

# Return to project root
Set-Location $ProjectRoot

Write-Host "`n✓ Test environment setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Run individual service tests: .\testing\scripts\test-<service>.ps1"
Write-Host "  2. Run all tests: .\testing\scripts\run-all-tests.ps1"
Write-Host "  3. Generate report: .\testing\scripts\generate-report.ps1"