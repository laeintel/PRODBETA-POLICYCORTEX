# PolicyCortex Local Development Startup Script
param(
    [Parameter(Mandatory=$false)]
    [switch]$BackendOnly = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$FrontendOnly = $false,
    
    [Parameter(Mandatory=$false)]
    [int]$BackendPort = 8000,
    
    [Parameter(Mandatory=$false)]
    [int]$FrontendPort = 3000
)

Write-Host "PolicyCortex Local Development Environment" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path "backend\venv\Scripts\python.exe")) {
    Write-Host "Virtual environment not found. Creating one..." -ForegroundColor Yellow
    
    Push-Location backend
    python -m venv venv
    .\venv\Scripts\pip.exe install -r services\api_gateway\requirements.txt
    Pop-Location
    
    Write-Host "Virtual environment created and dependencies installed!" -ForegroundColor Green
}

# Check if node modules exist
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Frontend dependencies not found. Installing..." -ForegroundColor Yellow
    
    Push-Location frontend
    npm install
    Pop-Location
    
    Write-Host "Frontend dependencies installed!" -ForegroundColor Green
}

# Set environment variables
$env:ENVIRONMENT = "development"
$env:SERVICE_NAME = "api_gateway"
$env:SERVICE_PORT = $BackendPort
$env:JWT_SECRET_KEY = "dev-secret-key-change-in-production"
$env:LOG_LEVEL = "DEBUG"
$env:VITE_API_BASE_URL = "http://localhost:$BackendPort"
$env:VITE_PORT = $FrontendPort
$env:PORT = $FrontendPort

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Backend: http://localhost:$BackendPort" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:$FrontendPort" -ForegroundColor Green
Write-Host "  API Docs: http://localhost:$BackendPort/docs" -ForegroundColor Green

# Function to start backend
function Start-Backend {
    Write-Host "`nStarting API Gateway backend service..." -ForegroundColor Yellow
    
    Push-Location backend\services\api_gateway
    
    # Start the backend service
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {
        Write-Host 'PolicyCortex API Gateway' -ForegroundColor Cyan
        Write-Host '========================' -ForegroundColor Cyan
        Write-Host 'Starting on http://localhost:$BackendPort' -ForegroundColor Green
        Write-Host ''
        ..\..\venv\Scripts\python.exe main_simple.py
    }" -WindowStyle Normal
    
    Pop-Location
    
    Write-Host "Backend service started!" -ForegroundColor Green
}

# Function to start frontend
function Start-Frontend {
    Write-Host "`nStarting Frontend development server..." -ForegroundColor Yellow
    
    Push-Location frontend
    
    # Set frontend environment variables
    $env:VITE_PORT = $FrontendPort
    $env:PORT = $FrontendPort
    
    # Start the frontend development server
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {
        Write-Host 'PolicyCortex Frontend' -ForegroundColor Cyan
        Write-Host '=====================' -ForegroundColor Cyan
        Write-Host 'Starting on http://localhost:$FrontendPort' -ForegroundColor Green
        Write-Host ''
        npm run dev
    }" -WindowStyle Normal
    
    Pop-Location
    
    Write-Host "Frontend service started!" -ForegroundColor Green
}

# Start services based on parameters
if ($BackendOnly) {
    Start-Backend
} elseif ($FrontendOnly) {
    Start-Frontend
} else {
    # Start both services
    Start-Backend
    
    Write-Host "`nWaiting 3 seconds for backend to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    Start-Frontend
}

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Services are starting up!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`nUseful commands:" -ForegroundColor Yellow
Write-Host "  Backend only: .\start-local-dev.ps1 -BackendOnly" -ForegroundColor Cyan
Write-Host "  Frontend only: .\start-local-dev.ps1 -FrontendOnly" -ForegroundColor Cyan
Write-Host "  Custom ports: .\start-local-dev.ps1 -BackendPort 8010 -FrontendPort 3000" -ForegroundColor Cyan

Write-Host "`nURLs:" -ForegroundColor Yellow
Write-Host "  🌐 Frontend: http://localhost:$FrontendPort" -ForegroundColor Green
Write-Host "  🔧 Backend API: http://localhost:$BackendPort" -ForegroundColor Green
Write-Host "  📚 API Documentation: http://localhost:$BackendPort/docs" -ForegroundColor Green
Write-Host "  💾 Interactive API: http://localhost:$BackendPort/redoc" -ForegroundColor Green

Write-Host "`nPress Ctrl+C to stop this script (services will continue running)" -ForegroundColor Yellow
Write-Host "To stop services, close their respective terminal windows" -ForegroundColor Yellow