# Start Local Development Environment Script
# This script starts all necessary services for local development

Write-Host "Starting PolicyCortex Local Development Environment..." -ForegroundColor Green

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check if Node.js is available
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Node.js is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Navigate to project root
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Start API Gateway
Write-Host "`n1. Starting API Gateway on port 8010..." -ForegroundColor Yellow
$apiGateway = Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd backend\services\api_gateway && start-api-8010.bat" -PassThru -WindowStyle Normal

# Give API Gateway time to start
Start-Sleep -Seconds 3

# Start Frontend
Write-Host "`n2. Starting Frontend on port 3000..." -ForegroundColor Yellow
$frontend = Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd frontend && npm run dev" -PassThru -WindowStyle Normal

# Display status
Write-Host "`n✅ Development environment started!" -ForegroundColor Green
Write-Host "`nServices running:" -ForegroundColor Cyan
Write-Host "- API Gateway: http://localhost:8010" -ForegroundColor White
Write-Host "- Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "`nAPI Endpoints:" -ForegroundColor Cyan
Write-Host "- Policies: http://localhost:8010/api/v1/policies" -ForegroundColor White
Write-Host "- Health: http://localhost:8010/health" -ForegroundColor White
Write-Host "- Docs: http://localhost:8010/docs" -ForegroundColor White

Write-Host "`nPress any key to stop all services..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Stop processes
Write-Host "`nStopping services..." -ForegroundColor Yellow
Stop-Process -Id $apiGateway.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue

Write-Host "✅ All services stopped." -ForegroundColor Green