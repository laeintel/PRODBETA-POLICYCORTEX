# Restart Services Script
# This script stops any running services and restarts them

Write-Host "Restarting PolicyCortex Services..." -ForegroundColor Green

# Kill processes on ports
Write-Host "`nStopping existing services..." -ForegroundColor Yellow

# Kill process on port 3000 (Frontend)
$port3000 = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($port3000) {
    foreach ($pid in $port3000) {
        Write-Host "Killing process on port 3000 (PID: $pid)" -ForegroundColor Red
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }
}

# Kill process on port 8010 (API Gateway)
$port8010 = Get-NetTCPConnection -LocalPort 8010 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($port8010) {
    foreach ($pid in $port8010) {
        Write-Host "Killing process on port 8010 (PID: $pid)" -ForegroundColor Red
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }
}

# Wait a moment for ports to be released
Start-Sleep -Seconds 2

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
Write-Host "`n✅ Services restarted successfully!" -ForegroundColor Green
Write-Host "`nServices running:" -ForegroundColor Cyan
Write-Host "- API Gateway: http://localhost:8010" -ForegroundColor White
Write-Host "- Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "`nAPI Endpoints:" -ForegroundColor Cyan
Write-Host "- Policies: http://localhost:8010/api/v1/policies" -ForegroundColor White
Write-Host "- Health: http://localhost:8010/health" -ForegroundColor White
Write-Host "- Docs: http://localhost:8010/docs" -ForegroundColor White
Write-Host "`n🌐 Open your browser at: http://localhost:3000" -ForegroundColor Green