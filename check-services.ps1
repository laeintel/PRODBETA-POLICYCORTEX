# Check if PolicyCortex services are running
Write-Host "PolicyCortex Service Health Check" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Check backend API
Write-Host "`nChecking Backend API..." -ForegroundColor Yellow
try {
    $backendResponse = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -UseBasicParsing
    if ($backendResponse.StatusCode -eq 200) {
        Write-Host "✅ Backend API is running on http://localhost:8000" -ForegroundColor Green
        Write-Host "   Status: $($backendResponse.StatusCode)" -ForegroundColor Gray
    }
} catch {
    Write-Host "❌ Backend API is not responding on http://localhost:8000" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Gray
}

# Check frontend
Write-Host "`nChecking Frontend..." -ForegroundColor Yellow
try {
    $frontendResponse = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 5 -UseBasicParsing
    if ($frontendResponse.StatusCode -eq 200) {
        Write-Host "✅ Frontend is running on http://localhost:5173" -ForegroundColor Green
        Write-Host "   Status: $($frontendResponse.StatusCode)" -ForegroundColor Gray
    }
} catch {
    Write-Host "❌ Frontend is not responding on http://localhost:5173" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Gray
}

Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "Health check completed!" -ForegroundColor Green

Write-Host "`nQuick Links:" -ForegroundColor Yellow
Write-Host "🌐 Frontend App: http://localhost:5173" -ForegroundColor Cyan
Write-Host "🔧 Backend API: http://localhost:8000" -ForegroundColor Cyan  
Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "💾 ReDoc: http://localhost:8000/redoc" -ForegroundColor Cyan