# Simple PolicyCortex Testing Script
Write-Host "PolicyCortex Local Testing Setup" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Step 1: Check Docker
Write-Host "`nStep 1: Checking Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "Docker found: $dockerVersion" -ForegroundColor Green
}
catch {
    Write-Host "Docker is not running or not installed" -ForegroundColor Red
    exit 1
}

# Step 2: Start Services
Write-Host "`nStep 2: Starting Services..." -ForegroundColor Yellow
Write-Host "Stopping any existing containers..." -ForegroundColor Yellow
docker-compose -f docker-compose.local.yml down

Write-Host "Starting all services..." -ForegroundColor Yellow
docker-compose -f docker-compose.local.yml up -d --build

# Step 3: Wait for Services
Write-Host "`nStep 3: Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Step 4: Check Service Health
Write-Host "`nStep 4: Checking Service Health..." -ForegroundColor Yellow

$services = @(
    @{ Name = "API Gateway"; Port = "8000" },
    @{ Name = "Azure Integration"; Port = "8001" },
    @{ Name = "AI Engine"; Port = "8002" },
    @{ Name = "Data Processing"; Port = "8003" },
    @{ Name = "Conversation"; Port = "8004" },
    @{ Name = "Notification"; Port = "8005" }
)

$healthyCount = 0
foreach ($service in $services) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$($service.Port)/health" -Method Get -TimeoutSec 5
        Write-Host "$($service.Name): HEALTHY" -ForegroundColor Green
        $healthyCount++
    }
    catch {
        Write-Host "$($service.Name): FAILED" -ForegroundColor Red
    }
}

Write-Host "`nHealth Check Results: $healthyCount/$($services.Count) services healthy" -ForegroundColor Cyan

# Step 5: Test Patent APIs
Write-Host "`nStep 5: Testing Patent APIs with Python script..." -ForegroundColor Yellow
try {
    python test_patent_apis.py
}
catch {
    Write-Host "Python test script failed or Python not available" -ForegroundColor Yellow
}

# Step 6: Final Results
Write-Host "`nTesting Complete!" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Host "Services Running: $healthyCount/$($services.Count)" -ForegroundColor White
Write-Host "Frontend URL: http://localhost:5173" -ForegroundColor White
Write-Host "API Gateway: http://localhost:8000" -ForegroundColor White
Write-Host "AI Engine: http://localhost:8002" -ForegroundColor White

Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "1. Open http://localhost:5173 in your browser" -ForegroundColor White
Write-Host "2. Go to AI Assistant page to test conversational AI" -ForegroundColor White
Write-Host "3. Check logs if needed: docker-compose -f docker-compose.local.yml logs" -ForegroundColor White