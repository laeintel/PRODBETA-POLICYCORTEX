# Test Demo Mode Authentication Bypass

Write-Host "Testing PolicyCortex Demo Mode..." -ForegroundColor Green
Write-Host ""

# Check if demo mode is enabled
Write-Host "1. Checking demo mode configuration..." -ForegroundColor Yellow
$envPath = "frontend\.env.local"
if (Test-Path $envPath) {
    $demoMode = Select-String -Path $envPath -Pattern "NEXT_PUBLIC_DEMO_MODE=true"
    if ($demoMode) {
        Write-Host "  Demo mode is ENABLED in .env.local" -ForegroundColor Green
    } else {
        Write-Host "  Demo mode is DISABLED in .env.local" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "2. Testing demo session endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000/api/auth/demo" -Method GET -UseBasicParsing
    $content = $response.Content | ConvertFrom-Json
    if ($content.demoModeEnabled) {
        Write-Host "  Demo mode endpoint confirms demo mode is ENABLED" -ForegroundColor Green
    } else {
        Write-Host "  Demo mode endpoint says demo mode is DISABLED" -ForegroundColor Red
    }
    if ($content.authenticated) {
        Write-Host "  User is authenticated" -ForegroundColor Green
    } else {
        Write-Host "  User is not authenticated, creating session..." -ForegroundColor Yellow
        # Create demo session
        $postResponse = Invoke-WebRequest -Uri "http://localhost:3000/api/auth/demo" -Method POST -UseBasicParsing
        if ($postResponse.StatusCode -eq 200) {
            Write-Host "  Demo session created successfully" -ForegroundColor Green
        }
    }
} catch {
    Write-Host "  Failed to check demo session" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "3. Testing protected endpoints..." -ForegroundColor Yellow

$endpoints = @(
    "/dashboard",
    "/tactical",
    "/ai/unified",
    "/resources"
)

foreach ($endpoint in $endpoints) {
    try {
        # Note: This won't follow redirects, which is what we want
        $response = Invoke-WebRequest -Uri "http://localhost:3000$endpoint" -Method GET -UseBasicParsing -MaximumRedirection 0 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "  $endpoint - Accessible (Status: 200)" -ForegroundColor Green
        } else {
            Write-Host "  $endpoint - Status: $($response.StatusCode)" -ForegroundColor Yellow
        }
    } catch {
        if ($_.Exception.Response.StatusCode -eq 307 -or $_.Exception.Response.StatusCode -eq 308) {
            Write-Host "  $endpoint - Redirecting (authentication required)" -ForegroundColor Red
        } else {
            Write-Host "  $endpoint - Error occurred" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DEMO MODE SETUP INSTRUCTIONS:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "If demo mode is not working properly:" -ForegroundColor White
Write-Host "1. Ensure NEXT_PUBLIC_DEMO_MODE=true in frontend\.env.local" -ForegroundColor White
Write-Host "2. Restart the Next.js development server" -ForegroundColor White
Write-Host "3. Clear browser cookies for localhost:3000" -ForegroundColor White
Write-Host "4. Navigate to http://localhost:3000" -ForegroundColor White
Write-Host ""
Write-Host "The app is running at: http://localhost:3000" -ForegroundColor Green
Write-Host ""