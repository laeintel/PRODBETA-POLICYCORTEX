@echo off
echo ===============================================
echo Testing Azure Connection and Data Retrieval
echo ===============================================
echo.

REM Test health endpoint
echo Testing health endpoint...
curl -s http://localhost:8080/health | findstr "healthy" >nul
if %errorlevel% equ 0 (
    echo   ✓ Health check passed
) else (
    echo   ✗ Health check failed
    echo   Make sure the backend is running: docker logs policycortex-core
    exit /b 1
)

REM Test metrics endpoint (should return real Azure data)
echo.
echo Testing metrics endpoint for real Azure data...
curl -s http://localhost:8080/api/v1/metrics > metrics.json

REM Check if we got data
findstr "policies" metrics.json >nul
if %errorlevel% equ 0 (
    echo   ✓ Metrics endpoint returned data
    
    REM Display some key metrics
    echo.
    echo Key Metrics from Azure:
    echo ----------------------------------------
    type metrics.json | findstr /C:"total" /C:"compliance_rate" /C:"current_spend" /C:"users" /C:"resources"
    echo ----------------------------------------
) else (
    echo   ✗ Metrics endpoint failed or returned no data
    echo   Check backend logs: docker logs policycortex-core
)

REM Test predictions endpoint
echo.
echo Testing predictions endpoint...
curl -s http://localhost:8080/api/v1/predictions > predictions.json
if %errorlevel% equ 0 (
    echo   ✓ Predictions endpoint accessible
) else (
    echo   ✗ Predictions endpoint failed
)

REM Test recommendations endpoint
echo.
echo Testing recommendations endpoint...
curl -s http://localhost:8080/api/v1/recommendations > recommendations.json
if %errorlevel% equ 0 (
    echo   ✓ Recommendations endpoint accessible
) else (
    echo   ✗ Recommendations endpoint failed
)

REM Check for Azure authentication
echo.
echo Checking Azure authentication in backend...
docker logs policycortex-core 2>&1 | findstr "Azure client initialized" >nul
if %errorlevel% equ 0 (
    echo   ✓ Azure client initialized successfully
) else (
    echo   ✗ Azure client initialization failed
    echo.
    echo   Please ensure these environment variables are set:
    echo   - AZURE_SUBSCRIPTION_ID
    echo   - AZURE_TENANT_ID  
    echo   - AZURE_CLIENT_ID
    echo.
    echo   And that you're logged in: az login
)

REM Clean up temp files
del metrics.json predictions.json recommendations.json 2>nul

echo.
echo ===============================================
echo Test complete!
echo ===============================================