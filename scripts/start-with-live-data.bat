@echo off
echo ========================================
echo PolicyCortex - Live Azure Data Mode
echo ========================================
echo.

REM Set Azure credentials - UPDATED TO CORRECT SUBSCRIPTION
set AZURE_SUBSCRIPTION_ID=632a3b06-2a6c-4b07-8f4f-6bf4c6184095
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c

REM CRITICAL: Enable Azure integration and disable ALL mock data
set USE_AZURE_DATA=true
set USE_REAL_DATA=true
set DISABLE_MOCK_DATA=true
set FAIL_FAST_MODE=true
set AZURE_AUTH_MODE=cli
set AZURE_USE_CLI_AUTH=true

REM CRITICAL: Disable demo/mock modes
set NEXT_PUBLIC_DEMO_MODE=false
set DEMO_MODE=false
set USE_MOCK_DATA=false
set ENABLE_MOCK_FALLBACK=false

REM Enable debugging
set RUST_LOG=debug,policycortex_core=debug
set LOG_AZURE_REQUESTS=true

echo Checking Azure connection...
call az account show >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Not logged into Azure. Please run: az login
    pause
    exit /b 1
)

echo Azure connection verified.
echo.

echo Starting backend with live Azure data...
cd core
start /B cargo run --release

echo Waiting for backend to start...
timeout /t 10 /nobreak >nul

echo.
echo Starting frontend...
cd ..\frontend
start /B npm run dev

echo.
echo ========================================
echo PolicyCortex is running with LIVE DATA!
echo ========================================
echo.
echo Backend:  http://localhost:8080
echo Frontend: http://localhost:3000
echo.
echo Health Check: http://localhost:8080/api/v1/health/azure
echo.
echo Press Ctrl+C to stop all services
echo.

pause