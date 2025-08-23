@echo off
echo ========================================
echo PolicyCortex - Live Azure Data Mode
echo ========================================
echo.

REM Set Azure credentials
set AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c

REM Enable Azure integration
set USE_AZURE_DATA=true
set AZURE_AUTH_MODE=cli

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