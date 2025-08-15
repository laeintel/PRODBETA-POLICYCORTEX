@echo off
echo ===============================================
echo Starting PolicyCortex Simple Development Mode
echo ===============================================
echo.

REM Set environment variables
echo Setting environment variables...
set AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c
set USE_REAL_AZURE=true
set USE_REAL_DATA=true
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex
set REDIS_URL=redis://localhost:6379
set JWT_SECRET=dev-secret-key
set RUST_LOG=info

echo.
echo Starting Core Backend Service...
start "PolicyCortex Backend" cmd /k "cd core && cargo run"

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo Starting Frontend Service...
start "PolicyCortex Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ===============================================
echo Services Starting:
echo   Backend:  http://localhost:8080
echo   Frontend: http://localhost:3000
echo ===============================================
echo.
echo Login page: http://localhost:3000/login
echo.
echo Press any key to stop all services...
pause >nul

echo.
echo Stopping services...
taskkill /FI "WindowTitle eq PolicyCortex Backend*" /T /F 2>nul
taskkill /FI "WindowTitle eq PolicyCortex Frontend*" /T /F 2>nul
echo Services stopped.