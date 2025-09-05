@echo off
echo ========================================
echo Starting PolicyCortex Production Build
echo With Live Azure Data
echo ========================================

REM Set Azure credentials for live data
set AZURE_SUBSCRIPTION_ID=6dc7cfa2-0332-4740-98b6-bac9f1a23de9
set AZURE_TENANT_ID=e1f3e196-aa55-4709-9c55-0e334c0b444f
set AZURE_CLIENT_ID=232c44f7-d0cf-4825-a9b5-beba9f587ffb

REM Set other environment variables
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex
set REDIS_URL=redis://localhost:6379
set RUST_LOG=info
set NODE_ENV=production

REM Disable simulated data mode
set USE_SIMULATED_DATA=false
set USE_REAL_DATA=true

echo.
echo [1/4] Building Rust backend...
cd ..\..\core
cargo build --release
if %errorlevel% neq 0 (
    echo Failed to build Rust backend
    exit /b 1
)

echo.
echo [2/4] Starting Rust backend on port 8080...
start "PolicyCortex Core API" cmd /c "cargo run --release"

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo [3/4] Building Next.js frontend...
cd ..\frontend
call npm run build
if %errorlevel% neq 0 (
    echo Failed to build frontend
    exit /b 1
)

echo.
echo [4/4] Starting Next.js frontend on port 3000...
start "PolicyCortex Frontend" cmd /c "npm start"

echo.
echo ========================================
echo PolicyCortex is starting up!
echo ========================================
echo.
echo Backend API:  http://localhost:8080
echo Frontend UI:  http://localhost:3000
echo.
echo Using LIVE Azure data from subscription:
echo %AZURE_SUBSCRIPTION_ID%
echo.
echo Press any key to open the browser...
pause >nul

start http://localhost:3000

echo.
echo Application is running. Close this window to keep services running.
echo To stop all services, close the individual command windows.
pause