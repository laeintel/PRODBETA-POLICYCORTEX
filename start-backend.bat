@echo off
echo Starting PolicyCortex Backend Services...
echo.

REM Check Azure credentials
echo Checking Azure authentication...
call az account show >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Azure CLI not authenticated. Please run 'az login' first.
    echo.
    pause
    exit /b 1
)

echo Azure authentication verified ✓
echo.

REM Set environment variables for real Azure connection
set AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c
set USE_REAL_AZURE=true
set USE_REAL_DATA=true
set RUST_LOG=info
set REQUIRE_AUTH=false

echo Environment configured ✓
echo.

REM Start API Gateway (Python)
echo Starting API Gateway on port 8090...
cd backend\services\api_gateway
start "PolicyCortex API Gateway" cmd /k "python main.py"
cd ..\..\..

REM Give API Gateway time to start
timeout /t 3 /nobreak >nul

REM Start Core API (Rust) - Skip if compilation fails
echo Starting Core API on port 8080...
cd core
cargo build 2>nul
if %errorlevel% equ 0 (
    start "PolicyCortex Core API" cmd /k "cargo run"
    echo Core API starting...
) else (
    echo WARNING: Core API compilation failed, skipping...
    echo API Gateway will handle requests with Python services
)
cd ..

echo.
echo Services are starting:
echo - API Gateway: http://localhost:8090 (Python)
echo - Core API: http://localhost:8080 (Rust - if compiled)
echo - Frontend: http://localhost:3000
echo.
echo Azure Integration: ENABLED
echo.
echo Press any key to exit...
pause >nul