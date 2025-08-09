@echo off
echo Starting PolicyCortex Backend with Real Azure Connection...
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
set RUST_LOG=debug

echo Environment configured for real Azure connection ✓
echo.
echo AZURE_SUBSCRIPTION_ID: %AZURE_SUBSCRIPTION_ID%
echo USE_REAL_AZURE: %USE_REAL_AZURE%
echo.

REM Try to compile and run
echo Attempting to compile backend...
cd core
cargo build --release 2>nul
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Backend compilation has errors. The system has:
    echo - Async Azure client with authentication ready
    echo - Data mode system (real vs simulated)
    echo - Core API endpoints configured
    echo - Caching and connection pooling set up
    echo.
    echo To complete real Azure connection:
    echo 1. Fix remaining compilation errors in security_graph and finops modules
    echo 2. Run this script again
    echo.
    echo For now, you can use the frontend with simulated data.
    pause
    exit /b 1
)

echo Backend compiled successfully!
echo Starting backend server...
cargo run --release