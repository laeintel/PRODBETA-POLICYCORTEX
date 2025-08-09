@echo off
echo Starting PolicyCortex Backend with Tenant-Level Architecture...
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

REM Set environment variables for tenant-level operation
REM Only tenant ID is required - subscriptions will be discovered dynamically
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set RUST_LOG=info

echo Environment configured for tenant-level governance ✓
echo.
echo AZURE_TENANT_ID: %AZURE_TENANT_ID%
echo.
echo The system will now discover all subscriptions you have access to...
echo.

REM Run the backend
cd core
cargo run --release