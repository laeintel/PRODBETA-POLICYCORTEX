@echo off
echo ========================================
echo Starting PolicyCortex Backend with LIVE Azure Data
echo ========================================

REM Set Azure credentials
set AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c

REM Set database and cache
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex
set REDIS_URL=redis://localhost:6379

REM Force LIVE data mode
set USE_SIMULATED_DATA=false
set USE_REAL_DATA=true
set USE_MOCK_DATA=false
set DATA_MODE=live

REM Set logging
set RUST_LOG=info,policycortex_core=debug

REM Disable SQLx offline mode
set SQLX_OFFLINE=

echo.
echo Environment Variables Set:
echo AZURE_SUBSCRIPTION_ID=%AZURE_SUBSCRIPTION_ID%
echo AZURE_TENANT_ID=%AZURE_TENANT_ID%
echo AZURE_CLIENT_ID=%AZURE_CLIENT_ID%
echo USE_REAL_DATA=%USE_REAL_DATA%
echo DATA_MODE=%DATA_MODE%
echo.

cd core
echo Building and starting backend...
cargo run --release