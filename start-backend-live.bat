@echo off
echo ========================================
echo Starting PolicyCortex Backend - LIVE DATA MODE
echo ========================================

REM Set all critical flags for real data mode
set AZURE_SUBSCRIPTION_ID=632a3b06-2a6c-4b07-8f4f-6bf4c6184095
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c
set AZURE_CLIENT_SECRET=8fx8Q~K.Bas6Pv9Z4qA.hIKW_nnCQJTm8yP0da8G

REM CRITICAL: Enable real data mode
set USE_REAL_DATA=true
set USE_AZURE_DATA=true
set DISABLE_MOCK_DATA=true
set FAIL_FAST_MODE=true
set AZURE_USE_CLI_AUTH=true

REM Disable all mock/demo modes
set DEMO_MODE=false
set USE_MOCK_DATA=false
set ENABLE_MOCK_FALLBACK=false

REM Database and server config
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex
set HOST=0.0.0.0
set PORT=8080
set CORS_ORIGIN=http://localhost:3000

REM Logging
set RUST_LOG=info,policycortex_core=debug,core::azure=debug
set LOG_AZURE_REQUESTS=true

REM Disable Key Vault for local dev
set DISABLE_KEY_VAULT=true

echo.
echo Configuration:
echo - Azure Subscription: %AZURE_SUBSCRIPTION_ID%
echo - Azure Tenant: %AZURE_TENANT_ID%
echo - Real Data Mode: ENABLED
echo - Mock Data: DISABLED
echo - Fail Fast: ENABLED
echo.

echo Starting backend...
cd core
cargo run --release

pause