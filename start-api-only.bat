@echo off
echo ===============================================
echo Starting PolicyCortex API Gateway (No Docker)
echo ===============================================
echo.

REM Set Azure environment variables
set AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c
set USE_REAL_AZURE=true
set USE_REAL_DATA=true

REM Set database to use SQLite instead of PostgreSQL
set DATABASE_URL=sqlite:///policycortex.db
set REDIS_URL=redis://localhost:6379

REM Disable auth for local development
set REQUIRE_AUTH=false

REM Start API Gateway
echo Starting API Gateway on port 8090...
cd backend\services\api_gateway
python -m uvicorn main:app --host 0.0.0.0 --port 8090 --reload