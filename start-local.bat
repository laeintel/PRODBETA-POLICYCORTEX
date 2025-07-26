@echo off
echo Starting PolicyCortex API Gateway...

REM Set environment variables for local development
set ENVIRONMENT=development
set SERVICE_NAME=api_gateway
set SERVICE_PORT=8000
set JWT_SECRET_KEY=local-dev-secret-key
set DEBUG=true
set LOG_LEVEL=INFO

REM Mock Azure settings for local dev
set AZURE_CLIENT_ID=dummy
set AZURE_TENANT_ID=dummy
set AZURE_CLIENT_SECRET=dummy
set AZURE_SUBSCRIPTION_ID=dummy

REM Mock service URLs for local dev
set AZURE_INTEGRATION_URL=http://localhost:8001
set AI_ENGINE_URL=http://localhost:8002
set DATA_PROCESSING_URL=http://localhost:8003
set CONVERSATION_URL=http://localhost:8004
set NOTIFICATION_URL=http://localhost:8005

REM Mock database settings
set AZURE_SQL_SERVER=localhost
set AZURE_SQL_DATABASE=policortex_dev
set AZURE_SQL_USERNAME=sa
set AZURE_SQL_PASSWORD=YourStrong@Passw0rd

REM Mock Cosmos DB
set AZURE_COSMOS_ENDPOINT=https://localhost:8081/
set AZURE_COSMOS_KEY=dummy-key
set AZURE_COSMOS_DATABASE=policortex_dev

REM Mock Redis
set REDIS_HOST=localhost
set REDIS_PASSWORD=
set REDIS_PORT=6379
set REDIS_SSL=false

REM Navigate to API Gateway and run
cd backend\services\api_gateway
echo.
echo Running API Gateway on http://localhost:8000
echo Press Ctrl+C to stop
echo.
..\..\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000 --host 0.0.0.0