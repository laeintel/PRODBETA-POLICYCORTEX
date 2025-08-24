@echo off
echo ========================================
echo Starting PolicyCortex Backend Services
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    exit /b 1
)

REM Start databases if not running
echo [1/7] Checking databases...
docker ps | findstr policycortex-postgres >nul
if %errorlevel% neq 0 (
    echo Starting PostgreSQL...
    docker run -d --name policycortex-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=policycortex -p 5432:5432 postgres:15-alpine
) else (
    echo PostgreSQL already running
)

docker ps | findstr policycortex-redis >nul
if %errorlevel% neq 0 (
    echo Starting Redis...
    docker run -d --name policycortex-redis -p 6379:6379 redis:7-alpine
) else (
    echo Redis already running
)

echo.
echo [2/7] Starting Core API (Rust)...
start "Core API" cmd /c "cd core && cargo run"
timeout /t 3 >nul

echo [3/7] Starting GraphQL Gateway...
start "GraphQL Gateway" cmd /c "cd graphql && npm run dev"
timeout /t 3 >nul

echo [4/7] Starting Python API Gateway...
REM Fix Python imports first
set PYTHONPATH=%CD%\backend;%CD%\backend\services
start "Python API Gateway" cmd /c "cd backend\services\api_gateway && python -m uvicorn main:app --reload --port 8000"
timeout /t 3 >nul

echo [5/7] Starting ML Services...
start "ML Prediction" cmd /c "cd ml-service && python app.py"
timeout /t 3 >nul

echo [6/7] Starting WebSocket Server...
start "WebSocket" cmd /c "cd backend\services\websocket && npm run dev"
timeout /t 3 >nul

echo [7/7] Frontend is already running on port 3000
echo.

echo ========================================
echo Waiting for services to be ready...
echo ========================================
timeout /t 10 >nul

echo.
echo Testing service endpoints...
echo ========================================

REM Test Core API
curl -s http://localhost:8080/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Core API: http://localhost:8080
) else (
    echo [FAIL] Core API not responding
)

REM Test GraphQL
curl -s http://localhost:4000/graphql -H "Content-Type: application/json" -d "{\"query\": \"{ __typename }\"}" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] GraphQL: http://localhost:4000
) else (
    echo [FAIL] GraphQL not responding
)

REM Test Python API
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Python API: http://localhost:8000
) else (
    echo [FAIL] Python API not responding
)

REM Test Frontend
curl -s http://localhost:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Frontend: http://localhost:3000
) else (
    echo [FAIL] Frontend not responding
)

echo.
echo ========================================
echo Services Status:
echo ========================================
echo Frontend:     http://localhost:3000
echo Core API:     http://localhost:8080/docs
echo GraphQL:      http://localhost:4000/graphql
echo Python API:   http://localhost:8000/docs
echo PostgreSQL:   localhost:5432
echo Redis:        localhost:6379
echo ========================================
echo.
echo Press any key to stop all services...
pause >nul

echo.
echo Stopping services...
taskkill /F /FI "WindowTitle eq Core API*" >nul 2>&1
taskkill /F /FI "WindowTitle eq GraphQL Gateway*" >nul 2>&1
taskkill /F /FI "WindowTitle eq Python API Gateway*" >nul 2>&1
taskkill /F /FI "WindowTitle eq ML Prediction*" >nul 2>&1
taskkill /F /FI "WindowTitle eq WebSocket*" >nul 2>&1
echo Services stopped.