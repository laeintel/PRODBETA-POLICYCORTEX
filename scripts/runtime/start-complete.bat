@echo off
REM PolicyCortex Complete Local Testing Environment
REM This script starts all services for comprehensive local testing

echo ===============================================
echo PolicyCortex Complete Testing Environment
echo ===============================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

REM Check for .env.development file
if not exist .env.development (
    echo [WARNING] .env.development file not found!
    echo Creating from .env.example...
    if exist .env.example (
        copy .env.example .env.development
    ) else (
        echo [INFO] No .env.example file found. Continuing without environment file.
    )
)

REM Load environment variables
if exist .env.development (
    echo Loading environment variables from .env.development...
    for /f "tokens=1,2 delims==" %%a in (.env.development) do (
        if not "%%a"=="" if not "%%b"=="" (
            echo %%a | findstr /b "#" >nul
            if errorlevel 1 (
                set "%%a=%%b"
            )
        )
    )
)

echo.
echo [Phase 1/4] Cleaning up...
echo ----------------------------------------
docker-compose -f docker-compose.local.yml down >nul 2>&1
docker-compose -f docker-compose.dev.yml down >nul 2>&1

echo.
echo [Phase 2/4] Starting Infrastructure Services...
echo ----------------------------------------
echo Starting PostgreSQL, Redis, and EventStore...
docker-compose -f docker-compose.dev.yml up -d postgres redis eventstore
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start infrastructure services
    echo Try running: docker-compose -f docker-compose.dev.yml down -v
    pause
    exit /b 1
)

REM Wait for databases to be ready
echo Waiting for databases to initialize...
timeout /t 10 /nobreak >nul

REM Test database connections
echo Testing database connections...
docker exec policycortex-postgres-dev psql -U postgres -d policycortex -c "SELECT 1;" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] PostgreSQL is ready
) else (
    echo   [WARNING] PostgreSQL may not be ready yet
)

docker exec policycortex-redis-dev redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Redis/DragonflyDB is ready
) else (
    echo   [WARNING] Redis may not be ready yet
)

echo.
echo [Phase 3/4] Starting Application Services...
echo ----------------------------------------

REM Option 1: Try Docker Compose first for all services
echo Attempting to start services via Docker Compose...
docker-compose -f docker-compose.local.yml up -d --build
if %errorlevel% equ 0 (
    echo   [OK] Services started via Docker Compose
    set "START_MODE=docker"
) else (
    echo   [INFO] Docker Compose failed, starting services locally...
    set "START_MODE=local"
    
    REM Start Core service locally
    echo Starting Core service locally...
    start "PolicyCortex Core" cmd /k "cd core && cargo run"
    
    REM Start GraphQL service locally
    echo Starting GraphQL service locally...
    start "PolicyCortex GraphQL" cmd /k "cd graphql && npm install && npm run dev"
    
    REM Start Frontend service locally
    echo Starting Frontend service locally...
    start "PolicyCortex Frontend" cmd /k "cd frontend && npm install && npm run dev"
)

REM Wait for services to start
echo.
echo Waiting for services to initialize (30 seconds)...
timeout /t 30 /nobreak >nul

echo.
echo [Phase 4/4] Verifying Services...
echo ----------------------------------------

REM Test service endpoints
echo Testing service endpoints...

curl -f -s http://localhost:8080/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Core API is responding
) else (
    echo   [WARNING] Core API not responding (may still be starting)
)

curl -f -s http://localhost:4000/.well-known/apollo/server-health >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] GraphQL Gateway is responding
) else (
    echo   [WARNING] GraphQL not responding (may still be starting)
)

curl -f -s http://localhost:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Frontend is responding
) else (
    echo   [WARNING] Frontend not responding (may still be starting)
)

echo.
echo ===============================================
echo PolicyCortex Testing Environment Ready!
echo ===============================================
echo.
echo Service Endpoints:
echo   Frontend:         http://localhost:3000
echo   Core API:         http://localhost:8080
echo   GraphQL:          http://localhost:4000/graphql
echo   PostgreSQL:       localhost:5432 (user: postgres, pass: postgres)
echo   Redis/Dragonfly:  localhost:6379
echo   EventStore UI:    http://localhost:2113 (admin/changeit)
echo.
echo Key API Endpoints:
echo   Health Check:     http://localhost:8080/health
echo   Metrics:          http://localhost:8080/api/v1/metrics
echo   Predictions:      http://localhost:8080/api/v1/predictions
echo   Correlations:     http://localhost:8080/api/v1/correlations
echo   Conversation:     http://localhost:8080/api/v1/conversation
echo.

if "%START_MODE%"=="docker" (
    echo Services started via: Docker Compose
    echo.
    echo To view logs:
    echo   docker-compose -f docker-compose.local.yml logs -f [service]
    echo.
    echo To stop all services:
    echo   docker-compose -f docker-compose.local.yml down
    echo   docker-compose -f docker-compose.dev.yml down
) else (
    echo Services started via: Local Development Mode
    echo.
    echo To view logs:
    echo   Check the individual command windows
    echo.
    echo To stop all services:
    echo   Close all command windows
    echo   docker-compose -f docker-compose.dev.yml down
)

echo.
echo ===============================================
echo Ready for testing! Press Ctrl+C to stop.
echo ===============================================
echo.

if "%START_MODE%"=="docker" (
    REM Follow logs if using Docker
    docker-compose -f docker-compose.local.yml logs -f
) else (
    REM Just keep the window open if running locally
    pause
)