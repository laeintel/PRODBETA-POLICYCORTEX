@echo off
REM PolicyCortex v2 Local Development Startup Script for Windows

echo =============================
echo PolicyCortex v2 Local Development
echo =============================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo [1/5] Cleaning up existing containers...
docker-compose -f docker-compose.local.yml down >nul 2>&1

echo [2/5] Building services...
docker-compose -f docker-compose.local.yml build
if %errorlevel% neq 0 (
    echo [ERROR] Build failed. Check Docker logs.
    pause
    exit /b 1
)

echo [3/5] Starting services...
docker-compose -f docker-compose.local.yml up -d
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start services.
    pause
    exit /b 1
)

echo [4/5] Waiting for services to initialize...
timeout /t 10 /nobreak >nul

echo [5/5] Service Status:
echo.
echo Core Service:     http://localhost:8080/health
echo GraphQL Gateway:  http://localhost:4000/graphql
echo Frontend:         http://localhost:3000
echo EventStore UI:    http://localhost:2113
echo.
echo PostgreSQL:       localhost:5432 (user: postgres, pass: postgres)
echo DragonflyDB:      localhost:6379
echo Edge Simulator:   http://localhost:8787
echo.
echo =============================
echo Services are starting up...
echo Press Ctrl+C to stop watching logs
echo =============================
echo.

docker-compose -f docker-compose.local.yml logs -f