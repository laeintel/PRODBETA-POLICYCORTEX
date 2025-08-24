@echo off
echo =============================================================
echo PolicyCortex Quick Setup Script
echo =============================================================
echo.

echo [INFO] Checking Docker...
docker --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop.
    echo Download from: https://docker.com/get-started
    pause
    exit /b 1
)
echo [OK] Docker is installed

echo.
echo [INFO] Starting Docker services...
cd /d "%~dp0"
docker-compose -f docker-services.yml up -d
if %errorLevel% neq 0 (
    echo [ERROR] Failed to start Docker services
    echo Please check if Docker Desktop is running
    pause
    exit /b 1
)

echo.
echo [INFO] Waiting for services to start (30 seconds)...
timeout /t 30 /nobreak >nul

echo.
echo [INFO] Checking service health...
docker ps --format "table {{.Names}}\t{{.Status}}" | findstr "policycortex"

echo.
echo [INFO] Initializing database...
docker exec -i policycortex-postgres psql -U postgres -c "CREATE DATABASE IF NOT EXISTS policycortex;" 2>nul
docker exec -i policycortex-postgres psql -U postgres -d policycortex < init-db.sql
if %errorLevel% equ 0 (
    echo [OK] Database initialized
) else (
    echo [WARNING] Database initialization had issues (may already exist)
)

echo.
echo =============================================================
echo [SUCCESS] Setup completed!
echo =============================================================
echo.
echo Services are running at:
echo - PostgreSQL: localhost:5432
echo - Redis/DragonflyDB: localhost:6379
echo - EventStore: http://localhost:2113
echo - Grafana: http://localhost:3010 (admin/admin)
echo - Prometheus: http://localhost:9090
echo - Jaeger: http://localhost:16686
echo - MLflow: http://localhost:5000
echo - Adminer: http://localhost:8081
echo.
echo To start the application:
echo 1. Frontend: cd frontend ^&^& npm run dev
echo 2. Backend: cd core ^&^& cargo run
echo.
echo To stop services: docker-compose -f docker-services.yml down
echo.
pause