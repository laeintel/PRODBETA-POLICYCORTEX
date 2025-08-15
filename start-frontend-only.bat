@echo off
echo ===============================================
echo Starting PolicyCortex Frontend-Only Mode
echo ===============================================
echo.

REM Start PostgreSQL and Redis for data
echo Starting core services...
docker-compose up -d postgres redis 2>nul

REM Wait for services
echo Waiting for services to be ready...
timeout /t 5 /nobreak >nul

REM Check services
echo Checking service health...
docker exec policycortex-postgres psql -U postgres -d policycortex -c "SELECT 1" >nul 2>&1 && echo   PostgreSQL: Ready || echo   PostgreSQL: Failed
docker exec policycortex-redis redis-cli ping >nul 2>&1 && echo   Redis: Ready || echo   Redis: Failed

REM Start frontend
echo.
echo Starting Frontend (with demo data)...
cd frontend
start "PolicyCortex Frontend" cmd /k "npm run dev"

echo.
echo ===============================================
echo PolicyCortex Frontend Started!
echo ===============================================
echo.
echo Services:
echo   Frontend:     http://localhost:3000 (demo mode)
echo   PostgreSQL:   localhost:5432
echo   Redis:        localhost:6379
echo.
echo Note: Backend API not running - using demo data
echo To stop: Close frontend window and run:
echo   docker-compose down
echo ===============================================