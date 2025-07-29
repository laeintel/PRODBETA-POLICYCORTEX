@echo off
echo ========================================
echo PolicyCortex Frontend Development Setup
echo ========================================
echo.

echo Checking if Docker is running...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    echo Then run this script again.
    pause
    exit /b 1
)
echo ✓ Docker is running

echo.
echo IMPORTANT: Before continuing, please update .env.development with your real Azure values:
echo - AZURE_SUBSCRIPTION_ID
echo - AZURE_TENANT_ID  
echo - AZURE_CLIENT_ID
echo - AZURE_CLIENT_SECRET
echo - AZURE_COSMOS_ENDPOINT and AZURE_COSMOS_KEY
echo - AZURE_SQL_SERVER and AZURE_SQL_PASSWORD
echo - AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY
echo.
set /p continue="Have you updated .env.development with real Azure values? (y/n): "
if /i not "%continue%"=="y" (
    echo Please update .env.development first, then run this script again.
    pause
    exit /b 1
)

echo.
echo Starting PolicyCortex frontend development environment...
echo This will start:
echo - Redis (caching)
echo - API Gateway (backend)
echo - Azure Integration Service  
echo - Frontend (React app)
echo.

echo Building and starting containers...
docker-compose -f docker-compose.frontend-dev.yml up --build -d

echo.
echo Waiting for services to start...
timeout /t 10 >nul

echo.
echo ========================================
echo Services Status:
echo ========================================
docker-compose -f docker-compose.frontend-dev.yml ps

echo.
echo ========================================
echo URLs for Monitoring:
echo ========================================
echo Frontend:          http://localhost:5173
echo API Gateway:       http://localhost:8000
echo API Gateway Health: http://localhost:8000/health
echo API Gateway Docs:  http://localhost:8000/docs
echo Azure Integration: http://localhost:8001
echo Redis:             localhost:6379
echo.

echo ========================================
echo Monitoring Commands:
echo ========================================
echo View logs:         docker-compose -f docker-compose.frontend-dev.yml logs -f
echo Stop services:     docker-compose -f docker-compose.frontend-dev.yml down
echo Restart frontend:  docker-compose -f docker-compose.frontend-dev.yml restart frontend
echo.

echo Frontend development environment is ready!
echo Open http://localhost:5173 in your browser to start monitoring.
pause