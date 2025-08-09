@echo off
echo ========================================
echo PolicyCortex Docker Deployment
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Docker is running ✓
echo.

REM Load environment variables
if exist .env.docker (
    echo Loading environment variables from .env.docker...
    for /f "delims=" %%x in (.env.docker) do (
        set "%%x"
    )
    echo Environment variables loaded ✓
) else (
    echo WARNING: .env.docker file not found. Using default values.
)
echo.

REM Stop any existing containers
echo Stopping existing containers...
docker-compose down
echo.

REM Build images
echo Building Docker images...
echo.
echo [1/3] Building Backend (Rust)...
docker-compose build backend --no-cache

echo.
echo [2/3] Building Frontend (Next.js)...
docker-compose build frontend --no-cache

echo.
echo [3/3] Building GraphQL Gateway...
docker-compose build graphql --no-cache

echo.
echo All images built successfully ✓
echo.

REM Start services
echo Starting all services...
docker-compose up -d

echo.
echo Waiting for services to be healthy...
timeout /t 10 /nobreak >nul

REM Check service health
echo.
echo Checking service status...
echo.
docker-compose ps

echo.
echo ========================================
echo PolicyCortex is starting up!
echo ========================================
echo.
echo Access the application at:
echo   Frontend:    http://localhost:3000
echo   Backend API: http://localhost:8080
echo   GraphQL:     http://localhost:4000
echo   PostgreSQL:  localhost:5432
echo   Redis:       localhost:6379
echo   EventStore:  http://localhost:2113
echo.
echo To view logs: docker-compose logs -f [service_name]
echo To stop:      docker-compose down
echo.
pause