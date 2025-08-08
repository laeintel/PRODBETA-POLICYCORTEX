@echo off
echo ===============================================
echo Starting PolicyCortex Development Environment
echo ===============================================
echo.

REM Check if Azure CLI is logged in
echo Checking Azure authentication...
az account show >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Not logged in to Azure CLI
    echo Please run: az login
    exit /b 1
)

REM Display current Azure subscription
echo Current Azure Subscription:
az account show --query "{Name:name, ID:id}" --output table

REM Load environment variables from .env.development
echo.
echo Loading environment variables from .env.development...
if exist .env.development (
    for /f "tokens=1,2 delims==" %%a in (.env.development) do (
        if not "%%a"=="" if not "%%b"=="" (
            REM Skip comments
            echo %%a | findstr /b "#" >nul
            if errorlevel 1 (
                set "%%a=%%b"
                echo   Set %%a
            )
        )
    )
) else (
    echo WARNING: .env.development file not found!
)

REM Stop any existing containers
echo.
echo Stopping any existing containers...
docker-compose -f docker-compose.dev.yml down 2>nul

REM Start Redis first (needed for caching)
echo.
echo Starting Redis cache...
docker run -d --name policycortex-redis -p 6379:6379 redis:alpine 2>nul || docker start policycortex-redis

REM Wait for Redis to be ready
echo Waiting for Redis to be ready...
timeout /t 3 /nobreak >nul

REM Build and start the core service
echo.
echo Building and starting Core service...
docker build -t policycortex-core ./core
if %errorlevel% neq 0 (
    echo ERROR: Failed to build Core service
    exit /b 1
)

REM Start core with all required environment variables
docker run -d --name policycortex-core -p 8080:8080 ^
  -e AZURE_SUBSCRIPTION_ID=%AZURE_SUBSCRIPTION_ID% ^
  -e AZURE_TENANT_ID=%AZURE_TENANT_ID% ^
  -e AZURE_CLIENT_ID=%AZURE_CLIENT_ID% ^
  -e REDIS_URL=redis://host.docker.internal:6379 ^
  -e RUST_LOG=debug ^
  -e ENABLE_REAL_AZURE_DATA=true ^
  -e ENABLE_CACHE=true ^
  policycortex-core 2>nul || docker restart policycortex-core

REM Wait for core to be ready
echo Waiting for Core service to be ready...
:wait_core
timeout /t 2 /nobreak >nul
curl -f http://localhost:8080/health >nul 2>&1
if %errorlevel% neq 0 (
    echo   Still waiting...
    goto wait_core
)
echo   Core service is ready!

REM Start the frontend in a new window
echo.
echo Starting Frontend service in new window...
start "PolicyCortex Frontend" cmd /k "cd frontend && npm run dev"

REM Display status
echo.
echo ===============================================
echo PolicyCortex Development Environment Started!
echo ===============================================
echo.
echo Services:
echo   Core API:     http://localhost:8080
echo   Frontend:     http://localhost:3000
echo   Redis:        localhost:6379
echo.
echo API Endpoints:
echo   Health:       http://localhost:8080/health
echo   Metrics:      http://localhost:8080/api/v1/metrics
echo   Predictions:  http://localhost:8080/api/v1/predictions
echo   Correlations: http://localhost:8080/api/v1/correlations
echo.
echo Azure Configuration:
echo   Subscription: %AZURE_SUBSCRIPTION_ID%
echo   Tenant:       %AZURE_TENANT_ID%
echo   Client ID:    %AZURE_CLIENT_ID%
echo.
echo To view logs:
echo   docker logs policycortex-core -f
echo.
echo To stop all services:
echo   docker stop policycortex-core policycortex-redis
echo   Close the Frontend window
echo ===============================================