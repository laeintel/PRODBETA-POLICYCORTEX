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

REM Start infrastructure services (PostgreSQL, Redis, EventStore)
echo.
echo Starting infrastructure services...
docker-compose -f docker-compose.dev.yml up -d postgres redis eventstore
if %errorlevel% neq 0 (
    echo ERROR: Failed to start infrastructure services
    echo Try running: docker-compose -f docker-compose.dev.yml down -v
    exit /b 1
)

REM Wait for services to be ready
echo Waiting for infrastructure services to be ready...
timeout /t 10 /nobreak >nul

REM Start the core service locally with cargo
echo.
echo Starting Core service locally...
start "PolicyCortex Core" cmd /k "cd core && cargo run"

REM Start the GraphQL service locally
echo.
echo Starting GraphQL service locally...
start "PolicyCortex GraphQL" cmd /k "cd graphql && npm install && npm run dev"

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
echo   GraphQL:      http://localhost:4000/graphql
echo   Frontend:     http://localhost:3000
echo   PostgreSQL:   localhost:5432
echo   Redis:        localhost:6379
echo   EventStore:   http://localhost:2113
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
echo To view Docker logs:
echo   docker-compose -f docker-compose.dev.yml logs -f
echo.
echo To stop all services:
echo   docker-compose -f docker-compose.dev.yml down
echo   Close all command windows (Core, GraphQL, Frontend)
echo ===============================================