@echo off
REM ========================================================================
REM PolicyCortex v2 - Windows Bootstrap Script
REM One-click setup for development environment
REM ========================================================================

echo.
echo ================================================================
echo PolicyCortex v2 - Enterprise Bootstrap
echo ================================================================
echo.

REM Check for required tools
echo [1/7] Checking prerequisites...

where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js is not installed
    echo Please install Node.js from https://nodejs.org/
    exit /b 1
)

where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: npm is not installed
    echo Please install Node.js from https://nodejs.org/
    exit /b 1
)

where cargo >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Rust is not installed
    echo Please install Rust from https://rustup.rs/
    exit /b 1
)

where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Docker is not installed
    echo Docker is optional but recommended for full functionality
    echo Install from https://www.docker.com/
)

where az >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Azure CLI is not installed
    echo Azure CLI is optional for Azure integration
    echo Install from https://aka.ms/installazurecliwindows
)

echo Prerequisites check complete.
echo.

REM Check Azure authentication
echo [2/7] Checking Azure authentication...
az account show >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Not logged into Azure
    echo Run 'az login' to enable Azure features
    set USE_REAL_DATA=false
) else (
    echo Azure authentication detected
    set USE_REAL_DATA=true
)
echo.

REM Install frontend dependencies
echo [3/7] Installing frontend dependencies...
cd frontend
call npm install
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install frontend dependencies
    exit /b 1
)
cd ..
echo Frontend dependencies installed.
echo.

REM Build Rust backend
echo [4/7] Building Rust backend...
cd core
cargo build --release
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to build Rust backend
    exit /b 1
)
cd ..
echo Rust backend built successfully.
echo.

REM Setup environment file
echo [5/7] Setting up environment configuration...
if not exist frontend\.env.local (
    echo Creating frontend/.env.local...
    (
        echo NEXT_PUBLIC_API_URL=http://localhost:8080
        echo NEXT_PUBLIC_WEBSOCKET_URL=ws://localhost:8080
        echo NEXT_PUBLIC_GRAPHQL_URL=http://localhost:4000/graphql
        echo NEXT_PUBLIC_ENABLE_TELEMETRY=false
        echo NEXT_PUBLIC_DATA_MODE=%USE_REAL_DATA%
    ) > frontend\.env.local
)

if not exist core\.env (
    echo Creating core/.env...
    (
        echo RUST_LOG=info,policycortex_core=debug
        echo PORT=8080
        echo DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex
        echo REDIS_URL=redis://localhost:6379
        echo USE_REAL_DATA=%USE_REAL_DATA%
        echo AZURE_SUBSCRIPTION_ID=%AZURE_SUBSCRIPTION_ID%
        echo AZURE_TENANT_ID=%AZURE_TENANT_ID%
        echo AZURE_CLIENT_ID=%AZURE_CLIENT_ID%
    ) > core\.env
)
echo Environment configuration complete.
echo.

REM Database setup
echo [6/7] Setting up database...
docker ps >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Starting PostgreSQL with Docker...
    docker run -d --name policycortex-postgres ^
        -e POSTGRES_USER=postgres ^
        -e POSTGRES_PASSWORD=postgres ^
        -e POSTGRES_DB=policycortex ^
        -p 5432:5432 ^
        postgres:15 >nul 2>nul
    
    echo Starting Redis with Docker...
    docker run -d --name policycortex-redis ^
        -p 6379:6379 ^
        redis:7-alpine >nul 2>nul
    
    echo Database services started.
) else (
    echo WARNING: Docker not running, skipping database setup
    echo You'll need to manually set up PostgreSQL and Redis
)
echo.

REM Preflight checks
echo [7/7] Running preflight checks...
echo.
echo System Information:
echo -------------------
node --version | findstr /r "^v" && echo Node.js: OK || echo Node.js: ERROR
npm --version | findstr /r "^[0-9]" && echo npm: OK || echo npm: ERROR
cargo --version | findstr /r "^cargo" && echo Rust: OK || echo Rust: ERROR
echo.

echo Service Status:
echo ---------------
curl -s http://localhost:8080/health >nul 2>nul && echo Backend API: RUNNING || echo Backend API: NOT RUNNING
curl -s http://localhost:3000 >nul 2>nul && echo Frontend: RUNNING || echo Frontend: NOT RUNNING
curl -s http://localhost:5432 >nul 2>nul && echo PostgreSQL: RUNNING || echo PostgreSQL: NOT RUNNING
curl -s http://localhost:6379 >nul 2>nul && echo Redis: RUNNING || echo Redis: NOT RUNNING
echo.

echo ================================================================
echo Bootstrap Complete!
echo ================================================================
echo.
echo Next steps:
echo 1. Start the backend:  cd core && cargo run
echo 2. Start the frontend: cd frontend && npm run dev
echo 3. Open browser:       http://localhost:3000
echo.
echo Data Mode: %USE_REAL_DATA%
if "%USE_REAL_DATA%"=="false" (
    echo Note: Running in SIMULATED mode. Run 'az login' for real Azure data.
)
echo.
echo For production deployment, see docs/deployment.md
echo ================================================================