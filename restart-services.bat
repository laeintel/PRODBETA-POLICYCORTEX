@echo off
REM Script to restart all PolicyCortex services
echo =========================================
echo PolicyCortex Services Restart Script
echo =========================================
echo.

REM Kill existing processes
echo [1/2] Stopping all services...
echo.

REM Stop Node.js processes (frontend, GraphQL)
echo Stopping Node.js services...
taskkill /F /IM node.exe 2>nul
if %errorlevel% equ 0 (
    echo - Node.js processes stopped
) else (
    echo - No Node.js processes running
)

REM Stop Rust backend
echo Stopping Rust backend...
taskkill /F /IM policycortex-core.exe 2>nul
taskkill /F /IM cargo.exe 2>nul
if %errorlevel% equ 0 (
    echo - Rust backend stopped
) else (
    echo - No Rust backend running
)

REM Stop Python services
echo Stopping Python services...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM uvicorn.exe 2>nul
if %errorlevel% equ 0 (
    echo - Python services stopped
) else (
    echo - No Python services running
)

REM Wait for processes to fully terminate
echo.
echo Waiting for processes to terminate...
timeout /t 3 /nobreak >nul

echo.
echo [2/2] Starting all services...
echo.

REM Set environment variables
echo Setting environment variables...
set NODE_ENV=development
set RUST_LOG=info
set USE_REAL_DATA=true
set AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c

REM Start Core API (Rust)
echo Starting Core API (Rust)...
cd /d "%~dp0core"
start "PolicyCortex Core API" cmd /k "cargo run --release"

REM Wait for core to initialize
timeout /t 5 /nobreak >nul

REM Start GraphQL Gateway
echo Starting GraphQL Gateway...
cd /d "%~dp0graphql"
if exist node_modules (
    start "PolicyCortex GraphQL" cmd /k "npm run dev"
) else (
    echo Installing GraphQL dependencies first...
    call npm install
    start "PolicyCortex GraphQL" cmd /k "npm run dev"
)

REM Start API Gateway (Python)
echo Starting API Gateway (Python)...
cd /d "%~dp0backend\services\api_gateway"
start "PolicyCortex API Gateway" cmd /k "uvicorn main:app --reload --port 8000"

REM Start AI Engine (Python)
echo Starting AI Engine...
cd /d "%~dp0backend\services\ai_engine"
start "PolicyCortex AI Engine" cmd /k "python app.py"

REM Wait for backend services to be ready
timeout /t 5 /nobreak >nul

REM Start Frontend (Next.js)
echo Starting Frontend (Next.js)...
cd /d "%~dp0frontend"
if exist node_modules (
    start "PolicyCortex Frontend" cmd /k "npm run dev"
) else (
    echo Installing frontend dependencies first...
    call npm install
    start "PolicyCortex Frontend" cmd /k "npm run dev"
)

echo.
echo =========================================
echo All services are starting...
echo =========================================
echo.
echo Service URLs:
echo - Frontend:     http://localhost:3000
echo - Core API:     http://localhost:8080
echo - GraphQL:      http://localhost:4000/graphql
echo - API Gateway:  http://localhost:8000
echo - AI Engine:    http://localhost:8001
echo.
echo Check the individual command windows for service status.
echo Press any key to open the frontend in your browser...
pause >nul

REM Open frontend in default browser
start http://localhost:3000

cd /d "%~dp0"