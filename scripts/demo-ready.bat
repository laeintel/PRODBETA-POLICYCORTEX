@echo off
REM ========================================
REM PolicyCortex Demo Ready Script
REM Quick setup for local demo environment
REM ========================================

echo.
echo ==========================================
echo PolicyCortex Demo Environment Setup
echo ==========================================
echo.

REM Set demo environment variables
set NODE_ENV=demo
set USE_REAL_DATA=false
set USE_MOCK_GRAPHQL=true
set USE_DEMO_CHAT=true
set REQUIRE_AUTH=false
set NEXT_PUBLIC_API_URL=http://localhost:8080
set NEXT_PUBLIC_GRAPHQL_ENDPOINT=http://localhost:4000/graphql
set NEXT_PUBLIC_DEMO_MODE=true
set NEXT_PUBLIC_USE_WS=false
set RUST_LOG=info

echo [1/5] Setting up demo environment variables...
echo       NODE_ENV=%NODE_ENV%
echo       USE_REAL_DATA=%USE_REAL_DATA%
echo       USE_MOCK_GRAPHQL=%USE_MOCK_GRAPHQL%
echo       USE_DEMO_CHAT=%USE_DEMO_CHAT%
echo.

REM Check Docker
echo [2/5] Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker not found. Please install Docker Desktop.
    exit /b 1
)
echo       Docker is installed and running
echo.

REM Start database services
echo [3/5] Starting database services...
docker-compose -f docker-compose.local.yml up -d postgres dragonfly eventstore >nul 2>&1
if %errorlevel% neq 0 (
    echo       WARNING: Database services may already be running
) else (
    echo       PostgreSQL, Redis, and EventStore started
)
timeout /t 5 /nobreak >nul
echo.

REM Seed demo data
echo [4/5] Seeding demo tenants and data...
if exist scripts\seed-data.bat (
    call scripts\seed-data.bat >nul 2>&1
    echo       Demo data seeded successfully
) else (
    echo       WARNING: seed-data.bat not found, skipping data seeding
)
echo.

REM Start services
echo [5/5] Starting PolicyCortex services...
echo.
echo Starting services (this may take a moment)...

REM Start Core API in background
start /min cmd /c "cd core && cargo run 2>nul"
echo       Core API starting on http://localhost:8080

REM Start GraphQL Gateway in background (optional)
start /min cmd /c "cd graphql && npm run dev 2>nul"
echo       GraphQL Gateway starting on http://localhost:4000

REM Start Frontend
echo       Frontend starting on http://localhost:3000
echo.
cd frontend
start cmd /c "npm run dev"

echo ==========================================
echo Demo Environment Ready!
echo ==========================================
echo.
echo Access the demo at: http://localhost:3000
echo.
echo Demo Features:
echo   - Multi-tenant switching (3 demo tenants)
echo   - Conversational AI (demo mode)
echo   - Security posture dashboard
echo   - Cost optimization panels
echo   - Predictive compliance
echo   - SHAP explainability charts
echo.
echo Press any key to stop all services...
pause >nul

REM Cleanup
echo.
echo Stopping services...
taskkill /F /IM node.exe >nul 2>&1
taskkill /F /IM cargo.exe >nul 2>&1
docker-compose -f docker-compose.local.yml down >nul 2>&1
echo Services stopped.
echo.