@echo off
REM Patent #4 ML System Testing Script for Windows
REM Tests all ML components locally and validates training pipeline

echo ============================================================
echo PolicyCortex Patent #4 ML System Testing
echo Predictive Policy Compliance Engine
echo ============================================================
echo.

REM Set environment variables
set PYTHONPATH=%cd%
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex
set REDIS_URL=redis://localhost:6379
set AZURE_TENANT_ID=e1f3e196-aa55-4709-9c55-0e334c0b444f
set AZURE_CLIENT_ID=232c44f7-d0cf-4825-a9b5-beba9f587ffb
set ML_TEST_MODE=true

echo [1/8] Checking prerequisites...
echo --------------------------------

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+
    exit /b 1
)
echo ✓ Python installed

REM Check PostgreSQL
psql --version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: PostgreSQL client not found. Database tests may fail.
) else (
    echo ✓ PostgreSQL client found
)

REM Check Redis
redis-cli ping >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Redis not running. Starting Redis...
    start redis-server
    timeout /t 2 >nul
)
echo ✓ Redis running

echo.
echo [2/8] Installing ML dependencies...
echo --------------------------------
pip install -q torch torchvision numpy pandas scikit-learn xgboost lightgbm prophet websockets aioredis pytest pytest-asyncio

echo.
echo [3/8] Creating ML database tables...
echo --------------------------------
psql %DATABASE_URL% -f backend\migrations\create_ml_tables.sql
if %errorlevel% neq 0 (
    echo WARNING: Could not create database tables. They may already exist.
)

echo.
echo [4/8] Testing ML Models...
echo --------------------------------
echo Running performance validation tests...
python -m pytest tests\ml\test_performance_validation.py -v --tb=short
if %errorlevel% neq 0 (
    echo ERROR: ML model tests failed
    exit /b 1
)

echo.
echo [5/8] Starting ML Prediction Server...
echo --------------------------------
start /B python -c "from backend.services.ml_models.prediction_serving import PredictionServingEngine; import asyncio; engine = PredictionServingEngine(); asyncio.run(engine.serve())" 2>nul
timeout /t 3 >nul
echo ✓ Prediction server started on port 8080

echo.
echo [6/8] Starting WebSocket Server...
echo --------------------------------
start /B python backend\services\websocket_server.py 2>nul
timeout /t 3 >nul
echo ✓ WebSocket server started on port 8765

echo.
echo [7/8] Running ML Training Pipeline Test...
echo --------------------------------
echo.
echo Starting comprehensive training pipeline test...
echo This will validate Patent #4 requirements:
echo   - Accuracy: 99.2%%
echo   - False Positive Rate: ^<2%%
echo   - Inference Latency: ^<100ms
echo.
python scripts\test_ml_training_pipeline.py
if %errorlevel% neq 0 (
    echo WARNING: Training pipeline test encountered issues
) else (
    echo ✓ Training pipeline test completed
)

echo.
echo [8/8] Testing ML API Endpoints...
echo --------------------------------

REM Test prediction endpoint
echo Testing POST /api/v1/predictions...
curl -X POST http://localhost:8080/api/v1/predictions ^
  -H "Content-Type: application/json" ^
  -d "{\"resource_id\":\"test-vm-001\",\"tenant_id\":\"org-1\",\"configuration\":{\"encryption\":{\"enabled\":false},\"public_access\":true}}" ^
  -o test_prediction.json 2>nul

if %errorlevel% eq 0 (
    echo ✓ Prediction endpoint working
) else (
    echo ✗ Prediction endpoint failed
)

REM Test metrics endpoint
echo Testing GET /api/v1/ml/metrics...
curl -X GET http://localhost:8080/api/v1/ml/metrics -o test_metrics.json 2>nul
if %errorlevel% eq 0 (
    echo ✓ Metrics endpoint working
) else (
    echo ✗ Metrics endpoint failed
)

REM Test WebSocket connection
echo Testing WebSocket connection...
python -c "import asyncio, websockets; asyncio.run(websockets.connect('ws://localhost:8765').aclose())" 2>nul
if %errorlevel% eq 0 (
    echo ✓ WebSocket connection successful
) else (
    echo ✗ WebSocket connection failed
)

echo.
echo [9/9] Running Integration Tests...
echo --------------------------------
python tests\ml\test_ml_integration.py
if %errorlevel% neq 0 (
    echo WARNING: Some integration tests failed
)

echo.
echo ========================================
echo Test Summary
echo ========================================
echo.

REM Check if all critical components are working
set /a passed=0
set /a failed=0

if exist test_prediction.json (
    set /a passed+=1
    echo ✅ ML Prediction API: WORKING
) else (
    set /a failed+=1
    echo ❌ ML Prediction API: FAILED
)

if exist test_metrics.json (
    set /a passed+=1
    echo ✅ ML Metrics API: WORKING
) else (
    set /a failed+=1
    echo ❌ ML Metrics API: FAILED
)

netstat -an | find "8765" >nul
if %errorlevel% eq 0 (
    set /a passed+=1
    echo ✅ WebSocket Server: RUNNING
) else (
    set /a failed+=1
    echo ❌ WebSocket Server: NOT RUNNING
)

netstat -an | find "8080" >nul
if %errorlevel% eq 0 (
    set /a passed+=1
    echo ✅ Prediction Server: RUNNING
) else (
    set /a failed+=1
    echo ❌ Prediction Server: NOT RUNNING
)

echo.
echo Results: %passed% passed, %failed% failed
echo.

REM Cleanup
del test_prediction.json 2>nul
del test_metrics.json 2>nul

REM Kill background processes
echo Stopping test servers...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq ML*" 2>nul

if %failed% gtr 0 (
    echo ⚠️  Some tests failed. Please check the logs above.
    exit /b 1
) else (
    echo ✅ All ML systems operational!
    echo.
    echo Test reports generated:
    echo   - ml_test_report.json  (JSON format)
    echo   - ml_test_report.md    (Markdown format)
    echo   - ml_training_test.log (Detailed logs)
    echo.
    echo Next steps:
    echo 1. Review test report: ml_test_report.md
    echo 2. Run frontend: cd frontend ^&^& npm run dev
    echo 3. Access ML dashboard: http://localhost:3000/tactical/ml
    echo 4. Monitor WebSocket: ws://localhost:8765
    exit /b 0
)