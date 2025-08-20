@echo off
REM ==================================================================
REM Quick ML Docker Test - Build and Basic Validation
REM ==================================================================

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
cd /d "%PROJECT_ROOT%"

echo.
echo ========================================
echo Quick ML Docker Test
echo ========================================
echo.

REM ==================================================================
REM 1. Build CPU-only Docker image
REM ==================================================================
echo [INFO] Building ML Docker image (CPU-only)...
docker build -f Dockerfile.ml-cpu -t policycortex-ml-cpu:test --progress=plain .
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to build Docker image
    exit /b 1
)
echo [SUCCESS] Docker image built successfully
echo.

REM ==================================================================
REM 2. Test container startup
REM ==================================================================
echo [INFO] Testing container startup...
docker run --rm -d --name ml-test policycortex-ml-cpu:test sleep 30
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to start container
    exit /b 1
)
echo [SUCCESS] Container started successfully
echo.

REM ==================================================================
REM 3. Check Python and packages
REM ==================================================================
echo [INFO] Checking Python installation...
docker exec ml-test python3 --version
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in container
    docker stop ml-test >nul 2>&1
    exit /b 1
)

echo.
echo [INFO] Checking key packages...
docker exec ml-test python3 -c "import torch; import sklearn; import fastapi; import websockets; print('Core packages imported successfully')"
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some packages may not be installed correctly
) else (
    echo [SUCCESS] Core packages are available
)

REM ==================================================================
REM 4. Check file structure
REM ==================================================================
echo.
echo [INFO] Checking file structure...
docker exec ml-test ls -la /app/ml_models/ 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] ML models directory may not exist
)

REM ==================================================================
REM 5. Cleanup
REM ==================================================================
echo.
echo [INFO] Cleaning up test container...
docker stop ml-test >nul 2>&1
echo [SUCCESS] Test container removed
echo.

echo ========================================
echo Quick test completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Run full test: scripts\test-ml-docker.bat
echo 2. Start services: docker-compose -f docker-compose.ml-windows.yml up
echo.

exit /b 0