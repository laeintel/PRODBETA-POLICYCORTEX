@echo off
REM Test ML API Endpoints for PolicyCortex Patent #4
REM This script tests all ML endpoints with performance validation

echo ========================================
echo PolicyCortex ML API Endpoint Test Suite
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    exit /b 1
)

REM Check if requests library is installed
python -c "import requests" >nul 2>&1
if errorlevel 1 (
    echo Installing required Python packages...
    pip install requests
)

REM Run the test script
echo Starting ML endpoint tests...
echo.
python "%~dp0test-ml-endpoints.py"

REM Capture exit code
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% EQU 0 (
    echo ========================================
    echo All tests completed successfully!
    echo ========================================
) else (
    echo ========================================
    echo Some tests failed. Please review output.
    echo ========================================
)

exit /b %EXIT_CODE%