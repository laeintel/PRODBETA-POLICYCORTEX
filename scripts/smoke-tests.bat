@echo off
REM ========================================
REM PolicyCortex Smoke Tests for Windows
REM Quick validation for CI/CD pipeline
REM ========================================

echo =========================================
echo PolicyCortex Smoke Tests
echo =========================================
echo.

set TESTS_PASSED=0
set TESTS_FAILED=0

REM Wait for services
echo Waiting for services to start...
timeout /t 10 /nobreak >nul

echo.
echo Running smoke tests...
echo -----------------------------------------

REM Test 1: Core API Health
echo Testing Core API Health...
curl -f -s http://localhost:8080/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] Core API Health
    set /a TESTS_PASSED+=1
) else (
    echo   [FAIL] Core API Health
    set /a TESTS_FAILED+=1
)

REM Test 2: Core API Metrics
echo Testing Core API Metrics...
curl -f -s http://localhost:8080/api/v1/metrics | findstr "metrics" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] Core API Metrics
    set /a TESTS_PASSED+=1
) else (
    echo   [FAIL] Core API Metrics
    set /a TESTS_FAILED+=1
)

REM Test 3: Frontend Root
echo Testing Frontend Root...
curl -f -s http://localhost:3000 | findstr "PolicyCortex" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] Frontend Root
    set /a TESTS_PASSED+=1
) else (
    echo   [FAIL] Frontend Root
    set /a TESTS_FAILED+=1
)

REM Test 4: Frontend Dashboard
echo Testing Frontend Dashboard...
curl -f -s http://localhost:3000/dashboard >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] Frontend Dashboard
    set /a TESTS_PASSED+=1
) else (
    echo   [FAIL] Frontend Dashboard
    set /a TESTS_FAILED+=1
)

REM Test 5: GraphQL Health
echo Testing GraphQL Health...
curl -f -s http://localhost:4000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] GraphQL Health
    set /a TESTS_PASSED+=1
) else (
    echo   [FAIL] GraphQL Health
    set /a TESTS_FAILED+=1
)

REM Test 6: Conversation API
echo Testing Conversation API...
curl -f -s http://localhost:8080/api/v1/conversation >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] Conversation API
    set /a TESTS_PASSED+=1
) else (
    echo   [FAIL] Conversation API
    set /a TESTS_FAILED+=1
)

REM Test 7: Predictions API
echo Testing Predictions API...
curl -f -s http://localhost:8080/api/v1/predictions >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] Predictions API
    set /a TESTS_PASSED+=1
) else (
    echo   [FAIL] Predictions API
    set /a TESTS_FAILED+=1
)

REM Test 8: Knowledge Graph API
echo Testing Knowledge Graph API...
curl -f -s http://localhost:3000/api/v1/graph | findstr "nodes" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] Knowledge Graph API
    set /a TESTS_PASSED+=1
) else (
    echo   [FAIL] Knowledge Graph API
    set /a TESTS_FAILED+=1
)

REM Test 9: Database connectivity
echo Testing PostgreSQL connectivity...
docker exec policycortex-postgres pg_isready -U postgres >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] PostgreSQL connectivity
    set /a TESTS_PASSED+=1
) else (
    echo   [FAIL] PostgreSQL connectivity
    set /a TESTS_FAILED+=1
)

echo.
echo -----------------------------------------
echo Smoke Test Results:
echo   Passed: %TESTS_PASSED%
echo   Failed: %TESTS_FAILED%
echo -----------------------------------------

if %TESTS_FAILED% gtr 0 (
    echo Smoke tests FAILED
    exit /b 1
) else (
    echo All smoke tests PASSED
    exit /b 0
)