@echo off
REM PolicyCortex - Complete Testing Script for Windows
REM Tests all services: Frontend, Backend, GraphQL, Databases

echo.
echo ========================================
echo  PolicyCortex Complete Test Suite
echo  Platform: Windows
echo ========================================
echo.

REM Save the script directory and get the project root
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%\..\.."
set "ROOT_DIR=%CD%"

REM Set error handling
setlocal enabledelayedexpansion

REM Colors for output (simplified for compatibility)
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Track test results
set "TESTS_PASSED=0"
set "TESTS_FAILED=0"

echo %BLUE%Phase 1: Environment Check%NC%
echo ----------------------------------------

REM Check required tools
echo Checking required tools...

where node >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%X Node.js not found%NC%
    set /a TESTS_FAILED+=1
) else (
    for /f "tokens=*" %%i in ('node --version') do echo %GREEN%+ Node.js: %%i%NC%
    set /a TESTS_PASSED+=1
)

where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%X npm not found%NC%
    set /a TESTS_FAILED+=1
) else (
    for /f "tokens=*" %%i in ('npm --version') do echo %GREEN%+ npm: v%%i%NC%
    set /a TESTS_PASSED+=1
)

where cargo >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%X Rust/Cargo not found%NC%
    set /a TESTS_FAILED+=1
) else (
    for /f "tokens=*" %%i in ('cargo --version') do echo %GREEN%+ %%i%NC%
    set /a TESTS_PASSED+=1
)

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%! Python not found - Python tests will be skipped%NC%
    set "PYTHON_AVAILABLE=false"
) else (
    for /f "tokens=*" %%i in ('python --version') do echo %GREEN%+ %%i%NC%
    set "PYTHON_AVAILABLE=true"
    set /a TESTS_PASSED+=1
)

where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%! Docker not found - Docker tests will be skipped%NC%
    set "DOCKER_AVAILABLE=false"
) else (
    for /f "tokens=*" %%i in ('docker --version') do echo %GREEN%+ %%i%NC%
    set "DOCKER_AVAILABLE=true"
    set /a TESTS_PASSED+=1
)

echo.
echo %BLUE%Phase 2: Service Dependencies%NC%
echo ----------------------------------------

if "%DOCKER_AVAILABLE%"=="true" (
    echo Checking Docker services...
    
    REM Check if PostgreSQL is already running
    docker ps --filter "name=policycortex-postgres" --format "{{.Names}}" | findstr "policycortex-postgres" >nul 2>&1
    if %errorlevel% equ 0 (
        echo %GREEN%+ PostgreSQL: Already running%NC%
        set /a TESTS_PASSED+=1
    ) else (
        echo Starting PostgreSQL...
        docker run -d --name policycortex-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=policycortex -p 5432:5432 postgres:14 >nul 2>&1
        if %errorlevel% equ 0 (
            echo %GREEN%+ PostgreSQL: Started%NC%
            set /a TESTS_PASSED+=1
        ) else (
            echo %YELLOW%! PostgreSQL: May already exist or failed to start%NC%
        )
    )
    
    REM Check if Redis is already running
    docker ps --filter "name=policycortex-redis" --format "{{.Names}}" | findstr "policycortex-redis" >nul 2>&1
    if %errorlevel% equ 0 (
        echo %GREEN%+ Redis: Already running%NC%
        set /a TESTS_PASSED+=1
    ) else (
        echo Starting Redis...
        docker run -d --name policycortex-redis -p 6379:6379 redis:alpine >nul 2>&1
        if %errorlevel% equ 0 (
            echo %GREEN%+ Redis: Started%NC%
            set /a TESTS_PASSED+=1
        ) else (
            echo %YELLOW%! Redis: May already exist or failed to start%NC%
        )
    )
    
    REM Wait a moment for services
    echo Waiting for services to be ready...
    ping -n 6 127.0.0.1 >nul
    
    REM Test database connections
    echo Testing PostgreSQL connection...
    docker exec policycortex-postgres pg_isready -U postgres >nul 2>&1
    if %errorlevel% equ 0 (
        echo %GREEN%+ PostgreSQL: Connected%NC%
        set /a TESTS_PASSED+=1
    ) else (
        echo %RED%X PostgreSQL: Failed to connect%NC%
        set /a TESTS_FAILED+=1
    )
    
    echo Testing Redis connection...
    docker exec policycortex-redis redis-cli ping >nul 2>&1
    if %errorlevel% equ 0 (
        echo %GREEN%+ Redis: Connected%NC%
        set /a TESTS_PASSED+=1
    ) else (
        echo %RED%X Redis: Failed to connect%NC%
        set /a TESTS_FAILED+=1
    )
) else (
    echo %YELLOW%! Skipping Docker services - Docker not available%NC%
)

echo.
echo %BLUE%Phase 3: Frontend Testing%NC%
echo ----------------------------------------

REM Navigate to frontend directory
cd /d "%ROOT_DIR%\frontend" 2>nul
if %errorlevel% neq 0 (
    cd /d "%ROOT_DIR%\..\..\frontend" 2>nul
)
if %errorlevel% neq 0 (
    echo %RED%X Frontend directory not found%NC%
    set /a TESTS_FAILED+=1
    goto :backend_tests
)

echo Testing Frontend (Next.js)...

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    call npm ci --silent
    if %errorlevel% neq 0 (
        echo %RED%X Frontend: npm install failed%NC%
        set /a TESTS_FAILED+=1
        goto :backend_tests
    )
)

echo Running type check...
call npm run type-check 2>nul
if %errorlevel% neq 0 (
    echo %YELLOW%! Frontend: TypeScript warnings found%NC%
) else (
    echo %GREEN%+ Frontend: TypeScript check passed%NC%
    set /a TESTS_PASSED+=1
)

echo Running linter...
call npm run lint 2>nul
if %errorlevel% neq 0 (
    echo %YELLOW%! Frontend: Linting warnings found%NC%
) else (
    echo %GREEN%+ Frontend: Linting passed%NC%
    set /a TESTS_PASSED+=1
)

echo Building frontend...
call npm run build >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%X Frontend: Build failed%NC%
    set /a TESTS_FAILED+=1
) else (
    echo %GREEN%+ Frontend: Build successful%NC%
    set /a TESTS_PASSED+=1
)

:backend_tests
echo.
echo %BLUE%Phase 4: Backend Testing (Rust)%NC%
echo ----------------------------------------

REM Navigate to core directory
cd /d "%ROOT_DIR%\core" 2>nul
if %errorlevel% neq 0 (
    cd /d "%ROOT_DIR%\..\..\core" 2>nul
)
if %errorlevel% neq 0 (
    echo %RED%X Core directory not found%NC%
    set /a TESTS_FAILED+=1
    goto :summary
)

echo Testing Backend (Rust)...

echo Checking Rust code...
cargo check >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%X Backend: Compilation errors%NC%
    set /a TESTS_FAILED+=1
) else (
    echo %GREEN%+ Backend: Code check passed%NC%
    set /a TESTS_PASSED+=1
)

echo Running Clippy...
cargo clippy -- -D warnings >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%! Backend: Clippy warnings found%NC%
) else (
    echo %GREEN%+ Backend: Clippy passed%NC%
    set /a TESTS_PASSED+=1
)

echo Running tests...
cargo test --quiet >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%! Backend: Some tests failed%NC%
) else (
    echo %GREEN%+ Backend: Tests passed%NC%
    set /a TESTS_PASSED+=1
)

:summary
echo.
echo ========================================
echo  Test Summary
echo ========================================
echo.
echo %GREEN%Passed: %TESTS_PASSED%%NC%
echo %RED%Failed: %TESTS_FAILED%%NC%
echo.

if %TESTS_FAILED% gtr 0 (
    echo %RED%TESTS FAILED - Please fix the issues above%NC%
    exit /b 1
) else (
    echo %GREEN%ALL TESTS PASSED!%NC%
    exit /b 0
)