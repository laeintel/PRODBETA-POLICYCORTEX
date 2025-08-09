@echo off
REM Script to stop all PolicyCortex services
echo =========================================
echo Stopping PolicyCortex Services
echo =========================================
echo.

REM Stop Node.js processes (frontend, GraphQL)
echo Stopping Node.js services...
taskkill /F /IM node.exe 2>nul
if %errorlevel% equ 0 (
    echo - Node.js processes stopped
) else (
    echo - No Node.js processes were running
)

REM Stop Rust backend
echo Stopping Rust backend...
taskkill /F /IM policycortex-core.exe 2>nul
taskkill /F /IM cargo.exe 2>nul
if %errorlevel% equ 0 (
    echo - Rust backend stopped
) else (
    echo - No Rust backend was running
)

REM Stop Python services
echo Stopping Python services...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM uvicorn.exe 2>nul
if %errorlevel% equ 0 (
    echo - Python services stopped
) else (
    echo - No Python services were running
)

REM Stop any Docker containers if running
echo Checking for Docker containers...
docker ps -q --filter "name=policycortex" >nul 2>&1
if %errorlevel% equ 0 (
    echo Stopping Docker containers...
    docker-compose down 2>nul
    echo - Docker containers stopped
) else (
    echo - No Docker containers running
)

echo.
echo =========================================
echo All PolicyCortex services have been stopped
echo =========================================
echo.
pause