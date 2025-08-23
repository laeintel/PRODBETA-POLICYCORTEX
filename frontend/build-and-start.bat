@echo off
echo ========================================
echo PolicyCortex Frontend Build and Start
echo ========================================
echo.

REM Kill any process using port 3000
echo [1/4] Killing any processes on port 3000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000 ^| findstr LISTENING') do (
    echo Killing process with PID: %%a
    taskkill /PID %%a /F >nul 2>&1
)

REM Also kill any node.exe processes that might be hanging
echo Cleaning up any hanging Node processes...
taskkill /IM node.exe /F >nul 2>&1

REM Wait a moment for ports to be released
timeout /t 2 /nobreak >nul

echo [2/4] Port 3000 is now free
echo.

REM Build the production version
echo [3/4] Building production version...
echo This may take a few minutes...
call npm run build
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo [4/4] Starting production server on port 3000...
echo.
echo ========================================
echo Server is starting at http://localhost:3000
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the production server
npm run start