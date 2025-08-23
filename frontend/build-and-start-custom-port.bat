@echo off
echo ========================================
echo PolicyCortex Frontend Build and Start
echo ========================================
echo.

REM Ask for port number
set /p PORT="Enter port number (default 3000): "
if "%PORT%"=="" set PORT=3000

REM Kill any process using the specified port
echo [1/4] Killing any processes on port %PORT%...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do (
    echo Killing process with PID: %%a
    taskkill /PID %%a /F >nul 2>&1
)

REM Wait a moment for ports to be released
timeout /t 2 /nobreak >nul

echo [2/4] Port %PORT% is now free
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
echo [4/4] Starting production server on port %PORT%...
echo.
echo ========================================
echo Server is starting at http://localhost:%PORT%
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the production server on custom port
npx next start -p %PORT%