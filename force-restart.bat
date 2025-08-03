@echo off
echo Forcefully restarting PolicyCortex Backend...
echo ===========================================

echo Killing all Python processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 3 /nobreak >nul

echo Starting backend on new port...
cd backend\services\api_gateway
set SERVICE_PORT=8013
start "PolicyCortex Backend" cmd /k "..\..\venv\Scripts\python.exe main_simple.py"
cd ..\..\..

echo.
echo ✅ Backend restarted on port 8013
echo.
echo Update your frontend to use: http://localhost:8013
echo Or update the API base URL in your frontend configuration
echo.
echo Press any key to continue...
pause >nul