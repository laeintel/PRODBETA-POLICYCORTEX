@echo off
echo Restarting PolicyCortex Backend...
echo =================================
echo.
echo Step 1: Killing existing Python processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo Step 2: Starting new backend instance...
cd backend\services\api_gateway
start "PolicyCortex API Gateway" cmd /k "..\..\venv\Scripts\python.exe main_simple.py"
cd ..\..\..

echo.
echo Backend restarted successfully!
echo.
echo Please refresh your browser to see the changes.
echo Resources will now show your real Azure resource groups:
echo - rg-policycortex-shared
echo - rg-policortex001-network-dev
echo - rg-policortex001-app-dev
echo - NetworkWatcherRG
echo - And others from your environment
echo.
pause