@echo off
echo Starting PolicyCortex Local Development Environment...
echo =====================================================

REM Start Backend API Gateway on port 8012
echo Starting Backend API Gateway on port 8012...
cd backend\services\api_gateway
start "PolicyCortex API Gateway" cmd /k "..\..\venv\Scripts\python.exe main_simple.py"
cd ..\..\..

REM Wait a moment for backend to start
timeout /t 3 /nobreak > nul

REM Start Frontend on port 3000
echo Starting Frontend on port 3000...
cd frontend
start "PolicyCortex Frontend" cmd /k "npm run dev"
cd ..

echo =====================================================
echo Local environment started!
echo.
echo Backend API Gateway: http://localhost:8012
echo Frontend Application: http://localhost:3000
echo.
echo Azure Policy Discovery: Automatic with 5-minute refresh
echo Policy Initiatives: All 4 initiatives will be detected
echo.
echo Press any key to exit...
pause > nul