@echo off
echo Starting PolicyCortex Local Development Environment
echo ==================================================

echo.
echo Setting up environment variables...
set ENVIRONMENT=development
set SERVICE_NAME=api_gateway
set SERVICE_PORT=8000
set JWT_SECRET_KEY=dev-secret-key-change-in-production
set LOG_LEVEL=DEBUG

echo Backend will run on: http://localhost:8000
echo Frontend will run on: http://localhost:3000

echo.
echo Starting backend services...
echo 1. API Gateway (Port 8000)
echo 2. Frontend (Port 3000)

echo.
echo Press any key to start both services...
pause >nul

echo.
echo Starting API Gateway backend service...
start "PolicyCortex API Gateway" cmd /k "cd /d backend\services\api_gateway && ..\..\venv\Scripts\python.exe main_simple.py"

echo.
echo Waiting 3 seconds for backend to start...
timeout /t 3 /nobreak >nul

echo.
echo Starting Frontend development server...
start "PolicyCortex Frontend" cmd /k "cd /d frontend && set PORT=3000 && set VITE_PORT=3000 && npm run dev"

echo.
echo Both services are starting...
echo - API Gateway: http://localhost:8000
echo - Frontend: http://localhost:3000
echo - API Docs: http://localhost:8000/docs

echo.
echo Press any key to exit (this will NOT stop the services)
pause >nul