@echo off
echo Starting PolicyCortex API Gateway...

REM Set up environment
call setup-env.bat

REM Navigate to API Gateway
cd backend\services\api_gateway

echo.
echo Running API Gateway on http://localhost:8000
echo Press Ctrl+C to stop
echo.

REM Run the service
..\..\venv\Scripts\python.exe -m uvicorn main:app --port 8000 --host 0.0.0.0