@echo off
echo Starting PolicyCortex Services...

echo.
echo Starting API Gateway on port 8012...
start cmd /k "cd backend\services\api_gateway && start-api-8012.bat"

echo.
echo Waiting for API to start...
timeout /t 5 /nobreak > nul

echo.
echo Starting Frontend on port 3000...
start cmd /k "cd frontend && set PORT=3000 && npm run dev"

echo.
echo Services started successfully!
echo API Gateway: http://localhost:8012
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit...
pause > nul