@echo off
echo Starting PolicyCortex Frontend...

cd frontend

REM Set environment variables for frontend
set VITE_API_BASE_URL=http://localhost:8000
set VITE_WS_URL=ws://localhost:8000
set VITE_AZURE_CLIENT_ID=dummy
set VITE_AZURE_TENANT_ID=dummy
set VITE_AZURE_REDIRECT_URI=http://localhost:5173

echo.
echo Running Frontend on http://localhost:5173
echo Press Ctrl+C to stop
echo.
npm run dev