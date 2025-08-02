@echo off
echo Killing any processes using port 3000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000') do (
    echo Killing process %%a
    taskkill /pid %%a /f >nul 2>&1
)

echo Starting frontend on port 3000...
set PORT=3000
npm run dev