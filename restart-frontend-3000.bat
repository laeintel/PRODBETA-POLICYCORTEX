@echo off
echo Killing processes on port 3000...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":3000" ^| find "LISTENING"') do taskkill /F /PID %%a
timeout /t 2
echo Starting frontend on port 3000...
cd frontend
npm run dev