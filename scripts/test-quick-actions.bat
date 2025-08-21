@echo off
echo ===============================================
echo Testing Quick Actions Bar Implementation
echo ===============================================
echo.

cd /d "%~dp0\..\frontend"

echo Checking TypeScript compilation...
call npm run type-check
if %errorlevel% neq 0 (
    echo TypeScript compilation failed!
    exit /b 1
)

echo.
echo Starting development server to test Quick Actions Bar...
echo.
echo Quick Actions Bar Features:
echo - Check Compliance Status button
echo - View Cost Savings button
echo - Chat with AI button (opens floating chat)
echo - View Predictions button
echo - Check Active Risks button
echo - View Resources button
echo.
echo Global AI Assistant Features:
echo - Press Cmd/Ctrl + K to open AI chat
echo - Voice activation: "Hey PolicyCortex"
echo - Context-aware suggestions
echo - Natural language commands
echo.
echo Navigate to http://localhost:3000 and test:
echo 1. Quick action buttons in the header
echo 2. Hover tooltips on each button
echo 3. Real-time metrics display
echo 4. AI chat interface (Ctrl+K)
echo 5. Voice commands (click mic icon)
echo.

start http://localhost:3000
call npm run dev

pause