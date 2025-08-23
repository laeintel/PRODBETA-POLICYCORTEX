@echo off
echo ========================================
echo PolicyCortex Navigation Test Suite
echo ========================================
echo.

cd frontend

echo Starting development server...
start /B npm run dev

echo Waiting for server to start...
timeout /t 10 /nobreak >nul

echo.
echo Running navigation tests...
npx playwright test tests/navigation.spec.ts --reporter=html

echo.
echo ========================================
echo Test Results:
echo ========================================
echo Tests completed. HTML report available.
echo.

pause