@echo off
echo Opening PolicyCortex in your browser...
echo =====================================
echo.
echo Frontend: http://localhost:3000
echo Backend API: http://localhost:8012
echo API Docs: http://localhost:8012/docs
echo.
start http://localhost:3000
echo.
echo If the page doesn't load:
echo 1. Make sure both services are running
echo 2. Check that ports 3000 and 8012 are not blocked
echo 3. Try refreshing the page
echo.
pause