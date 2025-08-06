@echo off
echo Starting PolicyCortex Authentication Demo
echo ==========================================

echo.
echo Setting up environment variables...
set ENVIRONMENT=development
set SERVICE_NAME=api_gateway_auth_demo
set SERVICE_PORT=8010
set JWT_SECRET_KEY=dev-secret-key-super-secure-demo
set LOG_LEVEL=DEBUG

echo.
echo Demo service will run on: http://localhost:8010
echo Frontend will run on: http://localhost:3000
echo.
echo 🎯 Features enabled:
echo - Zero-configuration authentication (DEMO)
echo - Automatic organization detection (DEMO)
echo - Multi-tier organization support (DEMO)
echo - Azure policy integration (LIVE)
echo - Real-time compliance monitoring (LIVE)

echo.
echo 📧 Demo emails to try:
echo - admin@microsoft.com (Enterprise/Azure AD)
echo - user@google.com (Enterprise/OAuth2)
echo - ceo@amazon.com (Enterprise/SAML)
echo - manager@startup.com (Professional/Internal)
echo - trial@company.org (Trial/Internal)

echo.
echo Press any key to start the authentication demo...
pause >nul

echo.
echo Starting Authentication Demo API Gateway...
start "PolicyCortex Auth Demo" cmd /k "cd /d backend\services\api_gateway && ..\..\venv\Scripts\python.exe main_auth_demo.py"

echo.
echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo Starting Frontend with auth integration...
start "PolicyCortex Frontend" cmd /k "cd /d frontend && set PORT=3000 && set VITE_PORT=3000 && set VITE_API_BASE_URL=http://localhost:8010 && npm run dev"

echo.
echo Both services are starting with authentication demo...
echo.
echo 🎯 Authentication Demo API: http://localhost:8010
echo 🎨 Frontend: http://localhost:3000
echo 📚 API Documentation: http://localhost:8010/docs
echo 🔐 Authentication: DEMO MODE ENABLED
echo.
echo 🧪 New Authentication Endpoints:
echo - POST /api/auth/detect-organization
echo - POST /api/auth/login
echo - POST /api/auth/refresh
echo - POST /api/auth/logout
echo - GET  /api/auth/me
echo.
echo 🔍 Debug Endpoints:
echo - GET  /debug/auth-test
echo - GET  /debug/tokens
echo.
echo 📊 Enhanced Existing Endpoints (now with auth context):
echo - GET  /api/v1/dashboard/overview
echo - GET  /api/v1/azure/resources
echo - GET  /api/v1/status
echo.
echo Press any key to exit (this will NOT stop the services)
pause >nul