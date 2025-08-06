@echo off
echo Starting PolicyCortex with Integrated Authentication System
echo ========================================================

echo.
echo Setting up environment variables...
set ENVIRONMENT=development
set SERVICE_NAME=api_gateway_integrated
set SERVICE_PORT=8010
set JWT_SECRET_KEY=dev-secret-key-change-in-production-super-secure
set LOG_LEVEL=DEBUG

:: Redis configuration for session management
set REDIS_URL=redis://localhost:6379
set REDIS_PASSWORD=
set REDIS_SSL=false

:: Development Azure configuration
set AZURE_CLIENT_ID=your-dev-client-id
set AZURE_CLIENT_SECRET=your-dev-client-secret
set AZURE_TENANT_ID=common
set AZURE_KEY_VAULT_URL=https://dev-keyvault.vault.azure.net
set AZURE_COSMOS_ENDPOINT=https://dev-cosmos.documents.azure.com
set AZURE_COSMOS_KEY=dev-cosmos-key

echo.
echo Integrated service will run on: http://localhost:8010
echo Frontend will run on: http://localhost:3000
echo.
echo Features enabled:
echo - Zero-configuration authentication
echo - Multi-tenant data isolation
echo - Comprehensive audit logging
echo - Azure policy integration
echo - Real-time compliance monitoring

echo.
echo Press any key to start the integrated service...
pause >nul

echo.
echo Installing/updating dependencies...
cd backend\services\api_gateway
..\..\venv\Scripts\pip.exe install -r requirements_integrated.txt

echo.
echo Starting integrated API Gateway with authentication...
start "PolicyCortex Integrated Gateway" cmd /k "..\..\venv\Scripts\python.exe main_integrated.py"

echo.
echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo Starting Frontend development server...
start "PolicyCortex Frontend" cmd /k "cd /d frontend && set PORT=3000 && set VITE_PORT=3000 && set VITE_API_BASE_URL=http://localhost:8010 && npm run dev"

echo.
echo Both services are starting with authentication enabled...
echo.
echo 🎯 API Gateway (Integrated): http://localhost:8010
echo 🎨 Frontend: http://localhost:3000
echo 📚 API Documentation: http://localhost:8010/docs
echo 🔐 Authentication: ENABLED
echo.
echo New Authentication Endpoints:
echo - POST /api/auth/detect-organization
echo - POST /api/auth/login
echo - POST /api/auth/refresh
echo - POST /api/auth/logout
echo - GET  /api/auth/me
echo.
echo Enhanced Existing Endpoints (now with auth):
echo - GET  /api/v1/dashboard/overview
echo - GET  /api/v1/azure/policies
echo - GET  /api/v1/azure/resources
echo.
echo Tenant Management:
echo - GET  /api/tenant/info
echo.
echo Audit Logging:
echo - GET  /api/audit/logs
echo.
echo Press any key to exit (this will NOT stop the services)
pause >nul