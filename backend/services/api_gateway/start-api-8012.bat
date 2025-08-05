@echo off
SET SERVICE_PORT=8012
SET ENVIRONMENT=development
SET SERVICE_NAME=api_gateway
SET JWT_SECRET_KEY=dev-secret-key-change-in-production
SET LOG_LEVEL=debug

echo Starting API Gateway on port %SERVICE_PORT%...
echo.
echo Environment Variables:
echo - SERVICE_PORT: %SERVICE_PORT%
echo - ENVIRONMENT: %ENVIRONMENT%
echo - SERVICE_NAME: %SERVICE_NAME%
echo - LOG_LEVEL: %LOG_LEVEL%
echo.
echo Starting server...
python main_simple.py