@echo off
REM Set environment variables for PolicyCortex

REM Azure Configuration
set AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c

REM Enable real Azure data
set USE_REAL_AZURE=true
set USE_REAL_DATA=true

REM Database
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex

REM Redis
set REDIS_URL=redis://localhost:6379

REM API Keys (if needed)
set JWT_SECRET=your-secret-key-here

echo Environment variables set successfully!
echo.
echo AZURE_SUBSCRIPTION_ID=%AZURE_SUBSCRIPTION_ID%
echo AZURE_TENANT_ID=%AZURE_TENANT_ID%
echo USE_REAL_AZURE=%USE_REAL_AZURE%
echo USE_REAL_DATA=%USE_REAL_DATA%
echo.
echo Now run your application with these environment variables set.