@echo off
echo Starting PolicyCortex with Azure OpenAI support...

REM Set Azure credentials (already configured)
set AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c

REM Azure OpenAI credentials (Optional - leave empty if not using AI chat)
set AZURE_OPENAI_ENDPOINT=
set AZURE_OPENAI_API_KEY=
set AZURE_OPENAI_REALTIME_DEPLOYMENT=gpt-4
set AZURE_OPENAI_API_VERSION=2024-05-01-preview

REM Optional: Set to true to use real Azure data
set USE_REAL_DATA=true

echo.
echo Environment variables set:
echo - Azure Subscription: %AZURE_SUBSCRIPTION_ID%
echo - Azure OpenAI: %AZURE_OPENAI_ENDPOINT%
echo.

REM Start Docker Compose
docker-compose -f docker-compose.local.yml up --build

pause