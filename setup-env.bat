@echo off
echo Setting up PolicyCortex environment variables...

REM Core settings
set ENVIRONMENT=development
set DEBUG=true
set SERVICE_NAME=api_gateway
set SERVICE_PORT=8000
set LOG_LEVEL=INFO

REM Azure Mock Settings (for local development)
set AZURE_SUBSCRIPTION_ID=00000000-0000-0000-0000-000000000000
set AZURE_TENANT_ID=00000000-0000-0000-0000-000000000000
set AZURE_CLIENT_ID=00000000-0000-0000-0000-000000000000
set AZURE_CLIENT_SECRET=dummy-secret
set RESOURCE_GROUP=dummy-rg
set KEY_VAULT_NAME=dummy-kv
set STORAGE_ACCOUNT_NAME=dummystorage

REM Service URLs
set API_GATEWAY_URL=http://localhost:8000
set AZURE_INTEGRATION_URL=http://localhost:8001
set AI_ENGINE_URL=http://localhost:8002
set DATA_PROCESSING_URL=http://localhost:8003
set CONVERSATION_URL=http://localhost:8004
set NOTIFICATION_URL=http://localhost:8005

REM Database Configuration
set SQL_SERVER=localhost
set SQL_DATABASE=policortex_dev
set SQL_USERNAME=sa
set SQL_PASSWORD=YourStrong@Passw0rd
set AZURE_SQL_SERVER=localhost
set AZURE_SQL_DATABASE=policortex_dev
set AZURE_SQL_USERNAME=sa
set AZURE_SQL_PASSWORD=YourStrong@Passw0rd

REM Cosmos DB
set COSMOS_ENDPOINT=https://localhost:8081/
set COSMOS_KEY=C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==
set COSMOS_DATABASE=policortex_dev
set AZURE_COSMOS_ENDPOINT=https://localhost:8081/
set AZURE_COSMOS_KEY=C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==
set AZURE_COSMOS_DATABASE=policortex_dev

REM Redis
set REDIS_CONNECTION_STRING=localhost:6379,password=,ssl=False,abortConnect=False
set REDIS_HOST=localhost
set REDIS_PASSWORD=
set REDIS_PORT=6379
set REDIS_SSL=false

REM Security
set JWT_SECRET_KEY=dev-secret-key-change-in-production
set JWT_ALGORITHM=HS256
set JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

REM AI Services
set OPENAI_API_KEY=dummy-openai-key
set AZURE_OPENAI_ENDPOINT=https://dummy.openai.azure.com/
set AZURE_OPENAI_API_KEY=dummy-azure-openai-key
set AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
set COGNITIVE_SERVICES_KEY=dummy-key
set COGNITIVE_SERVICES_ENDPOINT=https://dummy.cognitiveservices.azure.com/

REM Service Bus
set SERVICE_BUS_NAMESPACE=dummy-namespace
set SERVICE_BUS_CONNECTION_STRING=Endpoint=sb://dummy.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=dummy
set AZURE_SERVICE_BUS_CONNECTION_STRING=Endpoint=sb://dummy.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=dummy

REM Storage
set AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=dummy;AccountKey=dummy;EndpointSuffix=core.windows.net

REM Application Insights
set APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=00000000-0000-0000-0000-000000000000;IngestionEndpoint=https://dummy.in.applicationinsights.azure.com/

REM ML Workspace
set ML_WORKSPACE_NAME=dummy-ml-workspace

REM Other settings
set TENANT_ID=00000000-0000-0000-0000-000000000000
set CLIENT_ID=00000000-0000-0000-0000-000000000000
set CLIENT_SECRET=dummy-secret
set SUBSCRIPTION_ID=00000000-0000-0000-0000-000000000000
set USE_LOCAL_STORAGE=true
set ENABLE_TELEMETRY=false
set ENABLE_METRICS=false

echo Environment variables set successfully!
echo.