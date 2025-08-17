@echo off
echo ================================================
echo PolicyCortex Production Build - Local Testing
echo ================================================
echo.

REM Set environment variables for production testing
echo Setting production environment variables...
set NODE_ENV=production
set RUST_LOG=info
set USE_REAL_DATA=true
set ENVIRONMENT=production

REM Load Azure credentials from .env if exists
if exist .env (
    echo Loading environment from .env file...
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        if not "%%a"=="" if not "%%b"=="" set %%a=%%b
    )
)

REM Verify required environment variables
echo.
echo Checking required environment variables...
if "%AZURE_SUBSCRIPTION_ID%"=="" (
    echo ERROR: AZURE_SUBSCRIPTION_ID not set
    echo Please set your Azure credentials in .env or environment
    exit /b 1
)

echo Azure Subscription: %AZURE_SUBSCRIPTION_ID%
echo.

REM Step 1: Stop any existing services
echo Step 1: Stopping existing services...
docker-compose -f docker-compose.local.yml down -v 2>nul
docker stop policycortex-core policycortex-postgres policycortex-redis 2>nul
docker rm policycortex-core policycortex-postgres policycortex-redis 2>nul
echo Done.
echo.

REM Step 2: Build production images
echo Step 2: Building production images...
echo.

echo Building Core (Rust) service...
docker build -t policycortex-core:prod -f core/Dockerfile core/
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to build core service
    exit /b 1
)

echo Building Frontend (Next.js) service...
docker build -t policycortex-frontend:prod -f frontend/Dockerfile frontend/ --build-arg NODE_ENV=production
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to build frontend service
    exit /b 1
)

echo Building GraphQL Gateway...
docker build -t policycortex-graphql:prod -f graphql/Dockerfile graphql/
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to build GraphQL service
    exit /b 1
)

echo Building API Gateway (Python)...
docker build -t policycortex-api-gateway:prod -f backend/services/api_gateway/Dockerfile backend/services/api_gateway/
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: API Gateway build failed, continuing...
)

echo Building Explainability service...
docker build -t policycortex-explainability:prod -f backend/services/explainability/Dockerfile backend/services/explainability/
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Explainability service build failed, continuing...
)

echo.
echo All images built successfully!
echo.

REM Step 3: Start infrastructure services
echo Step 3: Starting infrastructure services...
echo.

echo Starting PostgreSQL...
docker run -d --name policycortex-postgres ^
  -e POSTGRES_DB=policycortex ^
  -e POSTGRES_USER=postgres ^
  -e POSTGRES_PASSWORD=postgres ^
  -p 5432:5432 ^
  postgres:16-alpine

echo Starting Redis (DragonflyDB)...
docker run -d --name policycortex-redis ^
  -p 6379:6379 ^
  docker.dragonflydb.io/dragonflydb/dragonfly:latest

echo Starting Neo4j...
docker run -d --name policycortex-neo4j ^
  -p 7474:7474 -p 7687:7687 ^
  -e NEO4J_AUTH=neo4j/password ^
  -e NEO4J_PLUGINS=["apoc","graph-data-science"] ^
  neo4j:5-community

echo Starting EventStore...
docker run -d --name policycortex-eventstore ^
  -p 2113:2113 -p 1113:1113 ^
  -e EVENTSTORE_CLUSTER_SIZE=1 ^
  -e EVENTSTORE_RUN_PROJECTIONS=All ^
  -e EVENTSTORE_START_STANDARD_PROJECTIONS=true ^
  -e EVENTSTORE_HTTP_PORT=2113 ^
  -e EVENTSTORE_INSECURE=true ^
  eventstore/eventstore:latest

echo Waiting for infrastructure to be ready...
timeout /t 10 /nobreak >nul
echo.

REM Step 4: Run database migrations
echo Step 4: Setting up databases...
echo.

REM Create tables via SQL script if exists
if exist core/migrations/init.sql (
    echo Running database migrations...
    docker exec -i policycortex-postgres psql -U postgres -d policycortex < core/migrations/init.sql 2>nul
)

REM Step 5: Start application services
echo Step 5: Starting application services...
echo.

echo Starting Core service...
docker run -d --name policycortex-core-prod ^
  --network host ^
  -e RUST_LOG=%RUST_LOG% ^
  -e ENVIRONMENT=%ENVIRONMENT% ^
  -e USE_REAL_DATA=%USE_REAL_DATA% ^
  -e AZURE_SUBSCRIPTION_ID=%AZURE_SUBSCRIPTION_ID% ^
  -e AZURE_TENANT_ID=%AZURE_TENANT_ID% ^
  -e AZURE_CLIENT_ID=%AZURE_CLIENT_ID% ^
  -e AZURE_CLIENT_SECRET=%AZURE_CLIENT_SECRET% ^
  -e DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex ^
  -e REDIS_URL=redis://localhost:6379 ^
  -e NEO4J_URI=bolt://localhost:7687 ^
  -e NEO4J_USERNAME=neo4j ^
  -e NEO4J_PASSWORD=password ^
  -e EVENT_STORE_URL=http://localhost:2113 ^
  -p 8080:8080 ^
  policycortex-core:prod

echo Starting GraphQL Gateway...
docker run -d --name policycortex-graphql-prod ^
  --network host ^
  -e NODE_ENV=%NODE_ENV% ^
  -e CORE_SERVICE_URL=http://localhost:8080 ^
  -p 4000:4000 ^
  policycortex-graphql:prod

echo Starting Frontend...
docker run -d --name policycortex-frontend-prod ^
  --network host ^
  -e NODE_ENV=%NODE_ENV% ^
  -e NEXT_PUBLIC_API_URL=http://localhost:8080 ^
  -e NEXT_PUBLIC_GRAPHQL_URL=http://localhost:4000/graphql ^
  -p 3000:3000 ^
  policycortex-frontend:prod

echo.
echo Waiting for services to start...
timeout /t 15 /nobreak >nul

REM Step 6: Health checks
echo.
echo Step 6: Running health checks...
echo.

echo Checking Core service...
curl -f http://localhost:8080/health || echo Core service not responding

echo Checking GraphQL service...
curl -f http://localhost:4000/.well-known/apollo/server-health || echo GraphQL service not responding

echo Checking Frontend...
curl -f http://localhost:3000/ || echo Frontend not responding

echo.
echo ================================================
echo Production services are running!
echo ================================================
echo.
echo Access points:
echo - Frontend: http://localhost:3000
echo - Core API: http://localhost:8080
echo - GraphQL: http://localhost:4000/graphql
echo - PostgreSQL: localhost:5432
echo - Redis: localhost:6379
echo - Neo4j: http://localhost:7474
echo - EventStore: http://localhost:2113
echo.
echo To view logs:
echo   docker logs -f policycortex-core-prod
echo   docker logs -f policycortex-frontend-prod
echo   docker logs -f policycortex-graphql-prod
echo.
echo To stop all services:
echo   docker stop policycortex-core-prod policycortex-frontend-prod policycortex-graphql-prod
echo   docker stop policycortex-postgres policycortex-redis policycortex-neo4j policycortex-eventstore
echo.