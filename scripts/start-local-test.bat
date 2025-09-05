@echo off
echo ================================================
echo PolicyCortex - Local Development Test
echo ================================================
echo.

REM Stop any existing services
echo Cleaning up existing services...
docker-compose -f docker-compose.local.yml down 2>nul
docker stop policycortex-core policycortex-postgres policycortex-redis 2>nul
docker rm policycortex-core policycortex-postgres policycortex-redis 2>nul
echo.

REM Start infrastructure only
echo Starting infrastructure services...
docker run -d --name policycortex-postgres ^
  -e POSTGRES_DB=policycortex ^
  -e POSTGRES_USER=postgres ^
  -e POSTGRES_PASSWORD=postgres ^
  -p 5432:5432 ^
  postgres:16-alpine

docker run -d --name policycortex-redis ^
  -p 6379:6379 ^
  redis:7-alpine

echo Waiting for services to start...
timeout /t 5 /nobreak >nul
echo.

REM Run the frontend in development mode
echo Starting Frontend (Next.js) in development mode...
cd frontend
start /B cmd /c "npm run dev"
cd ..

REM Run the core service locally with Rust
echo Starting Core service (Rust) locally...
cd core
set RUST_LOG=info
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex
set REDIS_URL=redis://localhost:6379
set USE_REAL_DATA=false
set AZURE_SUBSCRIPTION_ID=6dc7cfa2-0332-4740-98b6-bac9f1a23de9
set AZURE_TENANT_ID=e1f3e196-aa55-4709-9c55-0e334c0b444f
set AZURE_CLIENT_ID=232c44f7-d0cf-4825-a9b5-beba9f587ffb
start /B cmd /c "cargo run"
cd ..

echo.
echo Waiting for services to initialize...
timeout /t 10 /nobreak >nul

echo.
echo ================================================
echo Services should be running at:
echo ================================================
echo Frontend: http://localhost:3000
echo Core API: http://localhost:8080
echo PostgreSQL: localhost:5432
echo Redis: localhost:6379
echo.
echo Press Ctrl+C to stop all services
echo ================================================
pause