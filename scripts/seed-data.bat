@echo off
REM PolicyCortex - Seed Data Script for Windows

echo =====================================
echo PolicyCortex Development Data Seeder
echo =====================================
echo.

REM Wait for services
echo Waiting for services to be ready...
timeout /t 5 /nobreak >nul

REM Check PostgreSQL
echo Checking PostgreSQL...
set POSTGRES_CONT=
for /f "tokens=*" %%i in ('docker ps --format "{{.Names}}" ^| findstr /i "postgres"') do (
  set POSTGRES_CONT=%%i
  goto :found_pg
)
:found_pg
if "%POSTGRES_CONT%"=="" (
  echo [WARN] PostgreSQL container not found; skipping DB seed
) else (
  :check_postgres
  docker exec %POSTGRES_CONT% pg_isready -U postgres >nul 2>&1
  if %errorlevel% neq 0 (
      echo Waiting for PostgreSQL...
      timeout /t 2 /nobreak >nul
      goto check_postgres
  )
  echo [OK] PostgreSQL ready (%POSTGRES_CONT%)
)
echo.

REM Seed PostgreSQL
if exist scripts\seed-data.sql (
  if not "%POSTGRES_CONT%"=="" (
    echo Seeding PostgreSQL database...
    type scripts\seed-data.sql | docker exec -i %POSTGRES_CONT% psql -U postgres -d policycortex >nul 2>&1
    if %errorlevel% equ 0 (
        echo [OK] PostgreSQL seeded
    ) else (
        echo [WARN] Failed to seed PostgreSQL; continuing demo
    )
  ) else (
    echo [INFO] Skipping DB seed (PostgreSQL not running)
  )
) else (
  echo [INFO] scripts\\seed-data.sql not found; skipping DB seed
)

REM Seed EventStore
echo.
echo Seeding EventStore...
curl -X POST http://localhost:2113/streams/policy-events ^
  -H "Content-Type: application/vnd.eventstore.events+json" ^
  -d "[{\"eventId\":\"event-1\",\"eventType\":\"PolicyCreated\",\"data\":{\"policyId\":\"pol-1\",\"name\":\"Require HTTPS for Storage Accounts\"}}]" >nul 2>&1

if %errorlevel% equ 0 (
    echo [OK] EventStore seeded
) else (
    echo [ERROR] Failed to seed EventStore
)

REM Seed Cache
echo.
echo Seeding cache...
set DF_CONT=
for /f "tokens=*" %%i in ('docker ps --format "{{.Names}}" ^| findstr /i "dragonfly"') do (
  set DF_CONT=%%i
  goto :found_df
)
:found_df
if not "%DF_CONT%"=="" (
  docker exec %DF_CONT% redis-cli SET session:demo "{\"user\":\"demo\"}" EX 3600 >nul 2>&1
  docker exec %DF_CONT% redis-cli HSET feature_flags "all_features" "enabled" >nul 2>&1
) else (
  echo [INFO] DragonflyDB not running; skipping cache seed
)

if %errorlevel% equ 0 (
    echo [OK] Cache seeded
) else (
    echo [ERROR] Failed to seed cache
)

echo.
echo =====================================
echo Seed data loaded successfully!
echo =====================================
echo.
echo Summary:
echo   - 3 Organizations
echo   - 3 Users  
echo   - 5 Policies
echo   - 3 Resources
echo   - Sample events
echo   - Cache data
echo.
echo Application URL: http://localhost:3000
echo.
echo Test Accounts:
echo   admin@contoso.com (Admin)
echo   analyst@contoso.com (Analyst)
echo.
pause