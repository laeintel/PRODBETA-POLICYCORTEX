@echo off
REM PolicyCortex v2 - Seed Data Script for Windows

echo =====================================
echo PolicyCortex v2 Development Data Seeder
echo =====================================
echo.

REM Wait for services
echo Waiting for services to be ready...
timeout /t 5 /nobreak >nul

REM Check PostgreSQL
echo Checking PostgreSQL...
:check_postgres
docker exec policycortex-v2-postgres-1 pg_isready -U postgres >nul 2>&1
if %errorlevel% neq 0 (
    echo Waiting for PostgreSQL...
    timeout /t 2 /nobreak >nul
    goto check_postgres
)

echo [OK] PostgreSQL ready
echo.

REM Seed PostgreSQL
echo Seeding PostgreSQL database...
docker exec -i policycortex-v2-postgres-1 psql -U postgres -d policycortex < scripts\seed-data.sql
if %errorlevel% equ 0 (
    echo [OK] PostgreSQL seeded
) else (
    echo [ERROR] Failed to seed PostgreSQL
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
docker exec policycortex-v2-dragonfly-1 redis-cli SET session:demo "{\"user\":\"demo\"}" EX 3600 >nul 2>&1
docker exec policycortex-v2-dragonfly-1 redis-cli HSET feature_flags "all_features" "enabled" >nul 2>&1

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