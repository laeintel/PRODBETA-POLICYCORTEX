@echo off
echo Starting minimal PolicyCortex services...
echo.

echo Stopping any existing services...
docker-compose -f docker-minimal.yml down 2>nul

echo Starting PostgreSQL and Redis...
docker-compose -f docker-minimal.yml up -d

echo.
echo Waiting for services to be ready...
timeout /t 10 /nobreak >nul

echo.
echo Services status:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo Services are available at:
echo - PostgreSQL: localhost:5432 (user: postgres, pass: postgres)
echo - Redis: localhost:6379
echo.
echo To stop: docker-compose -f docker-minimal.yml down
echo.
pause