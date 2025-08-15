@echo off
echo ====================================
echo Testing PolicyCortex Local Services
echo ====================================
echo.

echo Testing Frontend...
curl -s -o nul -w "Frontend (http://localhost:3000): %%{http_code}\n" http://localhost:3000

echo.
echo Testing Core API (if running)...
curl -s -o nul -w "Core API (http://localhost:8080/health): %%{http_code}\n" http://localhost:8080/health 2>nul || echo Core API: Not running

echo.
echo Testing GraphQL (if running)...
curl -s -o nul -w "GraphQL (http://localhost:4000): %%{http_code}\n" http://localhost:4000 2>nul || echo GraphQL: Not running

echo.
echo Testing Database Connection...
docker exec policycortex-postgres psql -U postgres -d policycortex -c "SELECT 1" >nul 2>&1 && echo PostgreSQL: Connected || echo PostgreSQL: Failed

echo.
echo Testing Redis Connection...
docker exec policycortex-redis redis-cli ping >nul 2>&1 && echo Redis: Connected || echo Redis: Failed

echo.
echo ====================================
echo Test Complete!
echo ====================================