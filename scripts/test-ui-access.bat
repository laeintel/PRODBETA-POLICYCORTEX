@echo off
echo Testing PolicyCortex UI Access...
echo ================================

echo.
echo Testing login page (http://localhost:3000)...
curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:3000

echo.
echo Testing tactical dashboard (http://localhost:3000/tactical)...
curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:3000/tactical

echo.
echo Testing AI chat interface (http://localhost:3000/chat)...
curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:3000/chat

echo.
echo Testing correlations page (http://localhost:3000/correlations)...
curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:3000/correlations

echo.
echo Testing API endpoints...
echo - Core API (http://localhost:8080/health)...
curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:8080/health

echo - GraphQL (http://localhost:4000/graphql)...
curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:4000/graphql

echo - ML Server (http://localhost:8081/health)...
curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:8081/health

echo.
echo ================================
echo All services tested!
echo.
echo To access the tactical dashboard:
echo 1. Open http://localhost:3000 in your browser
echo 2. Click the "Guest" button OR wait for auto-redirect
echo 3. You'll be taken to the tactical dashboard at /tactical
echo.
echo Demo mode is ENABLED - no Azure AD authentication required!