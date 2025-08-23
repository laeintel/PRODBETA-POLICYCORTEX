@echo off
echo ========================================
echo Testing PolicyCortex Live Azure Data
echo ========================================
echo.

set BASE_URL=http://localhost:8080

echo Testing Azure Health Check...
curl -s %BASE_URL%/api/v1/health/azure | python -m json.tool
echo.

echo ----------------------------------------
echo Testing Dashboard Metrics (Live)...
curl -s %BASE_URL%/api/v1/dashboard/metrics | python -m json.tool | head -20
echo.

echo ----------------------------------------
echo Testing Governance Compliance (Live)...
curl -s %BASE_URL%/api/v1/governance/compliance/status | python -m json.tool | head -20
echo.

echo ----------------------------------------
echo Testing Security IAM (Live)...
curl -s %BASE_URL%/api/v1/security/iam/users | python -m json.tool | head -20
echo.

echo ----------------------------------------
echo Testing Operations Resources (Live)...
curl -s %BASE_URL%/api/v1/operations/resources | python -m json.tool | head -20
echo.

echo ----------------------------------------
echo Testing DevOps Deployments (Live)...
curl -s %BASE_URL%/api/v1/devops/deployments | python -m json.tool | head -20
echo.

echo ========================================
echo Live Data Test Complete!
echo ========================================
pause