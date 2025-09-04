@echo off
REM Test script for PolicyCortex PROVE Pillar Evidence Chain

echo Testing PolicyCortex PROVE Pillar - Immutable Evidence Chain
echo ============================================================

REM Base URL
set BASE_URL=http://localhost:8080

REM Test 1: Collect Evidence
echo.
echo 1. Testing Evidence Collection...
curl -X POST "%BASE_URL%/api/v1/evidence/collect" ^
  -H "Content-Type: application/json" ^
  -d "{\"event_type\":\"PolicyCheck\",\"resource_id\":\"vm-test-001\",\"policy_id\":\"policy-test-001\",\"policy_name\":\"Test VM Compliance Policy\",\"compliance_status\":\"Compliant\",\"actor\":\"test-system\",\"subscription_id\":\"00000000-0000-0000-0000-000000000001\",\"resource_group\":\"rg-test-001\",\"resource_type\":\"Microsoft.Compute/virtualMachines\",\"details\":{\"test\":\"data\"},\"metadata\":{\"environment\":\"test\"}}"

REM Test 2: Get Chain Status
echo.
echo 2. Testing Chain Status...
curl "%BASE_URL%/api/v1/evidence/chain"

REM Test 3: Get Block 0 (Genesis)
echo.
echo 3. Testing Block Retrieval...
curl "%BASE_URL%/api/v1/evidence/block/0"

REM Test 4: Generate Report
echo.
echo 4. Testing Report Generation...
curl -X POST "%BASE_URL%/api/v1/evidence/report" ^
  -H "Content-Type: application/json" ^
  -d "{\"subscription_id\":\"00000000-0000-0000-0000-000000000001\",\"format\":\"PDF\",\"include_qr_code\":true,\"digital_signature\":true,\"include_evidence\":true,\"evidence_limit\":100}"

echo.
echo PROVE Pillar Evidence Chain Tests Complete!
pause