#!/bin/bash

# Test script for PolicyCortex PROVE Pillar Evidence Chain

echo "Testing PolicyCortex PROVE Pillar - Immutable Evidence Chain"
echo "============================================================"

# Base URL
BASE_URL="http://localhost:8080"

# Test 1: Collect Evidence
echo -e "\n1. Testing Evidence Collection..."
EVIDENCE_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/evidence/collect" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "PolicyCheck",
    "resource_id": "vm-test-001",
    "policy_id": "policy-test-001",
    "policy_name": "Test VM Compliance Policy",
    "compliance_status": "Compliant",
    "actor": "test-system",
    "subscription_id": "00000000-0000-0000-0000-000000000001",
    "resource_group": "rg-test-001",
    "resource_type": "Microsoft.Compute/virtualMachines",
    "details": {"test": "data"},
    "metadata": {"environment": "test"}
  }')

echo "Response: $EVIDENCE_RESPONSE"

# Extract evidence ID and hash
EVIDENCE_ID=$(echo $EVIDENCE_RESPONSE | grep -oP '"evidence_id":\s*"\K[^"]+')
HASH=$(echo $EVIDENCE_RESPONSE | grep -oP '"hash":\s*"\K[^"]+')

echo "Evidence ID: $EVIDENCE_ID"
echo "Hash: $HASH"

# Test 2: Get Chain Status
echo -e "\n2. Testing Chain Status..."
CHAIN_STATUS=$(curl -s "$BASE_URL/api/v1/evidence/chain")
echo "Chain Status: $CHAIN_STATUS"

# Test 3: Verify Evidence
echo -e "\n3. Testing Evidence Verification..."
if [ ! -z "$HASH" ]; then
  VERIFY_RESPONSE=$(curl -s "$BASE_URL/api/v1/evidence/verify/$HASH")
  echo "Verification Response: $VERIFY_RESPONSE"
fi

# Test 4: Get Evidence by ID
echo -e "\n4. Testing Evidence Retrieval..."
if [ ! -z "$EVIDENCE_ID" ]; then
  EVIDENCE=$(curl -s "$BASE_URL/api/v1/evidence/$EVIDENCE_ID")
  echo "Evidence: $EVIDENCE"
fi

# Test 5: Generate Report
echo -e "\n5. Testing Report Generation..."
REPORT_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/evidence/report" \
  -H "Content-Type: application/json" \
  -d '{
    "subscription_id": "00000000-0000-0000-0000-000000000001",
    "format": "PDF",
    "include_qr_code": true,
    "digital_signature": true,
    "include_evidence": true,
    "evidence_limit": 100
  }')

echo "Report Response: $REPORT_RESPONSE"

echo -e "\nPROVE Pillar Evidence Chain Tests Complete!"