#!/bin/bash
# ========================================
# PolicyCortex Smoke Tests
# Quick validation for CI/CD pipeline
# ========================================

set -e

echo "========================================="
echo "PolicyCortex Smoke Tests"
echo "========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Function to test endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local expected_status=$3
    local check_content=$4
    
    echo -n "Testing $name... "
    
    # Make request and capture status
    response=$(curl -s -o /tmp/response.txt -w "%{http_code}" $url)
    
    if [ "$response" = "$expected_status" ]; then
        if [ -n "$check_content" ]; then
            if grep -q "$check_content" /tmp/response.txt; then
                echo -e "${GREEN}✓ PASS${NC} (Status: $response, Content verified)"
                ((TESTS_PASSED++))
            else
                echo -e "${RED}✗ FAIL${NC} (Status OK, but content check failed)"
                ((TESTS_FAILED++))
            fi
        else
            echo -e "${GREEN}✓ PASS${NC} (Status: $response)"
            ((TESTS_PASSED++))
        fi
    else
        echo -e "${RED}✗ FAIL${NC} (Expected: $expected_status, Got: $response)"
        ((TESTS_FAILED++))
    fi
}

# Function to test JSON response
test_json_endpoint() {
    local name=$1
    local url=$2
    local expected_field=$3
    
    echo -n "Testing $name... "
    
    # Make request
    response=$(curl -s $url)
    
    # Check if response is valid JSON and contains expected field
    if echo "$response" | jq -e ".$expected_field" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC} (Valid JSON with field: $expected_field)"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC} (Invalid JSON or missing field: $expected_field)"
        ((TESTS_FAILED++))
    fi
}

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

echo ""
echo "Running smoke tests..."
echo "----------------------------------------"

# Test 1: Core API Health
test_endpoint "Core API Health" "http://localhost:8080/health" "200" "healthy"

# Test 2: Core API Metrics (should return non-empty)
test_json_endpoint "Core API Metrics" "http://localhost:8080/api/v1/metrics" "metrics"

# Test 3: Frontend Root
test_endpoint "Frontend Root" "http://localhost:3000" "200" "PolicyCortex"

# Test 4: Frontend Dashboard
test_endpoint "Frontend Dashboard" "http://localhost:3000/dashboard" "200" ""

# Test 5: GraphQL Health (Mock resolver)
test_endpoint "GraphQL Health" "http://localhost:4000/health" "200" ""

# Test 6: API Conversation Endpoint
test_endpoint "Conversation API" "http://localhost:8080/api/v1/conversation" "200" ""

# Test 7: Predictions API
test_json_endpoint "Predictions API" "http://localhost:8080/api/v1/predictions" "predictions"

# Test 8: Correlations API
test_json_endpoint "Correlations API" "http://localhost:8080/api/v1/correlations" "correlations"

# Test 9: Knowledge Graph API
test_endpoint "Knowledge Graph" "http://localhost:3000/api/v1/graph" "200" "nodes"

# Test 10: Database connectivity
echo -n "Testing PostgreSQL connectivity... "
if docker exec policycortex-postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

echo ""
echo "----------------------------------------"
echo "Smoke Test Results:"
echo "  Passed: ${GREEN}$TESTS_PASSED${NC}"
echo "  Failed: ${RED}$TESTS_FAILED${NC}"
echo "----------------------------------------"

# Exit with appropriate code
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Smoke tests FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}All smoke tests PASSED${NC}"
    exit 0
fi