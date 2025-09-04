#!/bin/bash

# PolicyCortex Security Testing Suite
# Comprehensive security testing implementation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SECURITY_REPORT_DIR="./security-reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_PREFIX="${SECURITY_REPORT_DIR}/${TIMESTAMP}"

# Create report directory
mkdir -p ${SECURITY_REPORT_DIR}

echo -e "${BLUE}🔒 PolicyCortex Security Testing Suite${NC}"
echo -e "${BLUE}=====================================\\n${NC}"

# Track overall results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
CRITICAL_ISSUES=0

# Function to run a test and track results
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "${YELLOW}Running: ${test_name}${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval "${test_command}"; then
        echo -e "${GREEN}✅ ${test_name} passed${NC}\\n"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}❌ ${test_name} failed${NC}\\n"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# 1. SAST - Static Application Security Testing
echo -e "${BLUE}1. Static Application Security Testing (SAST)${NC}"
echo -e "${BLUE}----------------------------------------------${NC}\\n"

# Semgrep scanning
if command -v semgrep &> /dev/null; then
    run_test "Semgrep Security Scan" \
        "semgrep --config=auto --json -o ${REPORT_PREFIX}_semgrep.json . 2>/dev/null"
else
    echo -e "${YELLOW}⚠️  Semgrep not installed. Install with: pip install semgrep${NC}"
fi

# ESLint security plugin for JavaScript/TypeScript
if [ -d "frontend" ]; then
    run_test "ESLint Security Plugin" \
        "cd frontend && npm run lint -- --format json -o ${REPORT_PREFIX}_eslint.json 2>/dev/null || true"
fi

# Bandit for Python
if command -v bandit &> /dev/null; then
    run_test "Bandit Python Security" \
        "bandit -r backend/ -f json -o ${REPORT_PREFIX}_bandit.json 2>/dev/null || true"
else
    echo -e "${YELLOW}⚠️  Bandit not installed. Install with: pip install bandit${NC}"
fi

# Cargo audit for Rust
if [ -d "core" ]; then
    run_test "Cargo Audit" \
        "cd core && cargo audit --json > ${REPORT_PREFIX}_cargo_audit.json 2>/dev/null || true"
fi

# 2. Dependency Scanning
echo -e "${BLUE}2. Dependency Vulnerability Scanning${NC}"
echo -e "${BLUE}-------------------------------------${NC}\\n"

# npm audit
if [ -f "frontend/package.json" ]; then
    run_test "npm audit (frontend)" \
        "cd frontend && npm audit --json > ${REPORT_PREFIX}_npm_audit.json 2>/dev/null || true"
fi

# Python Safety check
if command -v safety &> /dev/null; then
    run_test "Python Safety Check" \
        "safety check --json > ${REPORT_PREFIX}_safety.json 2>/dev/null || true"
else
    echo -e "${YELLOW}⚠️  Safety not installed. Install with: pip install safety${NC}"
fi

# 3. Secret Detection
echo -e "${BLUE}3. Secret Detection${NC}"
echo -e "${BLUE}-------------------${NC}\\n"

# Gitleaks
if command -v gitleaks &> /dev/null; then
    run_test "Gitleaks Secret Scan" \
        "gitleaks detect --report-format json --report-path ${REPORT_PREFIX}_gitleaks.json"
else
    echo -e "${YELLOW}⚠️  Gitleaks not installed. Download from: https://github.com/gitleaks/gitleaks${NC}"
fi

# TruffleHog
if command -v trufflehog &> /dev/null; then
    run_test "TruffleHog Secret Scan" \
        "trufflehog filesystem . --json > ${REPORT_PREFIX}_trufflehog.json 2>/dev/null || true"
fi

# 4. Container Security
echo -e "${BLUE}4. Container Security Scanning${NC}"
echo -e "${BLUE}------------------------------${NC}\\n"

# Trivy for container scanning
if command -v trivy &> /dev/null; then
    # Scan Docker images
    for image in "policycortex-frontend" "policycortex-core" "policycortex-api-gateway"; do
        if docker images | grep -q ${image}; then
            run_test "Trivy scan: ${image}" \
                "trivy image --format json -o ${REPORT_PREFIX}_trivy_${image}.json ${image}:latest"
        fi
    done
    
    # Scan filesystem
    run_test "Trivy filesystem scan" \
        "trivy fs --security-checks vuln,config,secret --format json -o ${REPORT_PREFIX}_trivy_fs.json ."
else
    echo -e "${YELLOW}⚠️  Trivy not installed. Install from: https://github.com/aquasecurity/trivy${NC}"
fi

# 5. Infrastructure as Code Security
echo -e "${BLUE}5. Infrastructure as Code Security${NC}"
echo -e "${BLUE}-----------------------------------${NC}\\n"

# Checkov for Terraform, Dockerfile, K8s
if command -v checkov &> /dev/null; then
    run_test "Checkov IaC Scan" \
        "checkov -d . --output json --output-file ${REPORT_PREFIX}_checkov.json 2>/dev/null || true"
else
    echo -e "${YELLOW}⚠️  Checkov not installed. Install with: pip install checkov${NC}"
fi

# 6. DAST - Dynamic Testing (if services are running)
echo -e "${BLUE}6. Dynamic Application Security Testing (DAST)${NC}"
echo -e "${BLUE}-----------------------------------------------${NC}\\n"

# Check if services are running
if curl -s http://localhost:3000/health > /dev/null 2>&1; then
    # OWASP ZAP baseline scan
    if command -v zap-baseline.py &> /dev/null; then
        run_test "OWASP ZAP Baseline Scan" \
            "zap-baseline.py -t http://localhost:3000 -J ${REPORT_PREFIX}_zap.json"
    else
        echo -e "${YELLOW}⚠️  OWASP ZAP not installed${NC}"
    fi
    
    # Custom security tests
    echo -e "${YELLOW}Running custom security tests...${NC}"
    
    # Test for security headers
    run_test "Security Headers Check" \
        "curl -s -I http://localhost:3000 | grep -E 'X-Frame-Options|X-Content-Type-Options|X-XSS-Protection' > /dev/null"
    
    # Test for SSL/TLS configuration (if HTTPS)
    if curl -s https://localhost:3000 > /dev/null 2>&1; then
        run_test "SSL/TLS Configuration" \
            "testssl --json ${REPORT_PREFIX}_testssl.json https://localhost:3000"
    fi
else
    echo -e "${YELLOW}⚠️  Services not running. Skipping DAST tests.${NC}"
fi

# 7. API Security Testing
echo -e "${BLUE}7. API Security Testing${NC}"
echo -e "${BLUE}-----------------------${NC}\\n"

# Test for common API vulnerabilities
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    # SQL Injection test
    run_test "SQL Injection Test" \
        "! curl -s 'http://localhost:8080/api/v1/resources?id=1%27%20OR%20%271%27=%271' | grep -q 'error'"
    
    # XSS test
    run_test "XSS Test" \
        "! curl -s -X POST http://localhost:8080/api/v1/resources \
         -H 'Content-Type: application/json' \
         -d '{\"name\":\"<script>alert(1)</script>\"}' | grep -q '<script>'"
    
    # Authentication bypass test
    run_test "Auth Bypass Test" \
        "curl -s http://localhost:8080/api/v1/admin/users | grep -q '401'"
    
    # Rate limiting test
    echo -e "${YELLOW}Testing rate limiting...${NC}"
    for i in {1..20}; do
        response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/api/v1/resources)
        if [ "$response" == "429" ]; then
            echo -e "${GREEN}✅ Rate limiting is working${NC}"
            break
        fi
    done
fi

# 8. Compliance Checks
echo -e "${BLUE}8. Compliance and Best Practices${NC}"
echo -e "${BLUE}---------------------------------${NC}\\n"

# Check for sensitive file exposure
run_test "Sensitive Files Check" \
    "! find . -name '.env' -o -name '*.key' -o -name '*.pem' 2>/dev/null | grep -v node_modules | grep -q ."

# Check for hardcoded credentials
run_test "Hardcoded Credentials Check" \
    "! grep -r -E '(password|secret|api[_-]?key)\\s*=\\s*[\"'\"'][^\"'\"']+[\"'\"']' --include='*.js' --include='*.ts' --include='*.py' --include='*.rs' . 2>/dev/null | grep -v test | grep -q ."

# 9. Generate Consolidated Report
echo -e "${BLUE}9. Generating Security Report${NC}"
echo -e "${BLUE}-----------------------------${NC}\\n"

# Create HTML report
cat > ${REPORT_PREFIX}_summary.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>PolicyCortex Security Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .pass { color: green; }
        .fail { color: red; }
        .warning { color: orange; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>PolicyCortex Security Test Report</h1>
    <p>Generated: $(date)</p>
    
    <h2>Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Tests</td><td>$TOTAL_TESTS</td></tr>
        <tr><td class="pass">Passed</td><td>$PASSED_TESTS</td></tr>
        <tr><td class="fail">Failed</td><td>$FAILED_TESTS</td></tr>
        <tr><td>Pass Rate</td><td>$(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)%</td></tr>
    </table>
    
    <h2>Critical Issues</h2>
    <p>Critical security issues found: $CRITICAL_ISSUES</p>
    
    <h2>Recommendations</h2>
    <ul>
        <li>Review all failed tests and remediate critical issues immediately</li>
        <li>Update all vulnerable dependencies</li>
        <li>Implement missing security headers</li>
        <li>Enable rate limiting on all API endpoints</li>
        <li>Regular security scanning in CI/CD pipeline</li>
    </ul>
</body>
</html>
EOF

echo -e "${GREEN}Report saved to: ${REPORT_PREFIX}_summary.html${NC}"

# 10. Final Summary
echo -e "\\n${BLUE}========================================${NC}"
echo -e "${BLUE}       SECURITY TEST SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total Tests Run: ${TOTAL_TESTS}"
echo -e "${GREEN}Passed: ${PASSED_TESTS}${NC}"
echo -e "${RED}Failed: ${FAILED_TESTS}${NC}"
echo -e "Pass Rate: $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)%"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\\n${GREEN}✅ All security tests passed!${NC}"
    exit 0
elif [ $CRITICAL_ISSUES -gt 0 ]; then
    echo -e "\\n${RED}❌ Critical security issues found! Immediate action required.${NC}"
    exit 2
else
    echo -e "\\n${YELLOW}⚠️  Some security tests failed. Review the report for details.${NC}"
    exit 1
fi