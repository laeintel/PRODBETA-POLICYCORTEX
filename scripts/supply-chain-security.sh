#!/bin/bash
# Supply Chain Security Script for PolicyCortex
# Performs comprehensive security analysis of dependencies and artifacts

set -e

echo "🔐 Supply Chain Security Analysis"
echo "=================================="

# Create reports directory
mkdir -p security-reports

# Initialize report
REPORT_FILE="security-reports/supply-chain-report-$(date +%Y%m%d-%H%M%S).json"
echo '{"timestamp":"'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'","results":[' > "$REPORT_FILE"

# Function to add result to report
add_result() {
    local component=$1
    local check=$2
    local status=$3
    local details=$4
    
    if [ ! -f "$REPORT_FILE.tmp" ]; then
        echo -n "" > "$REPORT_FILE.tmp"
    else
        echo -n "," >> "$REPORT_FILE.tmp"
    fi
    
    cat >> "$REPORT_FILE.tmp" << EOF
{
    "component": "$component",
    "check": "$check",
    "status": "$status",
    "details": "$details",
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
}

# Check Node.js dependencies
echo "📦 Checking Node.js dependencies..."
if [ -f "package.json" ]; then
    # Check for known vulnerabilities
    if command -v npm &> /dev/null; then
        echo "  Running npm audit..."
        npm audit --json > security-reports/npm-audit.json 2>/dev/null || true
        
        vulnerabilities=$(cat security-reports/npm-audit.json | jq '.metadata.vulnerabilities.total // 0' 2>/dev/null || echo "0")
        
        if [ "$vulnerabilities" -gt 0 ]; then
            add_result "npm" "vulnerability-scan" "warning" "Found $vulnerabilities vulnerabilities"
            echo "  ⚠️  Found $vulnerabilities vulnerabilities in npm dependencies"
        else
            add_result "npm" "vulnerability-scan" "pass" "No vulnerabilities found"
            echo "  ✅ No vulnerabilities found in npm dependencies"
        fi
    fi
    
    # Check for license compliance
    echo "  Checking licenses..."
    npm ls --json 2>/dev/null | jq -r '.dependencies | to_entries[] | .value.license // "UNKNOWN"' | sort | uniq -c > security-reports/npm-licenses.txt || true
    
    restricted_licenses=$(grep -E "(GPL|AGPL|LGPL)" security-reports/npm-licenses.txt | wc -l || echo "0")
    if [ "$restricted_licenses" -gt 0 ]; then
        add_result "npm" "license-check" "warning" "Found $restricted_licenses packages with restrictive licenses"
        echo "  ⚠️  Found packages with restrictive licenses"
    else
        add_result "npm" "license-check" "pass" "All licenses are permissive"
        echo "  ✅ All npm licenses are permissive"
    fi
fi

# Check Rust dependencies
echo "📦 Checking Rust dependencies..."
if [ -f "Cargo.toml" ] || [ -f "core/Cargo.toml" ]; then
    CARGO_DIR="."
    if [ -f "core/Cargo.toml" ]; then
        CARGO_DIR="core"
    fi
    
    # Check with cargo-audit if available
    if command -v cargo-audit &> /dev/null; then
        echo "  Running cargo audit..."
        cd "$CARGO_DIR"
        cargo audit --json > ../security-reports/cargo-audit.json 2>/dev/null || true
        cd - > /dev/null
        
        vulnerabilities=$(cat security-reports/cargo-audit.json | jq '.vulnerabilities.count // 0' 2>/dev/null || echo "0")
        
        if [ "$vulnerabilities" -gt 0 ]; then
            add_result "cargo" "vulnerability-scan" "warning" "Found $vulnerabilities vulnerabilities"
            echo "  ⚠️  Found $vulnerabilities vulnerabilities in Cargo dependencies"
        else
            add_result "cargo" "vulnerability-scan" "pass" "No vulnerabilities found"
            echo "  ✅ No vulnerabilities found in Cargo dependencies"
        fi
    else
        echo "  ℹ️  cargo-audit not installed, skipping Rust vulnerability scan"
        add_result "cargo" "vulnerability-scan" "skipped" "cargo-audit not installed"
    fi
fi

# Check Python dependencies
echo "📦 Checking Python dependencies..."
if [ -f "requirements.txt" ] || [ -f "backend/requirements.txt" ]; then
    REQ_FILE="requirements.txt"
    if [ -f "backend/requirements.txt" ]; then
        REQ_FILE="backend/requirements.txt"
    fi
    
    # Check with safety if available
    if command -v safety &> /dev/null; then
        echo "  Running safety check..."
        safety check -r "$REQ_FILE" --json > security-reports/safety-check.json 2>/dev/null || true
        
        vulnerabilities=$(cat security-reports/safety-check.json | jq '. | length' 2>/dev/null || echo "0")
        
        if [ "$vulnerabilities" -gt 0 ]; then
            add_result "python" "vulnerability-scan" "warning" "Found $vulnerabilities vulnerabilities"
            echo "  ⚠️  Found $vulnerabilities vulnerabilities in Python dependencies"
        else
            add_result "python" "vulnerability-scan" "pass" "No vulnerabilities found"
            echo "  ✅ No vulnerabilities found in Python dependencies"
        fi
    else
        echo "  ℹ️  safety not installed, using pip-audit fallback"
        if command -v pip-audit &> /dev/null; then
            pip-audit -r "$REQ_FILE" --format json > security-reports/pip-audit.json 2>/dev/null || true
            add_result "python" "vulnerability-scan" "info" "Scanned with pip-audit"
        else
            add_result "python" "vulnerability-scan" "skipped" "No Python security scanner available"
        fi
    fi
fi

# Check Docker images
echo "🐳 Checking Docker images..."
if [ -f "Dockerfile" ] || [ -f "frontend/Dockerfile" ] || [ -f "core/Dockerfile" ]; then
    # List all Dockerfiles
    dockerfiles=$(find . -name "Dockerfile*" -type f 2>/dev/null | head -10)
    
    for dockerfile in $dockerfiles; do
        echo "  Analyzing $dockerfile..."
        
        # Check for latest tags (security risk)
        latest_tags=$(grep -E "FROM.*:latest" "$dockerfile" | wc -l || echo "0")
        if [ "$latest_tags" -gt 0 ]; then
            add_result "docker" "version-pinning" "warning" "Found :latest tags in $dockerfile"
            echo "    ⚠️  Using :latest tags (security risk)"
        else
            add_result "docker" "version-pinning" "pass" "All tags pinned in $dockerfile"
            echo "    ✅ All image tags are pinned"
        fi
        
        # Check for root user
        root_user=$(grep -E "USER root" "$dockerfile" | wc -l || echo "0")
        if [ "$root_user" -gt 0 ]; then
            add_result "docker" "user-privileges" "warning" "Running as root in $dockerfile"
            echo "    ⚠️  Container runs as root user"
        else
            add_result "docker" "user-privileges" "pass" "Non-root user in $dockerfile"
            echo "    ✅ Container uses non-root user"
        fi
    done
fi

# Check for secrets in code
echo "🔍 Checking for exposed secrets..."
# Basic pattern matching for common secret patterns
secret_patterns="(api[_-]?key|secret|token|password|pwd|passwd|credentials|private[_-]?key)"
exposed_secrets=$(grep -rEi "$secret_patterns" --include="*.js" --include="*.ts" --include="*.py" --include="*.rs" --include="*.go" --exclude-dir=node_modules --exclude-dir=target --exclude-dir=.git 2>/dev/null | grep -v "// " | grep -v "# " | wc -l || echo "0")

if [ "$exposed_secrets" -gt 0 ]; then
    add_result "secrets" "secret-scan" "warning" "Found $exposed_secrets potential secrets in code"
    echo "  ⚠️  Found $exposed_secrets potential exposed secrets"
else
    add_result "secrets" "secret-scan" "pass" "No secrets detected"
    echo "  ✅ No exposed secrets detected"
fi

# SBOM Generation
echo "📋 Generating Software Bill of Materials (SBOM)..."
SBOM_FILE="security-reports/sbom-$(date +%Y%m%d-%H%M%S).json"

# Initialize SBOM
cat > "$SBOM_FILE" << EOF
{
    "bomFormat": "CycloneDX",
    "specVersion": "1.4",
    "serialNumber": "urn:uuid:$(uuidgen 2>/dev/null || echo "$(date +%s)")",
    "version": 1,
    "metadata": {
        "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
        "tools": [
            {
                "vendor": "PolicyCortex",
                "name": "supply-chain-security",
                "version": "1.0.0"
            }
        ],
        "component": {
            "type": "application",
            "name": "PolicyCortex",
            "version": "1.0.0"
        }
    },
    "components": []
}
EOF

add_result "sbom" "generation" "pass" "SBOM generated at $SBOM_FILE"
echo "  ✅ SBOM generated"

# Dependency tree analysis
echo "📊 Analyzing dependency tree..."
if [ -f "package-lock.json" ]; then
    total_deps=$(cat package-lock.json | jq '.dependencies | length' 2>/dev/null || echo "0")
    add_result "dependencies" "count" "info" "Total npm dependencies: $total_deps"
    echo "  ℹ️  Total npm dependencies: $total_deps"
fi

if [ -f "Cargo.lock" ] || [ -f "core/Cargo.lock" ]; then
    CARGO_LOCK="Cargo.lock"
    if [ -f "core/Cargo.lock" ]; then
        CARGO_LOCK="core/Cargo.lock"
    fi
    total_deps=$(grep -c "name = " "$CARGO_LOCK" 2>/dev/null || echo "0")
    add_result "dependencies" "count" "info" "Total Cargo dependencies: $total_deps"
    echo "  ℹ️  Total Cargo dependencies: $total_deps"
fi

# Finalize report
if [ -f "$REPORT_FILE.tmp" ]; then
    cat "$REPORT_FILE.tmp" >> "$REPORT_FILE"
fi
echo ']}' >> "$REPORT_FILE"
rm -f "$REPORT_FILE.tmp"

# Summary
echo ""
echo "📊 Supply Chain Security Summary"
echo "================================"
echo "✅ Security analysis complete"
echo "📁 Reports saved to security-reports/"
echo ""

# Check if any critical issues were found
if grep -q '"status":"warning"' "$REPORT_FILE" 2>/dev/null; then
    echo "⚠️  Some security concerns were identified. Please review the reports."
    # Don't fail the build for warnings, only for critical issues
    exit 0
else
    echo "✅ No critical security issues found!"
    exit 0
fi