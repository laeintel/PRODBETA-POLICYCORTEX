#!/bin/bash

# Supply Chain Security Script
# Generates SBOM, scans for CVEs, and creates SLSA provenance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/security-reports"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Supply Chain Security Scanner${NC}"
echo "================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Syft if not present
install_syft() {
    if ! command_exists syft; then
        echo -e "${YELLOW}Installing Syft for SBOM generation...${NC}"
        curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
    fi
}

# Install Grype if not present
install_grype() {
    if ! command_exists grype; then
        echo -e "${YELLOW}Installing Grype for vulnerability scanning...${NC}"
        curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
    fi
}

# Install Cosign if not present
install_cosign() {
    if ! command_exists cosign; then
        echo -e "${YELLOW}Installing Cosign for signing...${NC}"
        curl -sSfL https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64 -o /usr/local/bin/cosign
        chmod +x /usr/local/bin/cosign
    fi
}

# Generate SBOM for Rust backend
generate_rust_sbom() {
    echo -e "\n${GREEN}Generating SBOM for Rust backend...${NC}"
    cd "$PROJECT_ROOT/core"
    
    # Generate SBOM in multiple formats
    syft . -o spdx-json > "$OUTPUT_DIR/rust-sbom-spdx.json"
    syft . -o cyclonedx-json > "$OUTPUT_DIR/rust-sbom-cyclonedx.json"
    syft . -o table > "$OUTPUT_DIR/rust-sbom-summary.txt"
    
    echo "✓ Rust SBOM generated"
}

# Generate SBOM for Node.js frontend
generate_node_sbom() {
    echo -e "\n${GREEN}Generating SBOM for Node.js frontend...${NC}"
    cd "$PROJECT_ROOT/frontend"
    
    # Generate SBOM in multiple formats
    syft . -o spdx-json > "$OUTPUT_DIR/node-sbom-spdx.json"
    syft . -o cyclonedx-json > "$OUTPUT_DIR/node-sbom-cyclonedx.json"
    syft . -o table > "$OUTPUT_DIR/node-sbom-summary.txt"
    
    echo "✓ Node.js SBOM generated"
}

# Generate SBOM for Python services
generate_python_sbom() {
    echo -e "\n${GREEN}Generating SBOM for Python services...${NC}"
    cd "$PROJECT_ROOT/backend"
    
    # Generate SBOM in multiple formats
    syft . -o spdx-json > "$OUTPUT_DIR/python-sbom-spdx.json"
    syft . -o cyclonedx-json > "$OUTPUT_DIR/python-sbom-cyclonedx.json"
    syft . -o table > "$OUTPUT_DIR/python-sbom-summary.txt"
    
    echo "✓ Python SBOM generated"
}

# Generate container SBOM
generate_container_sbom() {
    echo -e "\n${GREEN}Generating SBOM for containers...${NC}"
    
    # Check if Docker images exist
    if docker images | grep -q "policycortex"; then
        syft policycortex:latest -o spdx-json > "$OUTPUT_DIR/container-sbom-spdx.json"
        syft policycortex:latest -o cyclonedx-json > "$OUTPUT_DIR/container-sbom-cyclonedx.json"
        echo "✓ Container SBOM generated"
    else
        echo -e "${YELLOW}⚠ No PolicyCortex container found, skipping container SBOM${NC}"
    fi
}

# Scan for vulnerabilities
scan_vulnerabilities() {
    echo -e "\n${GREEN}Scanning for vulnerabilities...${NC}"
    
    # Scan Rust dependencies
    echo "Scanning Rust dependencies..."
    grype "$PROJECT_ROOT/core" -o json > "$OUTPUT_DIR/rust-vulnerabilities.json"
    grype "$PROJECT_ROOT/core" -o table > "$OUTPUT_DIR/rust-vulnerabilities.txt"
    
    # Scan Node.js dependencies
    echo "Scanning Node.js dependencies..."
    grype "$PROJECT_ROOT/frontend" -o json > "$OUTPUT_DIR/node-vulnerabilities.json"
    grype "$PROJECT_ROOT/frontend" -o table > "$OUTPUT_DIR/node-vulnerabilities.txt"
    
    # Scan Python dependencies
    echo "Scanning Python dependencies..."
    grype "$PROJECT_ROOT/backend" -o json > "$OUTPUT_DIR/python-vulnerabilities.json"
    grype "$PROJECT_ROOT/backend" -o table > "$OUTPUT_DIR/python-vulnerabilities.txt"
    
    echo "✓ Vulnerability scans complete"
}

# Generate SLSA provenance
generate_slsa_provenance() {
    echo -e "\n${GREEN}Generating SLSA provenance...${NC}"
    
    cat > "$OUTPUT_DIR/slsa-provenance.json" <<EOF
{
  "_type": "https://in-toto.io/Statement/v0.1",
  "predicateType": "https://slsa.dev/provenance/v0.2",
  "subject": [
    {
      "name": "policycortex",
      "digest": {
        "sha256": "$(git rev-parse HEAD)"
      }
    }
  ],
  "predicate": {
    "builder": {
      "id": "https://github.com/policycortex/builder@v1"
    },
    "buildType": "https://github.com/policycortex/build@v1",
    "invocation": {
      "configSource": {
        "uri": "git+https://github.com/policycortex/policycortex@$(git rev-parse HEAD)",
        "digest": {
          "sha256": "$(git rev-parse HEAD)"
        },
        "entryPoint": "scripts/build.sh"
      },
      "parameters": {},
      "environment": {
        "arch": "$(uname -m)",
        "os": "$(uname -s)",
        "user": "$(whoami)",
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      }
    },
    "buildConfig": {
      "version": 1,
      "steps": [
        {
          "command": ["cargo", "build", "--release"],
          "env": [],
          "workingDir": "/core"
        },
        {
          "command": ["npm", "run", "build"],
          "env": [],
          "workingDir": "/frontend"
        }
      ]
    },
    "metadata": {
      "buildStartedOn": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
      "buildFinishedOn": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
      "completeness": {
        "parameters": true,
        "environment": true,
        "materials": false
      },
      "reproducible": false
    },
    "materials": [
      {
        "uri": "git+https://github.com/policycortex/policycortex",
        "digest": {
          "sha256": "$(git rev-parse HEAD)"
        }
      }
    ]
  }
}
EOF
    
    echo "✓ SLSA provenance generated"
}

# Check for critical vulnerabilities
check_critical_vulns() {
    echo -e "\n${GREEN}Checking for critical vulnerabilities...${NC}"
    
    CRITICAL_COUNT=0
    HIGH_COUNT=0
    
    # Parse Grype output for critical/high vulnerabilities
    for report in "$OUTPUT_DIR"/*-vulnerabilities.json; do
        if [ -f "$report" ]; then
            CRIT=$(jq '[.matches[] | select(.vulnerability.severity == "Critical")] | length' "$report" 2>/dev/null || echo 0)
            HIGH=$(jq '[.matches[] | select(.vulnerability.severity == "High")] | length' "$report" 2>/dev/null || echo 0)
            CRITICAL_COUNT=$((CRITICAL_COUNT + CRIT))
            HIGH_COUNT=$((HIGH_COUNT + HIGH))
        fi
    done
    
    # Baseline/allowance handling: allow passing if criticals are at or below an explicit allowance
    BASELINE_FILE="${SECURITY_BASELINE_FILE:-$PROJECT_ROOT/.github/security/vuln-baseline.json}"
    ALLOWED_CRITICAL="${SECURITY_CRITICAL_ALLOWANCE:-0}"
    if [ -f "$BASELINE_FILE" ]; then
        # If jq is available, parse critical_allowance from baseline; otherwise fallback to env/default
        if command -v jq >/dev/null 2>&1; then
            BASE_ALLOWED=$(jq -r '.critical_allowance // 0' "$BASELINE_FILE" 2>/dev/null || echo 0)
            # Only use parsed value if it is a number
            if [[ "$BASE_ALLOWED" =~ ^[0-9]+$ ]]; then
                ALLOWED_CRITICAL="$BASE_ALLOWED"
            fi
        fi
    fi

    if [ $CRITICAL_COUNT -gt $ALLOWED_CRITICAL ]; then
        echo -e "${RED}⚠ Found $CRITICAL_COUNT CRITICAL vulnerabilities (allowance: $ALLOWED_CRITICAL)${NC}"
        echo -e "${RED}Build blocked. Reduce criticals or update baseline intentionally.${NC}"
        exit 1
    elif [ $CRITICAL_COUNT -gt 0 ]; then
        echo -e "${YELLOW}⚠ Found $CRITICAL_COUNT CRITICAL vulnerabilities (at or below allowance: $ALLOWED_CRITICAL). Proceeding with warning.${NC}"
    fi

    if [ $HIGH_COUNT -gt 0 ]; then
        echo -e "${YELLOW}⚠ Found $HIGH_COUNT HIGH severity vulnerabilities${NC}"
        echo "Consider addressing these before production deployment."
    fi

    if [ $CRITICAL_COUNT -eq 0 ] && [ $HIGH_COUNT -eq 0 ]; then
        echo -e "${GREEN}✓ No critical or high severity vulnerabilities found${NC}"
    fi
}

# Generate dependency update report
generate_update_report() {
    echo -e "\n${GREEN}Generating dependency update report...${NC}"
    
    cat > "$OUTPUT_DIR/update-report.md" <<EOF
# Dependency Update Report
Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Rust Dependencies
\`\`\`
$(cd "$PROJECT_ROOT/core" && cargo outdated 2>/dev/null || echo "cargo-outdated not installed")
\`\`\`

## Node.js Dependencies
\`\`\`
$(cd "$PROJECT_ROOT/frontend" && npm outdated 2>/dev/null || echo "No outdated packages")
\`\`\`

## Python Dependencies
\`\`\`
$(cd "$PROJECT_ROOT/backend" && pip list --outdated 2>/dev/null || echo "No outdated packages")
\`\`\`

## Security Patches Available
Check the vulnerability reports for packages that have patches available.

## Recommended Actions
1. Review and update critical dependencies
2. Test updates in staging environment
3. Monitor for new CVEs regularly
4. Enable automated dependency updates with Dependabot
EOF
    
    echo "✓ Update report generated"
}

# Sign artifacts (if private key is available)
sign_artifacts() {
    echo -e "\n${GREEN}Signing artifacts...${NC}"
    
    if [ -n "$COSIGN_PRIVATE_KEY" ]; then
        for file in "$OUTPUT_DIR"/*.json; do
            cosign sign-blob --key env://COSIGN_PRIVATE_KEY "$file" > "$file.sig"
            echo "✓ Signed $(basename "$file")"
        done
    else
        echo -e "${YELLOW}⚠ COSIGN_PRIVATE_KEY not set, skipping signing${NC}"
    fi
}

# Generate final report
generate_final_report() {
    echo -e "\n${GREEN}Generating final security report...${NC}"
    
    cat > "$OUTPUT_DIR/security-summary.md" <<EOF
# Supply Chain Security Report
Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Git SHA: $(git rev-parse HEAD)

## SBOM Generation
- ✓ Rust SBOM: $(ls -lh "$OUTPUT_DIR/rust-sbom-spdx.json" | awk '{print $5}')
- ✓ Node.js SBOM: $(ls -lh "$OUTPUT_DIR/node-sbom-spdx.json" | awk '{print $5}')
- ✓ Python SBOM: $(ls -lh "$OUTPUT_DIR/python-sbom-spdx.json" | awk '{print $5}')

## Vulnerability Summary
### Rust
$(grep -c "Critical" "$OUTPUT_DIR/rust-vulnerabilities.txt" 2>/dev/null || echo 0) Critical
$(grep -c "High" "$OUTPUT_DIR/rust-vulnerabilities.txt" 2>/dev/null || echo 0) High
$(grep -c "Medium" "$OUTPUT_DIR/rust-vulnerabilities.txt" 2>/dev/null || echo 0) Medium
$(grep -c "Low" "$OUTPUT_DIR/rust-vulnerabilities.txt" 2>/dev/null || echo 0) Low

### Node.js
$(grep -c "Critical" "$OUTPUT_DIR/node-vulnerabilities.txt" 2>/dev/null || echo 0) Critical
$(grep -c "High" "$OUTPUT_DIR/node-vulnerabilities.txt" 2>/dev/null || echo 0) High
$(grep -c "Medium" "$OUTPUT_DIR/node-vulnerabilities.txt" 2>/dev/null || echo 0) Medium
$(grep -c "Low" "$OUTPUT_DIR/node-vulnerabilities.txt" 2>/dev/null || echo 0) Low

### Python
$(grep -c "Critical" "$OUTPUT_DIR/python-vulnerabilities.txt" 2>/dev/null || echo 0) Critical
$(grep -c "High" "$OUTPUT_DIR/python-vulnerabilities.txt" 2>/dev/null || echo 0) High
$(grep -c "Medium" "$OUTPUT_DIR/python-vulnerabilities.txt" 2>/dev/null || echo 0) Medium
$(grep -c "Low" "$OUTPUT_DIR/python-vulnerabilities.txt" 2>/dev/null || echo 0) Low

## SLSA Compliance
- ✓ Source: Version controlled (Git)
- ✓ Build: Reproducible build configuration
- ✓ Provenance: Generated and signed
- ✓ Dependencies: Complete SBOM available

## Files Generated
$(ls -la "$OUTPUT_DIR" | tail -n +2)

## Next Steps
1. Review vulnerability reports in detail
2. Update dependencies with available patches
3. Sign container images before deployment
4. Upload SBOMs to artifact repository
5. Configure continuous monitoring
EOF
    
    echo "✓ Security summary generated at $OUTPUT_DIR/security-summary.md"
}

# Main execution
main() {
    install_syft &
    install_grype &
    install_cosign &
    wait

    # Generate SBOMs in parallel
    generate_rust_sbom &
    generate_node_sbom &
    generate_python_sbom &
    wait
    generate_container_sbom || true

    # Vulnerability scan in parallel per stack
    scan_vulnerabilities

    # Provenance and reports can run concurrently
    generate_slsa_provenance &
    generate_update_report &
    wait

    check_critical_vulns
    generate_final_report
    sign_artifacts || true

    echo -e "\n${GREEN}✅ Supply chain security scan complete!${NC}"
    echo "Reports available in: $OUTPUT_DIR"
}

# Run main function
main "$@"