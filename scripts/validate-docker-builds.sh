#!/bin/bash
# Validate Docker builds for all services

set -e

echo "PolicyCortex Docker Build Validation"
echo "===================================="

SERVICES=("api_gateway" "azure_integration" "ai_engine" "data_processing" "conversation" "notification")
BUILD_CONTEXT="backend"

for service in "${SERVICES[@]}"; do
    echo -e "\n🔍 Validating $service..."
    
    # Check if Dockerfile exists
    dockerfile_path="$BUILD_CONTEXT/services/$service/Dockerfile"
    if [ ! -f "$dockerfile_path" ]; then
        echo "❌ Dockerfile not found: $dockerfile_path"
        continue
    fi
    
    # Check if requirements.txt exists
    req_path="$BUILD_CONTEXT/services/$service/requirements.txt"
    if [ ! -f "$req_path" ]; then
        echo "❌ requirements.txt not found: $req_path"
        continue
    fi
    
    echo "✅ Files exist: Dockerfile and requirements.txt"
    
    # Validate Dockerfile structure
    echo "📋 Dockerfile validation:"
    
    # Check COPY requirements line
    if grep -q "COPY services/$service/requirements.txt /app/requirements.txt" "$dockerfile_path"; then
        echo "  ✅ COPY requirements.txt line correct"
    else
        echo "  ❌ COPY requirements.txt line incorrect or missing"
        echo "     Expected: COPY services/$service/requirements.txt /app/requirements.txt"
        echo "     Found:"
        grep "COPY.*requirements.txt" "$dockerfile_path" || echo "     (none found)"
    fi
    
    # Check pip install line
    if grep -q "pip install.*-r /app/requirements.txt" "$dockerfile_path"; then
        echo "  ✅ pip install line correct"
    else
        echo "  ❌ pip install line incorrect"
        echo "     Expected: pip install ... -r /app/requirements.txt"
        echo "     Found:"
        grep "pip install.*requirements.txt" "$dockerfile_path" || echo "     (none found)"
    fi
    
    # Show requirements.txt first few lines
    echo "📄 Requirements preview:"
    head -3 "$req_path" | sed 's/^/     /'
    
done

echo -e "\n===================================="
echo "Docker build validation completed!"
echo "===================================="

# Optional: Test actual build (commented out to avoid long execution)
# echo -e "\n🔨 Testing actual Docker build for data_processing..."
# docker build -t test-data-processing -f backend/services/data_processing/Dockerfile backend/ --no-cache