#!/bin/bash
set -e

# Services to build
services=(
    "api_gateway"
    "azure_integration"
    "ai_engine"
    "data_processing"
    "conversation"
    "notification"
)

# ACR details
ACR_NAME="crpolicortex001dev.azurecr.io"

echo "Building and pushing all services..."

# Build and push each service
for service in "${services[@]}"; do
    echo "======================================"
    echo "Building $service..."
    echo "======================================"
    
    docker build -t $ACR_NAME/policortex001-$service:latest -f backend/services/$service/Dockerfile backend/
    
    echo "Pushing $service..."
    docker push $ACR_NAME/policortex001-$service:latest
    
    echo "$service completed!"
done

# Build and push frontend
echo "======================================"
echo "Building frontend..."
echo "======================================"
docker build -t $ACR_NAME/policortex001-frontend:latest -f frontend/Dockerfile frontend/
echo "Pushing frontend..."
docker push $ACR_NAME/policortex001-frontend:latest

echo "All images built and pushed successfully!"