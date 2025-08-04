#!/bin/bash
# Deploy real application images to Container Apps

set -e

echo "🚀 Deploying real PolicyCortex applications to Container Apps"

# Variables
RESOURCE_GROUP="rg-pcx-app-dev"
ACR_NAME="crpcxdev"
ENVIRONMENT="dev"

# Login to Azure
echo "📝 Logging into Azure..."
az account show || az login

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query loginServer -o tsv)
echo "📦 Container Registry: $ACR_LOGIN_SERVER"

# Login to ACR
echo "🔐 Logging into Container Registry..."
az acr login --name $ACR_NAME

# Build and push backend services
echo "🏗️ Building backend services..."
cd backend

SERVICES=("api_gateway" "azure_integration" "ai_engine" "data_processing" "conversation" "notification")

for SERVICE in "${SERVICES[@]}"; do
    echo "📦 Building $SERVICE..."
    
    # Create Dockerfile if it doesn't exist or update it
    cat > services/$SERVICE/Dockerfile << EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy shared dependencies first
COPY shared /app/shared
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/$SERVICE /app/services/$SERVICE

# Set environment variables
ENV PYTHONPATH=/app
ENV SERVICE_NAME=$SERVICE

# Expose port
EXPOSE 8000

# Run the service
CMD ["python", "-m", "uvicorn", "services.$SERVICE.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    # Map service names to container app names
    case $SERVICE in
        "api_gateway") APP_NAME="api-gateway" ;;
        "azure_integration") APP_NAME="azure-integration" ;;
        "ai_engine") APP_NAME="ai-engine" ;;
        "data_processing") APP_NAME="data-processing" ;;
        "conversation") APP_NAME="conversation" ;;
        "notification") APP_NAME="notification" ;;
    esac
    
    # Build and push image
    docker build -f services/$SERVICE/Dockerfile -t $ACR_LOGIN_SERVER/pcx-$APP_NAME:latest .
    docker push $ACR_LOGIN_SERVER/pcx-$APP_NAME:latest
    
    echo "✅ $SERVICE image pushed"
done

# Build and push frontend
echo "🏗️ Building frontend..."
cd ../frontend

# Create optimized Dockerfile for frontend
cat > Dockerfile << EOF
# Build stage
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Runtime stage
FROM nginx:alpine

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
EOF

# Create nginx configuration
cat > nginx.conf << EOF
server {
    listen 80;
    server_name _;
    
    root /usr/share/nginx/html;
    index index.html;
    
    # Enable gzip
    gzip on;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Handle client-side routing
    location / {
        try_files \$uri \$uri/ /index.html;
    }
    
    # API proxy (handled by ingress in Container Apps)
    location /api {
        return 404;
    }
    
    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Build and push frontend image
docker build -t $ACR_LOGIN_SERVER/pcx-frontend:latest .
docker push $ACR_LOGIN_SERVER/pcx-frontend:latest

echo "✅ Frontend image pushed"

cd ..

# Update Container Apps with new images
echo "🔄 Updating Container Apps..."

# Update backend services
CONTAINER_APPS=(
    "ca-pcx-gateway-dev"
    "ca-pcx-azureint-dev"
    "ca-pcx-ai-dev"
    "ca-pcx-dataproc-dev"
    "ca-pcx-chat-dev"
    "ca-pcx-notify-dev"
)

for APP in "${CONTAINER_APPS[@]}"; do
    echo "📝 Updating $APP..."
    az containerapp update \
        --name $APP \
        --resource-group $RESOURCE_GROUP \
        --image $ACR_LOGIN_SERVER/${APP/ca-pcx-/pcx-}:latest \
        --query "properties.latestRevisionName" \
        -o tsv
done

# Update frontend
echo "📝 Updating frontend..."
az containerapp update \
    --name ca-pcx-web-dev \
    --resource-group $RESOURCE_GROUP \
    --image $ACR_LOGIN_SERVER/pcx-frontend:latest \
    --query "properties.latestRevisionName" \
    -o tsv

echo "✅ All Container Apps updated!"

# Get URLs
echo ""
echo "🌐 Application URLs:"
FRONTEND_URL=$(az containerapp show --name ca-pcx-web-dev --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)
API_URL=$(az containerapp show --name ca-pcx-gateway-dev --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)

echo "Frontend: https://$FRONTEND_URL"
echo "API Gateway: https://$API_URL"

echo ""
echo "🎉 Deployment complete! Your applications are now running with real builds."