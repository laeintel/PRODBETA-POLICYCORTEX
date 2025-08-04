#!/bin/bash
# Quick deployment script for frontend only

set -e

echo "🚀 Deploying PolicyCortex frontend to Container Apps"

# Variables
RESOURCE_GROUP="rg-pcx-app-dev"
ACR_NAME="crpcxdev"

# Login to ACR
echo "🔐 Logging into Container Registry..."
az acr login --name $ACR_NAME

ACR_LOGIN_SERVER="$ACR_NAME.azurecr.io"

# Build and push frontend
echo "🏗️ Building frontend..."
cd frontend

# Create production build
echo "📦 Creating production build..."
npm install
npm run build

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM nginx:alpine

# Copy built files
COPY dist /usr/share/nginx/html

# Create nginx config
RUN echo 'server { \
    listen 80; \
    location / { \
        root /usr/share/nginx/html; \
        try_files $uri $uri/ /index.html; \
    } \
    location /api { \
        proxy_pass http://ca-pcx-gateway-dev.internal.lemonfield-7e1ea681.eastus.azurecontainerapps.io:8000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
EOF

# Build and push
docker build -t $ACR_LOGIN_SERVER/pcx-frontend:latest .
docker push $ACR_LOGIN_SERVER/pcx-frontend:latest

# Update frontend container app
echo "🔄 Updating frontend Container App..."
az containerapp update \
    --name ca-pcx-web-dev \
    --resource-group $RESOURCE_GROUP \
    --image $ACR_LOGIN_SERVER/pcx-frontend:latest \
    --set-env-vars "VITE_API_BASE_URL=https://ca-pcx-gateway-dev.lemonfield-7e1ea681.eastus.azurecontainerapps.io/api" \
    --revision-suffix "v$(date +%s)" \
    --query "properties.latestRevisionName" \
    -o tsv

# Get URL
FRONTEND_URL=$(az containerapp show --name ca-pcx-web-dev --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)

echo ""
echo "✅ Frontend deployed!"
echo "🌐 URL: https://$FRONTEND_URL"