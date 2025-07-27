#!/bin/sh
# Generate runtime configuration from environment variables
# This script runs when the container starts

cat > /app/dist/config.js << EOF
// Runtime configuration for PolicyCortex frontend
// Generated from environment variables at container startup

window.POLICYCORTEX_CONFIG = {
  VITE_API_BASE_URL: '${VITE_API_BASE_URL:-https://ca-api-gateway-dev.delightfulsmoke-bbe56ef9.eastus.azurecontainerapps.io/api}',
  VITE_WS_URL: '${VITE_WS_URL:-wss://ca-api-gateway-dev.delightfulsmoke-bbe56ef9.eastus.azurecontainerapps.io/ws}',
  VITE_AZURE_CLIENT_ID: '${VITE_AZURE_CLIENT_ID}',
  VITE_AZURE_TENANT_ID: '${VITE_AZURE_TENANT_ID}',
  VITE_AZURE_REDIRECT_URI: '${VITE_AZURE_REDIRECT_URI:-https://ca-frontend-dev.delightfulsmoke-bbe56ef9.eastus.azurecontainerapps.io}',
  VITE_APP_VERSION: '${VITE_APP_VERSION:-1.0.0}'
};
EOF

echo "✅ Generated config.js with environment variables"