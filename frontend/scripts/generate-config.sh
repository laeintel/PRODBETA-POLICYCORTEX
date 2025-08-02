#!/bin/bash
# Generate runtime configuration for PolicyCortex Frontend

echo "Generating runtime configuration..."

# Create config.js with environment variables
cat > /usr/share/nginx/html/config.js << EOF
window.ENV = {
  VITE_API_BASE_URL: '${VITE_API_BASE_URL:-http://localhost:8000}',
  VITE_WS_URL: '${VITE_WS_URL:-ws://localhost:8000}',
  VITE_AZURE_CLIENT_ID: '${VITE_AZURE_CLIENT_ID:-your-client-id}',
  VITE_AZURE_TENANT_ID: '${VITE_AZURE_TENANT_ID:-your-tenant-id}',
  VITE_AZURE_REDIRECT_URI: '${VITE_AZURE_REDIRECT_URI:-http://localhost:5173}',
  NODE_ENV: '${NODE_ENV:-development}'
};
EOF

echo "Configuration generated successfully!"

# Start nginx
exec nginx -g "daemon off;"