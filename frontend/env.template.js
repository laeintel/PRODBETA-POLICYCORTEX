// Environment configuration template
// This file is used by the Docker container to inject runtime environment variables
window.ENV = {
  VITE_API_BASE_URL: '${VITE_API_BASE_URL}',
  VITE_WS_URL: '${VITE_WS_URL}',
  VITE_AZURE_CLIENT_ID: '${VITE_AZURE_CLIENT_ID}',
  VITE_AZURE_TENANT_ID: '${VITE_AZURE_TENANT_ID}',
  VITE_AZURE_REDIRECT_URI: '${VITE_AZURE_REDIRECT_URI}',
  VITE_APP_VERSION: '${VITE_APP_VERSION}'
}