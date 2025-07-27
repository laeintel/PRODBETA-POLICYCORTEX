// Runtime configuration for PolicyCortex frontend
// This file is served statically and loaded before the main app
// Container Apps will replace these values using environment variables

window.POLICYCORTEX_CONFIG = {
  VITE_API_BASE_URL: window.VITE_API_BASE_URL || 'https://ca-api-gateway-dev.delightfulsmoke-bbe56ef9.eastus.azurecontainerapps.io/api',
  VITE_WS_URL: window.VITE_WS_URL || 'wss://ca-api-gateway-dev.delightfulsmoke-bbe56ef9.eastus.azurecontainerapps.io/ws',
  VITE_AZURE_CLIENT_ID: window.VITE_AZURE_CLIENT_ID || '',
  VITE_AZURE_TENANT_ID: window.VITE_AZURE_TENANT_ID || '',
  VITE_AZURE_REDIRECT_URI: window.VITE_AZURE_REDIRECT_URI || window.location.origin,
  VITE_APP_VERSION: window.VITE_APP_VERSION || '1.0.0'
};