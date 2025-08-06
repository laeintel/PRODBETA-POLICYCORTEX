// Runtime configuration for PolicyCortex frontend
// This file is served statically and loaded before the main app
// Container Apps will replace these values using environment variables

window.POLICYCORTEX_CONFIG = {
  VITE_API_BASE_URL: window.VITE_API_BASE_URL || 'http://localhost:8000',
  VITE_WS_URL: window.VITE_WS_URL || 'ws://localhost:8000/ws',
  VITE_AZURE_CLIENT_ID: window.VITE_AZURE_CLIENT_ID || '9ff9bdf6-a2b2-4b39-a270-71a9b8b9178d',
  VITE_AZURE_TENANT_ID: window.VITE_AZURE_TENANT_ID || '9ef5b184-d371-462a-bc75-5024ce8baff7',
  VITE_AZURE_REDIRECT_URI: window.VITE_AZURE_REDIRECT_URI || 'http://localhost:3000',
  VITE_APP_VERSION: window.VITE_APP_VERSION || '1.0.0'
};