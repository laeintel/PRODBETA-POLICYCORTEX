declare global {
  interface Window {
    POLICYCORTEX_CONFIG?: {
      VITE_API_BASE_URL?: string
      VITE_WS_URL?: string
      VITE_AZURE_CLIENT_ID?: string
      VITE_AZURE_TENANT_ID?: string
      VITE_AZURE_REDIRECT_URI?: string
      VITE_APP_VERSION?: string
    }
    msalInstance?: any
  }
}

export {}