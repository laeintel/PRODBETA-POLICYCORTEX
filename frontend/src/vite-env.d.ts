/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_APP_NAME: string
  readonly VITE_APP_VERSION: string
  readonly VITE_APP_DESCRIPTION: string
  readonly VITE_API_BASE_URL: string
  readonly VITE_API_TIMEOUT: string
  readonly VITE_API_RETRY_ATTEMPTS: string
  readonly VITE_WS_URL: string
  readonly VITE_WS_RECONNECT_ATTEMPTS: string
  readonly VITE_WS_RECONNECT_DELAY: string
  readonly VITE_AZURE_CLIENT_ID: string
  readonly VITE_AZURE_TENANT_ID: string
  readonly VITE_AZURE_REDIRECT_URI: string
  readonly VITE_AZURE_AUTHORITY: string
  readonly VITE_ENABLE_ANALYTICS: string
  readonly VITE_ENABLE_NOTIFICATIONS: string
  readonly VITE_ENABLE_WEBSOCKET: string
  readonly VITE_ENABLE_PWA: string
  readonly VITE_ENABLE_DARK_MODE: string
  readonly VITE_LOG_LEVEL: string
  readonly VITE_ENABLE_DEBUG: string
  readonly VITE_ENABLE_PERFORMANCE_MONITORING: string
  readonly VITE_CACHE_TTL: string
  readonly VITE_CACHE_MAX_SIZE: string
  readonly VITE_DEFAULT_PAGE_SIZE: string
  readonly VITE_MAX_PAGE_SIZE: string
  readonly VITE_MAX_FILE_SIZE: string
  readonly VITE_ALLOWED_FILE_TYPES: string
  readonly VITE_CHART_ANIMATION_DURATION: string
  readonly VITE_CHART_REFRESH_INTERVAL: string
  readonly VITE_NOTIFICATION_DURATION: string
  readonly VITE_NOTIFICATION_MAX_COUNT: string
  readonly VITE_SESSION_TIMEOUT: string
  readonly VITE_IDLE_TIMEOUT: string
  readonly VITE_CSRF_HEADER_NAME: string
  readonly VITE_SENTRY_DSN: string
  readonly VITE_GOOGLE_ANALYTICS_ID: string
  readonly VITE_HOTJAR_ID: string
  readonly VITE_AZURE_STORAGE_ACCOUNT: string
  readonly VITE_AZURE_STORAGE_CONTAINER: string
  readonly VITE_AZURE_FUNCTIONS_URL: string
  readonly VITE_AZURE_COGNITIVE_SERVICES_KEY: string
  readonly VITE_ENABLE_ERROR_REPORTING: string
  readonly VITE_ENABLE_PERFORMANCE_TRACKING: string
  readonly VITE_ENABLE_USER_ANALYTICS: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}