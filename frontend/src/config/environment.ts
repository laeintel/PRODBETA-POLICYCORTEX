// Get runtime configuration (injected by Container Apps)
const getRuntimeConfig = () => {
  if (typeof window !== 'undefined' && (window as any).POLICYCORTEX_CONFIG) {
    return (window as any).POLICYCORTEX_CONFIG
  }
  return {}
}

const runtimeConfig = getRuntimeConfig()

// Environment configuration
export const env = {
  // App
  NODE_ENV: import.meta.env.NODE_ENV || 'development',
  APP_NAME: import.meta.env.VITE_APP_NAME || 'PolicyCortex',
  APP_VERSION: runtimeConfig.VITE_APP_VERSION || import.meta.env.VITE_APP_VERSION || '1.0.0',
  APP_DESCRIPTION: import.meta.env.VITE_APP_DESCRIPTION || 'AI-Powered Azure Governance Intelligence Platform',

  // API
  API_BASE_URL: runtimeConfig.VITE_API_BASE_URL || import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  API_TIMEOUT: parseInt(import.meta.env.VITE_API_TIMEOUT || '30000'),
  API_RETRY_ATTEMPTS: parseInt(import.meta.env.VITE_API_RETRY_ATTEMPTS || '3'),

  // WebSocket
  WS_URL: runtimeConfig.VITE_WS_URL || import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws',
  WS_RECONNECT_ATTEMPTS: parseInt(import.meta.env.VITE_WS_RECONNECT_ATTEMPTS || '5'),
  WS_RECONNECT_DELAY: parseInt(import.meta.env.VITE_WS_RECONNECT_DELAY || '1000'),

  // Azure AD
  AZURE_CLIENT_ID: runtimeConfig.VITE_AZURE_CLIENT_ID || import.meta.env.VITE_AZURE_CLIENT_ID || '',
  AZURE_TENANT_ID: runtimeConfig.VITE_AZURE_TENANT_ID || import.meta.env.VITE_AZURE_TENANT_ID || '',
  AZURE_REDIRECT_URI: runtimeConfig.VITE_AZURE_REDIRECT_URI || import.meta.env.VITE_AZURE_REDIRECT_URI || '',
  AZURE_AUTHORITY: import.meta.env.VITE_AZURE_AUTHORITY || '',

  // Features
  ENABLE_ANALYTICS: import.meta.env.VITE_ENABLE_ANALYTICS === 'true',
  ENABLE_NOTIFICATIONS: import.meta.env.VITE_ENABLE_NOTIFICATIONS === 'true',
  ENABLE_WEBSOCKET: import.meta.env.VITE_ENABLE_WEBSOCKET === 'true',
  ENABLE_PWA: import.meta.env.VITE_ENABLE_PWA === 'true',
  ENABLE_DARK_MODE: import.meta.env.VITE_ENABLE_DARK_MODE === 'true',

  // Logging
  LOG_LEVEL: import.meta.env.VITE_LOG_LEVEL || 'info',
  ENABLE_DEBUG: import.meta.env.VITE_ENABLE_DEBUG === 'true',
  ENABLE_PERFORMANCE_MONITORING: import.meta.env.VITE_ENABLE_PERFORMANCE_MONITORING === 'true',

  // Cache
  CACHE_TTL: parseInt(import.meta.env.VITE_CACHE_TTL || '300000'), // 5 minutes
  CACHE_MAX_SIZE: parseInt(import.meta.env.VITE_CACHE_MAX_SIZE || '100'),

  // Pagination
  DEFAULT_PAGE_SIZE: parseInt(import.meta.env.VITE_DEFAULT_PAGE_SIZE || '25'),
  MAX_PAGE_SIZE: parseInt(import.meta.env.VITE_MAX_PAGE_SIZE || '100'),

  // File Upload
  MAX_FILE_SIZE: parseInt(import.meta.env.VITE_MAX_FILE_SIZE || '10485760'), // 10MB
  ALLOWED_FILE_TYPES: import.meta.env.VITE_ALLOWED_FILE_TYPES || 'jpg,jpeg,png,gif,pdf,doc,docx,xls,xlsx,csv',

  // Charts and Visualization
  CHART_ANIMATION_DURATION: parseInt(import.meta.env.VITE_CHART_ANIMATION_DURATION || '300'),
  CHART_REFRESH_INTERVAL: parseInt(import.meta.env.VITE_CHART_REFRESH_INTERVAL || '30000'), // 30 seconds

  // Notification
  NOTIFICATION_DURATION: parseInt(import.meta.env.VITE_NOTIFICATION_DURATION || '5000'),
  NOTIFICATION_MAX_COUNT: parseInt(import.meta.env.VITE_NOTIFICATION_MAX_COUNT || '3'),

  // Security
  SESSION_TIMEOUT: parseInt(import.meta.env.VITE_SESSION_TIMEOUT || '3600000'), // 1 hour
  IDLE_TIMEOUT: parseInt(import.meta.env.VITE_IDLE_TIMEOUT || '1800000'), // 30 minutes
  CSRF_HEADER_NAME: import.meta.env.VITE_CSRF_HEADER_NAME || 'X-CSRF-Token',

  // External Services
  SENTRY_DSN: import.meta.env.VITE_SENTRY_DSN || '',
  GOOGLE_ANALYTICS_ID: import.meta.env.VITE_GOOGLE_ANALYTICS_ID || '',
  HOTJAR_ID: import.meta.env.VITE_HOTJAR_ID || '',

  // Azure Services
  AZURE_STORAGE_ACCOUNT: import.meta.env.VITE_AZURE_STORAGE_ACCOUNT || '',
  AZURE_STORAGE_CONTAINER: import.meta.env.VITE_AZURE_STORAGE_CONTAINER || '',
  AZURE_FUNCTIONS_URL: import.meta.env.VITE_AZURE_FUNCTIONS_URL || '',
  AZURE_COGNITIVE_SERVICES_KEY: import.meta.env.VITE_AZURE_COGNITIVE_SERVICES_KEY || '',

  // Monitoring
  ENABLE_ERROR_REPORTING: import.meta.env.VITE_ENABLE_ERROR_REPORTING === 'true',
  ENABLE_PERFORMANCE_TRACKING: import.meta.env.VITE_ENABLE_PERFORMANCE_TRACKING === 'true',
  ENABLE_USER_ANALYTICS: import.meta.env.VITE_ENABLE_USER_ANALYTICS === 'true',
} as const

// Environment validation
export const validateEnvironment = () => {
  // Skip validation if runtime config is available (will be loaded dynamically)
  if (typeof window !== 'undefined' && (window as any).POLICYCORTEX_CONFIG) {
    console.info('Using runtime configuration from config.js')
    return
  }

  const requiredVars = [
    { key: 'AZURE_CLIENT_ID', value: env.AZURE_CLIENT_ID },
    { key: 'AZURE_TENANT_ID', value: env.AZURE_TENANT_ID },
    { key: 'API_BASE_URL', value: env.API_BASE_URL },
  ]

  const missingVars = requiredVars.filter(({ value }) => !value).map(({ key }) => key)

  if (missingVars.length > 0) {
    console.warn('Environment variables not found at build time, expecting runtime configuration')
    console.info('Current configuration:', {
      AZURE_CLIENT_ID: env.AZURE_CLIENT_ID ? '***' : '(will be loaded from config.js)',
      AZURE_TENANT_ID: env.AZURE_TENANT_ID ? '***' : '(will be loaded from config.js)',
      API_BASE_URL: env.API_BASE_URL || '(will be loaded from config.js)',
    })
    // Don't throw in production - config.js will provide values
  }
}

// Development helpers
export const isDevelopment = env.NODE_ENV === 'development'
export const isProduction = env.NODE_ENV === 'production'
export const isTest = env.NODE_ENV === 'test'

// Feature flags
export const features = {
  analytics: env.ENABLE_ANALYTICS,
  notifications: env.ENABLE_NOTIFICATIONS,
  websocket: env.ENABLE_WEBSOCKET,
  pwa: env.ENABLE_PWA,
  darkMode: env.ENABLE_DARK_MODE,
  debug: env.ENABLE_DEBUG,
  performanceMonitoring: env.ENABLE_PERFORMANCE_MONITORING,
  errorReporting: env.ENABLE_ERROR_REPORTING,
  performanceTracking: env.ENABLE_PERFORMANCE_TRACKING,
  userAnalytics: env.ENABLE_USER_ANALYTICS,
} as const

// API endpoints based on environment
export const getApiUrl = (endpoint: string) => {
  return `${env.API_BASE_URL}${endpoint}`
}

export const getWsUrl = (endpoint: string = '') => {
  return `${env.WS_URL}${endpoint}`
}

// Local storage keys
export const storageKeys = {
  theme: 'policycortex_theme',
  user: 'policycortex_user',
  settings: 'policycortex_settings',
  preferences: 'policycortex_preferences',
  tokens: 'policycortex_tokens',
  cache: 'policycortex_cache',
  notifications: 'policycortex_notifications',
  dashboard: 'policycortex_dashboard',
  filters: 'policycortex_filters',
  viewState: 'policycortex_view_state',
} as const

// Session storage keys
export const sessionStorageKeys = {
  auth: 'policycortex_auth_session',
  navigation: 'policycortex_navigation',
  tempData: 'policycortex_temp_data',
} as const

// Default values
export const defaults = {
  theme: 'light' as const,
  locale: 'en-US',
  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  currency: 'USD',
  dateFormat: 'yyyy-MM-dd',
  timeFormat: 'HH:mm',
  pageSize: env.DEFAULT_PAGE_SIZE,
  refreshInterval: 30000, // 30 seconds
  debounceDelay: 300,
  animationDuration: 300,
} as const

// Initialize environment validation
if (typeof window !== 'undefined' && env.NODE_ENV !== 'test') {
  validateEnvironment()
}