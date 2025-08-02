// API Configuration
export const apiConfig = {
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001',
  timeout: 30000,
  retryAttempts: 3,
  retryDelay: 1000,
}

// WebSocket Configuration
export const wsConfig = {
  url: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws',
  options: {
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
    maxReconnectionDelay: 5000,
    timeout: 20000,
  },
}

// Service Endpoints
export const endpoints = {
  // Authentication
  auth: {
    login: '/auth/login',
    logout: '/auth/logout',
    refresh: '/auth/refresh',
    profile: '/auth/profile',
    permissions: '/auth/permissions',
  },
  
  // Dashboard
  dashboard: {
    overview: '/dashboard/overview',
    metrics: '/dashboard/metrics',
    charts: '/dashboard/charts',
    alerts: '/dashboard/alerts',
  },
  
  // Policies
  policies: {
    list: '/policies',
    create: '/policies',
    update: (id: string) => `/policies/${id}`,
    delete: (id: string) => `/policies/${id}`,
    get: (id: string) => `/policies/${id}`,
    validate: '/policies/validate',
    simulate: '/policies/simulate',
    compliance: '/policies/compliance',
    templates: '/policies/templates',
  },
  
  // Resources
  resources: {
    list: '/resources',
    get: (id: string) => `/resources/${id}`,
    search: '/resources/search',
    inventory: '/resources/inventory',
    tags: '/resources/tags',
    compliance: '/resources/compliance',
    topology: '/resources/topology',
  },
  
  // Cost Management
  costs: {
    overview: '/costs/overview',
    analysis: '/costs/analysis',
    budgets: '/costs/budgets',
    forecasts: '/costs/forecasts',
    recommendations: '/costs/recommendations',
    anomalies: '/costs/anomalies',
    exports: '/costs/exports',
  },
  
  // AI/ML Services
  ai: {
    chat: '/ai/chat',
    analyze: '/ai/analyze',
    recommendations: '/ai/recommendations',
    insights: '/ai/insights',
    predictions: '/ai/predictions',
    models: '/ai/models',
  },
  
  // Notifications
  notifications: {
    list: '/notifications',
    create: '/notifications',
    update: (id: string) => `/notifications/${id}`,
    delete: (id: string) => `/notifications/${id}`,
    mark_read: (id: string) => `/notifications/${id}/read`,
    mark_all_read: '/notifications/mark-all-read',
    preferences: '/notifications/preferences',
    channels: '/notifications/channels',
  },
  
  // Settings
  settings: {
    profile: '/settings/profile',
    preferences: '/settings/preferences',
    security: '/settings/security',
    integrations: '/settings/integrations',
    api_keys: '/settings/api-keys',
    audit_logs: '/settings/audit-logs',
  },
  
  // Analytics
  analytics: {
    events: '/analytics/events',
    metrics: '/analytics/metrics',
    reports: '/analytics/reports',
    export: '/analytics/export',
  },
  
  // Azure Integration
  azure: {
    subscriptions: '/azure/subscriptions',
    resource_groups: '/azure/resource-groups',
    resources: '/azure/resources',
    policies: '/azure/policies',
    compliance: '/azure/compliance',
    costs: '/azure/costs',
    recommendations: '/azure/recommendations',
    alerts: '/azure/alerts',
  },
  
  // File Upload
  upload: {
    file: '/upload/file',
    bulk: '/upload/bulk',
    template: '/upload/template',
  },
  
  // Health Check
  health: '/health',
  
  // System
  system: {
    info: '/system/info',
    status: '/system/status',
    metrics: '/system/metrics',
  },
} as const

// WebSocket Event Types
export const wsEvents = {
  // Connection
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  
  // Authentication
  AUTHENTICATE: 'authenticate',
  AUTHENTICATED: 'authenticated',
  UNAUTHENTICATED: 'unauthenticated',
  
  // Notifications
  NOTIFICATION: 'notification',
  NOTIFICATION_READ: 'notification_read',
  NOTIFICATION_DELETE: 'notification_delete',
  
  // Real-time Updates
  POLICY_UPDATE: 'policy_update',
  RESOURCE_UPDATE: 'resource_update',
  COST_UPDATE: 'cost_update',
  ALERT_UPDATE: 'alert_update',
  
  // Chat/Conversation
  MESSAGE: 'message',
  TYPING: 'typing',
  STOP_TYPING: 'stop_typing',
  
  // System
  SYSTEM_STATUS: 'system_status',
  MAINTENANCE: 'maintenance',
  
  // Error Handling
  ERROR: 'error',
  RECONNECT: 'reconnect',
} as const

// HTTP Status Codes
export const httpStatus = {
  OK: 200,
  CREATED: 201,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  TOO_MANY_REQUESTS: 429,
  INTERNAL_SERVER_ERROR: 500,
  BAD_GATEWAY: 502,
  SERVICE_UNAVAILABLE: 503,
  GATEWAY_TIMEOUT: 504,
} as const

// Request Headers
export const headers = {
  CONTENT_TYPE: 'Content-Type',
  AUTHORIZATION: 'Authorization',
  ACCEPT: 'Accept',
  X_REQUESTED_WITH: 'X-Requested-With',
  X_CSRF_TOKEN: 'X-CSRF-Token',
  X_API_KEY: 'X-API-Key',
  X_CLIENT_ID: 'X-Client-ID',
  X_REQUEST_ID: 'X-Request-ID',
} as const

// Content Types
export const contentTypes = {
  JSON: 'application/json',
  FORM_DATA: 'multipart/form-data',
  URL_ENCODED: 'application/x-www-form-urlencoded',
  TEXT: 'text/plain',
  XML: 'application/xml',
  CSV: 'text/csv',
  PDF: 'application/pdf',
  EXCEL: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
} as const

// Cache Configuration
export const cacheConfig = {
  defaultTTL: 5 * 60 * 1000, // 5 minutes
  longTTL: 60 * 60 * 1000, // 1 hour
  shortTTL: 30 * 1000, // 30 seconds
  maxSize: 100, // Maximum number of cached items
}

// Pagination Configuration
export const paginationConfig = {
  defaultPage: 1,
  defaultLimit: 25,
  maxLimit: 100,
  showSizeChanger: true,
  showQuickJumper: true,
  showTotal: (total: number, range: [number, number]) =>
    `${range[0]}-${range[1]} of ${total} items`,
}