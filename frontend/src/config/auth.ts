import { Configuration, PopupRequest, RedirectRequest } from '@azure/msal-browser'
import { env } from './environment'

// MSAL configuration
export const msalConfig: Configuration = {
  auth: {
    clientId: env.AZURE_CLIENT_ID || '',
    authority: `https://login.microsoftonline.com/${env.AZURE_TENANT_ID || 'common'}`,
    redirectUri: env.AZURE_REDIRECT_URI || window.location.origin,
    postLogoutRedirectUri: window.location.origin,
    navigateToLoginRequestUrl: false,
  },
  cache: {
    cacheLocation: 'localStorage',
    storeAuthStateInCookie: false,
  },
  system: {
    loggerOptions: {
      loggerCallback: (level, message, containsPii) => {
        if (containsPii) {
          return
        }
        switch (level) {
          case 0: // LogLevel.Error
            console.error(message)
            break
          case 1: // LogLevel.Warning
            console.warn(message)
            break
          case 2: // LogLevel.Info
            console.info(message)
            break
          case 3: // LogLevel.Verbose
            console.debug(message)
            break
          default:
            console.log(message)
        }
      },
    },
  },
}

// Add scopes for ID token to be used at Microsoft identity platform endpoints
export const loginRequest: RedirectRequest = {
  scopes: [
    'openid',
    'profile',
    'email',
    'User.Read',
    'https://graph.microsoft.com/User.Read',
  ],
}

// Add scopes for access token to be used at Azure Resource Manager endpoints
export const armRequest: PopupRequest = {
  scopes: [
    'https://management.azure.com/user_impersonation',
    'https://management.core.windows.net/user_impersonation',
  ],
}

// Add scopes for Microsoft Graph API
export const graphRequest: PopupRequest = {
  scopes: [
    'https://graph.microsoft.com/User.Read',
    'https://graph.microsoft.com/User.ReadBasic.All',
    'https://graph.microsoft.com/Directory.Read.All',
  ],
}

// Silent request for token renewal
export const silentRequest = {
  scopes: [
    'openid',
    'profile',
    'email',
    'User.Read',
  ],
  account: null as any,
}

// Token scopes for different services
export const tokenScopes = {
  arm: ['https://management.azure.com/user_impersonation'],
  graph: ['https://graph.microsoft.com/User.Read'],
  api: [`api://${env.AZURE_CLIENT_ID}/access_as_user`],
}

// Auth endpoints
export const authEndpoints = {
  login: '/auth/login',
  logout: '/auth/logout',
  refresh: '/auth/refresh',
  profile: '/auth/profile',
}

// Role-based access control
export const roles = {
  ADMIN: 'admin',
  USER: 'user',
  VIEWER: 'viewer',
  POLICY_MANAGER: 'policy_manager',
  COST_MANAGER: 'cost_manager',
  SECURITY_MANAGER: 'security_manager',
} as const

export type UserRole = (typeof roles)[keyof typeof roles]

// Permission groups
export const permissions = {
  DASHBOARD_VIEW: 'dashboard:view',
  DASHBOARD_EDIT: 'dashboard:edit',
  POLICIES_VIEW: 'policies:view',
  POLICIES_CREATE: 'policies:create',
  POLICIES_EDIT: 'policies:edit',
  POLICIES_DELETE: 'policies:delete',
  RESOURCES_VIEW: 'resources:view',
  RESOURCES_MANAGE: 'resources:manage',
  COSTS_VIEW: 'costs:view',
  COSTS_MANAGE: 'costs:manage',
  SETTINGS_VIEW: 'settings:view',
  SETTINGS_EDIT: 'settings:edit',
  USERS_VIEW: 'users:view',
  USERS_MANAGE: 'users:manage',
  AUDIT_VIEW: 'audit:view',
  ANALYTICS_VIEW: 'analytics:view',
  ANALYTICS_EDIT: 'analytics:edit',
} as const

export type Permission = (typeof permissions)[keyof typeof permissions]

// Role-permission mapping
export const rolePermissions: Record<UserRole, Permission[]> = {
  [roles.ADMIN]: Object.values(permissions),
  [roles.USER]: [
    permissions.DASHBOARD_VIEW,
    permissions.POLICIES_VIEW,
    permissions.RESOURCES_VIEW,
    permissions.COSTS_VIEW,
    permissions.SETTINGS_VIEW,
    permissions.ANALYTICS_VIEW,
  ],
  [roles.VIEWER]: [
    permissions.DASHBOARD_VIEW,
    permissions.POLICIES_VIEW,
    permissions.RESOURCES_VIEW,
    permissions.COSTS_VIEW,
    permissions.ANALYTICS_VIEW,
  ],
  [roles.POLICY_MANAGER]: [
    permissions.DASHBOARD_VIEW,
    permissions.POLICIES_VIEW,
    permissions.POLICIES_CREATE,
    permissions.POLICIES_EDIT,
    permissions.POLICIES_DELETE,
    permissions.RESOURCES_VIEW,
    permissions.AUDIT_VIEW,
    permissions.ANALYTICS_VIEW,
  ],
  [roles.COST_MANAGER]: [
    permissions.DASHBOARD_VIEW,
    permissions.COSTS_VIEW,
    permissions.COSTS_MANAGE,
    permissions.RESOURCES_VIEW,
    permissions.ANALYTICS_VIEW,
  ],
  [roles.SECURITY_MANAGER]: [
    permissions.DASHBOARD_VIEW,
    permissions.POLICIES_VIEW,
    permissions.POLICIES_CREATE,
    permissions.POLICIES_EDIT,
    permissions.RESOURCES_VIEW,
    permissions.AUDIT_VIEW,
    permissions.ANALYTICS_VIEW,
  ],
}

// Helper function to check if user has permission
export const hasPermission = (userRole: UserRole, permission: Permission): boolean => {
  return rolePermissions[userRole]?.includes(permission) || false
}

// Helper function to check if user has any of the permissions
export const hasAnyPermission = (userRole: UserRole, permissions: Permission[]): boolean => {
  return permissions.some(permission => hasPermission(userRole, permission))
}

// Helper function to check if user has all permissions
export const hasAllPermissions = (userRole: UserRole, permissions: Permission[]): boolean => {
  return permissions.every(permission => hasPermission(userRole, permission))
}