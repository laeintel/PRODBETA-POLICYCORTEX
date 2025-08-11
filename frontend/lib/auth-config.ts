/**
 * MSAL Configuration for PolicyCortex
 * Tenant-scoped Azure AD authentication for managing entire organization
 */

import { Configuration, LogLevel } from '@azure/msal-browser'

// Azure AD Application Configuration
const clientId = process.env.NEXT_PUBLIC_AZURE_CLIENT_ID || ''
const tenantId = process.env.NEXT_PUBLIC_AZURE_TENANT_ID || ''
const redirectUri = typeof window !== 'undefined'
  ? (process.env.NEXT_PUBLIC_MSAL_REDIRECT_URI || window.location.origin)
  : undefined
const postLogoutRedirectUri = typeof window !== 'undefined'
  ? (process.env.NEXT_PUBLIC_MSAL_POST_LOGOUT_REDIRECT_URI || window.location.origin)
  : undefined
// Optional: strict audience scope for Core API
export const coreApiScope = process.env.NEXT_PUBLIC_CORE_API_SCOPE
  || (clientId ? `api://${clientId}/access_as_user` : '')

// In local/dev without Azure config, we allow running without MSAL blocking the UI
if (typeof window !== 'undefined' && (!clientId || !tenantId)) {
  console.warn('MSAL config missing client/tenant; running in local dev without Azure auth.')
}

export const msalConfig: Configuration = {
  auth: {
    clientId,
    authority: tenantId ? `https://login.microsoftonline.com/${tenantId}` : undefined,
    redirectUri,
    postLogoutRedirectUri,
    navigateToLoginRequestUrl: false,
  },
  cache: {
    cacheLocation: 'sessionStorage', // or 'localStorage'
    storeAuthStateInCookie: false, // Set true only for legacy browsers
  },
  system: {
    loggerOptions: {
      loggerCallback: (level, message, containsPii) => {
        if (containsPii) return
        switch (level) {
          case LogLevel.Error:
            console.error(message)
            return
          case LogLevel.Info:
            console.info(message)
            return
          case LogLevel.Verbose:
            console.debug(message)
            return
          case LogLevel.Warning:
            console.warn(message)
            return
          default:
            return
        }
      },
    },
  },
}

// Basic permissions for PolicyCortex (can be expanded with admin consent)
// Note: Azure AD only allows scopes from one resource per request
export const loginRequest = {
  scopes: [
    // Basic profile and OpenID Connect scopes (same resource)
    'openid',
    'profile',
    'offline_access',
    'User.Read' // Microsoft Graph user profile
  ]
}

// Azure Management API request (separate from Graph API)
export const azureManagementRequest = {
  scopes: [
    'https://management.azure.com/user_impersonation'
  ]
}

// Core API request (strict audience) – only if scope configured
export const coreApiRequest = coreApiScope
  ? { scopes: [coreApiScope] }
  : null as any

// Microsoft Graph API request (separate from Azure Management)
export const graphApiRequest = {
  scopes: [
    'User.Read',
    'Directory.Read.All',
    'Organization.Read.All', 
    'GroupMember.Read.All',
    'SecurityEvents.Read.All',
    'Policy.Read.All'
  ]
}

// API request configuration for backend calls (Azure Management)
export const apiRequest = {
  scopes: [
    'https://management.azure.com/user_impersonation'
  ]
}

// Graph API configuration
export const graphConfig = {
  graphMeEndpoint: 'https://graph.microsoft.com/v1.0/me',
  graphOrganizationEndpoint: 'https://graph.microsoft.com/v1.0/organization',
  graphUsersEndpoint: 'https://graph.microsoft.com/v1.0/users',
  graphGroupsEndpoint: 'https://graph.microsoft.com/v1.0/groups',
  graphDirectoryRolesEndpoint: 'https://graph.microsoft.com/v1.0/directoryRoles',
}

// Azure Management API configuration  
export const azureConfig = {
  managementEndpoint: 'https://management.azure.com',
  tenantEndpoint: (tenantId: string) => `https://management.azure.com/providers/Microsoft.Management/managementGroups/${tenantId}`,
  subscriptionsEndpoint: 'https://management.azure.com/subscriptions',
  policyEndpoint: (subscriptionId: string) => `https://management.azure.com/subscriptions/${subscriptionId}/providers/Microsoft.Authorization`,
  costManagementEndpoint: (scope: string) => `https://management.azure.com/${scope}/providers/Microsoft.CostManagement`
}