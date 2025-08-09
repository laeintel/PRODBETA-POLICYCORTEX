/**
 * MSAL Configuration for PolicyCortex
 * Tenant-scoped Azure AD authentication for managing entire organization
 */

import { Configuration, LogLevel } from '@azure/msal-browser'

// Azure AD Application Configuration
export const msalConfig: Configuration = {
  auth: {
    clientId: process.env.NEXT_PUBLIC_AZURE_CLIENT_ID || '1ecc95d1-e5bb-43e2-9324-30a17cb6b01c', // From your GitHub secrets
    authority: `https://login.microsoftonline.com/${process.env.NEXT_PUBLIC_AZURE_TENANT_ID || '9ef5b184-d371-462a-bc75-5024ce8baff7'}`, // Your tenant ID
    redirectUri: typeof window !== 'undefined' ? window.location.origin : '/',
    postLogoutRedirectUri: typeof window !== 'undefined' ? window.location.origin : '/',
    navigateToLoginRequestUrl: false,
  },
  cache: {
    cacheLocation: 'sessionStorage', // or 'localStorage'
    storeAuthStateInCookie: false, // Set to true for IE11 or Edge
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