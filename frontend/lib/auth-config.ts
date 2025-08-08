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
export const loginRequest = {
  scopes: [
    // Core Azure Management (user's own access)
    'https://management.azure.com/user_impersonation',
    
    // Basic Microsoft Graph permissions  
    'https://graph.microsoft.com/User.Read'
  ]
}

// Extended permissions for tenant-wide access (requires admin consent)
export const adminConsentRequest = {
  scopes: [
    'https://management.azure.com/user_impersonation',
    'https://graph.microsoft.com/User.Read',
    'https://graph.microsoft.com/Directory.Read.All',
    'https://graph.microsoft.com/Organization.Read.All', 
    'https://graph.microsoft.com/GroupMember.Read.All',
    'https://graph.microsoft.com/SecurityEvents.Read.All',
    'https://graph.microsoft.com/Policy.Read.All'
  ]
}

// API request configuration for backend calls
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