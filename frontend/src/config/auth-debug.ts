import { Configuration } from '@azure/msal-browser'
import { env } from './environment'

// Debug MSAL configuration
export const debugAuthConfig = () => {
  console.group('🔐 Azure AD Configuration Debug')
  console.log('Environment:', env.NODE_ENV)
  console.log('Client ID:', env.AZURE_CLIENT_ID)
  console.log('Tenant ID:', env.AZURE_TENANT_ID)
  console.log('Redirect URI:', env.AZURE_REDIRECT_URI)
  console.log('API Base URL:', env.API_BASE_URL)
  
  // Check if we have valid GUIDs
  const clientIdValid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(env.AZURE_CLIENT_ID || '')
  const tenantIdValid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(env.AZURE_TENANT_ID || '')
  
  console.log('Client ID Valid GUID:', clientIdValid)
  console.log('Tenant ID Valid GUID:', tenantIdValid)
  
  if (!clientIdValid || !tenantIdValid) {
    console.error('❌ Invalid Azure AD configuration - GUIDs not properly formatted')
  } else {
    console.log('✅ Azure AD configuration looks valid')
  }
  
  console.groupEnd()
}