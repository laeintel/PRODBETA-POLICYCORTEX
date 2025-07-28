import { PublicClientApplication } from '@azure/msal-browser'
import { msalConfig } from '@/config/auth'

let msalInstance: PublicClientApplication | null = null

export const initializeMsal = async (): Promise<PublicClientApplication> => {
  if (msalInstance) {
    return msalInstance
  }

  // Wait for config.js to be loaded
  let attempts = 0
  while (!(window as any).POLICYCORTEX_CONFIG && attempts < 50) {
    await new Promise(resolve => setTimeout(resolve, 100))
    attempts++
  }

  if (!(window as any).POLICYCORTEX_CONFIG) {
    console.error('Failed to load config.js after 5 seconds')
    throw new Error('Configuration not loaded')
  }

  console.log('Creating MSAL instance with config:', {
    clientId: msalConfig.auth.clientId,
    authority: msalConfig.auth.authority,
    redirectUri: msalConfig.auth.redirectUri
  })

  msalInstance = new PublicClientApplication(msalConfig)
  
  try {
    await msalInstance.initialize()
    console.log('MSAL initialized successfully')
    
    // Handle redirect
    const response = await msalInstance.handleRedirectPromise()
    if (response) {
      console.log('Redirect response received:', response)
    }
    
    // Make instance available globally for debugging
    (window as any).msalInstance = msalInstance
    
    return msalInstance
  } catch (error) {
    console.error('MSAL initialization error:', error)
    throw error
  }
}

export const getMsalInstance = (): PublicClientApplication => {
  if (!msalInstance) {
    throw new Error('MSAL not initialized. Call initializeMsal() first.')
  }
  return msalInstance
}