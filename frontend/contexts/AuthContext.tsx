'use client'

import React, { createContext, useContext, useEffect, useState } from 'react'
import { 
  PublicClientApplication, 
  AccountInfo, 
  AuthenticationResult,
  InteractionRequiredAuthError,
  SilentRequest
} from '@azure/msal-browser'
import { MsalProvider, useMsal, useAccount, useIsAuthenticated } from '@azure/msal-react'
import { msalConfig, loginRequest, apiRequest } from '../lib/auth-config'

// Create the MSAL instance only on client side
let msalInstance: PublicClientApplication | null = null;

if (typeof window !== 'undefined') {
  msalInstance = new PublicClientApplication(msalConfig);
  
  // Initialize MSAL
  msalInstance.initialize().then(() => {
    // Check if there's an account already signed in
    const accounts = msalInstance!.getAllAccounts()
    if (accounts.length > 0) {
      msalInstance!.setActiveAccount(accounts[0])
    }
  }).catch((error) => {
    console.error('MSAL initialization failed:', error);
  });
}

interface AuthContextType {
  isAuthenticated: boolean
  user: AccountInfo | null
  login: () => Promise<void>
  logout: () => Promise<void>
  getAccessToken: () => Promise<string>
  loading: boolean
  error: string | null
}

// Default context for SSR
const defaultAuthContext: AuthContextType = {
  isAuthenticated: false,
  user: null,
  login: async () => { console.log('Login not available during SSR') },
  logout: async () => { console.log('Logout not available during SSR') },
  getAccessToken: async () => '',
  loading: false,
  error: null
}

const AuthContext = createContext<AuthContextType>(defaultAuthContext)

export const useAuth = () => {
  const context = useContext(AuthContext)
  // Return context (will have default value if not within provider)
  return context
}

interface AuthProviderInnerProps {
  children: React.ReactNode
}

const AuthProviderInner: React.FC<AuthProviderInnerProps> = ({ children }) => {
  const { instance, accounts } = useMsal()
  const isAuthenticated = useIsAuthenticated()
  const account = useAccount(accounts[0] || {})

  // Demo mode if Azure config is missing or explicitly enabled
  const demoMode = typeof window !== 'undefined' && (
    !process.env.NEXT_PUBLIC_AZURE_CLIENT_ID ||
    !process.env.NEXT_PUBLIC_AZURE_TENANT_ID ||
    process.env.NEXT_PUBLIC_DEMO_MODE === 'true'
  )

  const [demoUser, setDemoUser] = useState<AccountInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const login = async () => {
    setLoading(true)
    setError(null)
    
    try {
      // Demo mode bypass for local development
      if (demoMode) {
        const demoAccount: AccountInfo = {
          username: 'demo@policycortex.local',
          name: 'Demo User',
          homeAccountId: 'demo-account',
          environment: 'demo',
          tenantId: 'demo-tenant',
          localAccountId: 'demo-local'
        }
        setDemoUser(demoAccount)
        console.log('Demo mode: Login bypassed')
        return
      }
      
      const loginResponse = await instance.loginPopup(loginRequest)
      instance.setActiveAccount(loginResponse.account)
      console.log('Login successful:', loginResponse)
    } catch (err: any) {
      console.error('Login failed:', err)
      setError(err.message || 'Login failed')
      
      // Fallback to demo mode on error in development
      if (process.env.NODE_ENV === 'development' || demoMode) {
        const demoAccount: AccountInfo = {
          username: 'demo@policycortex.local',
          name: 'Demo User (Auth Failed)',
          homeAccountId: 'demo-account',
          environment: 'demo',
          tenantId: 'demo-tenant',
          localAccountId: 'demo-local'
        }
        setDemoUser(demoAccount)
        console.log('Auth failed - using demo mode')
        setError(null) // Clear error for demo mode
      }
    } finally {
      setLoading(false)
    }
  }

  const logout = async () => {
    setLoading(true)
    
    try {
      if (demoMode) {
        setDemoUser(null)
      } else {
        await instance.logoutPopup({
          postLogoutRedirectUri: window.location.origin,
          mainWindowRedirectUri: window.location.origin
        })
      }
    } catch (err: any) {
      console.error('Logout failed:', err)
      setError(err.message || 'Logout failed')
    } finally {
      setLoading(false)
    }
  }

  const getAccessToken = async (): Promise<string> => {
    if (demoMode) {
      return ''
    }
    if (!account) {
      // Local/dev: return empty string so backend can allow optional auth
      return ''
    }

    const silentRequest: SilentRequest = {
      ...apiRequest,
      account: account
    }

    try {
      const response = await instance.acquireTokenSilent(silentRequest)
      return response.accessToken
    } catch (error) {
      if (error instanceof InteractionRequiredAuthError) {
        // Fallback to interactive method
        const response = await instance.acquireTokenPopup(apiRequest)
        return response.accessToken
      }
      // Graceful dev fallback: no token
      return ''
    }
  }

  const contextValue: AuthContextType = {
    isAuthenticated: demoMode ? !!demoUser : isAuthenticated,
    user: demoMode ? demoUser : account,
    login,
    logout,
    getAccessToken,
    loading,
    error
  }

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  )
}

interface AuthProviderProps {
  children: React.ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  // Return with default context during SSR
  if (typeof window === 'undefined' || !msalInstance) {
    return (
      <AuthContext.Provider value={defaultAuthContext}>
        {children}
      </AuthContext.Provider>
    );
  }
  
  return (
    <MsalProvider instance={msalInstance}>
      <AuthProviderInner>
        {children}
      </AuthProviderInner>
    </MsalProvider>
  )
}

// Hook for making authenticated API calls
export const useAuthenticatedFetch = () => {
  const { getAccessToken } = useAuth()

  const authenticatedFetch = async (url: string, options: RequestInit = {}) => {
    try {
      const token = await getAccessToken()
      
      const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
        ...options.headers,
      }

      const response = await fetch(url, {
        ...options,
        headers,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return response
    } catch (error) {
      console.error('Authenticated fetch failed:', error)
      throw error
    }
  }

  return authenticatedFetch
}