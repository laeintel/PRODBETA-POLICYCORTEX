/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

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
    // Handle redirect response from Azure AD
    msalInstance!.handleRedirectPromise().then((response) => {
      if (response && response.account) {
        msalInstance!.setActiveAccount(response.account)
        // Don't auto-redirect - let components handle navigation
      } else {
        // Check if there's an account already signed in
        const accounts = msalInstance!.getAllAccounts()
        if (accounts.length > 0) {
          msalInstance!.setActiveAccount(accounts[0])
        }
      }
    }).catch((error) => {
      console.error('Error handling redirect:', error)
    })
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

  // Demo mode only in development with explicit flag
  const demoMode = typeof window !== 'undefined' && 
    process.env.NODE_ENV === 'development' &&
    process.env.NEXT_PUBLIC_DEMO_MODE === 'true'

  // Initialize demo user immediately if in demo mode
  const initialDemoUser: AccountInfo | null = demoMode ? {
    username: 'demo@policycortex.local',
    name: 'Demo User',
    homeAccountId: 'demo-account',
    environment: 'development',
    tenantId: 'demo-tenant',
    localAccountId: 'demo-local'
  } : null

  const [demoUser, setDemoUser] = useState<AccountInfo | null>(initialDemoUser)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Initialize demo session on mount if in demo mode
  useEffect(() => {
    if (demoMode && typeof window !== 'undefined') {
      // Set demo session cookies via API
      fetch('/api/auth/demo', {
        method: 'POST',
        credentials: 'include'
      }).then(() => {
        console.log('Demo session initialized')
      }).catch(err => {
        console.error('Failed to initialize demo session:', err)
      })
    }
  }, [demoMode])

  const login = async () => {
    setLoading(true)
    setError(null)
    
    try {
      // Demo mode bypass for development only
      if (demoMode) {
        console.warn('🚨 SECURITY WARNING: Demo mode is active! This bypasses authentication and should NEVER be used in production!')
        const demoAccount: AccountInfo = {
          username: 'demo@policycortex.local',
          name: 'Demo User (DEV)',
          homeAccountId: 'demo-account',
          environment: 'development',
          tenantId: 'demo-tenant',
          localAccountId: 'demo-local'
        }
        setDemoUser(demoAccount)
        console.log('Demo mode: Authentication bypassed for development')
        setLoading(false)
        return
      }
      
      console.log('Starting Azure AD login with config:', {
        clientId: msalConfig.auth.clientId,
        authority: msalConfig.auth.authority,
        redirectUri: msalConfig.auth.redirectUri
      })
      
      // Use popup to avoid redirect loop issues
      const loginResponse = await instance.loginPopup(loginRequest)
      instance.setActiveAccount(loginResponse.account)
      console.log('Login successful via popup:', loginResponse)
      
      // Acquire access token for API calls
      let accessToken = '';
      try {
        const tokenResp = await instance.acquireTokenSilent(apiRequest);
        accessToken = tokenResp?.accessToken || '';
      } catch (e) {
        try {
          const tokenResp = await instance.acquireTokenPopup(apiRequest);
          accessToken = tokenResp?.accessToken || '';
        } catch {
          console.warn('Could not acquire access token');
        }
      }
      
      // Create secure session with httpOnly cookies
      const sessionResponse = await fetch('/api/auth/session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          accessToken,
          idToken: loginResponse.idToken,
          user: loginResponse.account
        })
      });
      
      if (!sessionResponse.ok) {
        throw new Error('Failed to create session');
      }
      
      // Don't redirect here - let the component handle navigation
    } catch (err) {
      console.error('Login failed:', err)
      const errorMessage = (err as { errorMessage?: string; message?: string })?.errorMessage || (err as Error)?.message || 'Login failed'
      setError(errorMessage)
      
      // Only fallback to demo mode if explicitly enabled
      if (demoMode) {
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
      } else {
        // Keep the error for real authentication failures
        console.error('Authentication failed. Please check your Azure AD configuration.')
      }
    } finally {
      setLoading(false)
    }
  }

  const logout = async () => {
    setLoading(true)
    
    try {
      // Destroy server session and clear cookies
      await fetch('/api/auth/session', {
        method: 'DELETE',
        credentials: 'include'
      })
      
      if (demoMode) {
        setDemoUser(null)
      } else {
        await instance.logoutPopup({
          postLogoutRedirectUri: window.location.origin,
          mainWindowRedirectUri: window.location.origin
        })
      }
    } catch (err) {
      console.error('Logout failed:', err)
      setError((err as Error)?.message || 'Logout failed')
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

// Session refresh hook - ensures tokens stay fresh
export const SessionRefresher: React.FC = () => {
  const { user } = useAuth();
  
  useEffect(() => {
    if (!user || typeof window === 'undefined') return;
    
    // Check session status every 10 minutes
    const interval = setInterval(async () => {
      try {
        const response = await fetch('/api/auth/session', {
          credentials: 'include'
        });
        const data = await response.json();
        
        if (!data.authenticated && user) {
          // Session expired, trigger re-authentication
          console.warn('Session expired, please log in again');
          // You might want to trigger a logout or show a notification here
        }
      } catch (error) {
        console.error('Failed to check session:', error);
      }
    }, 10 * 60 * 1000); // 10 minutes
    
    return () => clearInterval(interval);
  }, [user]);
  
  return null;
}