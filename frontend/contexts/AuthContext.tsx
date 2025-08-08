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

// Create the MSAL instance
const msalInstance = new PublicClientApplication(msalConfig)

// Initialize MSAL
msalInstance.initialize().then(() => {
  // Check if there's an account already signed in
  const accounts = msalInstance.getAllAccounts()
  if (accounts.length > 0) {
    msalInstance.setActiveAccount(accounts[0])
  }
})

interface AuthContextType {
  isAuthenticated: boolean
  user: AccountInfo | null
  login: () => Promise<void>
  logout: () => Promise<void>
  getAccessToken: () => Promise<string>
  loading: boolean
  error: string | null
}

const AuthContext = createContext<AuthContextType | null>(null)

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}

interface AuthProviderInnerProps {
  children: React.ReactNode
}

const AuthProviderInner: React.FC<AuthProviderInnerProps> = ({ children }) => {
  const { instance, accounts } = useMsal()
  const isAuthenticated = useIsAuthenticated()
  const account = useAccount(accounts[0] || {})
  
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const login = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const loginResponse = await instance.loginPopup(loginRequest)
      instance.setActiveAccount(loginResponse.account)
      console.log('Login successful:', loginResponse)
    } catch (err: any) {
      console.error('Login failed:', err)
      setError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  const logout = async () => {
    setLoading(true)
    
    try {
      await instance.logoutPopup({
        postLogoutRedirectUri: window.location.origin,
        mainWindowRedirectUri: window.location.origin
      })
    } catch (err: any) {
      console.error('Logout failed:', err)
      setError(err.message || 'Logout failed')
    } finally {
      setLoading(false)
    }
  }

  const getAccessToken = async (): Promise<string> => {
    if (!account) {
      throw new Error('No account found')
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
      throw error
    }
  }

  const contextValue: AuthContextType = {
    isAuthenticated,
    user: account,
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