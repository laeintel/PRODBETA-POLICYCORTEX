import { useState, useEffect, useCallback } from 'react'
import { useMsal } from '@azure/msal-react'
import { AccountInfo, InteractionRequiredAuthError } from '@azure/msal-browser'
import { loginRequest, silentRequest, tokenScopes } from '@/config/auth'
import { useAuthStore } from '@/store/authStore'
import { User, UserRole } from '@/types'
import { authService } from '@/services/authService'
import toast from 'react-hot-toast'

export const useAuth = () => {
  const { instance, accounts, inProgress } = useMsal()
  const { user, isAuthenticated, setUser, setAuthenticated, logout: logoutStore } = useAuthStore()
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Initialize authentication
  const initialize = useCallback(async () => {
    try {
      setIsLoading(true)
      setError(null)

      // Check if we have an active account
      if (accounts.length > 0) {
        const account = accounts[0]
        await acquireTokenSilent(account)
      } else {
        // No active account, user needs to login
        setAuthenticated(false)
      }
    } catch (error) {
      console.error('Auth initialization error:', error)
      setError('Failed to initialize authentication')
      setAuthenticated(false)
    } finally {
      setIsLoading(false)
    }
  }, [accounts, setAuthenticated])

  // Login with redirect
  const login = useCallback(async () => {
    try {
      setIsLoading(true)
      setError(null)

      await instance.loginRedirect({
        ...loginRequest,
        prompt: 'select_account',
      })
    } catch (error) {
      console.error('Login error:', error)
      setError('Failed to login')
      toast.error('Login failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }, [instance])

  // Login with popup
  const loginPopup = useCallback(async () => {
    try {
      setIsLoading(true)
      setError(null)

      const result = await instance.loginPopup({
        ...loginRequest,
        prompt: 'select_account',
      })

      if (result.account) {
        await acquireTokenSilent(result.account)
      }
    } catch (error) {
      console.error('Login popup error:', error)
      setError('Failed to login')
      toast.error('Login failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }, [instance])

  // Logout
  const logout = useCallback(async () => {
    try {
      setIsLoading(true)
      
      // Clear local state
      logoutStore()
      
      // Logout from MSAL
      await instance.logoutRedirect({
        postLogoutRedirectUri: window.location.origin,
      })
    } catch (error) {
      console.error('Logout error:', error)
      setError('Failed to logout')
      toast.error('Logout failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }, [instance, logoutStore])

  // Acquire token silently
  const acquireTokenSilent = useCallback(async (account: AccountInfo) => {
    try {
      const silentTokenRequest = {
        ...silentRequest,
        account,
      }

      const response = await instance.acquireTokenSilent(silentTokenRequest)
      
      if (response.accessToken) {
        // Get user profile from backend
        const userProfile = await authService.getProfile(response.accessToken)
        setUser(userProfile)
        setAuthenticated(true)
        return response.accessToken
      }
    } catch (error) {
      if (error instanceof InteractionRequiredAuthError) {
        // Token expired or requires interaction
        console.warn('Token expired, requiring user interaction')
        setAuthenticated(false)
      } else {
        console.error('Silent token acquisition error:', error)
        setError('Failed to acquire token')
      }
      throw error
    }
  }, [instance, setUser, setAuthenticated])

  // Get access token for API calls
  const getAccessToken = useCallback(async (scopes: string[] = tokenScopes.api): Promise<string | null> => {
    try {
      if (accounts.length === 0) {
        return null
      }

      const account = accounts[0]
      const response = await instance.acquireTokenSilent({
        scopes,
        account,
      })

      return response.accessToken
    } catch (error) {
      if (error instanceof InteractionRequiredAuthError) {
        try {
          // Try to get token with popup
          const response = await instance.acquireTokenPopup({
            scopes,
            account: accounts[0],
          })
          return response.accessToken
        } catch (popupError) {
          console.error('Token acquisition popup error:', popupError)
          return null
        }
      } else {
        console.error('Token acquisition error:', error)
        return null
      }
    }
  }, [instance, accounts])

  // Check if user has permission
  const hasPermission = useCallback((permission: string): boolean => {
    return user?.permissions?.some(p => p.name === permission) || false
  }, [user])

  // Check if user has role
  const hasRole = useCallback((role: string): boolean => {
    return user?.role?.name === role || false
  }, [user])

  // Check if user has any of the roles
  const hasAnyRole = useCallback((roles: string[]): boolean => {
    return roles.some(role => hasRole(role))
  }, [hasRole])

  // Refresh user profile
  const refreshProfile = useCallback(async () => {
    try {
      const token = await getAccessToken()
      if (token) {
        const userProfile = await authService.getProfile(token)
        setUser(userProfile)
      }
    } catch (error) {
      console.error('Failed to refresh profile:', error)
    }
  }, [getAccessToken, setUser])

  // Effect to handle account changes
  useEffect(() => {
    if (inProgress === 'none' && accounts.length > 0) {
      const account = accounts[0]
      acquireTokenSilent(account).catch(() => {
        // Handle silent token acquisition failure
        setAuthenticated(false)
      })
    }
  }, [accounts, inProgress, acquireTokenSilent, setAuthenticated])

  return {
    // State
    user,
    isAuthenticated,
    isLoading,
    error,
    accounts,
    inProgress,

    // Methods
    initialize,
    login,
    loginPopup,
    logout,
    getAccessToken,
    refreshProfile,

    // Permission checks
    hasPermission,
    hasRole,
    hasAnyRole,
  }
}