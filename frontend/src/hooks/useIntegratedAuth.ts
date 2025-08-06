/**
 * React Hook for Integrated Authentication System
 * Provides zero-configuration authentication with automatic organization detection
 */

import { useState, useEffect, useCallback } from 'react'
import authService, { 
  OrganizationDetectionResponse, 
  LoginRequest, 
  UserInfo 
} from '../services/integratedAuthService'

export interface UseIntegratedAuthReturn {
  // Authentication state
  isAuthenticated: boolean
  isLoading: boolean
  user: UserInfo | null
  error: string | null

  // Organization detection
  organizationConfig: OrganizationDetectionResponse | null
  isDetectingOrganization: boolean
  
  // Authentication methods
  detectOrganization: (email: string) => Promise<OrganizationDetectionResponse>
  login: (loginRequest: LoginRequest) => Promise<void>
  logout: () => Promise<void>
  refreshUser: () => Promise<void>
  
  // Permissions and roles
  hasPermission: (permission: string) => boolean
  hasRole: (roles: string | string[]) => boolean
  
  // Tenant context
  tenantId: string | null
  organization: string | null
  
  // Utilities
  clearError: () => void
}

export const useIntegratedAuth = (): UseIntegratedAuthReturn => {
  const [isAuthenticated, setIsAuthenticated] = useState(authService.isAuthenticated())
  const [isLoading, setIsLoading] = useState(false)
  const [user, setUser] = useState<UserInfo | null>(authService.getUser())
  const [error, setError] = useState<string | null>(null)
  const [organizationConfig, setOrganizationConfig] = useState<OrganizationDetectionResponse | null>(null)
  const [isDetectingOrganization, setIsDetectingOrganization] = useState(false)

  // Initialize authentication state
  useEffect(() => {
    const initAuth = async () => {
      if (authService.isAuthenticated()) {
        try {
          setIsLoading(true)
          const currentUser = await authService.getCurrentUser()
          setUser(currentUser)
          setIsAuthenticated(true)
        } catch (e) {
          console.error('Failed to get current user:', e)
          setIsAuthenticated(false)
          setUser(null)
        } finally {
          setIsLoading(false)
        }
      }
    }

    initAuth()
  }, [])

  const detectOrganization = useCallback(async (email: string): Promise<OrganizationDetectionResponse> => {
    try {
      setIsDetectingOrganization(true)
      setError(null)
      
      const orgConfig = await authService.detectOrganization(email)
      setOrganizationConfig(orgConfig)
      
      return orgConfig
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'Organization detection failed'
      setError(errorMessage)
      throw e
    } finally {
      setIsDetectingOrganization(false)
    }
  }, [])

  const login = useCallback(async (loginRequest: LoginRequest): Promise<void> => {
    try {
      setIsLoading(true)
      setError(null)
      
      const loginResponse = await authService.login(loginRequest)
      
      setUser(loginResponse.user)
      setIsAuthenticated(true)
      
      // Clear organization config after successful login
      setOrganizationConfig(null)
      
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'Login failed'
      setError(errorMessage)
      throw e
    } finally {
      setIsLoading(false)
    }
  }, [])

  const logout = useCallback(async (): Promise<void> => {
    try {
      setIsLoading(true)
      await authService.logout()
      
      setUser(null)
      setIsAuthenticated(false)
      setOrganizationConfig(null)
      setError(null)
      
    } catch (e) {
      console.error('Logout failed:', e)
      // Even if logout request fails, clear local state
      setUser(null)
      setIsAuthenticated(false)
      setOrganizationConfig(null)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const refreshUser = useCallback(async (): Promise<void> => {
    if (!authService.isAuthenticated()) {
      return
    }

    try {
      setIsLoading(true)
      const currentUser = await authService.getCurrentUser()
      setUser(currentUser)
      setIsAuthenticated(true)
    } catch (e) {
      console.error('Failed to refresh user:', e)
      setError('Failed to refresh user information')
      // Don't clear authentication on refresh failure
    } finally {
      setIsLoading(false)
    }
  }, [])

  const hasPermission = useCallback((permission: string): boolean => {
    return authService.hasPermission(permission)
  }, [user])

  const hasRole = useCallback((roles: string | string[]): boolean => {
    return authService.hasRole(roles)
  }, [user])

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return {
    // Authentication state
    isAuthenticated,
    isLoading,
    user,
    error,

    // Organization detection
    organizationConfig,
    isDetectingOrganization,

    // Authentication methods
    detectOrganization,
    login,
    logout,
    refreshUser,

    // Permissions and roles
    hasPermission,
    hasRole,

    // Tenant context
    tenantId: user?.tenant_id || null,
    organization: user?.organization || null,

    // Utilities
    clearError
  }
}

export default useIntegratedAuth