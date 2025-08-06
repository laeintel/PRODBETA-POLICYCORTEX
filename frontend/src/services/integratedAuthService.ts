/**
 * Integrated Authentication Service for PolicyCortex Phase 1
 * Implements zero-configuration authentication with automatic organization detection
 */

import { env } from '../config/environment'

export interface OrganizationDetectionResponse {
  domain: string
  organization_name: string
  organization_type: string
  authentication_method: string
  sso_enabled: boolean
  tenant_id: string
  features: Record<string, any>
  settings: Record<string, any>
}

export interface LoginRequest {
  email: string
  password?: string
  auth_code?: string
  saml_response?: string
}

export interface LoginResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
  user: {
    id: string
    email: string
    name: string
    tenant_id: string
    organization: string
    roles: string[]
    permissions: string[]
  }
}

export interface UserInfo {
  id: string
  email: string
  name: string
  tenant_id: string
  organization: string
  roles: string[]
  permissions: string[]
}

class IntegratedAuthService {
  private apiBaseUrl: string
  private accessToken: string | null = null
  private refreshToken: string | null = null
  private user: UserInfo | null = null

  constructor() {
    this.apiBaseUrl = env.API_BASE_URL || 'http://localhost:8010'
    
    // Load tokens from localStorage
    this.accessToken = localStorage.getItem('policycortex_access_token')
    this.refreshToken = localStorage.getItem('policycortex_refresh_token')
    const userData = localStorage.getItem('policycortex_user')
    if (userData) {
      try {
        this.user = JSON.parse(userData)
      } catch (e) {
        console.error('Failed to parse user data from localStorage')
      }
    }
  }

  /**
   * Detect organization configuration from email domain
   * This is the first step in zero-configuration authentication
   */
  async detectOrganization(email: string): Promise<OrganizationDetectionResponse> {
    const response = await fetch(`${this.apiBaseUrl}/api/auth/detect-organization`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email })
    })

    if (!response.ok) {
      throw new Error(`Organization detection failed: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Login with automatic authentication method detection
   */
  async login(loginRequest: LoginRequest): Promise<LoginResponse> {
    const response = await fetch(`${this.apiBaseUrl}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(loginRequest)
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `Login failed: ${response.statusText}`)
    }

    const loginResponse: LoginResponse = await response.json()

    // Store tokens and user info
    this.accessToken = loginResponse.access_token
    this.refreshToken = loginResponse.refresh_token
    this.user = loginResponse.user

    localStorage.setItem('policycortex_access_token', this.accessToken)
    localStorage.setItem('policycortex_refresh_token', this.refreshToken)
    localStorage.setItem('policycortex_user', JSON.stringify(this.user))

    return loginResponse
  }

  /**
   * Refresh access token
   */
  async refreshAccessToken(): Promise<string> {
    if (!this.refreshToken) {
      throw new Error('No refresh token available')
    }

    const response = await fetch(`${this.apiBaseUrl}/api/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refresh_token: this.refreshToken })
    })

    if (!response.ok) {
      // Refresh token is invalid, clear storage
      this.clearAuth()
      throw new Error('Token refresh failed')
    }

    const tokens = await response.json()
    this.accessToken = tokens.access_token
    
    if (tokens.refresh_token) {
      this.refreshToken = tokens.refresh_token
      localStorage.setItem('policycortex_refresh_token', this.refreshToken)
    }
    
    localStorage.setItem('policycortex_access_token', this.accessToken)
    
    return this.accessToken
  }

  /**
   * Logout user
   */
  async logout(): Promise<void> {
    if (this.accessToken) {
      try {
        await fetch(`${this.apiBaseUrl}/api/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${this.accessToken}`,
            'Content-Type': 'application/json',
          }
        })
      } catch (e) {
        console.error('Logout request failed:', e)
      }
    }

    this.clearAuth()
  }

  /**
   * Get current user info from server
   */
  async getCurrentUser(): Promise<UserInfo> {
    const token = await this.getValidToken()
    
    const response = await fetch(`${this.apiBaseUrl}/api/auth/me`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      }
    })

    if (!response.ok) {
      throw new Error('Failed to get user info')
    }

    const user = await response.json()
    this.user = user
    localStorage.setItem('policycortex_user', JSON.stringify(user))
    
    return user
  }

  /**
   * Get valid access token, refreshing if necessary
   */
  async getValidToken(): Promise<string> {
    if (!this.accessToken) {
      throw new Error('No access token available')
    }

    // Check if token is expired (basic check)
    try {
      const tokenParts = this.accessToken.split('.')
      if (tokenParts.length === 3) {
        const payload = JSON.parse(atob(tokenParts[1]))
        const now = Math.floor(Date.now() / 1000)
        
        // If token expires in less than 5 minutes, refresh it
        if (payload.exp && payload.exp < now + 300) {
          await this.refreshAccessToken()
        }
      }
    } catch (e) {
      // If we can't parse the token, try to refresh
      await this.refreshAccessToken()
    }

    return this.accessToken!
  }

  /**
   * Make authenticated API request
   */
  async authenticatedRequest(url: string, options: RequestInit = {}): Promise<Response> {
    const token = await this.getValidToken()
    
    const headers = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
      ...options.headers
    }

    const response = await fetch(`${this.apiBaseUrl}${url}`, {
      ...options,
      headers
    })

    // If unauthorized, try refreshing token once
    if (response.status === 401 && !options.headers?.['X-Retry']) {
      try {
        await this.refreshAccessToken()
        const retryHeaders = {
          ...headers,
          'Authorization': `Bearer ${this.accessToken}`,
          'X-Retry': 'true'
        }
        
        return fetch(`${this.apiBaseUrl}${url}`, {
          ...options,
          headers: retryHeaders
        })
      } catch (e) {
        this.clearAuth()
        throw new Error('Authentication failed')
      }
    }

    return response
  }

  /**
   * Clear authentication data
   */
  private clearAuth(): void {
    this.accessToken = null
    this.refreshToken = null
    this.user = null
    
    localStorage.removeItem('policycortex_access_token')
    localStorage.removeItem('policycortex_refresh_token')
    localStorage.removeItem('policycortex_user')
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return !!this.accessToken && !!this.user
  }

  /**
   * Get current user (cached)
   */
  getUser(): UserInfo | null {
    return this.user
  }

  /**
   * Check if user has permission
   */
  hasPermission(permission: string): boolean {
    if (!this.user) return false
    
    // Global admin has all permissions
    if (this.user.permissions.includes('*')) return true
    
    // Check specific permission
    return this.user.permissions.includes(permission)
  }

  /**
   * Check if user has any of the specified roles
   */
  hasRole(roles: string | string[]): boolean {
    if (!this.user) return false
    
    const roleList = Array.isArray(roles) ? roles : [roles]
    return roleList.some(role => this.user!.roles.includes(role))
  }

  /**
   * Get tenant context
   */
  getTenantId(): string | null {
    return this.user?.tenant_id || null
  }

  /**
   * Get organization name
   */
  getOrganization(): string | null {
    return this.user?.organization || null
  }
}

// Export singleton instance
export const authService = new IntegratedAuthService()
export default authService