import { apiClient } from './apiClient'
import { User, ApiResponse } from '@/types'
import { endpoints } from '@/config/api'

export class AuthService {
  /**
   * Get user profile
   */
  async getProfile(token: string): Promise<User> {
    const response = await apiClient.get<ApiResponse<User>>(endpoints.auth.profile, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to get user profile')
    }
    
    return response.data.data
  }

  /**
   * Get user permissions
   */
  async getPermissions(token: string): Promise<string[]> {
    const response = await apiClient.get<ApiResponse<string[]>>(endpoints.auth.permissions, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to get user permissions')
    }
    
    return response.data.data
  }

  /**
   * Refresh access token
   */
  async refreshToken(refreshToken: string): Promise<{ accessToken: string; refreshToken: string }> {
    const response = await apiClient.post<ApiResponse<{ accessToken: string; refreshToken: string }>>(
      endpoints.auth.refresh,
      { refreshToken }
    )
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to refresh token')
    }
    
    return response.data.data
  }

  /**
   * Login with Azure AD token
   */
  async login(azureToken: string): Promise<{ user: User; accessToken: string; refreshToken: string }> {
    const response = await apiClient.post<ApiResponse<{ user: User; accessToken: string; refreshToken: string }>>(
      endpoints.auth.login,
      { azureToken }
    )
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to login')
    }
    
    return response.data.data
  }

  /**
   * Logout
   */
  async logout(token: string): Promise<void> {
    await apiClient.post(endpoints.auth.logout, {}, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
  }

  /**
   * Update user profile
   */
  async updateProfile(token: string, updates: Partial<User>): Promise<User> {
    const response = await apiClient.put<ApiResponse<User>>(endpoints.auth.profile, updates, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to update profile')
    }
    
    return response.data.data
  }

  /**
   * Change password
   */
  async changePassword(token: string, currentPassword: string, newPassword: string): Promise<void> {
    const response = await apiClient.post<ApiResponse<void>>(
      `${endpoints.auth.profile}/change-password`,
      { currentPassword, newPassword },
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    )
    
    if (!response.data.success) {
      throw new Error(response.data.message || 'Failed to change password')
    }
  }

  /**
   * Enable two-factor authentication
   */
  async enableTwoFactor(token: string): Promise<{ qrCode: string; secret: string }> {
    const response = await apiClient.post<ApiResponse<{ qrCode: string; secret: string }>>(
      `${endpoints.auth.profile}/two-factor/enable`,
      {},
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    )
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to enable two-factor authentication')
    }
    
    return response.data.data
  }

  /**
   * Disable two-factor authentication
   */
  async disableTwoFactor(token: string, code: string): Promise<void> {
    const response = await apiClient.post<ApiResponse<void>>(
      `${endpoints.auth.profile}/two-factor/disable`,
      { code },
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    )
    
    if (!response.data.success) {
      throw new Error(response.data.message || 'Failed to disable two-factor authentication')
    }
  }

  /**
   * Verify two-factor authentication code
   */
  async verifyTwoFactor(token: string, code: string): Promise<void> {
    const response = await apiClient.post<ApiResponse<void>>(
      `${endpoints.auth.profile}/two-factor/verify`,
      { code },
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    )
    
    if (!response.data.success) {
      throw new Error(response.data.message || 'Invalid two-factor authentication code')
    }
  }

  /**
   * Get session info
   */
  async getSessionInfo(token: string): Promise<{ expiresAt: string; lastActivity: string }> {
    const response = await apiClient.get<ApiResponse<{ expiresAt: string; lastActivity: string }>>(
      `${endpoints.auth.profile}/session`,
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    )
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to get session info')
    }
    
    return response.data.data
  }

  /**
   * Extend session
   */
  async extendSession(token: string): Promise<{ expiresAt: string }> {
    const response = await apiClient.post<ApiResponse<{ expiresAt: string }>>(
      `${endpoints.auth.profile}/session/extend`,
      {},
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    )
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to extend session')
    }
    
    return response.data.data
  }
}

export const authService = new AuthService()