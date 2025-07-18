import { apiClient } from './apiClient'
import { Notification, ApiResponse, PaginatedResponse } from '@/types'
import { endpoints } from '@/config/api'

export class NotificationService {
  /**
   * Get notifications for current user
   */
  async getNotifications(token: string, params?: {
    page?: number
    limit?: number
    unreadOnly?: boolean
    type?: string
  }): Promise<Notification[]> {
    const response = await apiClient.get<ApiResponse<PaginatedResponse<Notification>>>(
      endpoints.notifications.list,
      {
        params,
        headers: { Authorization: `Bearer ${token}` },
      }
    )
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to fetch notifications')
    }
    
    return response.data.data.data
  }

  /**
   * Mark notification as read
   */
  async markAsRead(token: string, notificationId: string): Promise<void> {
    const response = await apiClient.put<ApiResponse<void>>(
      endpoints.notifications.mark_read(notificationId),
      {},
      { headers: { Authorization: `Bearer ${token}` } }
    )
    
    if (!response.data.success) {
      throw new Error(response.data.message || 'Failed to mark notification as read')
    }
  }

  /**
   * Mark all notifications as read
   */
  async markAllAsRead(token: string): Promise<void> {
    const response = await apiClient.put<ApiResponse<void>>(
      endpoints.notifications.mark_all_read,
      {},
      { headers: { Authorization: `Bearer ${token}` } }
    )
    
    if (!response.data.success) {
      throw new Error(response.data.message || 'Failed to mark all notifications as read')
    }
  }

  /**
   * Delete notification
   */
  async deleteNotification(token: string, notificationId: string): Promise<void> {
    const response = await apiClient.delete<ApiResponse<void>>(
      endpoints.notifications.delete(notificationId),
      { headers: { Authorization: `Bearer ${token}` } }
    )
    
    if (!response.data.success) {
      throw new Error(response.data.message || 'Failed to delete notification')
    }
  }

  /**
   * Get notification preferences
   */
  async getPreferences(token: string): Promise<any> {
    const response = await apiClient.get<ApiResponse<any>>(
      endpoints.notifications.preferences,
      { headers: { Authorization: `Bearer ${token}` } }
    )
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to fetch notification preferences')
    }
    
    return response.data.data
  }

  /**
   * Update notification preferences
   */
  async updatePreferences(token: string, preferences: any): Promise<void> {
    const response = await apiClient.put<ApiResponse<void>>(
      endpoints.notifications.preferences,
      preferences,
      { headers: { Authorization: `Bearer ${token}` } }
    )
    
    if (!response.data.success) {
      throw new Error(response.data.message || 'Failed to update notification preferences')
    }
  }

  /**
   * Get notification channels
   */
  async getChannels(token: string): Promise<any[]> {
    const response = await apiClient.get<ApiResponse<any[]>>(
      endpoints.notifications.channels,
      { headers: { Authorization: `Bearer ${token}` } }
    )
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to fetch notification channels')
    }
    
    return response.data.data
  }

  /**
   * Create test notification
   */
  async createTestNotification(token: string, type: string): Promise<void> {
    const response = await apiClient.post<ApiResponse<void>>(
      `${endpoints.notifications.create}/test`,
      { type },
      { headers: { Authorization: `Bearer ${token}` } }
    )
    
    if (!response.data.success) {
      throw new Error(response.data.message || 'Failed to create test notification')
    }
  }
}

export const notificationService = new NotificationService()