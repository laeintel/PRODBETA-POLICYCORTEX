import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { notificationService } from '@/services/notificationService'
import { useAuth } from '@/hooks/useAuth'
import { Notification } from '@/types'
import toast from 'react-hot-toast'

export const useNotifications = () => {
  const { getAccessToken } = useAuth()
  const queryClient = useQueryClient()
  const [unreadCount, setUnreadCount] = useState(0)

  // Fetch notifications
  const {
    data: notifications,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['notifications'],
    queryFn: async () => {
      const token = await getAccessToken()
      if (!token) throw new Error('No access token')
      return notificationService.getNotifications(token)
    },
    enabled: !!getAccessToken,
    refetchInterval: 30000, // Refetch every 30 seconds
  })

  // Mark as read mutation
  const markAsReadMutation = useMutation({
    mutationFn: async (notificationId: string) => {
      const token = await getAccessToken()
      if (!token) throw new Error('No access token')
      return notificationService.markAsRead(token, notificationId)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] })
    },
    onError: (error) => {
      toast.error('Failed to mark notification as read')
      console.error('Mark as read error:', error)
    },
  })

  // Mark all as read mutation
  const markAllAsReadMutation = useMutation({
    mutationFn: async () => {
      const token = await getAccessToken()
      if (!token) throw new Error('No access token')
      return notificationService.markAllAsRead(token)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] })
      toast.success('All notifications marked as read')
    },
    onError: (error) => {
      toast.error('Failed to mark all notifications as read')
      console.error('Mark all as read error:', error)
    },
  })

  // Delete notification mutation
  const deleteNotificationMutation = useMutation({
    mutationFn: async (notificationId: string) => {
      const token = await getAccessToken()
      if (!token) throw new Error('No access token')
      return notificationService.deleteNotification(token, notificationId)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] })
      toast.success('Notification deleted')
    },
    onError: (error) => {
      toast.error('Failed to delete notification')
      console.error('Delete notification error:', error)
    },
  })

  // Update unread count when notifications change
  useEffect(() => {
    if (notifications) {
      const count = notifications.filter(n => !n.isRead).length
      setUnreadCount(count)
    }
  }, [notifications])

  // Helper functions
  const markAsRead = (notificationId: string) => {
    markAsReadMutation.mutate(notificationId)
  }

  const markAllAsRead = () => {
    markAllAsReadMutation.mutate()
  }

  const deleteNotification = (notificationId: string) => {
    deleteNotificationMutation.mutate(notificationId)
  }

  const getNotificationsByType = (type: string) => {
    return notifications?.filter(n => n.type === type) || []
  }

  const getUnreadNotifications = () => {
    return notifications?.filter(n => !n.isRead) || []
  }

  const getRecentNotifications = (limit = 5) => {
    return notifications?.slice(0, limit) || []
  }

  return {
    notifications: notifications || [],
    unreadCount,
    isLoading,
    error,
    refetch,
    markAsRead,
    markAllAsRead,
    deleteNotification,
    getNotificationsByType,
    getUnreadNotifications,
    getRecentNotifications,
    isMarkingAsRead: markAsReadMutation.isPending,
    isMarkingAllAsRead: markAllAsReadMutation.isPending,
    isDeleting: deleteNotificationMutation.isPending,
  }
}