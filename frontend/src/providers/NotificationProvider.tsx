import { createContext, useContext, useEffect, ReactNode } from 'react'
import { useWebSocket } from './WebSocketProvider'
import { useNotifications } from '@/hooks/useNotifications'
import { wsEvents } from '@/config/api'
import { Notification } from '@/types'

interface NotificationContextType {
  // Add any additional notification context methods here
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined)

export const useNotificationContext = (): NotificationContextType => {
  const context = useContext(NotificationContext)
  if (!context) {
    throw new Error('useNotificationContext must be used within a NotificationProvider')
  }
  return context
}

interface NotificationProviderProps {
  children: ReactNode
}

export const NotificationProvider = ({ children }: NotificationProviderProps) => {
  const { subscribe, unsubscribe, isConnected } = useWebSocket()
  const { refetch } = useNotifications()

  useEffect(() => {
    if (!isConnected) return

    // Subscribe to notification events
    const handleNotification = (notification: Notification) => {
      // Refetch notifications to update the list
      refetch()
    }

    const handleNotificationRead = (data: { notificationId: string }) => {
      // Refetch notifications to update read status
      refetch()
    }

    const handleNotificationDelete = (data: { notificationId: string }) => {
      // Refetch notifications to update the list
      refetch()
    }

    // Subscribe to WebSocket events
    subscribe(wsEvents.NOTIFICATION, handleNotification)
    subscribe(wsEvents.NOTIFICATION_READ, handleNotificationRead)
    subscribe(wsEvents.NOTIFICATION_DELETE, handleNotificationDelete)

    return () => {
      // Unsubscribe from WebSocket events
      unsubscribe(wsEvents.NOTIFICATION)
      unsubscribe(wsEvents.NOTIFICATION_READ)
      unsubscribe(wsEvents.NOTIFICATION_DELETE)
    }
  }, [isConnected, subscribe, unsubscribe, refetch])

  const contextValue: NotificationContextType = {
    // Add any additional methods here
  }

  return (
    <NotificationContext.Provider value={contextValue}>
      {children}
    </NotificationContext.Provider>
  )
}