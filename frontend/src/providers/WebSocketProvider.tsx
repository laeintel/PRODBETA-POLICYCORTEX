import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import { io, Socket } from 'socket.io-client'
import { wsConfig, wsEvents } from '@/config/api'
import { useAuth } from '@/hooks/useAuth'
import { WebSocketMessage, WebSocketState } from '@/types'
import { env } from '@/config/environment'
import toast from 'react-hot-toast'

interface WebSocketContextType {
  socket: Socket | null
  isConnected: boolean
  connectionId: string | null
  subscribe: (event: string, callback: (data: any) => void) => void
  unsubscribe: (event: string) => void
  emit: (event: string, data?: any) => void
  reconnect: () => void
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined)

export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}

interface WebSocketProviderProps {
  children: ReactNode
}

export const WebSocketProvider = ({ children }: WebSocketProviderProps) => {
  const { user, getAccessToken } = useAuth()
  const [socket, setSocket] = useState<Socket | null>(null)
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    connectionId: null,
    lastPing: null,
    reconnectAttempts: 0,
    subscriptions: [],
  })

  // Initialize WebSocket connection
  useEffect(() => {
    // Temporarily disable WebSocket connection until backend is ready
    console.log('WebSocket temporarily disabled for development')
    return
    
    if (!user || !env.ENABLE_WEBSOCKET) return

    const initializeSocket = async () => {
      try {
        const token = await getAccessToken()
        if (!token) return

        const socketInstance = io(wsConfig.url, {
          ...wsConfig.options,
          auth: {
            token,
          },
          query: {
            userId: user.id,
          },
        })

        // Connection event handlers
        socketInstance.on(wsEvents.CONNECT, () => {
          console.log('WebSocket connected')
          setState(prev => ({
            ...prev,
            isConnected: true,
            reconnectAttempts: 0,
          }))
        })

        socketInstance.on(wsEvents.DISCONNECT, (reason) => {
          console.log('WebSocket disconnected:', reason)
          setState(prev => ({
            ...prev,
            isConnected: false,
            connectionId: null,
          }))
        })

        socketInstance.on(wsEvents.AUTHENTICATED, (data) => {
          console.log('WebSocket authenticated:', data)
          setState(prev => ({
            ...prev,
            connectionId: data.connectionId,
          }))
        })

        socketInstance.on(wsEvents.UNAUTHENTICATED, () => {
          console.warn('WebSocket authentication failed')
          toast.error('WebSocket authentication failed')
        })

        // Real-time event handlers
        socketInstance.on(wsEvents.NOTIFICATION, (notification) => {
          console.log('New notification:', notification)
          // Handle notification display
          if (notification.severity === 'error') {
            toast.error(notification.message)
          } else if (notification.severity === 'warning') {
            toast.error(notification.message) // Using error toast for warnings too
          } else if (notification.severity === 'success') {
            toast.success(notification.message)
          } else {
            toast(notification.message)
          }
        })

        socketInstance.on(wsEvents.POLICY_UPDATE, (data) => {
          console.log('Policy update:', data)
          // Handle policy update
        })

        socketInstance.on(wsEvents.RESOURCE_UPDATE, (data) => {
          console.log('Resource update:', data)
          // Handle resource update
        })

        socketInstance.on(wsEvents.COST_UPDATE, (data) => {
          console.log('Cost update:', data)
          // Handle cost update
        })

        socketInstance.on(wsEvents.ALERT_UPDATE, (data) => {
          console.log('Alert update:', data)
          // Handle alert update
        })

        socketInstance.on(wsEvents.SYSTEM_STATUS, (data) => {
          console.log('System status:', data)
          // Handle system status update
        })

        socketInstance.on(wsEvents.MAINTENANCE, (data) => {
          console.log('Maintenance notice:', data)
          toast.error(`Maintenance: ${data.message}`)
        })

        socketInstance.on(wsEvents.ERROR, (error) => {
          console.error('WebSocket error:', error)
          toast.error(`WebSocket error: ${error.message}`)
        })

        socketInstance.on(wsEvents.RECONNECT, (attemptNumber) => {
          console.log('WebSocket reconnect attempt:', attemptNumber)
          setState(prev => ({
            ...prev,
            reconnectAttempts: attemptNumber,
          }))
        })

        // Heartbeat
        socketInstance.on('ping', () => {
          setState(prev => ({
            ...prev,
            lastPing: new Date().toISOString(),
          }))
        })

        setSocket(socketInstance)
      } catch (error) {
        console.error('Failed to initialize WebSocket:', error)
      }
    }

    initializeSocket()

    return () => {
      if (socket) {
        socket.disconnect()
        setSocket(null)
      }
    }
  }, [user, getAccessToken])

  // Subscribe to events
  const subscribe = (event: string, callback: (data: any) => void) => {
    if (!socket) return

    socket.on(event, callback)
    setState(prev => ({
      ...prev,
      subscriptions: [...prev.subscriptions.filter(s => s !== event), event],
    }))
  }

  // Unsubscribe from events
  const unsubscribe = (event: string) => {
    if (!socket) return

    socket.off(event)
    setState(prev => ({
      ...prev,
      subscriptions: prev.subscriptions.filter(s => s !== event),
    }))
  }

  // Emit events
  const emit = (event: string, data?: any) => {
    if (!socket || !state.isConnected) return

    socket.emit(event, data)
  }

  // Reconnect manually
  const reconnect = () => {
    if (socket) {
      socket.disconnect()
      socket.connect()
    }
  }

  const contextValue: WebSocketContextType = {
    socket,
    isConnected: state.isConnected,
    connectionId: state.connectionId,
    subscribe,
    unsubscribe,
    emit,
    reconnect,
  }

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  )
}