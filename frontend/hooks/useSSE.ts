import { useEffect, useState, useCallback } from 'react'

export interface SSEMessage {
  type: string
  data: any
  timestamp: string
}

export function useSSE(url: string) {
  const [messages, setMessages] = useState<SSEMessage[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (!url) return

    const eventSource = new EventSource(url)

    eventSource.onopen = () => {
      setIsConnected(true)
      setError(null)
    }

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        const message: SSEMessage = {
          type: data.type || 'message',
          data: data,
          timestamp: new Date().toISOString()
        }
        setMessages(prev => [...prev, message])
      } catch (err) {
        console.error('Failed to parse SSE message:', err)
      }
    }

    eventSource.onerror = (err) => {
      setIsConnected(false)
      setError(new Error('SSE connection failed'))
      eventSource.close()
    }

    return () => {
      eventSource.close()
      setIsConnected(false)
    }
  }, [url])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return {
    messages,
    isConnected,
    error,
    clearMessages
  }
}