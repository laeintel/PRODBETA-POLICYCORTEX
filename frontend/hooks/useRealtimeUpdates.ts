import { useEffect } from 'react'
import { useSSE } from './useSSE'
import { useGovernanceStore } from '@/store/governanceStore'

export function useRealtimeUpdates(enabled: boolean = true) {
  const { messages, isConnected, error } = useSSE(
    enabled ? `/api/v1/events` : ''
  )
  const updateMetric = useGovernanceStore(state => state.updateMetric)

  useEffect(() => {
    if (!enabled) return

    messages.forEach(message => {
      if (message.type === 'metric_update' && message.data) {
        const { category, updates } = message.data
        if (category) {
          updateMetric(category, updates)
        }
      }
    })
  }, [messages, updateMetric, enabled])

  return {
    isConnected,
    error,
    messagesCount: messages.length
  }
}