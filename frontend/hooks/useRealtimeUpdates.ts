/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

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