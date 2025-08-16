/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ApolloProvider } from '@apollo/client'
import { useState, useEffect } from 'react'
import { client } from '@/lib/apollo-client'
import dynamic from 'next/dynamic'
import { AuthProvider } from '../contexts/AuthContext'
import { DemoDataProvider } from '../contexts/DemoDataProvider'
import { ServiceWorkerRegistration } from '../components/ServiceWorkerRegistration'
import { OfflineIndicator, OfflineQueue, ConflictResolver } from '../components/OfflineIndicator'
import { I18nProvider } from '../lib/i18n'
import { AZURE_OPENAI } from '../lib/api-config'

// Dynamically import VoiceProvider to avoid SSR issues
const VoiceProvider = dynamic(
  () => import('../components/VoiceProvider'),
  { 
    ssr: false,
    loading: () => null
  }
)

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient())
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    // Optionally pre-warm Azure OpenAI endpoint (cheap no-op)
    const endpoint = AZURE_OPENAI.endpoint
    if (endpoint) {
      fetch(endpoint, { method: 'HEAD' }).catch(() => {})
    }
  }, [])

  return (
    <AuthProvider>
      <I18nProvider>
        <QueryClientProvider client={queryClient}>
          <ApolloProvider client={client}>
            <ServiceWorkerRegistration />
            <DemoDataProvider>
              {children}
            </DemoDataProvider>
            <OfflineIndicator />
            <OfflineQueue />
            <ConflictResolver />
            {mounted && <VoiceProvider />}
          </ApolloProvider>
        </QueryClientProvider>
      </I18nProvider>
    </AuthProvider>
  )
}