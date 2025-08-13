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