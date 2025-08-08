'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ApolloProvider } from '@apollo/client'
import { useState, useEffect } from 'react'
import { client } from '@/lib/apollo-client'
import dynamic from 'next/dynamic'
import { AuthProvider } from '../contexts/AuthContext'

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
      <QueryClientProvider client={queryClient}>
        <ApolloProvider client={client}>
          {children}
          {mounted && <VoiceProvider />}
        </ApolloProvider>
      </QueryClientProvider>
    </AuthProvider>
  )
}