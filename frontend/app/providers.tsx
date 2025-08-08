'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ApolloProvider } from '@apollo/client'
import { useState } from 'react'
import { client } from '@/lib/apollo-client'
import VoiceProvider from '../components/VoiceProvider'
import { AuthProvider } from '../contexts/AuthContext'

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient())

  return (
    <AuthProvider>
      <QueryClientProvider client={queryClient}>
        <ApolloProvider client={client}>
          {children}
          <VoiceProvider />
        </ApolloProvider>
      </QueryClientProvider>
    </AuthProvider>
  )
}