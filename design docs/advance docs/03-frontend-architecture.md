# Frontend Architecture Documentation

## Overview

PolicyCortex frontend is built with Next.js 14 using the App Router, Server Components, and modern React patterns. The architecture emphasizes performance, type safety, and real-time updates while maintaining a clean separation of concerns.

## Technology Stack

### Core Technologies

- **Framework**: Next.js 14.0.4 (App Router)
- **Language**: TypeScript 5.3
- **State Management**: Zustand 4.4 (NOT Redux)
- **Server State**: React Query (TanStack Query) 5.0
- **GraphQL**: Apollo Client 3.8
- **Styling**: Tailwind CSS 3.4
- **UI Components**: Custom components + Radix UI
- **Authentication**: Azure MSAL React 3.0
- **Real-time**: Server-Sent Events (SSE)
- **Animations**: Framer Motion 10.16

## Project Structure

```
frontend/
├── app/                          # Next.js App Router
│   ├── layout.tsx               # Root layout with providers
│   ├── page.tsx                 # Landing page
│   ├── providers.tsx            # Client-side providers
│   ├── globals.css              # Global styles
│   ├── dashboard/
│   │   ├── page.tsx            # Dashboard page
│   │   └── layout.tsx          # Dashboard layout
│   ├── policies/
│   │   ├── page.tsx            # Policies list
│   │   ├── [id]/
│   │   │   └── page.tsx        # Policy detail
│   │   └── new/
│   │       └── page.tsx        # Create policy
│   ├── resources/
│   │   ├── page.tsx            # Resources list
│   │   └── [id]/
│   │       ├── page.tsx        # Resource detail
│   │       └── loading.tsx     # Loading state
│   ├── ai-expert/
│   │   └── page.tsx            # AI assistant interface
│   ├── chat/
│   │   └── page.tsx            # Conversational AI
│   ├── rbac/
│   │   └── page.tsx            # RBAC management
│   └── settings/
│       └── page.tsx            # User settings
├── components/
│   ├── AppLayout.tsx            # Main app layout
│   ├── DashboardContent.tsx     # Dashboard container
│   ├── ActionDrawer/
│   │   ├── ActionDrawer.tsx    # Action execution UI
│   │   ├── BlastRadius.tsx     # Impact visualization
│   │   ├── PreflightDiff.tsx   # Change preview
│   │   └── ApprovalFlow.tsx    # Approval workflow
│   ├── Dashboard/
│   │   ├── DashboardGrid.tsx   # Grid layout
│   │   ├── KPITile.tsx         # KPI widget
│   │   ├── ComplianceChart.tsx # Compliance visualization
│   │   └── CostTrends.tsx      # Cost analytics
│   ├── Resources/
│   │   ├── ResourceCard.tsx    # Resource display
│   │   ├── ResourceFilters.tsx # Filter controls
│   │   └── ResourceActions.tsx # Action buttons
│   ├── Policies/
│   │   ├── PolicyEditor.tsx    # Policy CRUD
│   │   ├── PolicyValidator.tsx # Validation UI
│   │   └── PolicyHistory.tsx   # Version history
│   └── common/
│       ├── Button.tsx          # Reusable button
│       ├── Card.tsx            # Card component
│       ├── Modal.tsx           # Modal dialog
│       └── Toast.tsx           # Notifications
├── lib/
│   ├── api.ts                  # Core API client
│   ├── actions-api.ts          # Action orchestration
│   ├── apollo-client.ts        # GraphQL setup
│   ├── azure-api.ts            # Azure-specific APIs
│   ├── react-query.ts          # Query client config
│   └── utils.ts                # Utility functions
├── hooks/
│   ├── useAuth.ts              # Authentication hook
│   ├── useMetrics.ts           # Metrics fetching
│   ├── useActions.ts           # Action management
│   ├── useRealtime.ts          # SSE subscription
│   └── useDebounce.ts          # Debouncing hook
├── contexts/
│   ├── AuthContext.tsx         # Auth provider
│   ├── ThemeContext.tsx        # Theme provider
│   └── NotificationContext.tsx # Toast notifications
├── stores/
│   ├── appStore.ts             # Global app state
│   ├── uiStore.ts              # UI state
│   └── actionStore.ts          # Action state
└── types/
    ├── api.ts                  # API type definitions
    ├── models.ts               # Data models
    └── components.ts           # Component props

```

## Core Components

### 1. Root Layout (app/layout.tsx)

```typescript
import { Inter } from 'next/font/google'
import { Providers } from './providers'
import { AuthProvider } from '@/contexts/AuthContext'

const inter = Inter({ subsets: ['latin'] })

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>
          <AuthProvider>
            {children}
          </AuthProvider>
        </Providers>
      </body>
    </html>
  )
}
```

### 2. Providers Setup (app/providers.tsx)

```typescript
'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ApolloProvider } from '@apollo/client'
import { MsalProvider } from '@azure/msal-react'
import { apolloClient } from '@/lib/apollo-client'
import { msalInstance } from '@/lib/auth'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { NotificationProvider } from '@/contexts/NotificationContext'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000,   // 10 minutes
      refetchOnWindowFocus: false,
    },
  },
})

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <MsalProvider instance={msalInstance}>
      <QueryClientProvider client={queryClient}>
        <ApolloProvider client={apolloClient}>
          <ThemeProvider>
            <NotificationProvider>
              {children}
            </NotificationProvider>
          </ThemeProvider>
        </ApolloProvider>
      </QueryClientProvider>
    </MsalProvider>
  )
}
```

### 3. App Layout Component (components/AppLayout.tsx)

```typescript
'use client'

import { useState } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { useAuth } from '@/hooks/useAuth'
import { cn } from '@/lib/utils'

export function AppLayout({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false)
  const { user, logout } = useAuth()
  const router = useRouter()
  const pathname = usePathname()

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: 'dashboard' },
    { name: 'Resources', href: '/resources', icon: 'resources' },
    { name: 'Policies', href: '/policies', icon: 'policies' },
    { name: 'AI Expert', href: '/ai-expert', icon: 'ai' },
    { name: 'RBAC', href: '/rbac', icon: 'security' },
    { name: 'Settings', href: '/settings', icon: 'settings' },
  ]

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className={cn(
        "bg-gray-900 text-white transition-all duration-300",
        collapsed ? "w-16" : "w-64"
      )}>
        <div className="p-4">
          <h1 className={cn(
            "text-xl font-bold",
            collapsed && "hidden"
          )}>
            PolicyCortex
          </h1>
        </div>
        
        <nav className="mt-8">
          {navigation.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className={cn(
                "flex items-center px-4 py-2 hover:bg-gray-800",
                pathname === item.href && "bg-gray-800"
              )}
            >
              <span className={cn(
                "ml-3",
                collapsed && "hidden"
              )}>
                {item.name}
              </span>
            </a>
          ))}
        </nav>

        <div className="absolute bottom-0 w-full p-4">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="w-full py-2 bg-gray-800 rounded"
          >
            {collapsed ? '→' : '←'}
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <header className="bg-white shadow-sm px-6 py-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold">
              {navigation.find(n => n.href === pathname)?.name}
            </h2>
            <div className="flex items-center gap-4">
              <span>{user?.email}</span>
              <button
                onClick={logout}
                className="px-4 py-2 bg-red-500 text-white rounded"
              >
                Logout
              </button>
            </div>
          </div>
        </header>
        
        <div className="p-6">
          {children}
        </div>
      </main>
    </div>
  )
}
```

## State Management

### 1. Zustand Store (stores/appStore.ts)

```typescript
import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'

interface AppState {
  // State
  metrics: GovernanceMetrics | null
  selectedResource: Resource | null
  filters: FilterState
  
  // Actions
  setMetrics: (metrics: GovernanceMetrics) => void
  selectResource: (resource: Resource | null) => void
  updateFilters: (filters: Partial<FilterState>) => void
  resetFilters: () => void
}

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set) => ({
        // Initial state
        metrics: null,
        selectedResource: null,
        filters: {
          resourceType: 'all',
          complianceStatus: 'all',
          dateRange: '7d',
        },
        
        // Actions
        setMetrics: (metrics) => set({ metrics }),
        selectResource: (resource) => set({ selectedResource: resource }),
        updateFilters: (filters) => set((state) => ({
          filters: { ...state.filters, ...filters }
        })),
        resetFilters: () => set({
          filters: {
            resourceType: 'all',
            complianceStatus: 'all',
            dateRange: '7d',
          }
        }),
      }),
      {
        name: 'app-storage',
        partialize: (state) => ({ filters: state.filters }),
      }
    )
  )
)
```

### 2. React Query Hooks (hooks/useMetrics.ts)

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useMetrics() {
  return useQuery({
    queryKey: ['metrics'],
    queryFn: () => api.get('/metrics'),
    refetchInterval: 30000, // Refetch every 30 seconds
  })
}

export function usePredictions() {
  return useQuery({
    queryKey: ['predictions'],
    queryFn: () => api.get('/predictions'),
    staleTime: 5 * 60 * 1000, // Consider fresh for 5 minutes
  })
}

export function useCreatePolicy() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (policy: CreatePolicyInput) => 
      api.post('/policies', policy),
    onSuccess: () => {
      // Invalidate and refetch policies
      queryClient.invalidateQueries({ queryKey: ['policies'] })
    },
  })
}
```

## Real-time Features

### 1. Server-Sent Events Hook (hooks/useRealtime.ts)

```typescript
import { useEffect, useState } from 'react'
import { EventSourcePolyfill } from 'event-source-polyfill'

export function useActionEvents(actionId: string) {
  const [action, setAction] = useState<Action | null>(null)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    const token = localStorage.getItem('access_token')
    const eventSource = new EventSourcePolyfill(
      `${process.env.NEXT_PUBLIC_API_URL}/api/v1/actions/${actionId}/events`,
      {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      }
    )

    eventSource.addEventListener('action-update', (event) => {
      const data = JSON.parse(event.data)
      setAction(data)
    })

    eventSource.addEventListener('error', (event) => {
      setError(new Error('Connection failed'))
      eventSource.close()
    })

    return () => {
      eventSource.close()
    }
  }, [actionId])

  return { action, error }
}
```

### 2. Action Drawer Component (components/ActionDrawer/ActionDrawer.tsx)

```typescript
'use client'

import { useState, useEffect } from 'react'
import { useActionEvents } from '@/hooks/useRealtime'
import { BlastRadius } from './BlastRadius'
import { PreflightDiff } from './PreflightDiff'
import { ApprovalFlow } from './ApprovalFlow'

interface ActionDrawerProps {
  isOpen: boolean
  onClose: () => void
  actionId?: string
  resourceId?: string
  actionType?: string
}

export function ActionDrawer({
  isOpen,
  onClose,
  actionId,
  resourceId,
  actionType,
}: ActionDrawerProps) {
  const [currentAction, setCurrentAction] = useState<Action | null>(null)
  const { action: liveAction } = useActionEvents(actionId || '')

  useEffect(() => {
    if (liveAction) {
      setCurrentAction(liveAction)
    }
  }, [liveAction])

  const createAction = async () => {
    const response = await fetch('/api/v1/actions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
      },
      body: JSON.stringify({
        action_type: actionType,
        resource_id: resourceId,
      }),
    })
    
    const action = await response.json()
    setCurrentAction(action)
  }

  return (
    <div className={`fixed right-0 top-0 h-full w-96 bg-white shadow-xl transform transition-transform ${
      isOpen ? 'translate-x-0' : 'translate-x-full'
    }`}>
      <div className="p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold">Execute Action</h2>
          <button onClick={onClose}>×</button>
        </div>

        {!currentAction && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                Resource
              </label>
              <input
                value={resourceId}
                disabled
                className="w-full p-2 border rounded"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">
                Action Type
              </label>
              <select
                value={actionType}
                className="w-full p-2 border rounded"
              >
                <option value="remediate">Remediate</option>
                <option value="stop">Stop Resource</option>
                <option value="delete">Delete Resource</option>
                <option value="tag">Update Tags</option>
              </select>
            </div>

            <button
              onClick={createAction}
              className="w-full py-2 bg-blue-500 text-white rounded"
            >
              Analyze Impact
            </button>
          </div>
        )}

        {currentAction && (
          <div className="space-y-6">
            {/* Blast Radius Visualization */}
            <BlastRadius action={currentAction} />
            
            {/* Preflight Diff */}
            <PreflightDiff action={currentAction} />
            
            {/* Approval Flow */}
            {currentAction.blast_radius.risk_level !== 'low' && (
              <ApprovalFlow action={currentAction} />
            )}
            
            {/* Execution Status */}
            <div className="bg-gray-50 p-4 rounded">
              <h3 className="font-medium mb-2">Status</h3>
              <div className="flex items-center gap-2">
                <span className={`w-3 h-3 rounded-full ${
                  currentAction.status === 'completed' ? 'bg-green-500' :
                  currentAction.status === 'failed' ? 'bg-red-500' :
                  currentAction.status === 'executing' ? 'bg-yellow-500 animate-pulse' :
                  'bg-gray-400'
                }`} />
                <span className="capitalize">{currentAction.status}</span>
              </div>
            </div>
            
            {/* Execution Log */}
            <div className="bg-gray-50 p-4 rounded max-h-60 overflow-auto">
              <h3 className="font-medium mb-2">Execution Log</h3>
              <div className="space-y-1 text-xs font-mono">
                {currentAction.execution_log.map((log, i) => (
                  <div key={i} className="flex gap-2">
                    <span className="text-gray-500">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                    <span>{log.message}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
```

## API Integration

### 1. Core API Client (lib/api.ts)

```typescript
class ApiClient {
  private baseURL: string
  private token: string | null = null

  constructor() {
    this.baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'
  }

  setToken(token: string) {
    this.token = token
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}/api/v1${endpoint}`
    
    const config: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(this.token && { Authorization: `Bearer ${this.token}` }),
        ...options.headers,
      },
    }

    const response = await fetch(url, config)

    if (!response.ok) {
      throw new ApiError(response.status, await response.text())
    }

    return response.json()
  }

  // GET request
  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' })
  }

  // POST request
  async post<T>(endpoint: string, body?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(body),
    })
  }

  // PUT request
  async put<T>(endpoint: string, body?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: JSON.stringify(body),
    })
  }

  // DELETE request
  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' })
  }
}

export const api = new ApiClient()
```

### 2. GraphQL Setup (lib/apollo-client.ts)

```typescript
import { ApolloClient, InMemoryCache, createHttpLink } from '@apollo/client'
import { setContext } from '@apollo/client/link/context'

const httpLink = createHttpLink({
  uri: process.env.NEXT_PUBLIC_GRAPHQL_URL || 'http://localhost:4000/graphql',
})

const authLink = setContext((_, { headers }) => {
  const token = localStorage.getItem('access_token')
  
  return {
    headers: {
      ...headers,
      authorization: token ? `Bearer ${token}` : '',
    },
  }
})

export const apolloClient = new ApolloClient({
  link: authLink.concat(httpLink),
  cache: new InMemoryCache({
    typePolicies: {
      Query: {
        fields: {
          resources: {
            merge(existing = [], incoming) {
              return [...existing, ...incoming]
            },
          },
        },
      },
    },
  }),
})
```

## Authentication

### 1. MSAL Configuration (lib/auth.ts)

```typescript
import { PublicClientApplication, Configuration } from '@azure/msal-browser'

const msalConfig: Configuration = {
  auth: {
    clientId: process.env.NEXT_PUBLIC_AZURE_CLIENT_ID!,
    authority: `https://login.microsoftonline.com/${process.env.NEXT_PUBLIC_AZURE_TENANT_ID}`,
    redirectUri: process.env.NEXT_PUBLIC_REDIRECT_URI,
  },
  cache: {
    cacheLocation: 'sessionStorage',
    storeAuthStateInCookie: false,
  },
}

export const msalInstance = new PublicClientApplication(msalConfig)

export const loginRequest = {
  scopes: ['api://policycortex/.default'],
}
```

### 2. Auth Hook (hooks/useAuth.ts)

```typescript
import { useMsal } from '@azure/msal-react'
import { useState, useEffect } from 'react'
import { api } from '@/lib/api'

export function useAuth() {
  const { instance, accounts, inProgress } = useMsal()
  const [user, setUser] = useState<User | null>(null)

  useEffect(() => {
    if (accounts.length > 0) {
      const account = accounts[0]
      setUser({
        id: account.localAccountId,
        email: account.username,
        name: account.name || '',
        roles: account.idTokenClaims?.roles || [],
      })
      
      // Get and set access token
      instance.acquireTokenSilent({
        ...loginRequest,
        account,
      }).then((response) => {
        api.setToken(response.accessToken)
      })
    }
  }, [accounts, instance])

  const login = async () => {
    try {
      await instance.loginPopup(loginRequest)
    } catch (error) {
      console.error('Login failed:', error)
    }
  }

  const logout = () => {
    instance.logoutPopup()
  }

  return {
    user,
    isAuthenticated: accounts.length > 0,
    isLoading: inProgress === 'login',
    login,
    logout,
  }
}
```

## Performance Optimizations

### 1. Server Components

```typescript
// app/resources/page.tsx - Server Component
import { Suspense } from 'react'
import { ResourceList } from '@/components/Resources/ResourceList'
import { ResourceFilters } from '@/components/Resources/ResourceFilters'

async function getResources() {
  const res = await fetch(`${process.env.API_URL}/api/v1/resources`, {
    cache: 'no-store',
  })
  return res.json()
}

export default async function ResourcesPage() {
  const resources = await getResources()

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Resources</h1>
      
      <ResourceFilters />
      
      <Suspense fallback={<div>Loading resources...</div>}>
        <ResourceList resources={resources} />
      </Suspense>
    </div>
  )
}
```

### 2. Dynamic Imports

```typescript
import dynamic from 'next/dynamic'

// Lazy load heavy components
const ComplianceChart = dynamic(
  () => import('@/components/Dashboard/ComplianceChart'),
  { 
    loading: () => <div>Loading chart...</div>,
    ssr: false, // Disable SSR for client-only components
  }
)

const PolicyEditor = dynamic(
  () => import('@/components/Policies/PolicyEditor'),
  { 
    loading: () => <div>Loading editor...</div>,
  }
)
```

### 3. Image Optimization

```typescript
import Image from 'next/image'

export function ResourceCard({ resource }: { resource: Resource }) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <Image
        src={`/icons/${resource.type}.svg`}
        alt={resource.type}
        width={48}
        height={48}
        loading="lazy"
      />
      <h3 className="mt-2 font-medium">{resource.name}</h3>
      <p className="text-sm text-gray-500">{resource.location}</p>
    </div>
  )
}
```

### 4. Memoization

```typescript
import { memo, useMemo } from 'react'

export const ExpensiveComponent = memo(({ data }: { data: any[] }) => {
  const processedData = useMemo(() => {
    // Expensive computation
    return data.map(item => ({
      ...item,
      calculated: heavyCalculation(item),
    }))
  }, [data])

  return (
    <div>
      {processedData.map(item => (
        <div key={item.id}>{item.calculated}</div>
      ))}
    </div>
  )
})
```

## Testing

### 1. Component Tests

```typescript
// __tests__/components/KPITile.test.tsx
import { render, screen } from '@testing-library/react'
import { KPITile } from '@/components/Dashboard/KPITile'

describe('KPITile', () => {
  it('renders KPI value and label', () => {
    render(
      <KPITile
        label="Compliance Score"
        value="85%"
        trend="up"
        change="+5%"
      />
    )

    expect(screen.getByText('Compliance Score')).toBeInTheDocument()
    expect(screen.getByText('85%')).toBeInTheDocument()
    expect(screen.getByText('+5%')).toBeInTheDocument()
  })

  it('shows correct trend indicator', () => {
    const { rerender } = render(
      <KPITile label="Test" value="100" trend="up" />
    )
    
    expect(screen.getByTestId('trend-up')).toBeInTheDocument()
    
    rerender(<KPITile label="Test" value="100" trend="down" />)
    expect(screen.getByTestId('trend-down')).toBeInTheDocument()
  })
})
```

### 2. Hook Tests

```typescript
// __tests__/hooks/useMetrics.test.ts
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useMetrics } from '@/hooks/useMetrics'

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  })
  
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )
}

describe('useMetrics', () => {
  it('fetches metrics successfully', async () => {
    const { result } = renderHook(() => useMetrics(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })

    expect(result.current.data).toHaveProperty('compliance')
    expect(result.current.data).toHaveProperty('security')
    expect(result.current.data).toHaveProperty('costs')
  })
})
```

## Build & Deployment

### 1. Next.js Configuration (next.config.js)

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  images: {
    domains: ['localhost', 'policycortex.azurewebsites.net'],
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
    NEXT_PUBLIC_GRAPHQL_URL: process.env.NEXT_PUBLIC_GRAPHQL_URL,
  },
  experimental: {
    serverActions: true,
  },
  webpack: (config) => {
    config.externals.push({
      'utf-8-validate': 'commonjs utf-8-validate',
      'bufferutil': 'commonjs bufferutil',
    })
    return config
  },
}

module.exports = nextConfig
```

### 2. Docker Configuration

```dockerfile
# Multi-stage build for production
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --only=production

FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT 3000

CMD ["node", "server.js"]
```

## Environment Variables

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_GRAPHQL_URL=http://localhost:4000/graphql
NEXT_PUBLIC_AZURE_CLIENT_ID=your-client-id
NEXT_PUBLIC_AZURE_TENANT_ID=your-tenant-id
NEXT_PUBLIC_REDIRECT_URI=http://localhost:3000

# Production
NEXT_PUBLIC_API_URL=https://api.policycortex.com
NEXT_PUBLIC_GRAPHQL_URL=https://graphql.policycortex.com
NEXT_PUBLIC_AZURE_CLIENT_ID=prod-client-id
NEXT_PUBLIC_AZURE_TENANT_ID=prod-tenant-id
NEXT_PUBLIC_REDIRECT_URI=https://app.policycortex.com
```