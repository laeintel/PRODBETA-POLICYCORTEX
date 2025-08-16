/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

// React Query Configuration - Issue #38
// Advanced server state management with caching, background updates, and optimistic updates

import { QueryClient, QueryCache, MutationCache } from '@tanstack/react-query'
import { toast } from 'sonner'

// Custom error handling
const handleError = (error: any, context?: string) => {
  console.error(`React Query error${context ? ` in ${context}` : ''}:`, error)
  
  // Show user-friendly error message
  const message = error?.message || error?.response?.data?.message || 'Something went wrong'
  toast.error(message, {
    duration: 5000,
    action: {
      label: 'Retry',
      onClick: () => {
        // Trigger retry - this would be handled by the specific query
        console.log('User requested retry')
      }
    }
  })
}

// Global query cache with advanced error handling
const queryCache = new QueryCache({
  onError: (error, query) => {
    handleError(error, `query: ${query.queryKey.join('.')}`)
  },
  onSuccess: (data, query) => {
    // Log successful queries in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`✅ Query success: ${query.queryKey.join('.')}`, data)
    }
  },
  onSettled: (data, error, query) => {
    // Track query performance
    if (query.state.dataUpdatedAt && query.state.dataUpdatedAt > 0) {
      const duration = Date.now() - query.state.dataUpdatedAt
      if (duration > 5000) { // Log slow queries
        console.warn(`🐌 Slow query detected: ${query.queryKey.join('.')} took ${duration}ms`)
      }
    }
  }
})

// Global mutation cache with optimistic updates
const mutationCache = new MutationCache({
  onError: (error, variables, context, mutation) => {
    handleError(error, `mutation: ${mutation.options.mutationKey?.join('.') || 'unknown'}`)
  },
  onSuccess: (data, variables, context, mutation) => {
    if (process.env.NODE_ENV === 'development') {
      console.log(`✅ Mutation success: ${mutation.options.mutationKey?.join('.') || 'unknown'}`, {
        data,
        variables
      })
    }
  },
  onSettled: (data, error, variables, context, mutation) => {
    // Invalidate related queries after mutations
    const mutationKey = mutation.options.mutationKey?.[0]
    if (mutationKey) {
      // Auto-invalidate related queries based on mutation type
      const invalidationMap: Record<string, string[]> = {
        'updatePolicy': ['policies', 'compliance'],
        'createPolicy': ['policies', 'compliance'],
        'deletePolicy': ['policies', 'compliance'],
        'updateResource': ['resources', 'compliance'],
        'applyRemediation': ['policies', 'resources', 'compliance'],
        'generateReport': ['reports'],
      }
      
      const queriesToInvalidate = invalidationMap[mutationKey as string]
      if (queriesToInvalidate) {
        queriesToInvalidate.forEach(queryKey => {
          queryClient.invalidateQueries({ queryKey: [queryKey] })
        })
      }
    }
  }
})

// Import performance API client
import { createPerformanceQueryClient } from './performance-api'

// React Query Client with advanced configuration and performance optimizations
export const queryClient = createPerformanceQueryClient()

// Legacy query client with fallback configuration
export const legacyQueryClient = new QueryClient({
  queryCache,
  mutationCache,
  defaultOptions: {
    queries: {
      // Cache configuration
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
      
      // Network configuration
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
      refetchOnMount: true,
      
      // Retry configuration
      retry: (failureCount, error: any) => {
        // Don't retry on 4xx errors
        if (error?.response?.status >= 400 && error?.response?.status < 500) {
          return false
        }
        // Retry up to 3 times with exponential backoff
        return failureCount < 3
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      
      // Background updates
      refetchInterval: (query) => {
        // Refresh compliance data every 30 seconds
        if (query.queryKey.includes('compliance')) {
          return 30 * 1000
        }
        // Refresh policies every 2 minutes
        if (query.queryKey.includes('policies')) {
          return 2 * 60 * 1000
        }
        // Refresh resources every 1 minute
        if (query.queryKey.includes('resources')) {
          return 60 * 1000
        }
        // No automatic refresh for other queries
        return false
      },
      
      // Performance optimization
      notifyOnChangeProps: 'all',
      structuralSharing: true,
    },
    
    mutations: {
      // Retry configuration for mutations
      retry: (failureCount, error: any) => {
        // Only retry on network errors, not validation errors
        if (error?.response?.status >= 400 && error?.response?.status < 500) {
          return false
        }
        return failureCount < 2
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 5000),
      
      // Global mutation settings
      networkMode: 'online',
    },
  },
})

// Query Keys - Centralized key management
export const queryKeys = {
  // User and Authentication
  user: ['user'] as const,
  userPreferences: (userId: string) => ['user', userId, 'preferences'] as const,
  
  // Policies
  policies: ['policies'] as const,
  policiesList: (filters?: any, search?: string) => ['policies', 'list', filters, search] as const,
  policy: (id: string) => ['policies', id] as const,
  policyCompliance: (id: string) => ['policies', id, 'compliance'] as const,
  
  // Resources
  resources: ['resources'] as const,
  resourcesList: (filters?: any, search?: string, pagination?: any) => 
    ['resources', 'list', filters, search, pagination] as const,
  resource: (id: string) => ['resources', id] as const,
  resourceCompliance: (id: string) => ['resources', id, 'compliance'] as const,
  resourcePolicies: (id: string) => ['resources', id, 'policies'] as const,
  
  // Compliance and Reporting
  compliance: ['compliance'] as const,
  complianceDashboard: (dateRange?: any) => ['compliance', 'dashboard', dateRange] as const,
  complianceScore: ['compliance', 'score'] as const,
  complianceTrends: (dateRange?: any) => ['compliance', 'trends', dateRange] as const,
  
  // Reports
  reports: ['reports'] as const,
  reportsList: (type?: string, dateRange?: any) => ['reports', 'list', type, dateRange] as const,
  report: (id: string) => ['reports', id] as const,
  
  // Real-time data
  realTimeUpdates: ['realtime'] as const,
  notifications: ['notifications'] as const,
  
  // Search and suggestions
  search: (query: string, type?: string) => ['search', query, type] as const,
  suggestions: (query: string) => ['suggestions', query] as const,
}

// Custom hooks factory for type-safe queries
export const createQueryHook = <TData, TError = Error, TQueryKey extends ReadonlyArray<unknown> = ReadonlyArray<unknown>>(
  queryKeyFn: (...args: any[]) => TQueryKey,
  queryFn: (context: { queryKey: TQueryKey }) => Promise<TData>
) => {
  return (...args: Parameters<typeof queryKeyFn>) => {
    const queryKey = queryKeyFn(...args)
    return {
      queryKey,
      queryFn: ({ queryKey }: { queryKey: TQueryKey }) => queryFn({ queryKey })
    }
  }
}

// Optimistic update helpers
export const optimisticUpdateHelpers = {
  // Update a single item in a list
  updateItemInList: <T extends { id: string }>(
    queryKey: readonly unknown[],
    itemId: string,
    updates: Partial<T>
  ) => {
    queryClient.setQueryData(queryKey, (oldData: T[] | undefined) => {
      if (!oldData) return oldData
      
      return oldData.map(item => 
        item.id === itemId ? { ...item, ...updates } : item
      )
    })
  },
  
  // Add item to list optimistically
  addItemToList: <T>(queryKey: readonly unknown[], newItem: T) => {
    queryClient.setQueryData(queryKey, (oldData: T[] | undefined) => {
      if (!oldData) return [newItem]
      return [newItem, ...oldData]
    })
  },
  
  // Remove item from list optimistically
  removeItemFromList: <T extends { id: string }>(
    queryKey: readonly unknown[],
    itemId: string
  ) => {
    queryClient.setQueryData(queryKey, (oldData: T[] | undefined) => {
      if (!oldData) return oldData
      return oldData.filter(item => item.id !== itemId)
    })
  },
  
  // Update nested object property
  updateNestedProperty: (
    queryKey: readonly unknown[],
    propertyPath: string[],
    value: any
  ) => {
    queryClient.setQueryData(queryKey, (oldData: any) => {
      if (!oldData) return oldData
      
      const newData = { ...oldData }
      let current = newData
      
      for (let i = 0; i < propertyPath.length - 1; i++) {
        const key = propertyPath[i]
        current[key] = { ...current[key] }
        current = current[key]
      }
      
      current[propertyPath[propertyPath.length - 1]] = value
      return newData
    })
  }
}

// Performance monitoring and debugging
if (process.env.NODE_ENV === 'development') {
  // Query performance tracking
  let queryStartTimes = new Map<string, number>()
  
  queryClient.getQueryCache().subscribe((event) => {
    const queryKey = event.query.queryKey.join('.')
    
    if (event.type === 'added') {
      queryStartTimes.set(queryKey, Date.now())
    } else if (event.type === 'updated' && event.query.state.status === 'success') {
      const startTime = queryStartTimes.get(queryKey)
      if (startTime) {
        const duration = Date.now() - startTime
        console.log(`⚡ Query completed: ${queryKey} in ${duration}ms`)
        queryStartTimes.delete(queryKey)
        
        // Alert on slow queries
        if (duration > 3000) {
          console.warn(`🐌 Slow query detected: ${queryKey} took ${duration}ms`)
        }
      }
    }
  })
  
  // Cache size monitoring
  setInterval(() => {
    const cache = queryClient.getQueryCache()
    const totalQueries = cache.getAll().length
    const staleQueries = cache.getAll().filter(q => q.isStale()).length
    
    if (totalQueries > 100) {
      console.warn(`📊 Large query cache: ${totalQueries} queries, ${staleQueries} stale`)
    }
  }, 30000) // Check every 30 seconds
}

// Cache management utilities
export const cacheUtils = {
  // Clear all cache data
  clearAll: () => queryClient.clear(),
  
  // Clear specific query patterns
  clearByPattern: (pattern: string) => {
    queryClient.removeQueries({
      predicate: (query) => query.queryKey.some(key => 
        typeof key === 'string' && key.includes(pattern)
      )
    })
  },
  
  // Get cache statistics
  getStats: () => {
    const cache = queryClient.getQueryCache()
    const queries = cache.getAll()
    
    return {
      totalQueries: queries.length,
      staleQueries: queries.filter(q => q.isStale()).length,
      fetchingQueries: queries.filter(q => q.state.status === 'pending').length,
      errorQueries: queries.filter(q => q.state.status === 'error').length,
      cacheSize: JSON.stringify(cache).length, // Approximate size
    }
  },
  
  // Prefetch common queries
  prefetchCommon: async () => {
    await Promise.all([
      queryClient.prefetchQuery({
        queryKey: queryKeys.policies,
        queryFn: () => fetch('/api/policies').then(res => res.json()),
        staleTime: 10 * 60 * 1000 // 10 minutes
      }),
      queryClient.prefetchQuery({
        queryKey: queryKeys.resources,
        queryFn: () => fetch('/api/resources').then(res => res.json()),
        staleTime: 10 * 60 * 1000
      }),
      queryClient.prefetchQuery({
        queryKey: queryKeys.complianceScore,
        queryFn: () => fetch('/api/compliance/score').then(res => res.json()),
        staleTime: 5 * 60 * 1000 // 5 minutes
      })
    ])
  }
}