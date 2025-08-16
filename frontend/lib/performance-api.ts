/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

// High-Performance API Client with Intelligent Caching and Request Deduplication
import { QueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

// In-memory cache with LRU eviction
class PerformanceCache {
  private cache = new Map<string, { data: any; timestamp: number; ttl: number }>()
  private maxSize = 1000
  private accessOrder = new Map<string, number>()
  private accessCounter = 0

  set(key: string, data: any, ttl: number = 300000) { // 5 min default TTL
    // LRU eviction if cache is full
    if (this.cache.size >= this.maxSize) {
      const oldestKey = Array.from(this.accessOrder.entries())
        .sort(([,a], [,b]) => a - b)[0][0]
      this.cache.delete(oldestKey)
      this.accessOrder.delete(oldestKey)
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    })
    this.accessOrder.set(key, ++this.accessCounter)
  }

  get(key: string): any | null {
    const entry = this.cache.get(key)
    if (!entry) return null

    // Check TTL
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key)
      this.accessOrder.delete(key)
      return null
    }

    // Update access order
    this.accessOrder.set(key, ++this.accessCounter)
    return entry.data
  }

  invalidate(pattern: string) {
    const keysToDelete = Array.from(this.cache.keys()).filter(key => 
      key.includes(pattern) || new RegExp(pattern).test(key)
    )
    
    keysToDelete.forEach(key => {
      this.cache.delete(key)
      this.accessOrder.delete(key)
    })
  }

  getStats() {
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hitRate: this.accessCounter > 0 ? (this.cache.size / this.accessCounter) : 0
    }
  }
}

// Request deduplication to prevent duplicate API calls
class RequestDeduplicator {
  private pendingRequests = new Map<string, Promise<any>>()

  async dedupe<T>(key: string, requestFn: () => Promise<T>): Promise<T> {
    // If request is already pending, return the same promise
    if (this.pendingRequests.has(key)) {
      return this.pendingRequests.get(key)!
    }

    // Create new request
    const promise = requestFn().finally(() => {
      // Clean up after request completes
      this.pendingRequests.delete(key)
    })

    this.pendingRequests.set(key, promise)
    return promise
  }

  getPendingCount(): number {
    return this.pendingRequests.size
  }
}

// Circuit breaker to handle API failures gracefully
class CircuitBreaker {
  private failureCount = 0
  private lastFailureTime = 0
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED'
  
  constructor(
    private maxFailures = 5,
    private resetTimeout = 60000 // 1 minute
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime > this.resetTimeout) {
        this.state = 'HALF_OPEN'
      } else {
        throw new Error('Circuit breaker is OPEN - service temporarily unavailable')
      }
    }

    try {
      const result = await fn()
      this.onSuccess()
      return result
    } catch (error) {
      this.onFailure()
      throw error
    }
  }

  private onSuccess() {
    this.failureCount = 0
    this.state = 'CLOSED'
  }

  private onFailure() {
    this.failureCount++
    this.lastFailureTime = Date.now()
    
    if (this.failureCount >= this.maxFailures) {
      this.state = 'OPEN'
    }
  }

  getState() {
    return {
      state: this.state,
      failureCount: this.failureCount,
      lastFailureTime: this.lastFailureTime
    }
  }
}

// High-performance API client
class PerformanceApiClient {
  private cache = new PerformanceCache()
  private deduplicator = new RequestDeduplicator()
  private circuitBreaker = new CircuitBreaker()
  private baseURL: string
  private abortController = new AbortController()
  
  // Connection pooling simulation (browser handles actual pooling)
  private maxConcurrentRequests = 6
  private requestQueue: Array<() => void> = []
  private activeRequests = 0

  constructor(baseURL = '') {
    this.baseURL = baseURL
    
    // Warm up critical endpoints
    this.warmCache()
  }

  private async warmCache() {
    try {
      // Pre-fetch critical data that's likely to be needed
      await Promise.allSettled([
        // Avoid warming protected endpoints here (may require auth); warm only health
        this.get('/health', { cache: 'cold' })
      ])
    } catch (error) {
      console.debug('Cache warming failed:', error)
    }
  }

  private async executeWithBackpressure<T>(fn: () => Promise<T>): Promise<T> {
    if (this.activeRequests >= this.maxConcurrentRequests) {
      // Queue the request
      await new Promise<void>(resolve => {
        this.requestQueue.push(resolve)
      })
    }

    this.activeRequests++
    
    try {
      return await fn()
    } finally {
      this.activeRequests--
      
      // Process next queued request
      const next = this.requestQueue.shift()
      if (next) next()
    }
  }

  async get<T>(
    endpoint: string, 
    options: {
      cache?: 'hot' | 'warm' | 'cold' | 'none'
      ttl?: number
      headers?: Record<string, string>
      timeout?: number
    } = {}
  ): Promise<T> {
    const { cache = 'warm', ttl, headers = {}, timeout = 30000 } = options
    const cacheKey = `GET:${endpoint}:${JSON.stringify(headers)}`

    // Try cache first (except for 'none' cache mode)
    if (cache !== 'none') {
      const cached = this.cache.get(cacheKey)
      if (cached) {
        console.debug(`Cache HIT (${cache}):`, endpoint)
        return cached
      }
    }

    // Deduplicate identical requests
    return this.deduplicator.dedupe(cacheKey, async () => {
      return this.executeWithBackpressure(async () => {
        return this.circuitBreaker.execute(async () => {
          if (process.env.NODE_ENV === 'development') {
            console.debug(`API CALL (${cache}):`, endpoint)
          }
          
          const controller = new AbortController()
          const timeoutId = setTimeout(() => controller.abort(), timeout)

          try {
            const response = await fetch(`${this.baseURL}${endpoint}`, {
              method: 'GET',
              headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, br',
                ...headers
              },
              signal: controller.signal,
            })

            if (!response.ok) {
              throw new Error(`HTTP ${response.status}: ${response.statusText}`)
            }

            const data = await response.json()

            // Cache based on cache mode
            if (cache !== 'none') {
              const cacheTTL = ttl || this.getCacheTTL(cache)
              this.cache.set(cacheKey, data, cacheTTL)
            }

            return data
          } finally {
            clearTimeout(timeoutId)
          }
        })
      })
    })
  }

  async post<T>(
    endpoint: string,
    body: any,
    options: {
      headers?: Record<string, string>
      timeout?: number
      invalidateCache?: string[]
    } = {}
  ): Promise<T> {
    const { headers = {}, timeout = 30000, invalidateCache = [] } = options

    return this.executeWithBackpressure(async () => {
      return this.circuitBreaker.execute(async () => {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), timeout)

        try {
          const response = await fetch(`${this.baseURL}${endpoint}`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
              ...headers
            },
            body: JSON.stringify(body),
            signal: controller.signal,
          })

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`)
          }

          const data = await response.json()

          // Invalidate related cache entries
          invalidateCache.forEach(pattern => {
            this.cache.invalidate(pattern)
          })

          return data
        } finally {
          clearTimeout(timeoutId)
        }
      })
    })
  }

  // Batch multiple requests for efficiency
  async batch<T>(requests: Array<{
    endpoint: string
    options?: any
  }>): Promise<Array<T | Error>> {
    // Execute requests in parallel with concurrency control
    const chunks = []
    const chunkSize = this.maxConcurrentRequests
    
    for (let i = 0; i < requests.length; i += chunkSize) {
      chunks.push(requests.slice(i, i + chunkSize))
    }

    const results: Array<T | Error> = []

    for (const chunk of chunks) {
      const chunkPromises = chunk.map(async ({ endpoint, options = {} }) => {
        try {
          return await this.get<T>(endpoint, options)
        } catch (error) {
          return error instanceof Error ? error : new Error(String(error))
        }
      })

      const chunkResults = await Promise.all(chunkPromises)
      results.push(...chunkResults)
    }

    return results
  }

  // Stream large datasets
  async *stream<T>(
    endpoint: string,
    options: {
      pageSize?: number
      maxPages?: number
    } = {}
  ): AsyncIterableIterator<T[]> {
    const { pageSize = 100, maxPages = Infinity } = options
    let page = 0
    let hasMore = true

    while (hasMore && page < maxPages) {
      try {
        const response = await this.get<{
          data: T[]
          hasMore: boolean
          nextPage?: number
        }>(`${endpoint}?page=${page}&limit=${pageSize}`, {
          cache: 'none' // Don't cache streamed data
        })

        if (response.data.length > 0) {
          yield response.data
        }

        hasMore = response.hasMore
        page++
      } catch (error) {
        console.error('Stream error:', error)
        break
      }
    }
  }

  private getCacheTTL(cacheMode: string): number {
    switch (cacheMode) {
      case 'hot': return 30000    // 30 seconds
      case 'warm': return 300000  // 5 minutes  
      case 'cold': return 1800000 // 30 minutes
      default: return 300000      // 5 minutes default
    }
  }

  // Performance monitoring
  getStats() {
    return {
      cache: this.cache.getStats(),
      circuitBreaker: this.circuitBreaker.getState(),
      pendingRequests: this.deduplicator.getPendingCount(),
      activeRequests: this.activeRequests,
      queuedRequests: this.requestQueue.length
    }
  }

  // Cache management
  invalidateCache(pattern: string) {
    this.cache.invalidate(pattern)
  }

  // Graceful shutdown
  shutdown() {
    this.abortController.abort()
  }
}

// Create singleton instance
export const performanceApi = new PerformanceApiClient()

// Enhanced React Query client with performance optimizations
export const createPerformanceQueryClient = () => {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // Use our performance API client
        queryFn: async ({ queryKey, signal }) => {
          const [endpoint, options = {}] = queryKey as [string, any?]
          
          try {
            return await performanceApi.get(endpoint, {
              ...options,
              // Convert React Query signal to our timeout
              timeout: signal?.aborted ? 0 : (options.timeout || 30000)
            })
          } catch (error) {
            // Enhanced error handling
            if (error instanceof Error) {
              if (error.message.includes('Circuit breaker')) {
                toast.error('Service temporarily unavailable - please try again later')
              } else if (error.message.includes('timeout')) {
                toast.error('Request timed out - check your connection')
              } else {
                toast.error('An unexpected error occurred')
              }
            }
            throw error
          }
        },

        // Aggressive caching for performance
        staleTime: 5 * 60 * 1000, // 5 minutes
        gcTime: 30 * 60 * 1000,   // 30 minutes
        
        // Smart retries
        retry: (failureCount, error: any) => {
          // Don't retry 4xx errors or circuit breaker errors
          if (error?.message?.includes('HTTP 4') || error?.message?.includes('Circuit breaker')) {
            return false
          }
          return failureCount < 3
        },
        
        retryDelay: (attemptIndex) => Math.min(1000 * Math.pow(2, attemptIndex), 30000),
        
        // Background updates for real-time feel
        refetchOnWindowFocus: false,
        refetchOnReconnect: true,
        refetchInterval: false, // Controlled per-query
        
        // Performance optimizations
        notifyOnChangeProps: ['data', 'error', 'isLoading'],
        structuralSharing: true,
      },
      mutations: {
        retry: (failureCount, error: any) => {
          if (error?.message?.includes('HTTP 4')) return false
          return failureCount < 2
        },
        retryDelay: (attemptIndex) => Math.min(1000 * Math.pow(2, attemptIndex), 5000),
      },
    },
  })
}

// Query key factories for consistency
export const queryKeys = {
  governance: {
    metrics: (filters?: any) => ['api/v1/metrics', { cache: 'hot', filters }] as const,
    policies: (search?: string) => ['api/v1/policies', { cache: 'warm', search }] as const,
    resources: (filters?: any) => ['api/v1/resources', { cache: 'warm', filters }] as const,
    compliance: (dateRange?: any) => ['api/v1/compliance', { cache: 'hot', dateRange }] as const,
  },
  ai: {
    predictions: () => ['api/v1/predictions', { cache: 'warm' }] as const,
    recommendations: () => ['api/v1/recommendations', { cache: 'hot' }] as const,
    correlations: () => ['api/v1/correlations', { cache: 'warm' }] as const,
    conversation: (sessionId: string) => ['api/v1/conversation', { cache: 'none', sessionId }] as const,
  },
  realtime: {
    alerts: () => ['api/v1/alerts', { cache: 'hot', ttl: 10000 }] as const,
    status: () => ['api/v1/status', { cache: 'hot', ttl: 5000 }] as const,
  }
} as const

// Performance monitoring hook
export function usePerformanceStats() {
  return performanceApi.getStats()
}

// Batch loading hook
export function useBatchQuery<T>(
  requests: Array<{ endpoint: string; options?: any }>,
  options?: any
) {
  return {
    queryKey: ['batch', requests],
    queryFn: () => performanceApi.batch<T>(requests),
    ...options
  }
}

export default performanceApi