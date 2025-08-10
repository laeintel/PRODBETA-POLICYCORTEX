/**
 * Offline queue management with IndexedDB
 */

import { v4 as uuidv4 } from 'uuid'
import { useEffect, useState } from 'react'

export interface QueuedRequest {
  id: string
  url: string
  method: string
  headers: Record<string, string>
  body?: any
  timestamp: number
  retryCount: number
  maxRetries: number
  status: 'pending' | 'processing' | 'failed' | 'completed'
  error?: string
}

export interface ConflictResolution {
  strategy: 'client-wins' | 'server-wins' | 'merge' | 'manual'
  resolver?: (client: any, server: any) => any
}

class OfflineQueue {
  private dbName = 'policycortex-offline'
  private storeName = 'requests'
  private db: IDBDatabase | null = null
  private syncInProgress = false
  private listeners: Map<string, (result: any) => void> = new Map()

  async init() {
    return new Promise<void>((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1)

      request.onerror = () => reject(request.error)
      request.onsuccess = () => {
        this.db = request.result
        resolve()
      }

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result
        
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, { keyPath: 'id' })
          store.createIndex('status', 'status', { unique: false })
          store.createIndex('timestamp', 'timestamp', { unique: false })
        }
      }
    })
  }

  async add(request: Omit<QueuedRequest, 'id' | 'timestamp' | 'retryCount' | 'status'>): Promise<string> {
    if (!this.db) await this.init()

    const id = uuidv4()
    const queuedRequest: QueuedRequest = {
      ...request,
      id,
      timestamp: Date.now(),
      retryCount: 0,
      status: 'pending'
    }

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction([this.storeName], 'readwrite')
      const store = tx.objectStore(this.storeName)
      const request = store.add(queuedRequest)

      request.onsuccess = () => {
        resolve(id)
        this.notifyQueueChange()
      }
      request.onerror = () => reject(request.error)
    })
  }

  async get(id: string): Promise<QueuedRequest | null> {
    if (!this.db) await this.init()

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction([this.storeName], 'readonly')
      const store = tx.objectStore(this.storeName)
      const request = store.get(id)

      request.onsuccess = () => resolve(request.result || null)
      request.onerror = () => reject(request.error)
    })
  }

  async getAll(status?: QueuedRequest['status']): Promise<QueuedRequest[]> {
    if (!this.db) await this.init()

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction([this.storeName], 'readonly')
      const store = tx.objectStore(this.storeName)
      
      const request = status
        ? store.index('status').getAll(status)
        : store.getAll()

      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
  }

  async update(id: string, updates: Partial<QueuedRequest>): Promise<void> {
    if (!this.db) await this.init()

    const existing = await this.get(id)
    if (!existing) throw new Error('Request not found')

    const updated = { ...existing, ...updates }

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction([this.storeName], 'readwrite')
      const store = tx.objectStore(this.storeName)
      const request = store.put(updated)

      request.onsuccess = () => {
        resolve()
        this.notifyQueueChange()
      }
      request.onerror = () => reject(request.error)
    })
  }

  async remove(id: string): Promise<void> {
    if (!this.db) await this.init()

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction([this.storeName], 'readwrite')
      const store = tx.objectStore(this.storeName)
      const request = store.delete(id)

      request.onsuccess = () => {
        resolve()
        this.notifyQueueChange()
      }
      request.onerror = () => reject(request.error)
    })
  }

  async clear(): Promise<void> {
    if (!this.db) await this.init()

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction([this.storeName], 'readwrite')
      const store = tx.objectStore(this.storeName)
      const request = store.clear()

      request.onsuccess = () => {
        resolve()
        this.notifyQueueChange()
      }
      request.onerror = () => reject(request.error)
    })
  }

  async sync(): Promise<void> {
    if (this.syncInProgress) return
    this.syncInProgress = true

    try {
      const pending = await this.getAll('pending')
      
      for (const item of pending) {
        await this.processRequest(item)
      }
    } finally {
      this.syncInProgress = false
    }
  }

  private async processRequest(item: QueuedRequest): Promise<void> {
    try {
      await this.update(item.id, { status: 'processing' })

      const response = await fetch(item.url, {
        method: item.method,
        headers: item.headers,
        body: item.body ? JSON.stringify(item.body) : undefined
      })

      if (response.ok) {
        await this.update(item.id, { status: 'completed' })
        const result = await response.json()
        this.notifyListener(item.id, { success: true, data: result })
        
        // Clean up completed request after 5 seconds
        setTimeout(() => this.remove(item.id), 5000)
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      
      if (item.retryCount < item.maxRetries) {
        await this.update(item.id, {
          status: 'pending',
          retryCount: item.retryCount + 1,
          error: errorMessage
        })
      } else {
        await this.update(item.id, {
          status: 'failed',
          error: errorMessage
        })
        this.notifyListener(item.id, { success: false, error: errorMessage })
      }
    }
  }

  onResult(requestId: string, callback: (result: any) => void) {
    this.listeners.set(requestId, callback)
  }

  private notifyListener(requestId: string, result: any) {
    const listener = this.listeners.get(requestId)
    if (listener) {
      listener(result)
      this.listeners.delete(requestId)
    }
  }

  private notifyQueueChange() {
    window.dispatchEvent(new CustomEvent('offline-queue-change'))
  }

  async getQueueSize(): Promise<number> {
    const all = await this.getAll()
    return all.filter(item => item.status === 'pending').length
  }
}

// Singleton instance
export const offlineQueue = new OfflineQueue()

/**
 * Optimistic concurrency control with ETags
 */
export class OptimisticConcurrency {
  private etagCache: Map<string, string> = new Map()

  setETag(resource: string, etag: string) {
    this.etagCache.set(resource, etag)
  }

  getETag(resource: string): string | undefined {
    return this.etagCache.get(resource)
  }

  async fetchWithETag(url: string, options?: RequestInit): Promise<Response> {
    const etag = this.getETag(url)
    const headers = new Headers(options?.headers)
    
    if (etag) {
      headers.set('If-Match', etag)
    }

    const response = await fetch(url, { ...options, headers })
    
    // Update ETag from response
    const newETag = response.headers.get('ETag')
    if (newETag) {
      this.setETag(url, newETag)
    }

    // Handle 412 Precondition Failed (conflict)
    if (response.status === 412) {
      throw new ConflictError('Resource has been modified', url)
    }

    return response
  }

  async updateWithRetry(
    url: string,
    data: any,
    conflictResolution: ConflictResolution
  ): Promise<any> {
    const maxRetries = 3
    let retries = 0

    while (retries < maxRetries) {
      try {
        const response = await this.fetchWithETag(url, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        })

        if (response.ok) {
          return await response.json()
        }
      } catch (error) {
        if (error instanceof ConflictError) {
          // Handle conflict based on strategy
          const resolved = await this.resolveConflict(url, data, conflictResolution)
          if (resolved) {
            data = resolved
            retries++
            continue
          }
        }
        throw error
      }
    }

    throw new Error('Max retries exceeded')
  }

  private async resolveConflict(
    url: string,
    clientData: any,
    resolution: ConflictResolution
  ): Promise<any> {
    // Fetch current server version
    const response = await fetch(url)
    const serverData = await response.json()
    const newETag = response.headers.get('ETag')
    
    if (newETag) {
      this.setETag(url, newETag)
    }

    switch (resolution.strategy) {
      case 'client-wins':
        return clientData
      
      case 'server-wins':
        return null // Don't retry, accept server version
      
      case 'merge':
        return resolution.resolver
          ? resolution.resolver(clientData, serverData)
          : { ...serverData, ...clientData }
      
      case 'manual':
        // Emit event for manual resolution
        const event = new CustomEvent('conflict-detected', {
          detail: { url, clientData, serverData }
        })
        window.dispatchEvent(event)
        return null
      
      default:
        return null
    }
  }
}

export class ConflictError extends Error {
  constructor(message: string, public url: string) {
    super(message)
    this.name = 'ConflictError'
  }
}

/**
 * Offline-aware fetch wrapper
 */
export async function offlineFetch(
  url: string,
  options?: RequestInit & { 
    offline?: boolean
    conflictResolution?: ConflictResolution
    maxRetries?: number
  }
): Promise<Response> {
  const isOnline = typeof window !== 'undefined' ? navigator.onLine : true

  if (!isOnline && options?.offline !== false) {
    // Queue request for later
    const id = await offlineQueue.add({
      url,
      method: options?.method || 'GET',
      headers: Object.fromEntries(new Headers(options?.headers)),
      body: options?.body ? JSON.parse(options.body as string) : undefined,
      maxRetries: options?.maxRetries || 3
    })

    // Return a synthetic response
    return new Response(
      JSON.stringify({
        queued: true,
        id,
        message: 'Request queued for sync when online'
      }),
      {
        status: 202,
        headers: { 'Content-Type': 'application/json' }
      }
    )
  }

  // Online - use optimistic concurrency if configured
  if (options?.conflictResolution) {
    const concurrency = new OptimisticConcurrency()
    return concurrency.fetchWithETag(url, options)
  }

  return fetch(url, options)
}

/**
 * Hook for offline status
 */
export function useOfflineStatus() {
  const [isOnline, setIsOnline] = useState(typeof window !== 'undefined' ? navigator.onLine : true)
  const [queueSize, setQueueSize] = useState(0)

  useEffect(() => {
    const updateOnlineStatus = () => setIsOnline(navigator.onLine)
    const updateQueueSize = async () => {
      const size = await offlineQueue.getQueueSize()
      setQueueSize(size)
    }

    window.addEventListener('online', updateOnlineStatus)
    window.addEventListener('offline', updateOnlineStatus)
    window.addEventListener('offline-queue-change', updateQueueSize)

    // Sync when coming back online
    window.addEventListener('online', () => {
      offlineQueue.sync()
    })

    // Initial queue size
    updateQueueSize()

    return () => {
      window.removeEventListener('online', updateOnlineStatus)
      window.removeEventListener('offline', updateOnlineStatus)
      window.removeEventListener('offline-queue-change', updateQueueSize)
    }
  }, [])

  return { isOnline, queueSize }
}

/**
 * Hook for conflict resolution
 */
export function useConflictResolution() {
  const [conflicts, setConflicts] = useState<any[]>([])

  useEffect(() => {
    const handleConflict = (event: CustomEvent) => {
      setConflicts(prev => [...prev, event.detail])
    }

    window.addEventListener('conflict-detected', handleConflict as any)
    return () => {
      window.removeEventListener('conflict-detected', handleConflict as any)
    }
  }, [])

  const resolveConflict = (index: number, resolution: any) => {
    setConflicts(prev => prev.filter((_, i) => i !== index))
    // Emit resolution event
    window.dispatchEvent(new CustomEvent('conflict-resolved', { detail: resolution }))
  }

  return { conflicts, resolveConflict }
}