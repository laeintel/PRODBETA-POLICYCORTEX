/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

/**
 * Offline Manager for PolicyCortex
 * Handles offline data synchronization and conflict resolution
 */

import { openDB, IDBPDatabase } from 'idb'
import { useEffect, useState } from 'react'

interface OfflineAction {
  id: string
  type: 'CREATE' | 'UPDATE' | 'DELETE'
  resource: string
  data: any
  timestamp: number
  synced: boolean
  conflicted?: boolean
  error?: string
}

interface CachedData {
  key: string
  data: any
  timestamp: number
  expiresAt: number
}

interface ConflictResolution {
  strategy: 'client-wins' | 'server-wins' | 'merge' | 'manual'
  resolver?: (clientData: any, serverData: any) => any
}

class OfflineManager {
  private db: IDBPDatabase | null = null
  private syncQueue: OfflineAction[] = []
  private isOnline: boolean = typeof window !== 'undefined' ? navigator.onLine : true
  private syncInProgress: boolean = false
  private listeners: Map<string, Set<Function>> = new Map()
  
  constructor() {
    if (typeof window !== 'undefined') {
      this.initializeDB()
      this.setupEventListeners()
      this.startSyncWorker()
    }
  }

  private async initializeDB() {
    this.db = await openDB('PolicyCortexOffline', 1, {
      upgrade(db) {
        // Store for offline actions
        if (!db.objectStoreNames.contains('actions')) {
          const actionStore = db.createObjectStore('actions', { keyPath: 'id' })
          actionStore.createIndex('synced', 'synced')
          actionStore.createIndex('timestamp', 'timestamp')
        }
        
        // Store for cached data
        if (!db.objectStoreNames.contains('cache')) {
          const cacheStore = db.createObjectStore('cache', { keyPath: 'key' })
          cacheStore.createIndex('expiresAt', 'expiresAt')
        }
        
        // Store for conflict resolution
        if (!db.objectStoreNames.contains('conflicts')) {
          db.createObjectStore('conflicts', { keyPath: 'id' })
        }
      }
    })
  }

  private setupEventListeners() {
    window.addEventListener('online', () => {
      this.isOnline = true
      this.emit('online')
      this.syncPendingActions()
    })
    
    window.addEventListener('offline', () => {
      this.isOnline = false
      this.emit('offline')
    })
  }

  private startSyncWorker() {
    // Sync every 30 seconds when online
    setInterval(() => {
      if (this.isOnline && !this.syncInProgress) {
        this.syncPendingActions()
      }
    }, 30000)
    
    // Clean expired cache every 5 minutes
    setInterval(() => {
      this.cleanExpiredCache()
    }, 300000)
  }

  // Queue an action for offline sync
  async queueAction(type: OfflineAction['type'], resource: string, data: any): Promise<void> {
    if (!this.db) return
    
    const action: OfflineAction = {
      id: `${Date.now()}-${Math.random()}`,
      type,
      resource,
      data,
      timestamp: Date.now(),
      synced: false
    }
    
    await this.db.add('actions', action)
    this.syncQueue.push(action)
    
    // Try to sync immediately if online
    if (this.isOnline) {
      this.syncPendingActions()
    }
    
    this.emit('action-queued', action)
  }

  // Cache data for offline access
  async cacheData(key: string, data: any, ttl: number = 3600000): Promise<void> {
    if (!this.db) return
    
    const cached: CachedData = {
      key,
      data,
      timestamp: Date.now(),
      expiresAt: Date.now() + ttl
    }
    
    await this.db.put('cache', cached)
  }

  // Get cached data
  async getCachedData(key: string): Promise<any> {
    if (!this.db) return null
    
    const cached = await this.db.get('cache', key)
    
    if (!cached) return null
    
    // Check if expired
    if (cached.expiresAt < Date.now()) {
      await this.db.delete('cache', key)
      return null
    }
    
    return cached.data
  }

  // Sync pending actions
  async syncPendingActions(): Promise<void> {
    if (!this.isOnline || this.syncInProgress || !this.db) return
    
    this.syncInProgress = true
    this.emit('sync-start')
    
    try {
      // Get all unsynced actions (index value equals false)
      const actions = await this.db.getAllFromIndex('actions', 'synced', IDBKeyRange.only(false))
      
      for (const action of actions) {
        try {
          await this.syncAction(action)
          
          // Mark as synced
          action.synced = true
          await this.db.put('actions', action)
          
          this.emit('action-synced', action)
        } catch (error: any) {
          // Handle conflict
          if (error.status === 409) {
            await this.handleConflict(action, error.serverData)
          } else {
            action.error = error.message
            await this.db.put('actions', action)
            this.emit('action-error', { action, error })
          }
        }
      }
      
      this.emit('sync-complete')
    } catch (error) {
      this.emit('sync-error', error)
    } finally {
      this.syncInProgress = false
    }
  }

  // Sync individual action
  private async syncAction(action: OfflineAction): Promise<void> {
    const endpoint = `/api/v1/${action.resource}`
    
    let method: string
    let url: string
    let body: any = action.data
    
    switch (action.type) {
      case 'CREATE':
        method = 'POST'
        url = endpoint
        break
      case 'UPDATE':
        method = 'PATCH'
        url = `${endpoint}/${action.data.id}`
        break
      case 'DELETE':
        method = 'DELETE'
        url = `${endpoint}/${action.data.id}`
        body = undefined
        break
    }
    
    const response = await fetch(url, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'X-Offline-Action-ID': action.id,
        'X-Offline-Timestamp': action.timestamp.toString()
      },
      body: body ? JSON.stringify(body) : undefined
    })
    
    if (!response.ok) {
      const error: any = new Error(`Sync failed: ${response.statusText}`)
      error.status = response.status
      
      if (response.status === 409) {
        error.serverData = await response.json()
      }
      
      throw error
    }
  }

  // Handle conflict resolution
  private async handleConflict(
    action: OfflineAction, 
    serverData: any,
    resolution: ConflictResolution = { strategy: 'server-wins' }
  ): Promise<void> {
    if (!this.db) return
    
    const conflict = {
      id: action.id,
      action,
      serverData,
      clientData: action.data,
      timestamp: Date.now(),
      resolved: false
    }
    
    await this.db.put('conflicts', conflict)
    
    let resolvedData: any
    
    switch (resolution.strategy) {
      case 'client-wins':
        resolvedData = action.data
        break
      
      case 'server-wins':
        resolvedData = serverData
        break
      
      case 'merge':
        resolvedData = this.mergeData(action.data, serverData)
        break
      
      case 'manual':
        // Emit event for manual resolution
        this.emit('conflict-detected', conflict)
        return
    }
    
    // Apply resolved data
    if (resolvedData) {
      await this.applyResolvedData(action.resource, resolvedData)
      conflict.resolved = true
      await this.db.put('conflicts', conflict)
      this.emit('conflict-resolved', conflict)
    }
  }

  // Merge data (simple strategy - can be customized)
  private mergeData(clientData: any, serverData: any): any {
    // Use server data as base, overlay client changes
    const merged = { ...serverData }
    
    for (const key in clientData) {
      if (key === 'id' || key === 'created_at') continue
      
      // Use client value if it's newer (simplified logic)
      if (clientData[key] !== undefined) {
        merged[key] = clientData[key]
      }
    }
    
    merged._merged = true
    merged._merge_timestamp = Date.now()
    
    return merged
  }

  // Apply resolved data
  private async applyResolvedData(resource: string, data: any): Promise<void> {
    const response = await fetch(`/api/v1/${resource}/${data.id}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'X-Conflict-Resolution': 'true'
      },
      body: JSON.stringify(data)
    })
    
    if (!response.ok) {
      throw new Error(`Failed to apply resolved data: ${response.statusText}`)
    }
  }

  // Clean expired cache
  private async cleanExpiredCache(): Promise<void> {
    if (!this.db) return
    
    const now = Date.now()
    const tx = this.db.transaction('cache', 'readwrite')
    const index = tx.objectStore('cache').index('expiresAt')
    
    for await (const cursor of index.iterate(IDBKeyRange.upperBound(now))) {
      await cursor.delete()
    }
    
    await tx.done
  }

  // Get pending actions count
  async getPendingActionsCount(): Promise<number> {
    if (!this.db) return 0
    
    const actions = await this.db.getAllFromIndex('actions', 'synced', IDBKeyRange.only(false))
    return actions.length
  }

  // Get conflicts
  async getConflicts(): Promise<any[]> {
    if (!this.db) return []
    
    return await this.db.getAll('conflicts')
  }

  // Resolve conflict manually
  async resolveConflict(conflictId: string, resolution: 'client' | 'server' | any): Promise<void> {
    if (!this.db) return
    
    const conflict = await this.db.get('conflicts', conflictId)
    if (!conflict) return
    
    let resolvedData: any
    
    if (resolution === 'client') {
      resolvedData = conflict.clientData
    } else if (resolution === 'server') {
      resolvedData = conflict.serverData
    } else {
      resolvedData = resolution // Custom resolved data
    }
    
    await this.applyResolvedData(conflict.action.resource, resolvedData)
    
    conflict.resolved = true
    conflict.resolution = resolution
    await this.db.put('conflicts', conflict)
    
    // Remove from actions queue
    await this.db.delete('actions', conflictId)
    
    this.emit('conflict-resolved', conflict)
  }

  // Event emitter methods
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set())
    }
    this.listeners.get(event)!.add(callback)
  }

  off(event: string, callback: Function): void {
    this.listeners.get(event)?.delete(callback)
  }

  private emit(event: string, data?: any): void {
    this.listeners.get(event)?.forEach(callback => callback(data))
  }

  // Check if online
  getIsOnline(): boolean {
    return this.isOnline
  }

  // Clear all offline data
  async clearOfflineData(): Promise<void> {
    if (!this.db) return
    
    await this.db.clear('actions')
    await this.db.clear('cache')
    await this.db.clear('conflicts')
    
    this.syncQueue = []
    this.emit('offline-data-cleared')
  }

  // Export offline data for debugging
  async exportOfflineData(): Promise<any> {
    if (!this.db) return null
    
    return {
      actions: await this.db.getAll('actions'),
      cache: await this.db.getAll('cache'),
      conflicts: await this.db.getAll('conflicts'),
      isOnline: this.isOnline,
      syncInProgress: this.syncInProgress
    }
  }
}

// Singleton instance - only create on client side
export const offlineManager = typeof window !== 'undefined' ? new OfflineManager() : null as any

// React hook for offline status
export function useOfflineStatus() {
  const [isOnline, setIsOnline] = useState(typeof window !== 'undefined' ? offlineManager?.getIsOnline() ?? true : true)
  const [pendingActions, setPendingActions] = useState(0)
  const [conflicts, setConflicts] = useState<any[]>([])
  
  useEffect(() => {
    // Skip on SSR
    if (typeof window === 'undefined' || !offlineManager) return
    
    const updateStatus = () => setIsOnline(offlineManager.getIsOnline())
    const updatePending = async () => {
      const count = await offlineManager.getPendingActionsCount()
      setPendingActions(count)
    }
    const updateConflicts = async () => {
      const conflictsList = await offlineManager.getConflicts()
      setConflicts(conflictsList.filter((c: any) => !c.resolved))
    }
    
    offlineManager.on('online', updateStatus)
    offlineManager.on('offline', updateStatus)
    offlineManager.on('action-queued', updatePending)
    offlineManager.on('action-synced', updatePending)
    offlineManager.on('conflict-detected', updateConflicts)
    offlineManager.on('conflict-resolved', updateConflicts)
    
    // Initial load
    updatePending()
    updateConflicts()
    
    return () => {
      offlineManager.off('online', updateStatus)
      offlineManager.off('offline', updateStatus)
      offlineManager.off('action-queued', updatePending)
      offlineManager.off('action-synced', updatePending)
      offlineManager.off('conflict-detected', updateConflicts)
      offlineManager.off('conflict-resolved', updateConflicts)
    }
  }, [])
  
  return {
    isOnline,
    pendingActions,
    conflicts,
    syncNow: () => offlineManager?.syncPendingActions?.() ?? Promise.resolve(),
    resolveConflict: (id: string, resolution: any) => offlineManager?.resolveConflict?.(id, resolution) ?? Promise.resolve(),
    clearOfflineData: () => offlineManager?.clearOfflineData?.() ?? Promise.resolve()
  }
}

// Export for use in other modules
export default offlineManager