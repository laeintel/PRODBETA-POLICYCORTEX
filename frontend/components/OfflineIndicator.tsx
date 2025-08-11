'use client'

import { useOfflineStatus } from '@/lib/offline-queue'
import { useState, useEffect } from 'react'
import { WifiOff, Wifi, Cloud, CloudOff, AlertCircle } from 'lucide-react'

export function OfflineIndicator() {
  const { isOnline, queueSize } = useOfflineStatus()
  const [showNotification, setShowNotification] = useState(false)
  const [lastOnlineStatus, setLastOnlineStatus] = useState(isOnline)

  useEffect(() => {
    // Show notification when status changes
    if (lastOnlineStatus !== isOnline) {
      setShowNotification(true)
      setLastOnlineStatus(isOnline)
      
      // Hide notification after 5 seconds
      const timer = setTimeout(() => setShowNotification(false), 5000)
      return () => clearTimeout(timer)
    }
  }, [isOnline, lastOnlineStatus])

  if (isOnline && queueSize === 0 && !showNotification) {
    return null
  }

  return (
    <>
      {/* Status bar indicator */}
      <div className="fixed top-4 right-4 z-50 flex items-center space-x-2">
        {!isOnline && (
          <div className="flex items-center space-x-2 px-3 py-2 bg-yellow-100/90 dark:bg-yellow-900/60 backdrop-blur-md text-yellow-800 dark:text-yellow-200 rounded-lg shadow-[0_10px_30px_-10px_rgba(0,0,0,0.4)]">
            <WifiOff className="w-4 h-4" />
            <span className="text-sm font-medium">Offline Mode</span>
          </div>
        )}
        
        {queueSize > 0 && (
          <div className="flex items-center space-x-2 px-3 py-2 bg-blue-100/90 dark:bg-blue-900/60 backdrop-blur-md text-blue-800 dark:text-blue-200 rounded-lg shadow-[0_10px_30px_-10px_rgba(0,0,0,0.4)]">
            <CloudOff className="w-4 h-4" />
            <span className="text-sm font-medium">{queueSize} pending</span>
          </div>
        )}
      </div>

      {/* Notification toast */}
      {showNotification && (
        <div 
          className={`fixed bottom-4 right-4 z-50 max-w-sm p-4 rounded-lg shadow-[0_10px_30px_-10px_rgba(0,0,0,0.5)] backdrop-blur-md transition-all duration-300 ${
            isOnline 
              ? 'bg-green-100/90 dark:bg-green-900/60 text-green-800 dark:text-green-200' 
              : 'bg-yellow-100/90 dark:bg-yellow-900/60 text-yellow-800 dark:text-yellow-200'
          }`}
          role="alert"
          aria-live="polite"
        >
          <div className="flex items-start space-x-3">
            {isOnline ? (
              <Wifi className="w-5 h-5 mt-0.5" />
            ) : (
              <WifiOff className="w-5 h-5 mt-0.5" />
            )}
            <div className="flex-1">
              <h3 className="font-semibold">
                {isOnline ? 'Back Online' : 'Working Offline'}
              </h3>
              <p className="text-sm mt-1">
                {isOnline 
                  ? 'Your connection has been restored. Any pending changes will be synced.'
                  : 'You can continue working. Changes will be saved locally and synced when online.'}
              </p>
              {queueSize > 0 && isOnline && (
                <p className="text-sm mt-2 font-medium">
                  Syncing {queueSize} pending {queueSize === 1 ? 'change' : 'changes'}...
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  )
}

export function OfflineQueue() {
  const { queueSize } = useOfflineStatus()
  const [queue, setQueue] = useState<any[]>([])
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    const loadQueue = async () => {
      const { offlineQueue } = await import('@/lib/offline-queue')
      const items = await offlineQueue.getAll('pending')
      setQueue(items)
    }

    loadQueue()
    
    // Reload on queue change
    const handleQueueChange = () => loadQueue()
    window.addEventListener('offline-queue-change', handleQueueChange)
    
    return () => {
      window.removeEventListener('offline-queue-change', handleQueueChange)
    }
  }, [])

  if (queueSize === 0) return null

  return (
    <div className="fixed bottom-20 right-4 z-40 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
        aria-expanded={expanded}
        aria-controls="offline-queue-details"
      >
        <div className="flex items-center space-x-2">
          <Cloud className="w-4 h-4 text-blue-600 dark:text-blue-400" />
          <span className="font-medium text-sm">
            {queueSize} Pending {queueSize === 1 ? 'Request' : 'Requests'}
          </span>
        </div>
        <svg
          className={`w-4 h-4 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div id="offline-queue-details" className="border-t border-gray-200 dark:border-gray-700 max-h-64 overflow-y-auto">
          {queue.map((item) => (
            <div
              key={item.id}
              className="px-4 py-3 border-b border-gray-100 dark:border-gray-700 last:border-b-0"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                    {item.method} {new URL(item.url).pathname}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {new Date(item.timestamp).toLocaleTimeString()}
                    {item.retryCount > 0 && ` • Retry ${item.retryCount}/${item.maxRetries}`}
                  </p>
                  {item.error && (
                    <p className="text-xs text-red-600 dark:text-red-400 mt-1 flex items-center">
                      <AlertCircle className="w-3 h-3 mr-1" />
                      {item.error}
                    </p>
                  )}
                </div>
                <div className="ml-2">
                  {item.status === 'processing' && (
                    <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                  )}
                  {item.status === 'pending' && (
                    <div className="w-2 h-2 bg-yellow-500 rounded-full" />
                  )}
                  {item.status === 'failed' && (
                    <AlertCircle className="w-4 h-4 text-red-600" />
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function ConflictResolver() {
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

  if (conflicts.length === 0) return null

  const resolveConflict = (index: number, useClient: boolean) => {
    const conflict = conflicts[index]
    const resolution = useClient ? conflict.clientData : conflict.serverData
    
    window.dispatchEvent(new CustomEvent('conflict-resolved', {
      detail: { url: conflict.url, resolution }
    }))
    
    setConflicts(prev => prev.filter((_, i) => i !== index))
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Resolve Conflicts
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Choose which version to keep for each conflicting change
          </p>
        </div>

        <div className="overflow-y-auto max-h-[60vh]">
          {conflicts.map((conflict, index) => (
            <div key={index} className="p-6 border-b border-gray-200 dark:border-gray-700">
              <h3 className="font-medium text-gray-900 dark:text-gray-100 mb-4">
                {new URL(conflict.url).pathname}
              </h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Your Version (Local)
                  </h4>
                  <pre className="text-xs bg-gray-100 dark:bg-gray-900 p-3 rounded overflow-x-auto">
                    {JSON.stringify(conflict.clientData, null, 2)}
                  </pre>
                  <button
                    onClick={() => resolveConflict(index, true)}
                    className="w-full px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors text-sm font-medium"
                  >
                    Use This Version
                  </button>
                </div>
                
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Server Version (Latest)
                  </h4>
                  <pre className="text-xs bg-gray-100 dark:bg-gray-900 p-3 rounded overflow-x-auto">
                    {JSON.stringify(conflict.serverData, null, 2)}
                  </pre>
                  <button
                    onClick={() => resolveConflict(index, false)}
                    className="w-full px-3 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors text-sm font-medium"
                  >
                    Use This Version
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}