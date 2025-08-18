"use client"

import { useState, useEffect } from 'react'
import { Download, Wifi, WifiOff, RefreshCw, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

// PWA Install Prompt Component
export function PWAInstallPrompt() {
  const [installPrompt, setInstallPrompt] = useState<any>(null)
  const [showPrompt, setShowPrompt] = useState(false)
  const [isInstalled, setIsInstalled] = useState(false)

  useEffect(() => {
    // Check if already installed
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches
    const isInApp = (window.navigator as any).standalone === true
    setIsInstalled(isStandalone || isInApp)

    // Listen for beforeinstallprompt event
    const handleBeforeInstallPrompt = (event: any) => {
      event.preventDefault()
      setInstallPrompt(event)
      setShowPrompt(true)
    }

    // Listen for app installed event
    const handleAppInstalled = () => {
      setIsInstalled(true)
      setShowPrompt(false)
      setInstallPrompt(null)
    }

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt)
    window.addEventListener('appinstalled', handleAppInstalled)

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt)
      window.removeEventListener('appinstalled', handleAppInstalled)
    }
  }, [])

  const handleInstall = async () => {
    if (!installPrompt) return

    try {
      installPrompt.prompt()
      const result = await installPrompt.userChoice
      
      if (result.outcome === 'accepted') {
        setShowPrompt(false)
        setInstallPrompt(null)
      }
    } catch (error) {
      console.error('Error installing PWA:', error)
    }
  }

  const handleDismiss = () => {
    setShowPrompt(false)
    // Don't show again for this session
    sessionStorage.setItem('pwa-install-dismissed', 'true')
  }

  // Don't show if already installed or dismissed this session
  if (isInstalled || !showPrompt || sessionStorage.getItem('pwa-install-dismissed')) {
    return null
  }

  return (
    <Card className="fixed bottom-4 right-4 w-80 z-50 shadow-lg">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Install PolicyCortex</CardTitle>
          <Button variant="ghost" size="icon" onClick={handleDismiss}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <CardDescription>
          Install our app for a better experience with offline access and faster loading.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="flex gap-2">
          <Button onClick={handleInstall} className="flex-1">
            <Download className="h-4 w-4 mr-2" />
            Install
          </Button>
          <Button variant="outline" onClick={handleDismiss}>
            Later
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

// Network Status Component
export function NetworkStatus() {
  const [isOnline, setIsOnline] = useState(true)
  const [connectionType, setConnectionType] = useState<string>('')

  useEffect(() => {
    const updateNetworkStatus = () => {
      setIsOnline(navigator.onLine)
      
      // Get connection info if available
      const connection = (navigator as any).connection || 
                        (navigator as any).mozConnection || 
                        (navigator as any).webkitConnection
      
      if (connection) {
        setConnectionType(connection.effectiveType || connection.type || '')
      }
    }

    updateNetworkStatus()

    window.addEventListener('online', updateNetworkStatus)
    window.addEventListener('offline', updateNetworkStatus)

    // Listen for connection changes
    const connection = (navigator as any).connection
    if (connection) {
      connection.addEventListener('change', updateNetworkStatus)
    }

    return () => {
      window.removeEventListener('online', updateNetworkStatus)
      window.removeEventListener('offline', updateNetworkStatus)
      if (connection) {
        connection.removeEventListener('change', updateNetworkStatus)
      }
    }
  }, [])

  if (isOnline) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Wifi className="h-4 w-4 text-green-500" />
        <span>Online</span>
        {connectionType && (
          <Badge variant="outline" className="text-xs">
            {connectionType.toUpperCase()}
          </Badge>
        )}
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2 text-sm">
      <WifiOff className="h-4 w-4 text-red-500" />
      <span className="text-red-500">Offline</span>
      <Badge variant="destructive" className="text-xs">
        Limited functionality
      </Badge>
    </div>
  )
}

// Update Available Banner
export function UpdateAvailableBanner() {
  const [updateAvailable, setUpdateAvailable] = useState(false)
  const [isUpdating, setIsUpdating] = useState(false)

  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        setUpdateAvailable(true)
      })

      // Check for updates
      navigator.serviceWorker.ready.then((registration) => {
        registration.addEventListener('updatefound', () => {
          const newWorker = registration.installing
          if (newWorker) {
            newWorker.addEventListener('statechange', () => {
              if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                setUpdateAvailable(true)
              }
            })
          }
        })
      })
    }
  }, [])

  const handleUpdate = async () => {
    setIsUpdating(true)
    
    try {
      if ('serviceWorker' in navigator) {
        const registration = await navigator.serviceWorker.ready
        if (registration.waiting) {
          registration.waiting.postMessage({ type: 'SKIP_WAITING' })
        }
      }
      
      // Reload the page to get the new version
      setTimeout(() => {
        window.location.reload()
      }, 1000)
    } catch (error) {
      console.error('Error updating app:', error)
      setIsUpdating(false)
    }
  }

  if (!updateAvailable) return null

  return (
    <Card className="fixed top-4 right-4 w-80 z-50 shadow-lg border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950">
      <CardContent className="pt-4">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-medium text-blue-900 dark:text-blue-100">
              Update Available
            </h4>
            <p className="text-sm text-blue-700 dark:text-blue-300">
              A new version of the app is ready to install.
            </p>
          </div>
          <Button
            onClick={handleUpdate}
            disabled={isUpdating}
            size="sm"
            className="ml-2"
          >
            {isUpdating ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              'Update'
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

// Offline Content Cache Status
export function CacheStatus() {
  const [cacheSize, setCacheSize] = useState<number>(0)
  const [isCalculating, setIsCalculating] = useState(false)

  const calculateCacheSize = async () => {
    setIsCalculating(true)
    
    try {
      if ('storage' in navigator && 'estimate' in navigator.storage) {
        const estimate = await navigator.storage.estimate()
        setCacheSize(estimate.usage || 0)
      }
    } catch (error) {
      console.error('Error calculating cache size:', error)
    } finally {
      setIsCalculating(false)
    }
  }

  const clearCache = async () => {
    try {
      if ('caches' in window) {
        const cacheNames = await caches.keys()
        await Promise.all(cacheNames.map(name => caches.delete(name)))
        setCacheSize(0)
      }
    } catch (error) {
      console.error('Error clearing cache:', error)
    }
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  useEffect(() => {
    calculateCacheSize()
  }, [])

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-lg">Offline Cache</CardTitle>
        <CardDescription>
          Manage cached content for offline access
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Cache Size:</span>
          <div className="flex items-center gap-2">
            <span className="font-medium">
              {isCalculating ? 'Calculating...' : formatBytes(cacheSize)}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={calculateCacheSize}
              disabled={isCalculating}
            >
              <RefreshCw className={cn("h-4 w-4", isCalculating && "animate-spin")} />
            </Button>
          </div>
        </div>
        
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={clearCache}
            disabled={cacheSize === 0}
            className="flex-1"
          >
            Clear Cache
          </Button>
          <Button
            variant="outline"
            onClick={calculateCacheSize}
            disabled={isCalculating}
            className="flex-1"
          >
            Refresh
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

// Combined PWA Features Component
export function PWAFeatures() {
  return (
    <div className="space-y-4">
      <PWAInstallPrompt />
      <UpdateAvailableBanner />
      <div className="flex items-center justify-between">
        <NetworkStatus />
      </div>
    </div>
  )
}