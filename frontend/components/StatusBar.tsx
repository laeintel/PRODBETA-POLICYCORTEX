'use client'

import { useEffect, useState } from 'react'
import { Activity, AlertCircle, CheckCircle, HelpCircle, Command } from 'lucide-react'
import Link from 'next/link'

interface StatusBarProps {
  env?: string
  build?: string
  tenant?: string
  region?: string
}

export default function StatusBar({
  env = process.env.NEXT_PUBLIC_ENV || 'prod',
  build = process.env.NEXT_PUBLIC_BUILD_ID || process.env.NEXT_PUBLIC_VERCEL_GIT_COMMIT_SHA?.slice(0, 7) || 'dev',
  tenant = 'default',
  region = 'us-central'
}: StatusBarProps) {
  const [healthy, setHealthy] = useState(true)
  const [dataAge, setDataAge] = useState('—')
  const [showHelp, setShowHelp] = useState(false)

  useEffect(() => {
    // Check backend health
    const checkHealth = async () => {
      try {
        const res = await fetch('/api/health', { method: 'HEAD' })
        setHealthy(res.ok)
      } catch {
        setHealthy(false)
      }
    }

    // Update data freshness
    const updateDataAge = () => {
      const lastSync = localStorage.getItem('lastDataSync')
      if (lastSync) {
        const age = Date.now() - parseInt(lastSync)
        const minutes = Math.floor(age / 60000)
        if (minutes < 1) {
          setDataAge('just now')
        } else if (minutes < 60) {
          setDataAge(`${minutes}m ago`)
        } else {
          const hours = Math.floor(minutes / 60)
          setDataAge(`${hours}h ago`)
        }
      }
    }

    checkHealth()
    updateDataAge()
    const interval = setInterval(() => {
      checkHealth()
      updateDataAge()
    }, 30000) // Check every 30 seconds

    return () => clearInterval(interval)
  }, [])

  // Keyboard shortcut for help
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === '?') {
        e.preventDefault()
        setShowHelp(!showHelp)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [showHelp])

  return (
    <>
      <div className="h-7 border-t border-border/50 bg-background/60 backdrop-blur-sm flex items-center justify-between px-3 text-[11px] select-none z-40">
        {/* Left side - Environment info */}
        <div className="flex items-center gap-3 text-muted-foreground">
          <span className="font-medium">
            {tenant !== 'default' && `${tenant} • `}
            {env === 'prod' ? (
              <span className="text-green-600 dark:text-green-400">production</span>
            ) : env === 'staging' ? (
              <span className="text-yellow-600 dark:text-yellow-400">staging</span>
            ) : (
              <span className="text-blue-600 dark:text-blue-400">development</span>
            )}
            {` • ${region}`}
          </span>
          <span className="opacity-70">|</span>
          <span className="flex items-center gap-1">
            <Activity className="w-3 h-3" />
            data: {dataAge}
          </span>
        </div>

        {/* Right side - Status and help */}
        <div className="flex items-center gap-3">
          {/* Health status */}
          <Link 
            href="/status"
            className="flex items-center gap-1 hover:opacity-80 transition-opacity"
          >
            {healthy ? (
              <>
                <CheckCircle className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                <span className="text-emerald-600 dark:text-emerald-400">All systems operational</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-3 h-3 text-amber-600 dark:text-amber-400" />
                <span className="text-amber-600 dark:text-amber-400">Degraded performance</span>
              </>
            )}
          </Link>

          <span className="opacity-30">|</span>

          {/* Build version */}
          <span className="text-muted-foreground opacity-60">
            v{process.env.NEXT_PUBLIC_VERSION || '2.0.0'} • {build}
          </span>

          <span className="opacity-30">|</span>

          {/* Help menu */}
          <button
            onClick={() => setShowHelp(!showHelp)}
            className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Help menu"
          >
            <HelpCircle className="w-3 h-3" />
            <span>Help</span>
            <kbd className="ml-1 px-1 py-0.5 text-[9px] font-mono bg-muted rounded opacity-60">
              ⌘⇧?
            </kbd>
          </button>
        </div>
      </div>

      {/* Help dropdown */}
      {showHelp && (
        <div className="absolute bottom-7 right-3 z-50 w-48 bg-card dark:bg-gray-900 border border-border dark:border-gray-800 rounded-lg shadow-lg py-1 text-sm">
          <Link
            href="/docs"
            className="block px-3 py-2 hover:bg-accent dark:hover:bg-gray-800 transition-colors"
            onClick={() => setShowHelp(false)}
          >
            <div className="flex items-center gap-2">
              <span>Documentation</span>
            </div>
          </Link>
          <Link
            href="/support"
            className="block px-3 py-2 hover:bg-accent dark:hover:bg-gray-800 transition-colors"
            onClick={() => setShowHelp(false)}
          >
            <div className="flex items-center gap-2">
              <span>Support</span>
            </div>
          </Link>
          <button
            onClick={() => {
              const event = new KeyboardEvent('keydown', {
                key: 'k',
                metaKey: true,
                ctrlKey: false,
                bubbles: true
              })
              document.dispatchEvent(event)
              setShowHelp(false)
            }}
            className="w-full text-left px-3 py-2 hover:bg-accent dark:hover:bg-gray-800 transition-colors"
          >
            <div className="flex items-center justify-between">
              <span>Command Palette</span>
              <kbd className="px-1.5 py-0.5 text-[10px] font-mono bg-muted rounded">
                ⌘K
              </kbd>
            </div>
          </button>
          <Link
            href="/keyboard-shortcuts"
            className="block px-3 py-2 hover:bg-accent dark:hover:bg-gray-800 transition-colors"
            onClick={() => setShowHelp(false)}
          >
            <div className="flex items-center justify-between">
              <span>Keyboard Shortcuts</span>
              <kbd className="px-1.5 py-0.5 text-[10px] font-mono bg-muted rounded">
                ?
              </kbd>
            </div>
          </Link>
          <div className="border-t border-border dark:border-gray-800 my-1" />
          <Link
            href="/legal/privacy"
            className="block px-3 py-2 hover:bg-accent dark:hover:bg-gray-800 transition-colors text-muted-foreground"
            onClick={() => setShowHelp(false)}
          >
            Privacy Policy
          </Link>
          <Link
            href="/legal/terms"
            className="block px-3 py-2 hover:bg-accent dark:hover:bg-gray-800 transition-colors text-muted-foreground"
            onClick={() => setShowHelp(false)}
          >
            Terms of Service
          </Link>
          <Link
            href="/legal/compliance"
            className="block px-3 py-2 hover:bg-accent dark:hover:bg-gray-800 transition-colors text-muted-foreground"
            onClick={() => setShowHelp(false)}
          >
            Compliance
          </Link>
        </div>
      )}
    </>
  )
}