/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

'use client'

import { AlertCircle, Database, CloudOff, TestTube } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useDemoDataFlags } from '@/contexts/DemoDataProvider'

interface MockDataIndicatorProps {
  type?: 'badge' | 'banner' | 'inline' | 'floating'
  dataSource?: string
  className?: string
}

export default function MockDataIndicator({ 
  type = 'badge', 
  dataSource = 'Simulated Data',
  className = '' 
}: MockDataIndicatorProps) {
  const [isRealData, setIsRealData] = useState<boolean>(true)
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'mock'>('connected')

  const flags = useDemoDataFlags()

  useEffect(() => {
    // Check if we're using real data
    const checkDataSource = async () => {
      try {
        const resp = await fetch('/api/v1/health')
        const data = await resp.json()
        
        // Check for real Azure connection (default to true if unknown)
        const hasAzureConnection = typeof data.azure_connected === 'boolean' ? data.azure_connected : true
        const isUsingMockData = flags.useMockData
        
        if (isUsingMockData || !hasAzureConnection) {
          setIsRealData(false)
          setConnectionStatus('mock')
        } else {
          setIsRealData(true)
          setConnectionStatus('connected')
        }
      } catch (error) {
        setIsRealData(false)
        setConnectionStatus('disconnected')
      }
    }

    checkDataSource()
    
    // Re-check every 30 seconds
    const interval = setInterval(checkDataSource, 30000)
    return () => clearInterval(interval)
  }, [])

  if (isRealData && connectionStatus === 'connected') {
    // Show a subtle indicator that real data is being used
    if (type === 'badge') {
      return (
        <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-full bg-green-100 text-green-800 text-xs ${className}`}>
          <Database className="w-3 h-3" />
          <span>Live Data</span>
        </div>
      )
    }
    return null
  }

  // Different indicator styles for mock data
  switch (type) {
    case 'banner':
      return (
        <div className={`bg-amber-500/95 text-white px-4 py-3 flex items-center gap-3 shadow-lg ${className}`}>
          <div className="flex items-center gap-2">
            <TestTube className="w-5 h-5 animate-pulse" />
            <span className="font-semibold">Mock Data Mode</span>
          </div>
          <span className="text-sm opacity-90">
            This view is displaying simulated data. Connect to Azure for real-time information.
          </span>
          <button 
            className="ml-auto px-3 py-1 bg-white/20 hover:bg-white/30 rounded-md text-sm transition-colors"
            onClick={() => window.location.href = '/settings'}
          >
            Configure Azure
          </button>
        </div>
      )

    case 'inline':
      return (
        <span className={`inline-flex items-center gap-1 text-amber-600 text-sm ${className}`}>
          <AlertCircle className="w-4 h-4" />
          <span className="font-medium">{dataSource}</span>
        </span>
      )

    case 'floating':
      return (
        <div className={`fixed bottom-4 right-4 bg-amber-500 text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-2 z-50 ${className}`}>
          <CloudOff className="w-5 h-5" />
          <div>
            <div className="font-semibold text-sm">Mock Data Active</div>
            <div className="text-xs opacity-90">Real Azure connection unavailable</div>
          </div>
        </div>
      )

    case 'badge':
    default:
      return (
        <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-full bg-amber-100 text-amber-800 text-xs font-medium ${className}`}>
          <TestTube className="w-3 h-3" />
          <span>Mock Data</span>
        </div>
      )
  }
}

// Hook to check if using mock data
export function useMockDataStatus() {
  const [isMockData, setIsMockData] = useState<boolean>(false)
  const [loading, setLoading] = useState<boolean>(true)

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const resp = await fetch('/api/v1/health')
        const data = await resp.json()
        
        const hasAzureConnection = data.azure_connected || false
        const isUsingMockData = process.env.NEXT_PUBLIC_USE_MOCK_DATA === 'true' || 
                               process.env.NEXT_PUBLIC_DISABLE_DEEP === 'true'
        
        setIsMockData(isUsingMockData || !hasAzureConnection)
      } catch {
        setIsMockData(true)
      } finally {
        setLoading(false)
      }
    }

    checkStatus()
  }, [])

  return { isMockData, loading }
}

// Component to wrap data displays with mock indicator
export function DataWithIndicator({ 
  children, 
  dataSource = 'Simulated',
  showIndicator = true 
}: { 
  children: React.ReactNode
  dataSource?: string
  showIndicator?: boolean
}) {
  const { isMockData } = useMockDataStatus()

  return (
    <div className="relative">
      {showIndicator && isMockData && (
        <div className="absolute top-0 right-0 z-10">
          <MockDataIndicator type="badge" dataSource={dataSource} />
        </div>
      )}
      <div className={isMockData ? 'opacity-90' : ''}>
        {children}
      </div>
    </div>
  )
}

// Watermark for charts and graphs
export function MockDataWatermark() {
  const { isMockData } = useMockDataStatus()

  if (!isMockData) return null

  return (
    <div className="absolute inset-0 pointer-events-none flex items-center justify-center opacity-10">
      <div className="transform rotate-[-30deg]">
        <div className="text-6xl font-bold text-gray-500">MOCK DATA</div>
      </div>
    </div>
  )
}