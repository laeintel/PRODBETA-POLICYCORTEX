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

import dynamic from 'next/dynamic'

import { useState, useEffect } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import { 
  Shield, 
  Home,
  Brain,
  Users,
  DollarSign,
  Network,
  Server,
  Settings,
  LogOut,
  ChevronLeft,
  ChevronRight,
  Menu,
  X
} from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { useMemo, useState as useStateReact, useEffect as useEffectReact } from 'react'
import { useRouter as useNextRouter } from 'next/navigation'

interface AppLayoutProps {
  children: React.ReactNode
}

function AppLayoutInner({ children }: AppLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const { logout, user } = useAuth()
  const router = useNextRouter()
  const [isPaletteOpen, setPaletteOpen] = useStateReact(false)

  // Build commands from consolidated IA
  const commands = useMemo(() => {
    return [
      { id: 'nav-dashboard', title: 'Open Operations Center', category: 'navigation', action: () => router.push('/tactical') },
      { id: 'nav-observe', title: 'Go to Monitoring Overview', category: 'navigation', action: () => router.push('/tactical/monitoring-overview') },
      { id: 'nav-sec', title: 'Open Secure & Govern', category: 'navigation', action: () => router.push('/security/overview') },
      { id: 'nav-build', title: 'Open Build & Release', category: 'navigation', action: () => router.push('/tactical/pipelines') },
      { id: 'nav-opt', title: 'Open Optimize (FinOps)', category: 'navigation', action: () => router.push('/tactical/cost-governance') },
      { id: 'nav-comm', title: 'Open Communicate', category: 'navigation', action: () => router.push('/tactical/notifications') },
      { id: 'nav-admin', title: 'Open Admin', category: 'navigation', action: () => router.push('/tactical/users') },
      { id: 'action-search', title: 'Search Policies', category: 'search', action: () => router.push('/policies') },
      { id: 'action-conversation', title: 'Ask AI', category: 'actions', action: () => router.push('/chat') },
      { id: 'action-resources', title: 'Open Resource Manager', category: 'resources', action: () => router.push('/resources') },
      { id: 'action-compliance', title: 'View Compliance Scores', category: 'security', action: () => router.push('/tactical/compliance-scores') },
      { id: 'action-costs', title: 'Open Cost Analytics', category: 'resources', action: () => router.push('/tactical/cost-governance') },
    ]
  }, [router])

  // Global keyboard shortcut to open palette (Cmd/Ctrl+K or '/')
  useEffectReact(() => {
    const handler = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase()
      const isMeta = e.metaKey || e.ctrlKey
      if ((isMeta && key === 'k') || key === '/') {
        e.preventDefault()
        setPaletteOpen(true)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex">
      {/* Main Content */}
      <div className="flex-1">
        {children}
      </div>
    </div>
  )
}

// Export dynamically imported version that doesn't run during SSR
const AppLayout = dynamic(
  () => Promise.resolve(AppLayoutInner),
  { 
    ssr: false,
    loading: () => (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-purple-400 border-t-transparent rounded-full mx-auto mb-4 animate-spin" />
          <p className="text-white">Loading...</p>
        </div>
      </div>
    )
  }
)

export default AppLayout