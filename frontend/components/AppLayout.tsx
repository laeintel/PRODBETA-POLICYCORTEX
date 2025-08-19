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
import ConnectionStatusBanner from './ConnectionStatusBanner'
import DemoModeBanner from './DemoModeBanner'
import { ModernSideMenu } from './ModernSideMenu'

interface AppLayoutProps {
  children: React.ReactNode
}

function AppLayoutInner({ children }: AppLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const { logout, user } = useAuth()

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex">
      {/* Global banners */}
      <div className="fixed top-0 left-0 right-0 z-50">
        <ConnectionStatusBanner />
        <DemoModeBanner />
      </div>
      
      {/* Modern Side Menu */}
      <ModernSideMenu 
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        onLogout={logout}
        user={user}
      />

      {/* Sidebar Toggle Button (when collapsed) */}
      {!sidebarOpen && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          onClick={() => setSidebarOpen(true)}
          className="hidden lg:flex fixed left-4 top-4 z-30 w-10 h-10 bg-purple-600/20 backdrop-blur-xl border border-purple-500/50 rounded-lg items-center justify-center text-white hover:bg-purple-600/30 transition-colors"
        >
          <ChevronRight className="w-5 h-5" />
        </motion.button>
      )}

      {/* Main Content */}
      <div className={`flex-1 ${sidebarOpen ? 'lg:ml-80' : ''} transition-all duration-300`}>
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