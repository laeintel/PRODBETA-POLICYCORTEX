'use client'

import { useState } from 'react'
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

interface AppLayoutProps {
  children: React.ReactNode
}

export default function AppLayout({ children }: AppLayoutProps) {
  const router = useRouter()
  const pathname = usePathname()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const { isAuthenticated, login, logout, user } = useAuth()
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})

  const navigation = [
    {
      id: 'dashboard',
      title: 'Dashboard',
      icon: Home,
      path: '/dashboard',
      description: 'Overview',
      color: 'purple'
    },
    {
      id: 'ai-expert',
      title: 'AI Expert',
      icon: Brain,
      path: '/ai-expert',
      description: 'Domain Expert',
      color: 'pink'
    },
    {
      id: 'chat',
      title: 'AI Assistant',
      icon: Brain,
      path: '/chat',
      description: 'Chat with AI',
      color: 'indigo'
    },
    {
      id: 'policies',
      title: 'Policies',
      icon: Shield,
      path: '/policies',
      description: 'Compliance',
      color: 'blue',
      children: [
        { id: 'policies-overview', title: 'Overview', path: '/policies' },
        { id: 'policies-security', title: 'Security', path: '/policies?category=Security' },
        { id: 'policies-governance', title: 'Governance', path: '/policies?category=Governance' },
        { id: 'policies-compliance', title: 'Compliance', path: '/policies?category=Compliance' },
        { id: 'policies-network', title: 'Network', path: '/policies?category=Network' },
        { id: 'policies-monitoring', title: 'Monitoring', path: '/policies?category=Monitoring' },
        { id: 'policies-sql', title: 'SQL', path: '/policies?category=SQL' },
        { id: 'policies-k8s', title: 'Kubernetes', path: '/policies?category=Kubernetes' },
        { id: 'policies-noncompliant', title: 'Non-Compliant', path: '/policies?view=noncompliant' },
      ]
    },
    {
      id: 'rbac',
      title: 'RBAC',
      icon: Users,
      path: '/rbac',
      description: 'Permissions',
      color: 'green',
      children: [
        { id: 'rbac-overview', title: 'Overview', path: '/rbac' },
        { id: 'rbac-privileged', title: 'Privileged Accounts', path: '/rbac?filter=privileged' },
        { id: 'rbac-service-principals', title: 'Service Principals', path: '/rbac?type=ServicePrincipal' },
        { id: 'rbac-role-defs', title: 'Role Definitions', path: '/rbac?view=roles' },
        { id: 'rbac-access-reviews', title: 'Access Reviews', path: '/rbac?view=reviews' },
      ]
    },
    {
      id: 'costs',
      title: 'Costs',
      icon: DollarSign,
      path: '/dashboard?module=costs',
      description: 'FinOps',
      color: 'yellow',
      children: [
        { id: 'costs-overview', title: 'Overview', path: '/dashboard?module=costs' },
        { id: 'costs-breakdown', title: 'Breakdown', path: '/dashboard?module=costs&view=breakdown' },
        { id: 'costs-anomalies', title: 'Anomalies', path: '/dashboard?module=costs&view=anomalies' },
        { id: 'costs-optimizations', title: 'Optimizations', path: '/dashboard?module=costs&view=optimizations' },
      ]
    },
    {
      id: 'network',
      title: 'Network',
      icon: Network,
      path: '/dashboard?module=network',
      description: 'Security',
      color: 'red',
      children: [
        { id: 'network-overview', title: 'Overview', path: '/dashboard?module=network' },
        { id: 'network-nsg', title: 'NSG Rules', path: '/dashboard?module=network&view=nsg' },
        { id: 'network-endpoints', title: 'Public Endpoints', path: '/dashboard?module=network&view=endpoints' },
        { id: 'network-zero-trust', title: 'Zero Trust', path: '/dashboard?module=network&view=zero-trust' },
      ]
    },
    {
      id: 'resources',
      title: 'Resources',
      icon: Server,
      path: '/resources',
      description: 'Management',
      color: 'indigo',
      children: [
        { id: 'resources-all', title: 'All Resources', path: '/resources' },
        { id: 'resources-vm', title: 'Virtual Machines', path: '/resources?type=virtualMachines' },
        { id: 'resources-storage', title: 'Storage Accounts', path: '/resources?type=storageAccounts' },
        { id: 'resources-db', title: 'Databases', path: '/resources?type=databases' },
        { id: 'resources-k8s', title: 'Kubernetes', path: '/resources?type=managedClusters' },
        { id: 'resources-web', title: 'Web Apps', path: '/resources?type=Web' },
      ]
    }
  ]

  const handleNavigation = (path: string) => {
    try {
      router.push(path)
    } catch (e) {
      // Hard fallback if Next router stalls
      if (typeof window !== 'undefined') {
        window.location.href = path
      }
    } finally {
      setMobileMenuOpen(false)
    }
  }

  const toggleExpand = (id: string) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  const isActive = (path: string) => {
    if (path === '/dashboard' && pathname === '/dashboard' && !pathname.includes('module')) {
      return true
    }
    if (path.includes('?module=')) {
      const module = path.split('module=')[1]
      // Check if window is defined (client-side)
      if (typeof window !== 'undefined') {
        return pathname === '/dashboard' && window.location.search.includes(`module=${module}`)
      }
      return false
    }
    return pathname === path
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex">
      {/* Global banners */}
      <div className="fixed top-0 left-0 right-0 z-50">
        <ConnectionStatusBanner />
        <DemoModeBanner />
      </div>
      {/* Desktop Sidebar */}
      <motion.div
        initial={{ x: 0 }}
        animate={{ x: sidebarOpen ? 0 : -240 }}
        transition={{ duration: 0.3 }}
        className="hidden lg:block fixed left-0 top-0 h-full z-20"
      >
        <div className="w-60 h-full bg-black/40 backdrop-blur-xl border-r border-white/10 flex flex-col">
          {/* Logo */}
          <div className="p-6 border-b border-white/10">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg flex items-center justify-center">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-white font-bold">PolicyCortex</h1>
                  <p className="text-xs text-gray-400">AI Governance</p>
                </div>
              </div>
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
            {navigation.map((item) => {
              const Icon = item.icon
              const active = isActive(item.path)
              
              return (
                <div key={item.id}>
                  <div className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                      active 
                        ? 'bg-purple-600/20 border border-purple-500/50 text-white' 
                        : 'hover:bg-white/5 text-gray-300 hover:text-white'
                    }`}>
                    <button
                      onClick={() => handleNavigation(item.path)}
                      className="flex-1 flex items-center gap-3 text-left"
                    >
                      <Icon className={`w-5 h-5 ${active ? 'text-purple-400' : ''}`} />
                      <div>
                        <div className="text-sm font-medium">{item.title}</div>
                        <div className="text-xs text-gray-400">{item.description}</div>
                      </div>
                    </button>
                    {Array.isArray((item as any).children) && (
                      <button
                        onClick={() => toggleExpand(item.id)}
                        className="text-gray-400 hover:text-white"
                        aria-label={`Toggle ${item.title} submenu`}
                      >
                        {(expanded[item.id] ? <ChevronLeft className="w-4 h-4 rotate-90" /> : <ChevronRight className="w-4 h-4" />)}
                      </button>
                    )}
                  </div>
                  {Array.isArray((item as any).children) && expanded[item.id] && (
                    <div className="ml-10 mt-1 mb-2 space-y-1">
                      {(item as any).children.map((sub: any) => (
                        <button
                          key={sub.id}
                          onClick={() => handleNavigation(sub.path)}
                          className={`w-full text-left text-xs px-2 py-1 rounded hover:bg-white/5 ${isActive(sub.path) ? 'text-white' : 'text-gray-400 hover:text-white'}`}
                        >
                          {sub.title}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </nav>

          {/* User Section */}
          <div className="p-4 border-t border-white/10">
            <div className="mb-3 text-xs text-gray-400">
              {isAuthenticated ? `Signed in as ${user?.username || user?.name || 'user'}` : 'Not signed in'}
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => router.push('/settings')}
                className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg hover:bg-white/5 text-gray-300 hover:text-white transition-all"
              >
                <Settings className="w-4 h-4" />
                <span className="text-sm">Settings</span>
              </button>
              {isAuthenticated ? (
                <button
                  onClick={() => logout()}
                  className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg hover:bg-white/5 text-gray-300 hover:text-white transition-all"
                >
                  <LogOut className="w-4 h-4" />
                  <span className="text-sm">Sign out</span>
                </button>
              ) : (
                <button
                  onClick={() => login()}
                  className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg hover:bg-white/5 text-gray-300 hover:text-white transition-all"
                >
                  <Brain className="w-4 h-4" />
                  <span className="text-sm">Sign in</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </motion.div>

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

      {/* Mobile Menu Button */}
      <button
        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 w-10 h-10 bg-purple-600/20 backdrop-blur-xl border border-purple-500/50 rounded-lg flex items-center justify-center text-white"
      >
        {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      {/* Mobile Sidebar */}
      {mobileMenuOpen && (
        <motion.div
          initial={{ x: -300 }}
          animate={{ x: 0 }}
          exit={{ x: -300 }}
          className="lg:hidden fixed inset-0 z-40 bg-black/80 backdrop-blur-xl"
        >
          <div className="w-72 h-full bg-black/90 border-r border-white/10 flex flex-col">
            {/* Logo */}
            <div className="p-6 border-b border-white/10">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg flex items-center justify-center">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-white font-bold">PolicyCortex</h1>
                  <p className="text-xs text-gray-400">AI Governance</p>
                </div>
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
              {navigation.map((item) => {
                const Icon = item.icon
                const active = isActive(item.path)
                
                return (
                  <button
                    key={item.id}
                    onClick={() => handleNavigation(item.path)}
                    className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg transition-all ${
                      active 
                        ? 'bg-purple-600/20 border border-purple-500/50 text-white' 
                        : 'hover:bg-white/5 text-gray-300 hover:text-white'
                    }`}
                  >
                    <Icon className={`w-5 h-5 ${active ? 'text-purple-400' : ''}`} />
                    <div className="text-left">
                      <div className="text-sm font-medium">{item.title}</div>
                      <div className="text-xs text-gray-400">{item.description}</div>
                    </div>
                  </button>
                )
              })}
            </nav>
          </div>
        </motion.div>
      )}

      {/* Main Content */}
      <div className={`flex-1 ${sidebarOpen ? 'lg:ml-60' : ''} transition-all duration-300`}>
        {children}
      </div>
    </div>
  )
}