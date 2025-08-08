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

interface AppLayoutProps {
  children: React.ReactNode
}

export default function AppLayout({ children }: AppLayoutProps) {
  const router = useRouter()
  const pathname = usePathname()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

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
      path: '/dashboard?module=policies',
      description: 'Compliance',
      color: 'blue'
    },
    {
      id: 'rbac',
      title: 'RBAC',
      icon: Users,
      path: '/dashboard?module=rbac',
      description: 'Permissions',
      color: 'green'
    },
    {
      id: 'costs',
      title: 'Costs',
      icon: DollarSign,
      path: '/dashboard?module=costs',
      description: 'FinOps',
      color: 'yellow'
    },
    {
      id: 'network',
      title: 'Network',
      icon: Network,
      path: '/dashboard?module=network',
      description: 'Security',
      color: 'red'
    },
    {
      id: 'resources',
      title: 'Resources',
      icon: Server,
      path: '/dashboard?module=resources',
      description: 'Management',
      color: 'indigo'
    }
  ]

  const handleNavigation = (path: string) => {
    router.push(path)
    setMobileMenuOpen(false)
  }

  const isActive = (path: string) => {
    if (path === '/dashboard' && pathname === '/dashboard' && !pathname.includes('module')) {
      return true
    }
    if (path.includes('?module=')) {
      const module = path.split('module=')[1]
      return pathname === '/dashboard' && window.location.search.includes(`module=${module}`)
    }
    return pathname === path
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex">
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
                <button
                  key={item.id}
                  onClick={() => handleNavigation(item.path)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
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

          {/* User Section */}
          <div className="p-4 border-t border-white/10">
            <button
              onClick={() => router.push('/settings')}
              className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 text-gray-300 hover:text-white transition-all"
            >
              <Settings className="w-5 h-5" />
              <span className="text-sm">Settings</span>
            </button>
            <button
              onClick={() => router.push('/')}
              className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 text-gray-300 hover:text-white transition-all mt-2"
            >
              <LogOut className="w-5 h-5" />
              <span className="text-sm">Sign Out</span>
            </button>
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