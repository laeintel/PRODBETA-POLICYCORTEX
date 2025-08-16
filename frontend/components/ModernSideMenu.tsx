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

import React, { useState, useEffect, useMemo } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import {
  // Core Navigation Icons
  LayoutDashboard,
  Brain,
  MessageSquare,
  
  // Resource Management
  Box,
  Server,
  Database,
  Globe,
  Cloud,
  HardDrive,
  
  // Governance & Security
  Shield,
  Lock,
  Key,
  UserCheck,
  FileCheck,
  AlertTriangle,
  
  // Cost & Performance
  DollarSign,
  TrendingUp,
  BarChart3,
  Zap,
  Gauge,
  
  // Network & Infrastructure
  Network,
  Wifi,
  GitBranch,
  
  // Actions & Settings
  Settings,
  Search,
  Bell,
  HelpCircle,
  LogOut,
  ChevronRight,
  ChevronLeft,
  Menu,
  X,
  Sparkles,
  Command,
  Plus,
  Star,
  Clock,
  Filter,
  
  // Status Indicators
  CircleDot,
  Circle,
  CheckCircle,
  XCircle,
  AlertCircle
} from 'lucide-react'

interface MenuItem {
  id: string
  title: string
  icon: React.ElementType
  path?: string
  badge?: string | number
  badgeType?: 'success' | 'warning' | 'error' | 'info' | 'default'
  description?: string
  shortcut?: string
  category?: string
  children?: MenuItem[]
  isNew?: boolean
  isPremium?: boolean
  aiPowered?: boolean
}

interface ModernSideMenuProps {
  isOpen: boolean
  onToggle: () => void
  onLogout?: () => void
  user?: any
}

export function ModernSideMenu({ isOpen, onToggle, onLogout, user }: ModernSideMenuProps) {
  const router = useRouter()
  const pathname = usePathname()
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set())
  const [favorites, setFavorites] = useState<Set<string>>(new Set())
  const [recentItems, setRecentItems] = useState<string[]>([])
  const [hoveredItem, setHoveredItem] = useState<string | null>(null)

  // Redesigned menu structure with better organization
  const menuStructure: Record<string, MenuItem[]> = {
    'Quick Access': [
      {
        id: 'dashboard',
        title: 'Dashboard',
        icon: LayoutDashboard,
        path: '/dashboard',
        description: 'Executive overview',
        shortcut: '⌘D',
        badge: 'Live',
        badgeType: 'success'
      },
      {
        id: 'ai-assistant',
        title: 'AI Assistant',
        icon: Sparkles,
        path: '/chat',
        description: 'Conversational AI',
        shortcut: '⌘K',
        aiPowered: true,
        isNew: true
      },
      {
        id: 'command-center',
        title: 'Command Center',
        icon: Command,
        path: '/command',
        description: 'Quick actions',
        shortcut: '⌘/',
        isPremium: true
      }
    ],
    
    'Resources': [
      {
        id: 'all-resources',
        title: 'All Resources',
        icon: Box,
        path: '/resources',
        description: 'Unified view',
        badge: '1,247',
        badgeType: 'default'
      },
      {
        id: 'resources-v2',
        title: 'Resources Dashboard',
        icon: Box,
        path: '/resources/v2',
        description: 'Enhanced management',
        badge: 'NEW',
        badgeType: 'success',
        isNew: true,
        aiPowered: true
      },
      {
        id: 'virtual-machines',
        title: 'Virtual Machines',
        icon: Server,
        path: '/resources/vm',
        description: 'Compute instances',
        badge: '342',
        badgeType: 'default'
      },
      {
        id: 'databases',
        title: 'Databases',
        icon: Database,
        path: '/resources/db',
        description: 'Data services',
        badge: '89',
        badgeType: 'default'
      },
      {
        id: 'storage',
        title: 'Storage',
        icon: HardDrive,
        path: '/resources/storage',
        description: 'Storage accounts',
        badge: '156',
        badgeType: 'default'
      },
      {
        id: 'kubernetes',
        title: 'Kubernetes',
        icon: GitBranch,
        path: '/resources/k8s',
        description: 'Container orchestration',
        badge: '23',
        badgeType: 'default'
      }
    ],
    
    'Governance': [
      {
        id: 'policies',
        title: 'Policies',
        icon: Shield,
        path: '/policies',
        description: 'Compliance rules',
        badge: '12',
        badgeType: 'warning',
        children: [
          {
            id: 'policy-violations',
            title: 'Violations',
            icon: AlertTriangle,
            path: '/policies/violations',
            badge: '12',
            badgeType: 'error'
          },
          {
            id: 'policy-insights',
            title: 'AI Insights',
            icon: Brain,
            path: '/policies/insights',
            aiPowered: true
          }
        ]
      },
      {
        id: 'security',
        title: 'Security Center',
        icon: Lock,
        path: '/security',
        description: 'Security posture',
        badge: '98%',
        badgeType: 'success'
      },
      {
        id: 'compliance',
        title: 'Compliance',
        icon: FileCheck,
        path: '/compliance',
        description: 'Regulatory standards',
        badge: '3',
        badgeType: 'warning'
      },
      {
        id: 'access-control',
        title: 'Access Control',
        icon: UserCheck,
        path: '/rbac',
        description: 'RBAC & permissions'
      }
    ],
    
    'Intelligence': [
      {
        id: 'cost-optimization',
        title: 'Cost Intelligence',
        icon: DollarSign,
        path: '/costs',
        description: 'FinOps insights',
        badge: '$45K',
        badgeType: 'success',
        aiPowered: true
      },
      {
        id: 'performance',
        title: 'Performance',
        icon: Gauge,
        path: '/performance',
        description: 'System metrics',
        badge: '↑12%',
        badgeType: 'info'
      },
      {
        id: 'predictions',
        title: 'Predictions',
        icon: TrendingUp,
        path: '/predictions',
        description: 'Predictive analytics',
        aiPowered: true,
        isPremium: true
      },
      {
        id: 'correlations',
        title: 'Correlations',
        icon: GitBranch,
        path: '/correlations',
        description: 'Cross-domain insights',
        aiPowered: true
      }
      // Removed: Voice Assistant and Explainability for demo
    ],
    
    'Operations': [
      {
        id: 'network',
        title: 'Network',
        icon: Network,
        path: '/network',
        description: 'Network topology',
        badge: '5',
        badgeType: 'warning'
      },
      {
        id: 'monitoring',
        title: 'Monitoring',
        icon: BarChart3,
        path: '/monitoring',
        description: 'Real-time metrics'
      },
      {
        id: 'automation',
        title: 'Automation',
        icon: Zap,
        path: '/automation',
        description: 'Workflows & scripts',
        isNew: true
      },
      {
        id: 'exceptions',
        title: 'Exceptions',
        icon: AlertCircle,
        path: '/exceptions',
        description: 'Policy exceptions',
        badge: '8',
        badgeType: 'error'
      }
    ]
  }

  // Flatten menu for search
  const flattenedMenu = useMemo(() => {
    const items: MenuItem[] = []
    Object.values(menuStructure).forEach(category => {
      category.forEach(item => {
        items.push(item)
        if (item.children) {
          items.push(...item.children)
        }
      })
    })
    return items
  }, [])

  // Filter menu items based on search
  const filteredMenu = useMemo(() => {
    if (!searchQuery) return menuStructure
    
    const query = searchQuery.toLowerCase()
    const filtered: Record<string, MenuItem[]> = {}
    
    Object.entries(menuStructure).forEach(([category, items]) => {
      const filteredItems = items.filter(item => 
        item.title.toLowerCase().includes(query) ||
        item.description?.toLowerCase().includes(query) ||
        item.children?.some(child => 
          child.title.toLowerCase().includes(query) ||
          child.description?.toLowerCase().includes(query)
        )
      )
      
      if (filteredItems.length > 0) {
        filtered[category] = filteredItems
      }
    })
    
    return filtered
  }, [searchQuery, menuStructure])

  // Track navigation
  const handleNavigation = (path: string, itemId: string) => {
    router.push(path)
    
    // Update recent items
    setRecentItems(prev => {
      const updated = [itemId, ...prev.filter(id => id !== itemId)].slice(0, 5)
      localStorage.setItem('recentMenuItems', JSON.stringify(updated))
      return updated
    })
  }

  // Toggle favorite
  const toggleFavorite = (itemId: string) => {
    setFavorites(prev => {
      const updated = new Set(prev)
      if (updated.has(itemId)) {
        updated.delete(itemId)
      } else {
        updated.add(itemId)
      }
      localStorage.setItem('favoriteMenuItems', JSON.stringify(Array.from(updated)))
      return updated
    })
  }

  // Load saved preferences
  useEffect(() => {
    const savedFavorites = localStorage.getItem('favoriteMenuItems')
    const savedRecent = localStorage.getItem('recentMenuItems')
    
    if (savedFavorites) {
      setFavorites(new Set(JSON.parse(savedFavorites)))
    }
    if (savedRecent) {
      setRecentItems(JSON.parse(savedRecent))
    }
  }, [])

  // Auto-expand active category
  useEffect(() => {
    Object.entries(menuStructure).forEach(([category, items]) => {
      const hasActiveItem = items.some(item => 
        pathname === item.path || 
        item.children?.some(child => pathname === child.path)
      )
      
      if (hasActiveItem) {
        setExpandedCategories(prev => new Set(prev).add(category))
      }
    })
  }, [pathname])

  const isItemActive = (item: MenuItem): boolean => {
    if (pathname === item.path) return true
    if (item.children) {
      return item.children.some(child => pathname === child.path)
    }
    return false
  }

  const getBadgeClasses = (type?: string) => {
    switch (type) {
      case 'success': return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'warning': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'error': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'info': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.aside
          initial={{ x: -320 }}
          animate={{ x: 0 }}
          exit={{ x: -320 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="fixed left-0 top-0 z-40 h-screen w-80 bg-gradient-to-b from-slate-900 via-slate-900 to-slate-950 border-r border-white/10 flex flex-col"
        >
          {/* Header */}
          <div className="p-4 border-b border-white/10">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                  <Shield className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-white">PolicyCortex</h1>
                  <p className="text-xs text-gray-400">AI Governance Platform</p>
                </div>
              </div>
              <button
                onClick={onToggle}
                className="p-2 hover:bg-white/10 rounded-lg transition-colors"
              >
                <ChevronLeft className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {/* Search Bar */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search menu..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:bg-white/10 transition-all"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 hover:bg-white/10 rounded"
                >
                  <X className="w-3 h-3 text-gray-400" />
                </button>
              )}
            </div>
          </div>

          {/* Quick Stats */}
          <div className="px-4 py-3 border-b border-white/10">
            <div className="grid grid-cols-3 gap-2">
              <div className="text-center">
                <p className="text-xs text-gray-400">Health</p>
                <p className="text-sm font-bold text-green-400">98%</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-400">Issues</p>
                <p className="text-sm font-bold text-yellow-400">23</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-400">Cost/Day</p>
                <p className="text-sm font-bold text-white">$4.2K</p>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            {/* Favorites Section */}
            {favorites.size > 0 && !searchQuery && (
              <div className="px-4 py-3">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                  Favorites
                </h3>
                {Array.from(favorites).map(itemId => {
                  const item = flattenedMenu.find(i => i.id === itemId)
                  if (!item) return null
                  
                  return (
                    <MenuItemComponent
                      key={item.id}
                      item={item}
                      isActive={isItemActive(item)}
                      onNavigate={handleNavigation}
                      onToggleFavorite={toggleFavorite}
                      isFavorite={true}
                      getBadgeClasses={getBadgeClasses}
                    />
                  )
                })}
              </div>
            )}

            {/* Recent Items */}
            {recentItems.length > 0 && !searchQuery && !favorites.size && (
              <div className="px-4 py-3">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <Clock className="w-3 h-3" />
                  Recent
                </h3>
                {recentItems.slice(0, 3).map(itemId => {
                  const item = flattenedMenu.find(i => i.id === itemId)
                  if (!item) return null
                  
                  return (
                    <MenuItemComponent
                      key={item.id}
                      item={item}
                      isActive={isItemActive(item)}
                      onNavigate={handleNavigation}
                      onToggleFavorite={toggleFavorite}
                      isFavorite={favorites.has(item.id)}
                      getBadgeClasses={getBadgeClasses}
                    />
                  )
                })}
              </div>
            )}

            {/* Main Menu Categories */}
            {Object.entries(filteredMenu).map(([category, items]) => (
              <div key={category} className="px-4 py-3">
                <button
                  onClick={() => {
                    setExpandedCategories(prev => {
                      const updated = new Set(prev)
                      if (updated.has(category)) {
                        updated.delete(category)
                      } else {
                        updated.add(category)
                      }
                      return updated
                    })
                  }}
                  className="w-full flex items-center justify-between text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 hover:text-gray-300 transition-colors"
                >
                  <span>{category}</span>
                  <ChevronRight 
                    className={`w-3 h-3 transition-transform ${
                      expandedCategories.has(category) ? 'rotate-90' : ''
                    }`}
                  />
                </button>
                
                <AnimatePresence>
                  {expandedCategories.has(category) && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      {items.map(item => (
                        <MenuItemComponent
                          key={item.id}
                          item={item}
                          isActive={isItemActive(item)}
                          onNavigate={handleNavigation}
                          onToggleFavorite={toggleFavorite}
                          isFavorite={favorites.has(item.id)}
                          getBadgeClasses={getBadgeClasses}
                        />
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ))}
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-white/10">
            {/* User Profile */}
            {user && (
              <div className="flex items-center gap-3 p-3 rounded-lg hover:bg-white/5 transition-colors mb-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                  <span className="text-white font-semibold">
                    {user.name?.charAt(0) || 'U'}
                  </span>
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-white">{user.name || 'User'}</p>
                  <p className="text-xs text-gray-400">{user.email || 'user@example.com'}</p>
                </div>
              </div>
            )}

            {/* Footer Actions */}
            <div className="flex items-center justify-between">
              <div className="flex gap-1">
                <button className="p-2 hover:bg-white/10 rounded-lg transition-colors group">
                  <Settings className="w-4 h-4 text-gray-400 group-hover:text-white" />
                </button>
                <button className="p-2 hover:bg-white/10 rounded-lg transition-colors group">
                  <HelpCircle className="w-4 h-4 text-gray-400 group-hover:text-white" />
                </button>
                <button className="p-2 hover:bg-white/10 rounded-lg transition-colors group">
                  <Bell className="w-4 h-4 text-gray-400 group-hover:text-white" />
                </button>
              </div>
              {onLogout && (
                <button
                  onClick={onLogout}
                  className="p-2 hover:bg-white/10 rounded-lg transition-colors group"
                >
                  <LogOut className="w-4 h-4 text-gray-400 group-hover:text-red-400" />
                </button>
              )}
            </div>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}

// Menu Item Component
function MenuItemComponent({ 
  item, 
  isActive, 
  onNavigate, 
  onToggleFavorite, 
  isFavorite,
  getBadgeClasses,
  depth = 0 
}: any) {
  const [expanded, setExpanded] = useState(false)
  const Icon = item.icon

  return (
    <div>
      <div
        className={`
          group flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-all
          ${isActive 
            ? 'bg-gradient-to-r from-purple-600/20 to-pink-600/20 border border-purple-500/30' 
            : 'hover:bg-white/5 border border-transparent'
          }
          ${depth > 0 ? 'ml-6' : ''}
        `}
        onClick={() => {
          if (item.path) {
            onNavigate(item.path, item.id)
          } else if (item.children) {
            setExpanded(!expanded)
          }
        }}
      >
        {/* Icon */}
        <div className={`
          p-2 rounded-lg transition-all
          ${isActive 
            ? 'bg-gradient-to-br from-purple-500 to-pink-500' 
            : 'bg-white/10 group-hover:bg-white/20'
          }
        `}>
          <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-gray-300'}`} />
        </div>

        {/* Content */}
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <p className={`text-sm font-medium ${isActive ? 'text-white' : 'text-gray-300'}`}>
              {item.title}
            </p>
            {item.isNew && (
              <span className="px-1.5 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded">
                NEW
              </span>
            )}
            {item.aiPowered && (
              <Sparkles className="w-3 h-3 text-purple-400" />
            )}
            {item.isPremium && (
              <Star className="w-3 h-3 text-yellow-400" />
            )}
          </div>
          {item.description && (
            <p className="text-xs text-gray-500">{item.description}</p>
          )}
        </div>

        {/* Badge */}
        {item.badge && (
          <span className={`
            px-2 py-1 text-xs rounded-full border
            ${getBadgeClasses(item.badgeType)}
          `}>
            {item.badge}
          </span>
        )}

        {/* Actions */}
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={(e) => {
              e.stopPropagation()
              onToggleFavorite(item.id)
            }}
            className="p-1 hover:bg-white/10 rounded transition-colors"
          >
            <Star className={`w-3 h-3 ${
              isFavorite ? 'text-yellow-400 fill-yellow-400' : 'text-gray-400'
            }`} />
          </button>
          {item.children && (
            <ChevronRight className={`w-3 h-3 text-gray-400 transition-transform ${
              expanded ? 'rotate-90' : ''
            }`} />
          )}
        </div>

        {/* Shortcut */}
        {item.shortcut && (
          <kbd className="hidden lg:block px-2 py-1 text-xs bg-white/10 text-gray-400 rounded">
            {item.shortcut}
          </kbd>
        )}
      </div>

      {/* Children */}
      {item.children && expanded && (
        <div className="mt-1">
          {item.children.map((child: any) => (
            <MenuItemComponent
              key={child.id}
              item={child}
              isActive={isActive}
              onNavigate={onNavigate}
              onToggleFavorite={onToggleFavorite}
              isFavorite={isFavorite}
              getBadgeClasses={getBadgeClasses}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// Add custom scrollbar styles
const styles = `
.custom-scrollbar::-webkit-scrollbar {
  width: 6px;
}

.custom-scrollbar::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.05);
}

.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 3px;
}

.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.3);
}
`

export default ModernSideMenu