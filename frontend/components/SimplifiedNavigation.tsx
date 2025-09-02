'use client'

import Link from 'next/link'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { useEffect } from 'react'
import { 
  LayoutDashboard, Shield, Settings, Cpu, Activity, Lock,
  ChevronRight, Search, Command, GitBranch, Menu, X, History,
  Server, Package, AlertCircle, FileText, Database,
  DollarSign, Briefcase, GitMerge, Brain, Sparkles,
  Link as LinkIcon, MessageSquare, Bot, Blocks, Atom, Network
} from 'lucide-react'
import { useState } from 'react'
import QuickActionsBar from './QuickActionsBar'
import ThemeToggle from './ThemeToggle'
import { CORE, LABS } from '@/config/navigation'

export default function SimplifiedNavigation() {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const router = useRouter()
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  // Auto-expand the section matching the current path
  useEffect(() => {
    const match = navigation.find((n) => pathname.startsWith(n.href) && n.subsections)
    if (match) {
      // Keep only the matched section expanded to avoid jumpy UI
      setExpandedSections(new Set([match.name]))
    }
  }, [pathname])

  // Map icons to nav items
  const iconMap: Record<string, any> = {
    'Executive': Briefcase,
    'Policy': FileText,
    'Audit Trail': Shield,
    'Predict': Brain,
    'FinOps & ROI': DollarSign,
    'Access Governance': Lock,
    'Resources': Database,
    'DevSecOps': GitBranch,
    'Settings': Settings,
    'Blockchain Explorer': Blocks,
    'Copilot': Bot,
    'Cloud ITSM': Server,
    'Quantum-Safe': Atom,
    'Edge Governance': Network
  }

  // Build navigation with subsections
  const navigation = [
    // Core navigation items from config
    ...CORE.map(item => ({
      name: item.label,
      href: item.href,
      icon: iconMap[item.label] || LayoutDashboard,
      description: item.description,
      highlight: ['Executive', 'Policy', 'Audit Trail', 'Predict'].includes(item.label),
      subsections: getSubsections(item.label)
    })),
    // Labs section with subsections
    {
      name: 'Labs',
      href: '#',
      icon: Sparkles,
      description: 'Experimental features',
      badge: 'Labs' as const,
      subsections: LABS.map(lab => ({
        name: lab.label,
        href: lab.href
      }))
    }
  ]

  // Helper function to get subsections for specific items
  function getSubsections(label: string) {
    switch(label) {
      case 'Executive':
        return [
          { name: 'Dashboard', href: '/executive' },
          { name: 'ROI Analysis', href: '/executive/roi' },
          { name: 'Risk Map', href: '/executive/risk-map' },
          { name: 'Board Reports', href: '/executive/reports' }
        ]
      case 'Policy':
        return [
          { name: 'Policy Hub', href: '/policy' },
          { name: 'Policy Packs', href: '/policy/packs' },
          { name: 'Composer', href: '/policy/composer' },
          { name: 'Enforcement', href: '/policy/enforcement' },
          { name: 'Exceptions', href: '/policy/exceptions' },
          { name: 'Evidence', href: '/policy/evidence' }
        ]
      case 'FinOps & ROI':
        return [
          { name: 'Anomalies', href: '/finops/anomalies' },
          { name: 'Optimization', href: '/finops/optimization' },
          { name: 'Forecasting', href: '/finops/forecasting' },
          { name: 'Chargeback', href: '/finops/chargeback' },
          { name: 'Savings Plans', href: '/finops/savings-plans' },
          { name: 'Arbitrage', href: '/finops/arbitrage' }
        ]
      case 'Access Governance':
        return [
          { name: 'Identity & Access', href: '/rbac' },
          { name: 'Role Management', href: '/rbac/roles' },
          { name: 'Access Reviews', href: '/rbac/reviews' }
        ]
      case 'Resources':
        return [
          { name: 'Inventory', href: '/resources' },
          { name: 'Monitoring', href: '/resources/monitoring' },
          { name: 'Alerts', href: '/resources/alerts' }
        ]
      case 'DevSecOps':
        return [
          { name: 'Pipelines', href: '/devsecops/pipelines' },
          { name: 'Security Gates', href: '/devsecops/gates' },
          { name: 'Policy-as-Code', href: '/devsecops/policy-code' },
          { name: 'Gate Results', href: '/devsecops/results' },
          { name: 'Policies', href: '/devsecops/policies', badge: 'Beta' as const }
        ]
      default:
        return undefined
    }
  }

  // Command Palette for quick access (Cmd+K)
  const quickActions = [
    { name: 'View Predictions', action: '/predict' },
    { name: 'Verify Audit Chain', action: '/audit' },
    { name: 'View Cost Savings', action: '/finops' },
    { name: 'Check Resources', action: '/resources' },
    { name: 'Access Governance', action: '/rbac' },
    { name: 'DevSecOps Gates', action: '/devsecops' }
  ]

  const toggleSection = (sectionName: string) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(sectionName)) {
      newExpanded.delete(sectionName)
    } else {
      newExpanded.add(sectionName)
    }
    setExpandedSections(newExpanded)
  }

  return (
    <>
      {/* Simplified Top Bar - Responsive */}
      <div className="fixed top-0 left-0 right-0 h-16 bg-background dark:bg-gray-900 border-b border-border dark:border-gray-800 z-50 transition-colors">
        <div className="flex items-center justify-between h-full px-4">
          <div className="flex items-center gap-2 sm:gap-4">
            {/* Mobile menu toggle */}
            <button type="button"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            
            <h1 className="text-lg sm:text-xl font-bold text-foreground dark:text-white">PolicyCortex</h1>
            <span className="hidden sm:inline text-xs text-muted-foreground dark:text-gray-400">AI-Powered Governance</span>
          </div>
          
          <div className="flex items-center gap-2">
            {/* Audit Trail Quick Access */}
            <Link
              href="/audit"
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              aria-label="Audit Trail"
              title="View Audit Trail"
            >
              <History className="w-5 h-5 text-muted-foreground dark:text-gray-400" />
            </Link>
            
            {/* Theme Toggle */}
            <ThemeToggle />
            
            {/* Command Palette Trigger */}
            <button type="button"
              onClick={() => setCommandPaletteOpen(true)}
              className="flex items-center gap-2 px-2 sm:px-3 py-1.5 bg-muted dark:bg-gray-800 text-muted-foreground dark:text-gray-300 rounded-lg hover:bg-accent dark:hover:bg-gray-700 transition-colors"
            >
              <Search className="w-4 h-4" />
              <span className="hidden sm:inline text-sm">Quick Actions</span>
              <kbd className="hidden md:inline text-xs bg-background dark:bg-gray-900 px-1.5 py-0.5 rounded">⌘K</kbd>
            </button>
          </div>
        </div>
      </div>

      {/* Quick Actions Bar */}
      <QuickActionsBar />

      {/* Responsive Sidebar */}
      <div className={`
        fixed left-0 top-16 bottom-0 z-40
        w-64 lg:w-64 xl:w-72 2xl:w-80
        bg-background dark:bg-gray-900 
        border-r border-border dark:border-gray-800 
        flex flex-col transition-all duration-300
        ${mobileMenuOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <nav className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-1 sm:space-y-2 custom-scrollbar">
          {navigation.map((item) => {
            const Icon = item.icon
            const isActive = pathname.startsWith(item.href)
            const isExpanded = expandedSections.has(item.name)
            
            return (
              <div key={item.name}>
                <div
                  className={`
                    flex items-center gap-2 sm:gap-3 px-2 sm:px-3 py-1.5 sm:py-2 rounded-lg transition-all cursor-pointer border-l-2
                    ${isActive 
                      ? 'bg-primary text-primary-foreground dark:bg-blue-600 dark:text-white border-primary dark:border-blue-400' 
                      : 'text-muted-foreground dark:text-gray-300 hover:bg-accent dark:hover:bg-gray-800 border-transparent'
                    }
                    ${'highlight' in item && item.highlight ? 'ring-2 ring-purple-500 ring-offset-2 ring-offset-background dark:ring-offset-gray-900' : ''}
                  `}
                  role="button"
                  tabIndex={0}
                  aria-current={isActive ? 'page' : undefined}
                  onClick={() => {
                    router.push(item.href)
                    if (item.subsections) {
                      setExpandedSections(new Set([item.name]))
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      router.push(item.href)
                      if (item.subsections) {
                        setExpandedSections(new Set([item.name]))
                      }
                    }
                  }}
                >
                  <Icon className="w-4 h-4 sm:w-5 sm:h-5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm sm:text-base truncate">{item.name}</span>
                      {'badge' in item && item.badge && (
                        <span className="text-xs bg-orange-600 text-white px-1.5 py-0.5 rounded">{item.badge}</span>
                      )}
                    </div>
                    <div className="hidden xl:block text-xs opacity-70 truncate">{item.description}</div>
                  </div>
                  {item.subsections && (
                    <button type="button"
                      className="ml-2 rounded p-1 hover:bg-accent dark:hover:bg-gray-800"
                      aria-label={isExpanded ? 'Collapse section' : 'Expand section'}
                      onClick={(e) => {
                        e.stopPropagation()
                        setExpandedSections((prev) => {
                          const next = new Set<string>()
                          if (!prev.has(item.name)) {
                            next.add(item.name)
                          }
                          return next
                        })
                      }}
                    >
                      <ChevronRight 
                        className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-90' : ''}`} 
                      />
                    </button>
                  )}
                </div>
                
                {/* Show subsections when expanded */}
                {isExpanded && item.subsections && (
                  <div className="ml-6 sm:ml-8 mt-1 sm:mt-2 space-y-0.5 sm:space-y-1">
                    {item.subsections.map((sub) => {
                      const hasQuery = sub.href.includes('?')
                      let isSubActive = pathname === sub.href
                      if (hasQuery) {
                        const [base, query] = sub.href.split('?')
                        if (pathname.startsWith(base) && query) {
                          const q = new URLSearchParams(query)
                          isSubActive = Array.from(q.entries()).every(([k, v]) => searchParams.get(k) === v)
                        } else {
                          isSubActive = false
                        }
                      }
                      return (
                        <Link
                          key={sub.href}
                          href={sub.href}
                          aria-current={isSubActive ? 'page' : undefined}
                          className={`block px-2 sm:px-3 py-1 sm:py-1.5 text-xs sm:text-sm transition-colors border-l-2 ${
                            isSubActive 
                              ? 'text-primary-foreground dark:text-white bg-primary/20 dark:bg-blue-600/20 border-primary dark:border-blue-500 rounded'
                              : 'text-muted-foreground dark:text-gray-400 hover:text-foreground dark:hover:text-white border-transparent'
                          }`}
                        >
                          {sub.name}
                        </Link>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          })}
        </nav>

        {/* Quick Stats - Responsive */}
        <div className="p-3 sm:p-4 border-t border-border dark:border-gray-800">
          <div className="text-xs text-muted-foreground dark:text-gray-400 mb-2">Quick Stats</div>
          <div className="grid grid-cols-2 gap-1.5 sm:gap-2 text-xs">
            <div className="bg-muted dark:bg-gray-800 rounded p-1.5 sm:p-2">
              <div className="text-green-600 dark:text-green-400 font-bold">94%</div>
              <div className="text-muted-foreground dark:text-gray-400">Compliant</div>
            </div>
            <div className="bg-muted dark:bg-gray-800 rounded p-1.5 sm:p-2">
              <div className="text-yellow-600 dark:text-yellow-400 font-bold">3</div>
              <div className="text-muted-foreground dark:text-gray-400">Risks</div>
            </div>
            <div className="bg-muted dark:bg-gray-800 rounded p-1.5 sm:p-2">
              <div className="text-blue-600 dark:text-blue-400 font-bold">$45K</div>
              <div className="text-muted-foreground dark:text-gray-400">Saved/mo</div>
            </div>
            <div className="bg-muted dark:bg-gray-800 rounded p-1.5 sm:p-2">
              <div className="text-purple-600 dark:text-purple-400 font-bold">7</div>
              <div className="text-muted-foreground dark:text-gray-400">AI Alerts</div>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile menu overlay */}
      {mobileMenuOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-30 lg:hidden" 
          onClick={() => setMobileMenuOpen(false)}
        />
      )}
      
      {/* Command Palette Overlay */}
      {commandPaletteOpen && (
        <div className="fixed inset-0 bg-black/50 z-50" onClick={() => setCommandPaletteOpen(false)}>
          <div className="fixed top-20 left-1/2 -translate-x-1/2 w-[90%] sm:w-full max-w-2xl bg-card dark:bg-gray-900 rounded-lg shadow-2xl border border-border dark:border-gray-700"
               onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-3 p-4 border-b border-border dark:border-gray-800">
              <Command className="w-5 h-5 text-muted-foreground dark:text-gray-400" />
              <input
                type="text"
                placeholder="Type a command or search..."
                className="flex-1 bg-transparent text-foreground dark:text-white outline-none"
                autoFocus
              />
              <kbd className="text-xs text-muted-foreground dark:text-gray-400">ESC</kbd>
            </div>
            <div className="p-2 max-h-96 overflow-y-auto">
              {quickActions.map((action) => (
                <Link
                  key={action.action}
                  href={action.action}
                  className="block px-4 py-2 text-foreground dark:text-gray-300 hover:bg-accent dark:hover:bg-gray-800 rounded transition-colors"
                  onClick={() => setCommandPaletteOpen(false)}
                >
                  {action.name}
                </Link>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  )
}