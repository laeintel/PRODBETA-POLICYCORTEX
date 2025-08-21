'use client'

import Link from 'next/link'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { useEffect } from 'react'
import { 
  LayoutDashboard, Shield, Settings, Cpu, Activity, Lock,
  ChevronRight, Search, Command, GitBranch, Menu, X, History
} from 'lucide-react'
import { useState } from 'react'
import QuickActionsBar from './QuickActionsBar'
import ThemeToggle from './ThemeToggle'

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

  // 6 PRIMARY SECTIONS - OPTIMAL FOR ENTERPRISE (80% REDUCTION FROM 200+ ITEMS)
  const navigation = [
    {
      name: 'Dashboard',
      href: '/tactical',
      icon: LayoutDashboard,
      description: 'Executive overview',
      quickStats: { compliance: '94%', risks: 3, savings: '$45K/mo' }
    },
    {
      name: 'Governance',
      href: '/governance',
      icon: Shield,
      description: 'Policies, compliance & cost',
      subsections: [
        { name: 'Policies & Compliance', href: '/governance?tab=compliance' },
        { name: 'Risk Management', href: '/governance?tab=risk' },
        { name: 'Cost Optimization', href: '/governance?tab=cost' }
      ]
    },
    {
      name: 'Security & Access',
      href: '/security',
      icon: Lock,
      description: 'Identity, RBAC, PIM & policies',
      critical: true, // Visual indicator for security section
      subsections: [
        { name: 'Identity & Access (IAM)', href: '/security?tab=iam' },
        { name: 'Role Management (RBAC)', href: '/security/rbac' },
        { name: 'Privileged Identity (PIM)', href: '/security/pim', badge: 'JIT' },
        { name: 'Conditional Access', href: '/security/conditional-access' },
        { name: 'Zero Trust Policies', href: '/security/zero-trust' },
        { name: 'Entitlement Management', href: '/security/entitlements' },
        { name: 'Access Reviews', href: '/security/access-reviews' }
      ]
    },
    {
      name: 'Operations',
      href: '/operations',
      icon: Activity,
      description: 'Resources, monitoring & alerts',
      subsections: [
        { name: 'Resources', href: '/operations/resources' },
        { name: 'Monitoring', href: '/operations/monitoring' },
        { name: 'Automation', href: '/operations/automation' },
        { name: 'Notifications', href: '/operations/notifications' },
        { name: 'Alerts', href: '/operations/alerts' }
      ]
    },
    {
      name: 'DevOps & CI/CD',
      href: '/devops',
      icon: GitBranch,
      description: 'Pipelines, releases & artifacts',
      subsections: [
        { name: 'Pipelines', href: '/devops/pipelines' },
        { name: 'Releases', href: '/devops/releases' },
        { name: 'Artifacts', href: '/devops/artifacts' },
        { name: 'Deployments', href: '/devops/deployments' },
        { name: 'Build Status', href: '/devops/builds' },
        { name: 'Repositories', href: '/devops/repos' }
      ]
    },
    {
      name: 'AI Intelligence',
      href: '/ai',
      icon: Cpu,
      description: 'Patented AI features',
      highlight: true, // Make this stand out!
      subsections: [
        { name: '🚀 Predictive Compliance', href: '/ai/predictive', patent: '#4' },
        { name: '🔗 Cross-Domain Analysis', href: '/ai/correlations', patent: '#1' },
        { name: '💬 Conversational AI', href: '/ai/chat', patent: '#2' },
        { name: '📊 Unified Platform', href: '/ai/unified', patent: '#3' }
      ]
    },
    {
      name: 'Audit Trail',
      href: '/audit',
      icon: History,
      description: 'Activity history',
      badge: 'NEW'
    },
    {
      name: 'Settings',
      href: '/settings',
      icon: Settings,
      description: 'Configuration'
    }
  ]

  // Command Palette for quick access (Cmd+K)
  const quickActions = [
    { name: 'Check Compliance Status', action: '/governance/compliance' },
    { name: 'View Cost Savings', action: '/governance/cost' },
    { name: 'Chat with AI', action: '/ai/chat' },
    { name: 'View Predictions', action: '/ai/predictive' },
    { name: 'Check Active Risks', action: '/governance/risk' },
    { name: 'View Resources', action: '/operations/resources' }
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
            <button
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
            <button
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

      {/* Responsive Sidebar - Scales with screen size */}
      <div className={`
        fixed left-0 top-16 bottom-0 z-40
        w-64 lg:w-64 xl:w-72 2xl:w-80 3xl:w-96 4xl:w-[28rem] 5xl:w-[32rem]
        bg-background dark:bg-gray-900 
        border-r border-border dark:border-gray-800 
        flex flex-col transition-all duration-300
        ${mobileMenuOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <nav className="flex-1 overflow-y-auto p-3 sm:p-4 xl:p-6 2xl:p-8 3xl:p-10 space-y-1 sm:space-y-2 xl:space-y-3 custom-scrollbar">
          {navigation.map((item) => {
            const Icon = item.icon
            const isActive = pathname.startsWith(item.href)
            const isExpanded = expandedSections.has(item.name)
            
            return (
              <div key={item.name}>
                <div
                  className={`
                    flex items-center gap-2 sm:gap-3 xl:gap-4 px-2 sm:px-3 xl:px-4 2xl:px-5 py-1.5 sm:py-2 xl:py-3 2xl:py-3.5 rounded-lg xl:rounded-xl transition-all cursor-pointer border-l-2 xl:border-l-4
                    ${isActive 
                      ? 'bg-primary text-primary-foreground dark:bg-blue-600 dark:text-white border-primary dark:border-blue-400' 
                      : 'text-muted-foreground dark:text-gray-300 hover:bg-accent dark:hover:bg-gray-800 border-transparent'
                    }
                    ${item.highlight ? 'ring-2 ring-purple-500 ring-offset-2 ring-offset-background dark:ring-offset-gray-900' : ''}
                    ${item.critical ? 'border-l-2 xl:border-l-4 border-orange-500' : ''}
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
                  <Icon className="w-4 h-4 sm:w-5 sm:h-5 xl:w-6 xl:h-6 2xl:w-7 2xl:h-7 3xl:w-8 3xl:h-8 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm sm:text-base xl:text-lg 2xl:text-xl 3xl:text-2xl truncate">{item.name}</div>
                    <div className="hidden xl:block text-xs 2xl:text-sm 3xl:text-base opacity-70 truncate">{item.description}</div>
                  </div>
                  {item.subsections && (
                    <button
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
                          {'patent' in sub && sub.patent && (
                            <span className="ml-2 text-xs text-purple-400">Patent {sub.patent}</span>
                          )}
                          {'badge' in sub && sub.badge && (
                            <span className="ml-2 text-xs bg-orange-600 text-white px-1.5 py-0.5 rounded">{sub.badge}</span>
                          )}
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