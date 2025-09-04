'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState } from 'react'
import { 
  Home, Shield, CheckCircle, DollarSign, Menu, X, Search
} from 'lucide-react'
import ThemeToggle from './ThemeToggle'

export default function SimplifiedNavigation() {
  const pathname = usePathname()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // PCG-focused navigation
  const navigation = [
    {
      name: 'Home',
      href: '/',
      icon: Home,
      description: 'Dashboard'
    },
    {
      name: 'Prevent',
      href: '/prevent',
      icon: Shield,
      description: 'Predictive compliance & risk prevention'
    },
    {
      name: 'Prove',
      href: '/prove',
      icon: CheckCircle,
      description: 'Evidence chain & audit trail'
    },
    {
      name: 'Payback',
      href: '/payback',
      icon: DollarSign,
      description: 'ROI tracking & cost optimization'
    }
  ]

  // Navigation content component for reuse
  const NavigationContent = () => (
    <>
      {navigation.map((item) => {
        const Icon = item.icon
        const isActive = pathname === item.href || (item.href !== '/' && pathname.startsWith(item.href))
        
        return (
          <Link
            key={item.name}
            href={item.href}
            className={`
              flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all
              ${isActive 
                ? 'bg-primary text-primary-foreground dark:bg-blue-600 dark:text-white' 
                : 'text-muted-foreground dark:text-gray-300 hover:bg-accent dark:hover:bg-gray-800'
              }
            `}
            onClick={() => setMobileMenuOpen(false)}
          >
            <Icon className="w-5 h-5 flex-shrink-0" />
            <div className="flex-1">
              <div className="font-medium">{item.name}</div>
              <div className="text-xs opacity-70">{item.description}</div>
            </div>
          </Link>
        )
      })}
    </>
  )

  return (
    <>
      {/* Top Bar */}
      <header className="sticky top-0 left-0 right-0 h-16 bg-background/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-border dark:border-gray-800 z-50">
        <div className="flex items-center justify-between h-full px-4">
          <div className="flex items-center gap-4">
            {/* Mobile menu toggle */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'}
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            
            <h1 className="text-xl font-bold text-foreground dark:text-white">PolicyCortex</h1>
            <span className="hidden sm:inline text-xs text-muted-foreground dark:text-gray-400">PCG Platform</span>
          </div>
          
          <div className="flex items-center gap-2">
            {/* Theme Toggle */}
            <ThemeToggle />
            
            {/* Quick Search */}
            <button className="flex items-center gap-2 px-3 py-1.5 bg-muted dark:bg-gray-800 text-muted-foreground dark:text-gray-300 rounded-lg hover:bg-accent dark:hover:bg-gray-700 transition-colors">
              <Search className="w-4 h-4" />
              <span className="hidden sm:inline text-sm">Search</span>
            </button>
          </div>
        </div>
      </header>

      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex fixed left-0 top-16 bottom-0 z-40 w-64 bg-background dark:bg-gray-900 border-r border-border dark:border-gray-800 flex-col">
        <nav className="flex-1 overflow-y-auto p-4 space-y-2">
          <NavigationContent />
        </nav>

        {/* Quick Stats */}
        <div className="p-4 border-t border-border dark:border-gray-800">
          <div className="text-xs text-muted-foreground dark:text-gray-400 mb-2">Platform Stats</div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-muted dark:bg-gray-800 rounded p-2">
              <div className="text-green-600 dark:text-green-400 font-bold">98%</div>
              <div className="text-muted-foreground dark:text-gray-400">Prevented</div>
            </div>
            <div className="bg-muted dark:bg-gray-800 rounded p-2">
              <div className="text-blue-600 dark:text-blue-400 font-bold">100%</div>
              <div className="text-muted-foreground dark:text-gray-400">Verified</div>
            </div>
            <div className="bg-muted dark:bg-gray-800 rounded p-2">
              <div className="text-purple-600 dark:text-purple-400 font-bold">$2.5M</div>
              <div className="text-muted-foreground dark:text-gray-400">Saved</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 bg-background dark:bg-gray-900 z-50 lg:hidden">
          <div className="flex flex-col h-full">
            {/* Mobile Menu Header */}
            <div className="px-4 py-3 flex items-center justify-between border-b border-border dark:border-gray-800">
              <span className="font-semibold text-lg">Menu</span>
              <button
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                onClick={() => setMobileMenuOpen(false)}
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            {/* Mobile Navigation */}
            <nav className="flex-1 overflow-y-auto p-4 space-y-2">
              <NavigationContent />
            </nav>

            {/* Mobile Quick Stats */}
            <div className="p-4 border-t border-border dark:border-gray-800">
              <div className="text-xs text-muted-foreground dark:text-gray-400 mb-2">Platform Stats</div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="bg-muted dark:bg-gray-800 rounded p-2">
                  <div className="text-green-600 dark:text-green-400 font-bold">98%</div>
                  <div className="text-muted-foreground dark:text-gray-400">Prevented</div>
                </div>
                <div className="bg-muted dark:bg-gray-800 rounded p-2">
                  <div className="text-blue-600 dark:text-blue-400 font-bold">100%</div>
                  <div className="text-muted-foreground dark:text-gray-400">Verified</div>
                </div>
                <div className="bg-muted dark:bg-gray-800 rounded p-2">
                  <div className="text-purple-600 dark:text-purple-400 font-bold">$2.5M</div>
                  <div className="text-muted-foreground dark:text-gray-400">Saved</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}