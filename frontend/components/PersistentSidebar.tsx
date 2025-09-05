'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import {
  ChevronLeft,
  ChevronRight,
  Shield,
  CheckCircle,
  DollarSign,
  Brain,
  Activity,
  Users,
  Lock,
  Zap,
  LinkIcon,
  Cloud,
  Bot,
  BarChart3,
  FileCheck,
  Settings,
  HelpCircle,
  Home,
  TrendingUp,
  AlertTriangle,
  Package,
  Network,
  Database,
  GitBranch,
  Layers
} from 'lucide-react';

interface NavItem {
  label: string;
  href: string;
  icon: React.ReactNode;
  badge?: string;
  children?: NavItem[];
}

const navItems: NavItem[] = [
  {
    label: 'Executive',
    href: '/executive',
    icon: <Home className="h-5 w-5" />,
    badge: 'LIVE'
  },
  {
    label: 'PREVENT',
    href: '/prevent',
    icon: <Shield className="h-5 w-5" />,
    children: [
      { label: 'Predictions', href: '/ai/predictions', icon: <Brain className="h-4 w-4" /> },
      { label: 'Risk Analysis', href: '/ai/unified', icon: <AlertTriangle className="h-4 w-4" /> },
      { label: 'Anomalies', href: '/anomalies', icon: <Activity className="h-4 w-4" /> }
    ]
  },
  {
    label: 'PROVE',
    href: '/prove',
    icon: <CheckCircle className="h-5 w-5" />,
    children: [
      { label: 'Audit Trail', href: '/audit', icon: <FileCheck className="h-4 w-4" /> },
      { label: 'Compliance', href: '/policies', icon: <Shield className="h-4 w-4" /> },
      { label: 'Evidence Chain', href: '/blockchain', icon: <LinkIcon className="h-4 w-4" /> }
    ]
  },
  {
    label: 'PAYBACK',
    href: '/payback',
    icon: <DollarSign className="h-5 w-5" />,
    children: [
      { label: 'Governance P&L', href: '/finops/pnl', icon: <TrendingUp className="h-4 w-4" />, badge: 'NEW' },
      { label: 'Cost Analysis', href: '/costs', icon: <BarChart3 className="h-4 w-4" /> },
      { label: 'ROI Dashboard', href: '/roi', icon: <DollarSign className="h-4 w-4" /> }
    ]
  },
  {
    label: 'ITSM',
    href: '/itsm',
    icon: <Package className="h-5 w-5" />,
    children: [
      { label: 'Incidents', href: '/itsm/incidents', icon: <AlertTriangle className="h-4 w-4" /> },
      { label: 'Changes', href: '/itsm/changes', icon: <GitBranch className="h-4 w-4" /> },
      { label: 'Service Catalog', href: '/itsm/catalog', icon: <Layers className="h-4 w-4" /> }
    ]
  },
  {
    label: 'Policy Hub',
    href: '/policies',
    icon: <Shield className="h-5 w-5" />
  },
  {
    label: 'Resources',
    href: '/resources',
    icon: <Database className="h-5 w-5" />
  },
  {
    label: 'Network',
    href: '/network',
    icon: <Network className="h-5 w-5" />
  },
  {
    label: 'Quantum',
    href: '/quantum',
    icon: <Zap className="h-5 w-5" />
  },
  {
    label: 'Blockchain',
    href: '/blockchain',
    icon: <LinkIcon className="h-5 w-5" />
  },
  {
    label: 'Edge',
    href: '/edge',
    icon: <Cloud className="h-5 w-5" />
  },
  {
    label: 'Copilot',
    href: '/copilot',
    icon: <Bot className="h-5 w-5" />,
    badge: 'AI'
  },
  {
    label: 'Security',
    href: '/security',
    icon: <Lock className="h-5 w-5" />
  },
  {
    label: 'RBAC',
    href: '/rbac',
    icon: <Users className="h-5 w-5" />
  }
];

// Separate Settings and Help items for the footer
const footerItems: NavItem[] = [
  {
    label: 'Settings',
    href: '/settings',
    icon: <Settings className="h-5 w-5" />
  },
  {
    label: 'Help',
    href: '/help',
    icon: <HelpCircle className="h-5 w-5" />
  }
];

export default function PersistentSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [expandedItems, setExpandedItems] = useState<string[]>([]);

  // Auto-expand parent items when their children are active
  useEffect(() => {
    const activeParents: string[] = [];
    navItems.forEach(item => {
      if (item.children) {
        const hasActiveChild = item.children.some(child => 
          pathname.startsWith(child.href)
        );
        if (hasActiveChild) {
          activeParents.push(item.label);
        }
      }
    });
    
    setExpandedItems(prev => {
      // Merge with existing expanded items to maintain manually expanded ones
      const combined = [...new Set([...prev, ...activeParents])];
      // Save to localStorage for persistence
      localStorage.setItem('expandedItems', JSON.stringify(combined));
      return combined;
    });
  }, [pathname]);

  // Load collapsed and expanded states from localStorage on mount
  useEffect(() => {
    const savedCollapsed = localStorage.getItem('sidebarCollapsed');
    if (savedCollapsed === 'true') {
      setIsCollapsed(true);
    }
    
    const savedExpanded = localStorage.getItem('expandedItems');
    if (savedExpanded) {
      try {
        setExpandedItems(JSON.parse(savedExpanded));
      } catch (e) {
        console.error('Error parsing expandedItems:', e);
      }
    }
  }, []);

  // Save collapsed state to localStorage and emit event
  const toggleSidebar = () => {
    const newState = !isCollapsed;
    setIsCollapsed(newState);
    localStorage.setItem('sidebarCollapsed', newState.toString());
    // Emit custom event for same-tab updates
    window.dispatchEvent(new Event('sidebarToggle'));
  };

  const toggleExpanded = (label: string) => {
    setExpandedItems(prev => {
      const newItems = prev.includes(label)
        ? prev.filter(item => item !== label)
        : [...prev, label];
      
      // Save to localStorage for persistence
      localStorage.setItem('expandedItems', JSON.stringify(newItems));
      return newItems;
    });
  };

  const isActive = (href: string) => {
    if (href === '/') return pathname === href;
    return pathname.startsWith(href);
  };

  return (
      /* Sidebar - no wrapper needed since AppShell handles positioning */
      <div
        className={`fixed left-0 top-0 h-full bg-gradient-to-b from-gray-900 via-gray-900 to-black border-r border-gray-800 transition-all duration-300 z-50 ${
          isCollapsed ? 'w-16' : 'w-64'
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-800">
          {!isCollapsed && (
            <div className="flex items-center space-x-2">
              <Shield className="h-6 w-6 text-blue-500" />
              <span className="text-white font-bold text-lg">PolicyCortex</span>
            </div>
          )}
          <button
            onClick={toggleSidebar}
            className="p-1.5 rounded-lg hover:bg-gray-800 transition-colors text-gray-400 hover:text-white"
            aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {isCollapsed ? <ChevronRight className="h-5 w-5" /> : <ChevronLeft className="h-5 w-5" />}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-4">
          <ul className="space-y-1 px-2">
            {navItems.map((item) => (
              <li key={item.label}>
                {/* Main navigation item */}
                <div className="relative">
                  <div
                    className={`flex items-center justify-between px-3 py-2 rounded-lg transition-all duration-200 group cursor-pointer ${
                      isActive(item.href)
                        ? 'bg-blue-600 text-white'
                        : item.children?.some(child => isActive(child.href))
                        ? 'bg-gray-800 text-white border-l-2 border-blue-500'
                        : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                    }`}
                    onClick={(e) => {
                      if (item.children && !isCollapsed) {
                        e.preventDefault();
                        // Only toggle if we're not already in this section's context
                        const isInContext = item.children.some(child => isActive(child.href));
                        if (!isInContext || !expandedItems.includes(item.label)) {
                          toggleExpanded(item.label);
                        }
                        // If no children, navigate to the parent href
                      } else if (!item.children) {
                        router.push(item.href);
                      }
                    }}
                  >
                    <div className="flex items-center space-x-3">
                      <span className={`${isActive(item.href) ? 'text-white' : 'text-gray-400'}`}>
                        {item.icon}
                      </span>
                      {!isCollapsed && (
                        <span className="font-medium">{item.label}</span>
                      )}
                    </div>
                    {!isCollapsed && (
                      <>
                        {item.badge && (
                          <span className={`px-2 py-0.5 text-xs rounded-full ${
                            item.badge === 'LIVE' 
                              ? 'bg-green-500 text-white' 
                              : item.badge === 'NEW'
                              ? 'bg-yellow-500 text-black'
                              : 'bg-blue-500 text-white'
                          }`}>
                            {item.badge}
                          </span>
                        )}
                        {item.children && (
                          <ChevronRight className={`h-4 w-4 transition-transform ${
                            expandedItems.includes(item.label) ? 'rotate-90' : ''
                          }`} />
                        )}
                      </>
                    )}
                  </div>
                  
                  {/* Tooltip for collapsed state */}
                  {isCollapsed && (
                    <div className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-sm rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
                      {item.label}
                      {item.badge && (
                        <span className="ml-2 px-1.5 py-0.5 text-xs bg-blue-500 rounded-full">
                          {item.badge}
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {/* Children items */}
                {!isCollapsed && item.children && expandedItems.includes(item.label) && (
                  <ul className="mt-1 ml-4 space-y-1">
                    {item.children.map((child) => (
                      <li key={child.label}>
                        <Link
                          href={child.href}
                          className={`flex items-center justify-between px-3 py-1.5 rounded-md transition-all duration-200 ${
                            isActive(child.href)
                              ? 'bg-gray-700 text-white'
                              : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                          }`}
                        >
                          <div className="flex items-center space-x-2">
                            {child.icon}
                            <span className="text-sm">{child.label}</span>
                          </div>
                          {child.badge && (
                            <span className="px-1.5 py-0.5 text-xs bg-yellow-500 text-black rounded-full">
                              {child.badge}
                            </span>
                          )}
                        </Link>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </nav>

        {/* Footer with Settings and Help */}
        <div className="border-t border-gray-800">
          {/* Settings and Help Navigation */}
          <div className="p-2">
            {footerItems.map((item) => (
              <Link
                key={item.label}
                href={item.href}
                className={`flex items-center px-3 py-2 mb-1 rounded-lg transition-all duration-200 group ${
                  isActive(item.href)
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                }`}
              >
                <span className={`${isActive(item.href) ? 'text-white' : 'text-gray-400'}`}>
                  {item.icon}
                </span>
                {!isCollapsed && (
                  <span className="ml-3 font-medium">{item.label}</span>
                )}
                
                {/* Tooltip for collapsed state */}
                {isCollapsed && (
                  <div className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-sm rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
                    {item.label}
                  </div>
                )}
              </Link>
            ))}
          </div>

          {/* Connection Status */}
          <div className="border-t border-gray-800 p-4">
            {!isCollapsed ? (
              <div className="text-xs text-gray-500">
                <div className="flex items-center justify-between mb-1">
                  <span>Real Data</span>
                  <span className="flex items-center">
                    <span className="w-2 h-2 bg-green-500 rounded-full mr-1 animate-pulse"></span>
                    Connected
                  </span>
                </div>
                <div className="text-gray-600">Port 8084</div>
              </div>
            ) : (
              <div className="flex justify-center">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
              </div>
            )}
          </div>
        </div>
      </div>
  );
}