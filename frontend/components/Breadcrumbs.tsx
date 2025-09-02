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

import React from 'react'
import Link from 'next/link'
import { usePathname, useSearchParams } from 'next/navigation'
import { ChevronRight, Home } from 'lucide-react'

export interface Crumb {
  href: string
  label: string
}

interface BreadcrumbItem {
  name: string
  href: string
  current: boolean
}

// Legacy component interface for backward compatibility
export function LegacyBreadcrumbs({ items }: { items: Crumb[] }) {
  return (
    <nav aria-label="Breadcrumb" className="text-sm text-gray-400 mb-4">
      {items.map((item, idx) => (
        <span key={item.href}>
          {idx > 0 && <span className="mx-2">/</span>}
          {idx < items.length - 1 ? (
            <Link href={item.href} className="hover:text-white">{item.label}</Link>
          ) : (
            <span className="text-white">{item.label}</span>
          )}
        </span>
      ))}
    </nav>
  )
}

// New automatic breadcrumb component
export default function Breadcrumbs() {
  const pathname = usePathname()
  const searchParams = useSearchParams()

  // Navigation mapping for proper breadcrumb names
  const navigationMap = {
    // Root paths
    '/tactical': 'Dashboard',
    '/executive': 'Executive',
    '/finops': 'FinOps',
    '/governance': 'Governance',
    '/security': 'Security & Access',
    '/operations': 'Operations',
    '/devops': 'DevOps',
    '/itsm': 'Cloud ITSM',
    '/ai': 'AI Intelligence',
    '/copilot': 'Governance Copilot',
    '/blockchain': 'Blockchain Audit',
    '/quantum': 'Quantum-Safe Secrets',
    '/edge': 'Edge Governance',
    '/audit': 'Audit Trail',
    '/settings': 'Settings',
    
    // Sub-pages
    '/executive/dashboard': 'Business KPIs',
    '/executive/roi': 'ROI Calculator',
    '/executive/risk-map': 'Risk-to-Revenue Map',
    '/executive/reports': 'Board Reports',
    
    '/finops/anomalies': 'Real-time Anomalies',
    '/finops/optimization': 'Auto Optimization',
    '/finops/forecasting': 'Spend Forecasting',
    '/finops/chargeback': 'Department Billing',
    '/finops/savings-plans': 'Savings Plans',
    '/finops/arbitrage': 'Multi-Cloud Arbitrage',
    
    '/security/rbac': 'Role Management (RBAC)',
    '/security/pim': 'Privileged Identity (PIM)',
    '/security/conditional-access': 'Conditional Access',
    '/security/zero-trust': 'Zero Trust Policies',
    '/security/entitlements': 'Entitlement Management',
    '/security/access-reviews': 'Access Reviews',
    
    '/operations/resources': 'Resources',
    '/operations/monitoring': 'Monitoring',
    '/operations/automation': 'Automation',
    '/operations/notifications': 'Notifications',
    '/operations/alerts': 'Alerts',
    
    '/devops/pipelines': 'Pipelines',
    '/devops/deployments': 'Deployments',
    '/devops/releases': 'Releases',
    '/devops/builds': 'Build Status',
    '/devops/artifacts': 'Artifacts',
    '/devops/repos': 'Repositories',
    '/devsecops/gates': 'Security Gates',
    '/devsecops/policy-as-code': 'Policy-as-Code',
    '/devsecops/ide-plugins': 'IDE Plugins',
    
    '/itsm/inventory': 'Resource Inventory',
    '/itsm/applications': 'Applications',
    '/itsm/services': 'Service Catalog',
    '/itsm/incidents': 'Incidents',
    '/itsm/changes': 'Changes',
    '/itsm/problems': 'Problems',
    '/itsm/assets': 'Assets',
    '/itsm/cmdb': 'CMDB',
    
    '/ai/predictive': 'Predictive Compliance',
    '/ai/correlations': 'Cross-Domain Analysis',
    '/ai/chat': 'Conversational AI',
    '/ai/unified': 'Unified Platform',
    '/ai/policy-studio': 'Policy Studio'
  }

  // Handle query parameters for governance tabs
  const getTabName = (tab: string | null) => {
    switch (tab) {
      case 'compliance': return 'Policies & Compliance'
      case 'risk': return 'Risk Management'
      case 'cost': return 'Cost Optimization'
      case 'iam': return 'Identity & Access (IAM)'
      default: return null
    }
  }

  const generateBreadcrumbs = (): BreadcrumbItem[] => {
    const breadcrumbs: BreadcrumbItem[] = []
    
    // Always start with Home - PolicyCortex
    breadcrumbs.push({
      name: 'PolicyCortex',
      href: '/tactical',
      current: false
    })

    // Skip root path and login page
    if (pathname === '/' || pathname === '/tactical') {
      breadcrumbs[0].current = true
      return breadcrumbs
    }

    // Split pathname into segments
    const pathSegments = pathname.split('/').filter(Boolean)
    let currentPath = ''

    pathSegments.forEach((segment, index) => {
      currentPath += `/${segment}`
      const isLast = index === pathSegments.length - 1
      
      // Check if we have a specific name for this path
      let segmentName = navigationMap[currentPath as keyof typeof navigationMap]
      
      // If no specific mapping, capitalize the segment
      if (!segmentName) {
        segmentName = segment
          .split('-')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ')
      }

      breadcrumbs.push({
        name: segmentName,
        href: currentPath,
        current: isLast
      })
    })

    // Handle query parameters (like governance tabs)
    const tab = searchParams.get('tab')
    if (tab) {
      const tabName = getTabName(tab)
      if (tabName) {
        // Update the last breadcrumb to not be current
        if (breadcrumbs.length > 0) {
          breadcrumbs[breadcrumbs.length - 1].current = false
        }
        
        // Add the tab as the current breadcrumb
        breadcrumbs.push({
          name: tabName,
          href: `${pathname}?tab=${tab}`,
          current: true
        })
      }
    }

    return breadcrumbs
  }

  const breadcrumbs = generateBreadcrumbs()

  // Don't show breadcrumbs on login page
  if (pathname === '/') {
    return null
  }

  // Debug logging
  console.log('Breadcrumbs - Current pathname:', pathname)
  console.log('Breadcrumbs - Generated breadcrumbs:', breadcrumbs)

  return (
    <div className="bg-yellow-100 dark:bg-yellow-900 border border-yellow-300 dark:border-yellow-700 rounded-lg p-3 mb-4">
      <div className="text-xs text-yellow-800 dark:text-yellow-200 mb-1">
        DEBUG: Current path: {pathname}
      </div>
      <nav className="flex items-center space-x-1 text-sm" aria-label="Breadcrumb">
        <div className="flex items-center min-w-0">
          <Home className="w-4 h-4 text-muted-foreground dark:text-gray-400 mr-2 flex-shrink-0" />
          
          {breadcrumbs.map((crumb, index) => (
            <React.Fragment key={crumb.href}>
              {crumb.current ? (
                <span 
                  className="font-medium text-foreground dark:text-white truncate"
                  aria-current="page"
                >
                  {crumb.name}
                </span>
              ) : (
                <Link
                  href={crumb.href}
                  className="text-muted-foreground dark:text-gray-400 hover:text-foreground dark:hover:text-white transition-colors truncate"
                >
                  {crumb.name}
                </Link>
              )}
              
              {index < breadcrumbs.length - 1 && (
                <ChevronRight className="w-4 h-4 text-muted-foreground dark:text-gray-400 mx-2 flex-shrink-0" />
              )}
            </React.Fragment>
          ))}
        </div>
        
        {/* Show current page name prominently on mobile */}
        <div className="sm:hidden ml-auto">
          <span className="font-semibold text-foreground dark:text-white">
            {breadcrumbs[breadcrumbs.length - 1]?.name}
          </span>
        </div>
      </nav>
    </div>
  )
}


