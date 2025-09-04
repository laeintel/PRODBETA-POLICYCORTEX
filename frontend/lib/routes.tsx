'use client';

import dynamic from 'next/dynamic';
import { ComponentType } from 'react';
import { LoadingFallback } from '../components/LazyLoad';

// Route configuration with lazy loading
export interface Route {
  path: string;
  Component: ComponentType<any>;
  preload?: boolean;
  prefetch?: boolean;
}

// Helper to create dynamic imports with proper loading states
const createDynamicImport = (
  importPath: string,
  loadingMessage?: string
) => {
  return dynamic(
    () => import(importPath),
    {
      loading: () => <LoadingFallback message={loadingMessage} />,
      ssr: true, // Enable SSR for SEO
    }
  );
};

// Main application routes with code splitting
export const routes: Route[] = [
  // Dashboard - High priority, preload
  {
    path: '/dashboard',
    Component: createDynamicImport('../app/dashboard/page', 'Loading dashboard...'),
    preload: true,
    prefetch: true,
  },
  
  // AI Module Routes
  {
    path: '/ai',
    Component: createDynamicImport('../app/ai/page', 'Loading AI center...'),
    prefetch: true,
  },
  {
    path: '/ai/chat',
    Component: createDynamicImport('../app/ai/chat/page', 'Loading AI assistant...'),
  },
  {
    path: '/ai/correlations',
    Component: createDynamicImport('../app/ai/correlations/page', 'Loading correlation engine...'),
  },
  {
    path: '/ai/predictive',
    Component: createDynamicImport('../app/ai/predictive/page', 'Loading predictive analytics...'),
  },
  {
    path: '/ai/unified',
    Component: createDynamicImport('../app/ai/unified/page', 'Loading unified platform...'),
  },
  
  // Governance Routes
  {
    path: '/governance',
    Component: createDynamicImport('../app/governance/page', 'Loading governance...'),
    prefetch: true,
  },
  {
    path: '/governance/compliance',
    Component: createDynamicImport('../app/governance/compliance/page', 'Loading compliance...'),
  },
  {
    path: '/governance/policies',
    Component: createDynamicImport('../app/governance/policies/page', 'Loading policies...'),
  },
  {
    path: '/governance/risk',
    Component: createDynamicImport('../app/governance/risk/page', 'Loading risk assessment...'),
  },
  {
    path: '/governance/cost',
    Component: createDynamicImport('../app/governance/cost/page', 'Loading cost management...'),
  },
  
  // Security Routes
  {
    path: '/security',
    Component: createDynamicImport('../app/security/page', 'Loading security center...'),
    prefetch: true,
  },
  {
    path: '/security/iam',
    Component: createDynamicImport('../app/security/iam/page', 'Loading IAM...'),
  },
  {
    path: '/security/rbac',
    Component: createDynamicImport('../app/security/rbac/page', 'Loading RBAC...'),
  },
  {
    path: '/security/pim',
    Component: createDynamicImport('../app/security/pim/page', 'Loading PIM...'),
  },
  {
    path: '/security/conditional-access',
    Component: createDynamicImport('../app/security/conditional-access/page', 'Loading conditional access...'),
  },
  {
    path: '/security/zero-trust',
    Component: createDynamicImport('../app/security/zero-trust/page', 'Loading Zero Trust...'),
  },
  
  // Operations Routes
  {
    path: '/operations',
    Component: createDynamicImport('../app/operations/page', 'Loading operations...'),
    prefetch: true,
  },
  {
    path: '/operations/resources',
    Component: createDynamicImport('../app/operations/resources/page', 'Loading resources...'),
  },
  {
    path: '/operations/monitoring',
    Component: createDynamicImport('../app/operations/monitoring/page', 'Loading monitoring...'),
  },
  {
    path: '/operations/automation',
    Component: createDynamicImport('../app/operations/automation/page', 'Loading automation...'),
  },
  {
    path: '/operations/alerts',
    Component: createDynamicImport('../app/operations/alerts/page', 'Loading alerts...'),
  },
  
  // DevOps Routes
  {
    path: '/devops',
    Component: createDynamicImport('../app/devops/page', 'Loading DevOps...'),
    prefetch: true,
  },
  {
    path: '/devops/pipelines',
    Component: createDynamicImport('../app/devops/pipelines/page', 'Loading pipelines...'),
  },
  {
    path: '/devops/releases',
    Component: createDynamicImport('../app/devops/releases/page', 'Loading releases...'),
  },
  {
    path: '/devops/builds',
    Component: createDynamicImport('../app/devops/builds/page', 'Loading builds...'),
  },
  {
    path: '/devops/deployments',
    Component: createDynamicImport('../app/devops/deployments/page', 'Loading deployments...'),
  },
  
  // ITSM Routes - Heavy modules, lazy load
  {
    path: '/itsm',
    Component: createDynamicImport('../app/itsm/page', 'Loading ITSM portal...'),
  },
  {
    path: '/itsm/incidents',
    Component: createDynamicImport('../app/itsm/incidents/page', 'Loading incident management...'),
  },
  {
    path: '/itsm/problems',
    Component: createDynamicImport('../app/itsm/problems/page', 'Loading problem management...'),
  },
  {
    path: '/itsm/changes',
    Component: createDynamicImport('../app/itsm/changes/page', 'Loading change management...'),
  },
  {
    path: '/itsm/releases',
    Component: createDynamicImport('../app/itsm/releases/page', 'Loading release management...'),
  },
  {
    path: '/itsm/assets',
    Component: createDynamicImport('../app/itsm/assets/page', 'Loading asset management...'),
  },
  {
    path: '/itsm/knowledge',
    Component: createDynamicImport('../app/itsm/knowledge/page', 'Loading knowledge base...'),
  },
  {
    path: '/itsm/service-catalog',
    Component: createDynamicImport('../app/itsm/service-catalog/page', 'Loading service catalog...'),
  },
  {
    path: '/itsm/sla',
    Component: createDynamicImport('../app/itsm/sla/page', 'Loading SLA management...'),
  },
];

// Route groups for prefetching
export const routeGroups = {
  core: ['/dashboard', '/ai', '/governance', '/security', '/operations', '/devops'],
  ai: ['/ai/chat', '/ai/correlations', '/ai/predictive', '/ai/unified'],
  governance: ['/governance/compliance', '/governance/policies', '/governance/risk', '/governance/cost'],
  security: ['/security/iam', '/security/rbac', '/security/pim', '/security/conditional-access'],
  operations: ['/operations/resources', '/operations/monitoring', '/operations/automation'],
  devops: ['/devops/pipelines', '/devops/releases', '/devops/builds'],
  itsm: ['/itsm/incidents', '/itsm/problems', '/itsm/changes', '/itsm/releases'],
};

// Prefetch strategy based on user navigation patterns
export const prefetchStrategy = {
  onDashboard: ['ai', 'governance', 'security'],
  onAI: ['ai'],
  onGovernance: ['governance'],
  onSecurity: ['security'],
  onOperations: ['operations'],
  onDevOps: ['devops'],
  onITSM: ['itsm'],
};

// Helper to get routes to prefetch based on current path
export const getRoutesToPrefetch = (currentPath: string): string[] => {
  const pathSegments = currentPath.split('/').filter(Boolean);
  const module = pathSegments[0];
  
  if (module && module in routeGroups) {
    return routeGroups[module as keyof typeof routeGroups];
  }
  
  // Default to core routes
  return routeGroups.core;
};