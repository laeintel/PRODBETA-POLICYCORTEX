export type NavItem = {
  label: string;
  href: string;
  ready?: boolean;
  badge?: 'Labs' | 'Beta';
  icon?: string;
  description?: string;
}

// Final nav per spec: clean, enterprise, 7-8 core items
export const CORE: NavItem[] = [
  { 
    label: 'Home', 
    href: '/tactical', 
    ready: true,
    description: 'Executive dashboard with predictive insights and ROI metrics'
  },
  { 
    label: 'Audit Trail', 
    href: '/audit', 
    ready: true,
    description: 'Tamper-evident blockchain-secured audit logs with verification'
  },
  { 
    label: 'Predict', 
    href: '/predict', 
    ready: true,
    description: 'AI predictions with 7-day look-ahead and auto-remediation'
  },
  { 
    label: 'FinOps & ROI', 
    href: '/finops', 
    ready: true,
    description: 'Cost optimization, savings tracking, and executive ROI'
  },
  { 
    label: 'Access Governance', 
    href: '/rbac', 
    ready: true,
    description: 'Identity, permissions, and RBAC management'
  },
  { 
    label: 'Resources', 
    href: '/resources', 
    ready: true,
    description: 'Cloud resource inventory and management'
  },
  { 
    label: 'DevSecOps', 
    href: '/devsecops', 
    ready: true,
    description: 'CI/CD gates, policy-as-code, and auto-fix PRs'
  },
  {
    label: 'Settings',
    href: '/settings',
    ready: true,
    description: 'System configuration and preferences'
  }
];

// Labs: powerful but not-yet-central capabilities
export const LABS: NavItem[] = [
  { 
    label: 'Copilot', 
    href: '/copilot', 
    badge: 'Labs',
    description: 'AI-powered conversational assistant'
  },
  { 
    label: 'Cloud ITSM', 
    href: '/itsm', 
    badge: 'Labs',
    description: 'IT service management for cloud infrastructure'
  },
  { 
    label: 'Quantum-Safe', 
    href: '/quantum', 
    badge: 'Labs',
    description: 'Post-quantum cryptography and security'
  },
  { 
    label: 'Edge Governance', 
    href: '/edge', 
    badge: 'Labs',
    description: 'Edge computing governance and management'
  },
  {
    label: 'Governance Policies',
    href: '/governance',
    badge: 'Labs',
    description: 'Advanced policy management (beta)'
  }
];

// Feature flags for conditional rendering
export const FEATURE_FLAGS = {
  SHOW_LABS: process.env.NEXT_PUBLIC_SHOW_LABS !== 'false',
  SHOW_BETA: process.env.NEXT_PUBLIC_SHOW_BETA !== 'false',
  ENABLE_VOICE: process.env.NEXT_PUBLIC_ENABLE_VOICE === 'true',
  ENABLE_3D: process.env.NEXT_PUBLIC_ENABLE_3D === 'true'
};

// Helper to get navigation items based on feature flags
export function getNavigationItems() {
  const items = [...CORE];
  
  if (FEATURE_FLAGS.SHOW_LABS) {
    // Filter labs items based on specific feature flags
    const labsItems = LABS.filter(item => {
      if (item.href === '/voice' && !FEATURE_FLAGS.ENABLE_VOICE) return false;
      if (item.href === '/3d' && !FEATURE_FLAGS.ENABLE_3D) return false;
      return true;
    });
    
    return { core: items, labs: labsItems };
  }
  
  return { core: items, labs: [] };
}

// Helper to check if a path is a labs feature
export function isLabsPath(pathname: string): boolean {
  return LABS.some(item => pathname.startsWith(item.href));
}

// Helper to get item by path
export function getNavItemByPath(pathname: string): NavItem | undefined {
  const allItems = [...CORE, ...LABS];
  return allItems.find(item => pathname.startsWith(item.href));
}