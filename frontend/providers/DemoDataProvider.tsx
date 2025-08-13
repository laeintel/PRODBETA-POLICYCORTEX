'use client';

import React, { createContext, useContext, useCallback, useMemo } from 'react';
import { useSearchParams } from 'next/navigation';

interface DemoDataContextType {
  isDemoMode: boolean;
  useMockData: boolean;
  enableDemo: () => void;
  disableDemo: () => void;
  toggleDemo: () => void;
  getDemoData: <T>(key: string, fallback: T) => T;
}

const DemoDataContext = createContext<DemoDataContextType | undefined>(undefined);

// Centralized mock data store
const MOCK_DATA_STORE = {
  metrics: {
    compliance: 94,
    resources: 1247,
    policies: 156,
    violations: 23,
    trend: 'up',
  },
  policies: [
    {
      id: 'demo-1',
      name: 'Require HTTPS for Storage Accounts',
      status: 'active',
      compliance: 98,
      affected: 42,
    },
    {
      id: 'demo-2',
      name: 'Enable Disk Encryption',
      status: 'active',
      compliance: 87,
      affected: 156,
    },
  ],
  correlations: [
    {
      id: 'corr-1',
      type: 'security-cost',
      strength: 0.85,
      description: 'High security configurations correlate with 15% higher costs',
    },
  ],
  predictions: [
    {
      id: 'pred-1',
      resource: 'Storage Accounts',
      drift: 12,
      timeframe: '7 days',
      confidence: 0.89,
    },
  ],
};

export function DemoDataProvider({ children }: { children: React.ReactNode }) {
  const searchParams = useSearchParams();
  
  // Check multiple sources for demo mode
  const isDemoMode = useMemo(() => {
    // Priority order:
    // 1. URL parameter
    if (searchParams?.get('demo') === 'true') return true;
    
    // 2. Environment variable
    if (process.env.NEXT_PUBLIC_DEMO_MODE === 'true') return true;
    
    // 3. Local storage
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('demoMode');
      if (stored === 'true') return true;
    }
    
    // 4. No real API configured
    if (!process.env.NEXT_PUBLIC_API_URL || 
        process.env.NEXT_PUBLIC_API_URL === 'http://localhost:8080') {
      return true;
    }
    
    return false;
  }, [searchParams]);

  const enableDemo = useCallback(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('demoMode', 'true');
      window.location.reload();
    }
  }, []);

  const disableDemo = useCallback(() => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('demoMode');
      // Remove demo param from URL if present
      const url = new URL(window.location.href);
      url.searchParams.delete('demo');
      window.history.replaceState({}, '', url.toString());
      window.location.reload();
    }
  }, []);

  const toggleDemo = useCallback(() => {
    if (isDemoMode) {
      disableDemo();
    } else {
      enableDemo();
    }
  }, [isDemoMode, enableDemo, disableDemo]);

  const getDemoData = useCallback(<T,>(key: string, fallback: T): T => {
    if (!isDemoMode) return fallback;
    
    // Navigate nested keys (e.g., "metrics.compliance")
    const keys = key.split('.');
    let data: any = MOCK_DATA_STORE;
    
    for (const k of keys) {
      if (data && typeof data === 'object' && k in data) {
        data = data[k];
      } else {
        return fallback;
      }
    }
    
    return data as T;
  }, [isDemoMode]);

  const value = useMemo(
    () => ({
      isDemoMode,
      useMockData: isDemoMode,
      enableDemo,
      disableDemo,
      toggleDemo,
      getDemoData,
    }),
    [isDemoMode, enableDemo, disableDemo, toggleDemo, getDemoData]
  );

  return (
    <DemoDataContext.Provider value={value}>
      {children}
    </DemoDataContext.Provider>
  );
}

export function useDemoData() {
  const context = useContext(DemoDataContext);
  if (context === undefined) {
    throw new Error('useDemoData must be used within a DemoDataProvider');
  }
  return context;
}

// HOC for wrapping components with demo data
export function withDemoData<P extends object>(
  Component: React.ComponentType<P>,
  mockProps?: Partial<P>
) {
  return function WrappedComponent(props: P) {
    const { isDemoMode, getDemoData } = useDemoData();
    
    if (isDemoMode && mockProps) {
      const demoProps = Object.keys(mockProps).reduce((acc, key) => {
        acc[key] = getDemoData(key, mockProps[key]);
        return acc;
      }, {} as any);
      
      return <Component {...props} {...demoProps} />;
    }
    
    return <Component {...props} />;
  };
}