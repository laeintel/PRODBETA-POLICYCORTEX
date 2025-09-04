'use client';

import React, { Suspense, ComponentType, lazy } from 'react';
import { Loader2 } from 'lucide-react';

// Loading component with better UX
export const LoadingFallback: React.FC<{ message?: string }> = ({ message = 'Loading...' }) => (
  <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
    <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
    <p className="text-gray-600 dark:text-gray-400 text-sm">{message}</p>
  </div>
);

// Error boundary for lazy loaded components
interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class LazyErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode; fallback?: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Lazy loading error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4 p-6">
            <div className="text-red-500 text-xl">⚠️</div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Failed to load component
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 text-center max-w-md">
              {this.state.error?.message || 'An unexpected error occurred while loading this component.'}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
            >
              Reload Page
            </button>
          </div>
        )
      );
    }

    return this.props.children;
  }
}

// Wrapper for lazy loaded components
export function withLazyLoad<T extends ComponentType<any>>(
  importFunc: () => Promise<{ default: T }>,
  loadingMessage?: string
) {
  const LazyComponent = lazy(importFunc);

  return (props: React.ComponentProps<T>) => (
    <LazyErrorBoundary>
      <Suspense fallback={<LoadingFallback message={loadingMessage} />}>
        <LazyComponent {...props} />
      </Suspense>
    </LazyErrorBoundary>
  );
}

// Pre-configured lazy imports for PCG components
export const LazyPrevent = withLazyLoad(
  () => import('../app/prevent/page').then(mod => ({ default: mod.default })),
  'Loading Prevent...'
);

export const LazyProve = withLazyLoad(
  () => import('../app/prove/page').then(mod => ({ default: mod.default })),
  'Loading Prove...'
);

export const LazyPayback = withLazyLoad(
  () => import('../app/payback/page').then(mod => ({ default: mod.default })),
  'Loading Payback...'
);

// Utility to preload components
export const preloadComponent = (
  importFunc: () => Promise<any>
) => {
  // Start loading the component in the background
  importFunc().catch(err => {
    console.warn('Failed to preload component:', err);
  });
};

// Hook to preload components on hover or focus
export const usePreload = (
  importFunc: () => Promise<any>
) => {
  const handlePreload = React.useCallback(() => {
    preloadComponent(importFunc);
  }, [importFunc]);

  return {
    onMouseEnter: handlePreload,
    onFocus: handlePreload,
  };
};