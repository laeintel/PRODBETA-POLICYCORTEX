'use client';

import { useEffect, useState } from 'react';
import { useToast } from '../hooks/useToast';

export function ServiceWorkerRegistration() {
  const [registration, setRegistration] = useState<ServiceWorkerRegistration | null>(null);
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    // Only register service worker in production and if supported
    if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
      return;
    }

    // Register service worker
    const registerServiceWorker = async () => {
      try {
        const reg = await navigator.serviceWorker.register('/service-worker.js', {
          scope: '/',
        });
        
        setRegistration(reg);
        console.log('Service Worker registered successfully');

        // Check for updates every hour
        const interval = setInterval(() => {
          reg.update();
        }, 60 * 60 * 1000);

        // Listen for update found
        reg.addEventListener('updatefound', () => {
          const newWorker = reg.installing;
          if (newWorker) {
            newWorker.addEventListener('statechange', () => {
              if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                setUpdateAvailable(true);
                toast({
                  title: 'Update Available',
                  description: 'A new version of PolicyCortex is available. Reload to update.',
                  action: (
                    <button
                      onClick={() => window.location.reload()}
                      className="text-sm font-medium text-blue-600 hover:text-blue-500"
                    >
                      Reload Now
                    </button>
                  ),
                });
              }
            });
          }
        });

        return () => clearInterval(interval);
      } catch (error) {
        console.error('Service Worker registration failed:', error);
      }
    };

    // Wait for window load to register SW to not block initial render
    if (document.readyState === 'complete') {
      registerServiceWorker();
    } else {
      window.addEventListener('load', registerServiceWorker);
      return () => window.removeEventListener('load', registerServiceWorker);
    }
  }, [toast]);

  // Handle app update prompt
  useEffect(() => {
    if (!updateAvailable || !registration) return;

    // Skip waiting and activate new service worker
    const handleUpdate = () => {
      if (registration.waiting) {
        registration.waiting.postMessage({ type: 'SKIP_WAITING' });
        window.location.reload();
      }
    };

    // Add keyboard shortcut for update
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'U') {
        handleUpdate();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [updateAvailable, registration]);

  // Listen for controller change (new SW activated)
  useEffect(() => {
    if (!('serviceWorker' in navigator)) return;

    const handleControllerChange = () => {
      window.location.reload();
    };

    navigator.serviceWorker.addEventListener('controllerchange', handleControllerChange);
    return () => {
      navigator.serviceWorker.removeEventListener('controllerchange', handleControllerChange);
    };
  }, []);

  // Handle offline/online status
  useEffect(() => {
    const handleOnline = () => {
      toast({
        title: 'Back Online',
        description: 'Your connection has been restored.',
        variant: 'success',
      });
    };

    const handleOffline = () => {
      toast({
        title: 'Offline Mode',
        description: 'You are currently offline. Some features may be limited.',
        variant: 'warning',
      });
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [toast]);

  return null; // This component doesn't render anything
}