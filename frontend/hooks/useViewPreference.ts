'use client';

import { useState, useEffect } from 'react';

type ViewType = 'cards' | 'visualizations';

export function useViewPreference(storageKey: string, defaultView: ViewType = 'cards') {
  const [view, setView] = useState<ViewType>(defaultView);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const savedView = localStorage.getItem(storageKey) as ViewType;
      if (savedView && (savedView === 'cards' || savedView === 'visualizations')) {
        setView(savedView);
      }
      setIsLoaded(true);
    }
  }, [storageKey]);

  const setViewPreference = (newView: ViewType) => {
    setView(newView);
    if (typeof window !== 'undefined') {
      localStorage.setItem(storageKey, newView);
    }
  };

  return { view, setView: setViewPreference, isLoaded };
}