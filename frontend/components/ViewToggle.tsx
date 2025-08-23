'use client';

import React from 'react';
import { BarChart3, Grid3X3 } from 'lucide-react';

interface ViewToggleProps {
  view: 'cards' | 'visualizations';
  onViewChange: (view: 'cards' | 'visualizations') => void;
}

export default function ViewToggle({ view, onViewChange }: ViewToggleProps) {
  return (
    <div className="flex items-center space-x-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
      <button
        onClick={() => onViewChange('cards')}
        className={`inline-flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
          view === 'cards'
            ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
            : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
        }`}
      >
        <Grid3X3 className="h-4 w-4 mr-2" />
        Cards
      </button>
      <button
        onClick={() => onViewChange('visualizations')}
        className={`inline-flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
          view === 'visualizations'
            ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
            : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
        }`}
      >
        <BarChart3 className="h-4 w-4 mr-2" />
        Visualizations
      </button>
    </div>
  );
}