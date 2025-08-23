'use client';

import React, { useState, ReactNode } from 'react';
import { Maximize2, Download } from 'lucide-react';

interface ChartContainerProps {
  title: string;
  children: ReactNode;
  onExport?: () => void;
  onDrillIn?: () => void;
  className?: string;
  fullscreen?: boolean;
  onFullscreenToggle?: () => void;
}

export default function ChartContainer({
  title,
  children,
  onExport,
  onDrillIn,
  className = '',
  fullscreen = false,
  onFullscreenToggle
}: ChartContainerProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      className={`
        bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 
        shadow-sm transition-all duration-200 hover:shadow-md
        ${fullscreen ? 'fixed inset-4 z-50' : 'h-96'}
        ${className}
      `}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {title}
          </h3>
          <div className={`flex items-center space-x-2 transition-opacity ${isHovered ? 'opacity-100' : 'opacity-0'}`}>
            {onExport && (
              <button
                onClick={onExport}
                className="p-1.5 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                title="Export chart"
              >
                <Download className="h-4 w-4" />
              </button>
            )}
            {onFullscreenToggle && (
              <button
                onClick={onFullscreenToggle}
                className="p-1.5 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
              >
                <Maximize2 className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      </div>
      <div 
        className={`p-4 ${onDrillIn ? 'cursor-pointer' : ''} ${fullscreen ? 'h-full' : 'h-80'}`}
        onClick={onDrillIn}
      >
        <div className="w-full h-full">
          {children}
        </div>
      </div>
      {fullscreen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 -z-10"
          onClick={onFullscreenToggle}
        />
      )}
    </div>
  );
}