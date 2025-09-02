'use client';

import React, { useEffect, useRef } from 'react';
import { X, ChevronLeft, ChevronRight, Maximize2, Copy, ExternalLink } from 'lucide-react';

interface DrawerTab {
  id: string;
  label: string;
  icon?: React.ReactNode;
  content: React.ReactNode;
}

interface RightDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  tabs?: DrawerTab[];
  activeTab?: string;
  onTabChange?: (tabId: string) => void;
  children?: React.ReactNode;
  width?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  showOverlay?: boolean;
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
  footer?: React.ReactNode;
  actions?: React.ReactNode;
}

const widthConfig = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'max-w-full',
};

export default function RightDrawer({
  isOpen,
  onClose,
  title,
  subtitle,
  tabs,
  activeTab: controlledActiveTab,
  onTabChange,
  children,
  width = 'md',
  showOverlay = true,
  closeOnOverlayClick = true,
  closeOnEscape = true,
  footer,
  actions,
}: RightDrawerProps) {
  const [internalActiveTab, setInternalActiveTab] = React.useState(tabs?.[0]?.id || '');
  const drawerRef = useRef<HTMLDivElement>(null);
  
  const activeTab = controlledActiveTab || internalActiveTab;
  
  // Handle escape key
  useEffect(() => {
    if (!closeOnEscape || !isOpen) return;
    
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose, closeOnEscape]);
  
  // Focus trap
  useEffect(() => {
    if (!isOpen) return;
    
    const drawer = drawerRef.current;
    if (!drawer) return;
    
    // Focus first focusable element
    const focusableElements = drawer.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;
    
    firstElement?.focus();
    
    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;
      
      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };
    
    drawer.addEventListener('keydown', handleTabKey);
    return () => drawer.removeEventListener('keydown', handleTabKey);
  }, [isOpen]);
  
  const handleTabChange = (tabId: string) => {
    if (onTabChange) {
      onTabChange(tabId);
    } else {
      setInternalActiveTab(tabId);
    }
  };
  
  const activeTabContent = tabs?.find(tab => tab.id === activeTab)?.content;
  
  return (
    <>
      {/* Overlay */}
      {showOverlay && isOpen && (
        <div
          className="fixed inset-0 bg-black/30 backdrop-blur-sm z-40 transition-opacity"
          onClick={closeOnOverlayClick ? onClose : undefined}
          aria-hidden="true"
        />
      )}
      
      {/* Drawer */}
      <div
        ref={drawerRef}
        className={`
          fixed top-0 right-0 h-full z-50 
          bg-white dark:bg-gray-800 
          shadow-2xl
          transition-transform duration-300 ease-in-out
          ${widthConfig[width]} w-full
          ${isOpen ? 'translate-x-0' : 'translate-x-full'}
        `}
        role="dialog"
        aria-modal="true"
        aria-labelledby="drawer-title"
        aria-describedby={subtitle ? "drawer-subtitle" : undefined}
      >
        <div className="h-full flex flex-col">
          {/* Header */}
          <div className="flex items-start justify-between p-6 border-b border-gray-200 dark:border-gray-700">
            <div className="flex-1">
              <h2 
                id="drawer-title"
                className="text-xl font-semibold text-gray-900 dark:text-white"
              >
                {title}
              </h2>
              {subtitle && (
                <p 
                  id="drawer-subtitle"
                  className="mt-1 text-sm text-gray-500 dark:text-gray-400"
                >
                  {subtitle}
                </p>
              )}
            </div>
            <div className="flex items-center gap-2">
              {actions}
              <button
                onClick={onClose}
                className="p-2 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                aria-label="Close drawer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
          
          {/* Tabs */}
          {tabs && tabs.length > 0 && (
            <div className="flex border-b border-gray-200 dark:border-gray-700">
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => handleTabChange(tab.id)}
                  className={`
                    flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors
                    ${activeTab === tab.id
                      ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
                      : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                    }
                  `}
                  role="tab"
                  aria-selected={activeTab === tab.id}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>
          )}
          
          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6">
            {tabs && tabs.length > 0 ? activeTabContent : children}
          </div>
          
          {/* Footer */}
          {footer && (
            <div className="border-t border-gray-200 dark:border-gray-700 p-6">
              {footer}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// Compound components for common patterns
export function DrawerSection({ 
  title, 
  children,
  className = '',
}: { 
  title?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`mb-6 ${className}`}>
      {title && (
        <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
          {title}
        </h3>
      )}
      {children}
    </div>
  );
}

export function DrawerField({ 
  label, 
  value,
  copyable = false,
  href,
}: { 
  label: string;
  value: string | React.ReactNode;
  copyable?: boolean;
  href?: string;
}) {
  const handleCopy = () => {
    if (typeof value === 'string') {
      navigator.clipboard.writeText(value);
    }
  };
  
  return (
    <div className="flex items-start justify-between py-2">
      <span className="text-sm text-gray-500 dark:text-gray-400">
        {label}
      </span>
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-gray-900 dark:text-white">
          {href ? (
            <a 
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:underline"
            >
              {value}
              <ExternalLink className="w-3 h-3" />
            </a>
          ) : value}
        </span>
        {copyable && typeof value === 'string' && (
          <button
            onClick={handleCopy}
            className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label={`Copy ${label}`}
          >
            <Copy className="w-3 h-3 text-gray-400" />
          </button>
        )}
      </div>
    </div>
  );
}

export function DrawerJSON({ 
  data,
  className = '',
}: { 
  data: any;
  className?: string;
}) {
  return (
    <pre className={`
      bg-gray-50 dark:bg-gray-900 
      border border-gray-200 dark:border-gray-700 
      rounded-lg p-4 
      text-xs font-mono 
      overflow-x-auto
      ${className}
    `}>
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}