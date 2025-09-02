'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import {
  Search, Command, ArrowRight, FileText, Shield, Brain,
  DollarSign, Users, Database, GitBranch, Settings,
  AlertCircle, Activity, Home, X, Hash, User, Clock,
  Zap, CheckCircle, Lock, Globe, Cpu, BarChart3
} from 'lucide-react';
import { CORE, LABS, getNavigationItems } from '@/config/navigation';

interface QuickAction {
  id: string;
  label: string;
  description?: string;
  icon: React.ReactNode;
  action: () => void;
  category: 'navigation' | 'action' | 'search';
  keywords?: string[];
}

export default function GlobalOmnibar() {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [recentSearches, setRecentSearches] = useState<string[]>([]);
  
  const navigationItems = getNavigationItems();

  // Define all quick actions
  const quickActions = useMemo<QuickAction[]>(() => {
    const actions: QuickAction[] = [
      // Navigation items from config
      ...navigationItems.core.map(item => ({
        id: `nav-${item.href}`,
        label: item.label,
        description: item.description,
        icon: <FileText className="w-4 h-4" />,
        action: () => {
          router.push(item.href);
          setIsOpen(false);
        },
        category: 'navigation' as const,
        keywords: [item.label.toLowerCase(), item.href]
      })),
      
      // Quick actions
      {
        id: 'verify-chain',
        label: 'Verify Blockchain',
        description: 'Verify the entire audit chain integrity',
        icon: <Shield className="w-4 h-4" />,
        action: async () => {
          const res = await fetch('/api/v1/blockchain/verify');
          if (res.ok) {
            const data = await res.json();
            alert(`Chain ${data.chain_integrity ? 'Verified ✓' : 'Failed ✗'}`);
          }
          setIsOpen(false);
        },
        category: 'action',
        keywords: ['verify', 'blockchain', 'audit', 'integrity']
      },
      {
        id: 'create-fix-pr',
        label: 'Create Fix PR',
        description: 'Create a pull request from prediction',
        icon: <GitBranch className="w-4 h-4" />,
        action: () => {
          window.open('https://github.com/your-org/repo/pulls/new', '_blank');
          setIsOpen(false);
        },
        category: 'action',
        keywords: ['pr', 'pull request', 'fix', 'github']
      },
      {
        id: 'export-audit',
        label: 'Export Audit Evidence',
        description: 'Download audit trail with signatures',
        icon: <FileText className="w-4 h-4" />,
        action: () => {
          router.push('/audit?export=true');
          setIsOpen(false);
        },
        category: 'action',
        keywords: ['export', 'audit', 'download', 'evidence']
      },
      {
        id: 'view-predictions',
        label: 'View Active Predictions',
        description: 'See all AI predictions',
        icon: <Brain className="w-4 h-4" />,
        action: () => {
          router.push('/ai/predictions');
          setIsOpen(false);
        },
        category: 'action',
        keywords: ['predictions', 'ai', 'ml', 'forecast']
      },
      {
        id: 'roi-dashboard',
        label: 'Open ROI Dashboard',
        description: 'View financial metrics and savings',
        icon: <DollarSign className="w-4 h-4" />,
        action: () => {
          router.push('/finops');
          setIsOpen(false);
        },
        category: 'action',
        keywords: ['roi', 'finops', 'cost', 'savings', 'money']
      }
    ];
    
    // Add labs items if enabled
    if (navigationItems.labs.length > 0) {
      actions.push(...navigationItems.labs.map(item => ({
        id: `labs-${item.href}`,
        label: `[Labs] ${item.label}`,
        description: item.description,
        icon: <Cpu className="w-4 h-4" />,
        action: () => {
          router.push(item.href);
          setIsOpen(false);
        },
        category: 'navigation' as const,
        keywords: ['labs', item.label.toLowerCase(), item.href]
      })));
    }
    
    return actions;
  }, [router, navigationItems]);

  // Filter actions based on search
  const filteredActions = useMemo(() => {
    if (!searchQuery) return quickActions;
    
    const query = searchQuery.toLowerCase();
    return quickActions.filter(action => 
      action.label.toLowerCase().includes(query) ||
      action.description?.toLowerCase().includes(query) ||
      action.keywords?.some(kw => kw.includes(query))
    );
  }, [searchQuery, quickActions]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Open omnibar with Cmd/Ctrl + K
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
        setSearchQuery('');
        setSelectedIndex(0);
      }
      
      // Close on Escape
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
      
      // Navigate with arrow keys
      if (isOpen) {
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          setSelectedIndex(prev => 
            prev < filteredActions.length - 1 ? prev + 1 : 0
          );
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          setSelectedIndex(prev => 
            prev > 0 ? prev - 1 : filteredActions.length - 1
          );
        } else if (e.key === 'Enter' && filteredActions[selectedIndex]) {
          e.preventDefault();
          filteredActions[selectedIndex].action();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filteredActions, selectedIndex]);

  // Reset selection when search changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [searchQuery]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm"
        onClick={() => setIsOpen(false)}
      />
      
      {/* Modal */}
      <div className="flex min-h-full items-start justify-center p-4 pt-20">
        <div className="relative w-full max-w-2xl bg-white dark:bg-gray-800 rounded-xl shadow-2xl">
          {/* Search Header */}
          <div className="border-b border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-center gap-3">
              <Search className="w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search pages, run commands, or take quick actions..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-1 bg-transparent outline-none text-gray-900 dark:text-white placeholder-gray-400"
                autoFocus
              />
              <div className="flex items-center gap-2">
                <kbd className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 rounded">
                  ESC
                </kbd>
                <button
                  onClick={() => setIsOpen(false)}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
          
          {/* Results */}
          <div className="max-h-96 overflow-y-auto">
            {filteredActions.length === 0 ? (
              <div className="p-8 text-center text-gray-500">
                No results found for "{searchQuery}"
              </div>
            ) : (
              <div className="p-2">
                {/* Group by category */}
                {['navigation', 'action', 'search'].map(category => {
                  const categoryActions = filteredActions.filter(a => a.category === category);
                  if (categoryActions.length === 0) return null;
                  
                  return (
                    <div key={category} className="mb-4">
                      <div className="px-3 py-1 text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                        {category}
                      </div>
                      {categoryActions.map((action, idx) => {
                        const globalIndex = filteredActions.indexOf(action);
                        const isSelected = globalIndex === selectedIndex;
                        
                        return (
                          <button
                            key={action.id}
                            onClick={action.action}
                            onMouseEnter={() => setSelectedIndex(globalIndex)}
                            className={`
                              w-full px-3 py-2 flex items-center gap-3 rounded-lg text-left transition-colors
                              ${isSelected 
                                ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-900 dark:text-blue-100' 
                                : 'hover:bg-gray-50 dark:hover:bg-gray-700/50 text-gray-700 dark:text-gray-300'
                              }
                            `}
                          >
                            <div className={`
                              flex items-center justify-center w-8 h-8 rounded-lg
                              ${isSelected 
                                ? 'bg-blue-100 dark:bg-blue-800 text-blue-600 dark:text-blue-300' 
                                : 'bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
                              }
                            `}>
                              {action.icon}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="font-medium">{action.label}</div>
                              {action.description && (
                                <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
                                  {action.description}
                                </div>
                              )}
                            </div>
                            {isSelected && (
                              <div className="flex items-center gap-1">
                                <kbd className="px-1.5 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 rounded">
                                  Enter
                                </kbd>
                                <ArrowRight className="w-4 h-4" />
                              </div>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
          
          {/* Footer */}
          <div className="border-t border-gray-200 dark:border-gray-700 px-4 py-3 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded">↑↓</kbd>
                Navigate
              </span>
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded">Enter</kbd>
                Select
              </span>
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded">Esc</kbd>
                Close
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Command className="w-3 h-3" />
              <span>Command Palette</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}