'use client'

import React from 'react'
import { Zap, FileCode, Shield, AlertTriangle, GitBranch, Send } from 'lucide-react'

interface QuickAction {
  id: string
  label: string
  description?: string
  icon: 'fix' | 'policy' | 'shield' | 'alert' | 'git' | 'send'
  variant: 'primary' | 'secondary' | 'danger' | 'success'
  onClick: () => void
  disabled?: boolean
}

interface QuickActionsProps {
  actions: QuickAction[]
  title?: string
  layout?: 'grid' | 'list'
}

export default function QuickActions({
  actions,
  title = 'Quick Actions',
  layout = 'grid'
}: QuickActionsProps) {
  const getIcon = (iconType: string) => {
    switch (iconType) {
      case 'fix':
        return <Zap className="h-5 w-5" />
      case 'policy':
        return <FileCode className="h-5 w-5" />
      case 'shield':
        return <Shield className="h-5 w-5" />
      case 'alert':
        return <AlertTriangle className="h-5 w-5" />
      case 'git':
        return <GitBranch className="h-5 w-5" />
      case 'send':
        return <Send className="h-5 w-5" />
      default:
        return <Zap className="h-5 w-5" />
    }
  }

  const getVariantStyles = (variant: string) => {
    switch (variant) {
      case 'primary':
        return 'bg-blue-600 hover:bg-blue-700 text-white border-blue-700'
      case 'secondary':
        return 'bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-100 border-gray-300 dark:border-gray-600'
      case 'danger':
        return 'bg-red-600 hover:bg-red-700 text-white border-red-700'
      case 'success':
        return 'bg-green-600 hover:bg-green-700 text-white border-green-700'
      default:
        return 'bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-100'
    }
  }

  const containerClass = layout === 'grid' 
    ? 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4'
    : 'space-y-3'

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-md border border-gray-200 dark:border-gray-700 p-6">
      {title && (
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">{title}</h2>
          <Zap className="h-5 w-5 text-yellow-500" />
        </div>
      )}

      <div className={containerClass}>
        {actions.map((action) => (
          <button
            key={action.id}
            onClick={action.onClick}
            disabled={action.disabled}
            className={`
              relative overflow-hidden transition-all duration-200 rounded-lg border
              ${getVariantStyles(action.variant)}
              ${action.disabled ? 'opacity-50 cursor-not-allowed' : 'hover:scale-[1.02] active:scale-[0.98]'}
              ${layout === 'grid' ? 'p-4' : 'p-3 w-full text-left'}
            `}
          >
            <div className="flex items-center gap-3">
              <div className={`
                flex items-center justify-center rounded-lg
                ${layout === 'grid' ? 'w-10 h-10 bg-white/10' : 'w-8 h-8 bg-white/10'}
              `}>
                {getIcon(action.icon)}
              </div>
              <div className="flex-1">
                <p className={`font-semibold ${layout === 'grid' ? 'text-sm' : 'text-base'}`}>
                  {action.label}
                </p>
                {action.description && layout === 'list' && (
                  <p className="text-xs opacity-80 mt-0.5">{action.description}</p>
                )}
              </div>
            </div>
            
            {action.description && layout === 'grid' && (
              <p className="text-xs opacity-80 mt-2">{action.description}</p>
            )}

            {/* Animated background effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full hover:translate-x-full transition-transform duration-700 pointer-events-none" />
          </button>
        ))}
      </div>
    </div>
  )
}