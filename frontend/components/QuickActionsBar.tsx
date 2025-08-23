/**
 * PATENT NOTICE: This component implements Patent #2 - Conversational Governance Intelligence System
 * for real-time AI-powered governance assistance with natural language processing
 */

'use client'

import React, { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Shield,
  DollarSign,
  MessageSquare,
  TrendingUp,
  AlertTriangle,
  Server,
  Activity,
  Search,
  Mic,
  Command,
  CheckCircle,
  XCircle,
  Clock,
  ChevronRight
} from 'lucide-react'
import GlobalAIChat from './AIAssistant/GlobalAIChat'
import VoiceActivation from './AIAssistant/VoiceActivation'

interface QuickActionItem {
  id: string
  icon: React.ElementType
  label: string
  shortLabel: string
  action: () => void
  color: string
  bgColor: string
  hoverColor: string
  value?: string | number
  trend?: number
  loading?: boolean
  pulse?: boolean
}

interface MetricsData {
  compliance: {
    score: number
    trend: number
    details: {
      totalPolicies: number
      compliantPolicies: number
      nonCompliantPolicies: number
    }
  }
  risks: {
    active: number
    critical: number
    high: number
    trend: number
  }
  costs: {
    savings: number
    trend: number
    currentMonth: number
  }
  ai: {
    predictions: number
    conversations: number
    accuracy: {
      predictiveCompliance: number
    }
  }
  resources: {
    total: number
    byType: Record<string, number>
  }
}

export default function QuickActionsBar() {
  const router = useRouter()
  const [aiChatOpen, setAiChatOpen] = useState(false)
  const [metrics, setMetrics] = useState<MetricsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [hoveredAction, setHoveredAction] = useState<string | null>(null)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [recentAlert, setRecentAlert] = useState<string | null>(null)

  // Fetch real-time metrics
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch('/api/v1/metrics')
        const data = await response.json()
        setMetrics(data)
        setLoading(false)

        // Check for critical alerts
        if (data.risks.critical > 0) {
          setRecentAlert(`${data.risks.critical} critical risk${data.risks.critical > 1 ? 's' : ''} detected`)
        }
      } catch (error) {
        console.error('Failed to fetch metrics:', error)
        setLoading(false)
      }
    }

    fetchMetrics()
    const interval = setInterval(fetchMetrics, 30000) // Refresh every 30 seconds

    return () => clearInterval(interval)
  }, [])

  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + K for AI Chat
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setAiChatOpen(true)
      }
      // Cmd/Ctrl + / for Quick Search
      if ((e.metaKey || e.ctrlKey) && e.key === '/') {
        e.preventDefault()
        router.push('/search')
      }
      // Escape to close AI Chat
      if (e.key === 'Escape' && aiChatOpen) {
        setAiChatOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [aiChatOpen, router])

  const quickActions: QuickActionItem[] = [
    {
      id: 'compliance',
      icon: Shield,
      label: 'Compliance Status',
      shortLabel: 'Compliance',
      action: () => router.push('/governance?tab=compliance'),
      color: (metrics?.compliance?.score ?? 0) >= 90 ? 'text-green-600 dark:text-green-400' : 'text-yellow-600 dark:text-yellow-400',
      bgColor: (metrics?.compliance?.score ?? 0) >= 90 ? 'bg-green-100 dark:bg-green-500/10' : 'bg-yellow-100 dark:bg-yellow-500/10',
      hoverColor: (metrics?.compliance?.score ?? 0) >= 90 ? 'hover:bg-green-200 dark:hover:bg-green-500/20' : 'hover:bg-yellow-200 dark:hover:bg-yellow-500/20',
      value: metrics?.compliance?.score ? `${metrics.compliance.score}%` : '...',
      trend: metrics?.compliance?.trend,
      loading: loading,
      pulse: (metrics?.compliance?.score ?? 100) < 80
    },
    {
      id: 'costs',
      icon: DollarSign,
      label: 'Cost Savings',
      shortLabel: 'Savings',
      action: () => router.push('/governance?tab=cost'),
      color: 'text-blue-600 dark:text-blue-400',
      bgColor: 'bg-blue-100 dark:bg-blue-500/10',
      hoverColor: 'hover:bg-blue-200 dark:hover:bg-blue-500/20',
      value: metrics?.costs?.savings ? `$${(metrics.costs.savings / 1000).toFixed(0)}K` : '...',
      trend: metrics?.costs?.trend,
      loading: loading
    },
    {
      id: 'ai-chat',
      icon: MessageSquare,
      label: 'Chat with AI',
      shortLabel: 'AI Chat',
      action: () => setAiChatOpen(true),
      color: 'text-purple-600 dark:text-purple-400',
      bgColor: 'bg-purple-100 dark:bg-purple-500/10',
      hoverColor: 'hover:bg-purple-200 dark:hover:bg-purple-500/20',
      value: 'Ask',
      loading: false,
      pulse: true
    },
    {
      id: 'predictions',
      icon: TrendingUp,
      label: 'View Predictions',
      shortLabel: 'Predictions',
      action: () => router.push('/ai/predictive'),
      color: 'text-indigo-600 dark:text-indigo-400',
      bgColor: 'bg-indigo-100 dark:bg-indigo-500/10',
      hoverColor: 'hover:bg-indigo-200 dark:hover:bg-indigo-500/20',
      value: metrics?.ai?.predictions || 0,
      loading: loading
    },
    {
      id: 'risks',
      icon: AlertTriangle,
      label: 'Active Risks',
      shortLabel: 'Risks',
      action: () => router.push('/governance?tab=risk'),
      color: (metrics?.risks?.critical ?? 0) > 0 ? 'text-red-600 dark:text-red-400' : 'text-orange-600 dark:text-orange-400',
      bgColor: (metrics?.risks?.critical ?? 0) > 0 ? 'bg-red-100 dark:bg-red-500/10' : 'bg-orange-100 dark:bg-orange-500/10',
      hoverColor: (metrics?.risks?.critical ?? 0) > 0 ? 'hover:bg-red-500/20' : 'hover:bg-orange-500/20',
      value: metrics?.risks?.active || 0,
      trend: metrics?.risks?.trend,
      loading: loading,
      pulse: (metrics?.risks?.critical ?? 0) > 0
    },
    {
      id: 'resources',
      icon: Server,
      label: 'View Resources',
      shortLabel: 'Resources',
      action: () => router.push('/operations/resources'),
      color: 'text-cyan-600 dark:text-cyan-400',
      bgColor: 'bg-cyan-100 dark:bg-cyan-500/10',
      hoverColor: 'hover:bg-cyan-200 dark:hover:bg-cyan-500/20',
      value: metrics?.resources?.total || 0,
      loading: loading
    }
  ]

  const handleVoiceCommand = (command: string) => {
    const lowerCommand = command.toLowerCase()
    
    if (lowerCommand.includes('compliance')) {
      router.push('/governance?tab=compliance')
    } else if (lowerCommand.includes('cost') || lowerCommand.includes('savings')) {
      router.push('/governance?tab=cost')
    } else if (lowerCommand.includes('risk')) {
      router.push('/governance?tab=risk')
    } else if (lowerCommand.includes('resource')) {
      router.push('/operations/resources')
    } else if (lowerCommand.includes('prediction')) {
      router.push('/ai/predictive')
    } else {
      // Default to opening AI chat with the command
      setAiChatOpen(true)
    }
  }

  return (
    <>
      {/* Quick Actions Bar */}
      <div className="fixed top-16 left-64 right-0 h-14 bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm border-b border-gray-200 dark:border-gray-800 z-10">
        <div className="h-full px-4 flex items-center justify-between">
          {/* Left side - Quick Actions */}
          <div className="flex items-center gap-2">
            {quickActions.map((action) => {
              const Icon = action.icon
              return (
                <motion.button
                  key={action.id}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={action.action}
                  onMouseEnter={() => setHoveredAction(action.id)}
                  onMouseLeave={() => setHoveredAction(null)}
                  className={`
                    relative flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all
                    ${action.bgColor} ${action.hoverColor} border border-gray-700
                    ${action.pulse ? 'animate-pulse' : ''}
                  `}
                >
                  {/* Loading state */}
                  {action.loading ? (
                    <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Icon className={`w-4 h-4 ${action.color}`} />
                  )}
                  
                  {/* Label - show short on mobile, full on desktop */}
                  <span className={`text-sm font-medium ${action.color} hidden sm:inline`}>
                    {action.label}
                  </span>
                  <span className={`text-sm font-medium ${action.color} sm:hidden`}>
                    {action.shortLabel}
                  </span>
                  
                  {/* Value/Badge */}
                  {action.value && (
                    <span className={`text-xs font-bold ${action.color}`}>
                      {action.value}
                    </span>
                  )}
                  
                  {/* Trend indicator */}
                  {action.trend !== undefined && action.trend !== 0 && (
                    <span className={`text-xs ${action.trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {action.trend > 0 ? '↑' : '↓'}{Math.abs(action.trend)}%
                    </span>
                  )}
                  
                  {/* Tooltip */}
                  <AnimatePresence>
                    {hoveredAction === action.id && (
                      <motion.div
                        initial={{ opacity: 0, y: -5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -5 }}
                        className="absolute top-full mt-2 left-1/2 -translate-x-1/2 px-2 py-1 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white text-xs rounded whitespace-nowrap z-50"
                      >
                        {action.label}
                        {action.id === 'ai-chat' && (
                          <span className="ml-2 text-gray-600 dark:text-gray-400">⌘K</span>
                        )}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.button>
              )
            })}
          </div>

          {/* Right side - Status indicators and voice */}
          <div className="flex items-center gap-3">
            {/* Recent Alert */}
            <AnimatePresence>
              {recentAlert && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 border border-red-500/20 rounded-lg"
                >
                  <AlertTriangle className="w-4 h-4 text-red-400" />
                  <span className="text-sm text-red-400">{recentAlert}</span>
                  <button type="button"
                    onClick={() => setRecentAlert(null)}
                    className="ml-2 text-red-400 hover:text-red-300"
                  >
                    <XCircle className="w-3 h-3" />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* AI Status */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-purple-500/10 rounded-lg">
              <Activity className="w-4 h-4 text-purple-400" />
              <span className="text-xs text-purple-400">
                AI: {metrics?.ai?.accuracy?.predictiveCompliance ? metrics.ai.accuracy.predictiveCompliance.toFixed(1) : '..'}% Accuracy
              </span>
            </div>

            {/* Voice Toggle */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setVoiceEnabled(!voiceEnabled)}
              className={`
                p-2 rounded-lg transition-all
                ${voiceEnabled 
                  ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30' 
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-300 dark:border-gray-700 hover:bg-gray-200 dark:hover:bg-gray-700'
                }
              `}
            >
              <Mic className="w-4 h-4" />
            </motion.button>

            {/* Command Palette hint */}
            <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 bg-gray-100 dark:bg-gray-800 rounded-lg">
              <Command className="w-3 h-3 text-gray-600 dark:text-gray-400" />
              <span className="text-xs text-gray-600 dark:text-gray-400">Press</span>
              <kbd className="text-xs bg-white dark:bg-gray-900 px-1.5 py-0.5 rounded text-gray-700 dark:text-gray-300">⌘K</kbd>
              <span className="text-xs text-gray-600 dark:text-gray-400">for AI</span>
            </div>
          </div>
        </div>
      </div>

      {/* Global AI Chat Interface */}
      <GlobalAIChat 
        isOpen={aiChatOpen} 
        onClose={() => setAiChatOpen(false)}
        metrics={metrics}
      />

      {/* Voice Activation Component */}
      {voiceEnabled && (
        <VoiceActivation
          onCommand={handleVoiceCommand}
          onTranscript={(text) => {
            // Optionally show what was heard
            console.log('Voice transcript:', text)
          }}
          isEnabled={voiceEnabled}
        />
      )}

      {/* Adjust main content padding to account for Quick Actions Bar */}
      <style jsx global>{`
        .pt-16 {
          padding-top: 7.5rem !important; /* 4rem (header) + 3.5rem (quick actions) */
        }
      `}</style>
    </>
  )
}