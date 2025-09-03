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
  ChevronRight,
  Briefcase
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
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [lastScrollY, setLastScrollY] = useState(0)

  // Handle scroll behavior for collapsing
  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY
      
      // Collapse when scrolling down more than 50px, expand when scrolling up
      if (currentScrollY > 50 && currentScrollY > lastScrollY) {
        setIsCollapsed(true)
      } else if (currentScrollY < lastScrollY) {
        setIsCollapsed(false)
      }
      
      // Always show when at the top
      if (currentScrollY < 10) {
        setIsCollapsed(false)
      }
      
      setLastScrollY(currentScrollY)
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [lastScrollY])

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
      // Cmd/Ctrl + B to toggle Quick Actions Bar
      if ((e.metaKey || e.ctrlKey) && e.key === 'b') {
        e.preventDefault()
        setIsCollapsed(prev => !prev)
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
      id: 'executive',
      icon: Briefcase,
      label: 'Executive Dashboard',
      shortLabel: 'Executive',
      action: () => router.push('/executive'),
      color: 'text-indigo-600 dark:text-indigo-400',
      bgColor: 'bg-indigo-100 dark:bg-indigo-500/10',
      hoverColor: 'hover:bg-indigo-200 dark:hover:bg-indigo-500/20',
      value: '287% ROI',
      loading: false
    },
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
      {/* Hover Zone - Invisible area to trigger expansion when collapsed */}
      {isCollapsed && (
        <div 
          className="fixed top-16 left-0 lg:left-64 xl:left-72 2xl:left-80 right-0 h-20 z-20"
          onMouseEnter={() => setIsCollapsed(false)}
        />
      )}
      
      {/* Quick Actions Bar with Collapse Animation */}
      <motion.div 
        initial={{ y: 0, height: 48 }}
        animate={{ 
          y: isCollapsed ? -44 : 0,
          height: isCollapsed ? 4 : 48
        }}
        transition={{ 
          type: "spring", 
          stiffness: 300, 
          damping: 30 
        }}
        className="fixed top-16 left-0 lg:left-64 xl:left-72 2xl:left-80 right-0 bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm border-b border-gray-200 dark:border-gray-800 z-10 overflow-hidden"
        onMouseLeave={() => {
          // Auto-collapse when mouse leaves if scrolled down
          if (window.scrollY > 50) {
            setIsCollapsed(true)
          }
        }}
      >
        {/* Collapsed State Indicator */}
        {isCollapsed && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="h-1 bg-gradient-to-r from-purple-500/10 via-indigo-500/10 to-blue-500/10 flex items-center justify-center cursor-pointer group hover:bg-gradient-to-r hover:from-purple-500/20 hover:via-indigo-500/20 hover:to-blue-500/20 transition-all"
            onClick={() => setIsCollapsed(false)}
          >
            <div className="flex items-center gap-1">
              {/* Minimal dots */}
              <div className="flex gap-0.5">
                {[...Array(3)].map((_, i) => (
                  <motion.div 
                    key={i} 
                    className="w-0.5 h-0.5 bg-gradient-to-r from-purple-400 to-indigo-400 rounded-full"
                    animate={{ 
                      opacity: [0.3, 0.8, 0.3]
                    }}
                    transition={{ 
                      duration: 2,
                      delay: i * 0.3,
                      repeat: Infinity
                    }}
                  />
                ))}
              </div>
              
              {/* Alert indicator if there are critical issues */}
              {(metrics?.risks?.critical ?? 0) > 0 && (
                <motion.div
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                  className="w-1 h-1 bg-red-500 rounded-full ml-1"
                />
              )}
            </div>
          </motion.div>
        )}
        
        {/* Main Content */}
        <motion.div 
          animate={{ opacity: isCollapsed ? 0 : 1 }}
          transition={{ duration: 0.2 }}
          className="h-12 px-3 flex items-center justify-between"
        >
          {/* Left side - Quick Actions */}
          <div className="flex items-center gap-1.5">
            {quickActions.map((action) => {
              const Icon = action.icon
              return (
                <motion.button
                  key={action.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={action.action}
                  onMouseEnter={() => setHoveredAction(action.id)}
                  onMouseLeave={() => setHoveredAction(null)}
                  className={`
                    relative flex items-center gap-1.5 px-2 py-1 rounded-md transition-all
                    ${action.bgColor} ${action.hoverColor} border border-gray-700
                    ${action.pulse ? 'animate-pulse' : ''}
                  `}
                >
                  {/* Loading state */}
                  {action.loading ? (
                    <div className="w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Icon className={`w-3 h-3 ${action.color}`} />
                  )}
                  
                  {/* Label - show short on mobile, full on desktop */}
                  <span className={`text-xs font-medium ${action.color} hidden lg:inline`}>
                    {action.label}
                  </span>
                  <span className={`text-xs font-medium ${action.color} lg:hidden`}>
                    {action.shortLabel}
                  </span>
                  
                  {/* Value/Badge */}
                  {action.value && (
                    <span className={`text-[10px] font-bold ${action.color}`}>
                      {action.value}
                    </span>
                  )}
                  
                  {/* Trend indicator */}
                  {action.trend !== undefined && action.trend !== 0 && (
                    <span className={`text-[10px] ${action.trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {action.trend > 0 ? '↑' : '↓'}{Math.abs(action.trend)}%
                    </span>
                  )}
                  
                  {/* Tooltip */}
                  <AnimatePresence>
                    {hoveredAction === action.id && (
                      <motion.div
                        initial={{ opacity: 0, y: -3 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -3 }}
                        className="absolute top-full mt-1 left-1/2 -translate-x-1/2 px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white text-[10px] rounded whitespace-nowrap z-50"
                      >
                        {action.label}
                        {action.id === 'ai-chat' && (
                          <span className="ml-1 text-gray-600 dark:text-gray-400 text-[9px]">⌘K</span>
                        )}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.button>
              )
            })}
          </div>

          {/* Right side - Status indicators and voice */}
          <div className="flex items-center gap-2">
            {/* Recent Alert */}
            <AnimatePresence>
              {recentAlert && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="flex items-center gap-1 px-2 py-0.5 bg-red-500/10 border border-red-500/20 rounded-md"
                >
                  <AlertTriangle className="w-3 h-3 text-red-400" />
                  <span className="text-[10px] text-red-400">{recentAlert}</span>
                  <button type="button"
                    onClick={() => setRecentAlert(null)}
                    className="ml-1 text-red-400 hover:text-red-300"
                  >
                    <XCircle className="w-2.5 h-2.5" />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* AI Status */}
            <div className="hidden sm:flex items-center gap-1 px-2 py-0.5 bg-purple-500/10 rounded-md">
              <Activity className="w-3 h-3 text-purple-400" />
              <span className="text-[10px] text-purple-400">
                AI: {metrics?.ai?.accuracy?.predictiveCompliance ? metrics.ai.accuracy.predictiveCompliance.toFixed(0) : '..'}%
              </span>
            </div>

            {/* Voice Toggle */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setVoiceEnabled(!voiceEnabled)}
              className={`
                p-1 rounded-md transition-all
                ${voiceEnabled 
                  ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30' 
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-300 dark:border-gray-700 hover:bg-gray-200 dark:hover:bg-gray-700'
                }
              `}
            >
              <Mic className="w-3 h-3" />
            </motion.button>

            {/* Command Palette hint */}
            <div className="hidden xl:flex items-center gap-1 px-2 py-0.5 bg-gray-100 dark:bg-gray-800 rounded-md">
              <Command className="w-2.5 h-2.5 text-gray-600 dark:text-gray-400" />
              <span className="text-[10px] text-gray-600 dark:text-gray-400">Press</span>
              <kbd className="text-[10px] bg-white dark:bg-gray-900 px-1 py-0.5 rounded text-gray-700 dark:text-gray-300">⌘K</kbd>
              <span className="text-[10px] text-gray-600 dark:text-gray-400">AI</span>
            </div>
          </div>
        </motion.div>
      </motion.div>

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
          padding-top: ${isCollapsed ? '4.25rem' : '7rem'} !important;
          transition: padding-top 0.3s ease;
        }
        
        /* Smooth scroll behavior */
        html {
          scroll-behavior: smooth;
        }
        
        /* Keyboard shortcut hint */
        @media (min-width: 1024px) {
          .quick-actions-hint::after {
            content: 'Ctrl+B to toggle';
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 10px;
            color: var(--text-muted);
            opacity: 0;
            transition: opacity 0.2s;
          }
          
          .quick-actions-hint:hover::after {
            opacity: 1;
          }
        }
      `}</style>
    </>
  )
}