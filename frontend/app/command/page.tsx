/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2026 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Command,
  Search,
  Zap,
  Shield,
  DollarSign,
  Server,
  GitBranch,
  Settings,
  FileText,
  AlertTriangle,
  CheckCircle,
  Play,
  History,
  Star,
  Sparkles,
  Terminal,
  Keyboard,
  ArrowRight,
  Clock,
  Filter,
  Copy,
  ExternalLink
} from 'lucide-react'

interface CommandItem {
  id: string
  title: string
  description: string
  category: string
  icon: React.ElementType
  action: string
  shortcut?: string
  isPremium?: boolean
  aiPowered?: boolean
  usage: number
  lastUsed?: string
}

interface RecentCommand {
  id: string
  command: string
  timestamp: string
  status: 'success' | 'error' | 'pending'
  result?: string
}

export default function CommandCenterPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedCommand, setSelectedCommand] = useState<CommandItem | null>(null)
  const [recentCommands, setRecentCommands] = useState<RecentCommand[]>([])
  const [isExecuting, setIsExecuting] = useState(false)
  const [favorites, setFavorites] = useState<Set<string>>(new Set())
  const searchInputRef = useRef<HTMLInputElement>(null)

  const commands: CommandItem[] = [
    // Governance Commands
    {
      id: 'scan-compliance',
      title: 'Scan Compliance',
      description: 'Run comprehensive compliance scan across all resources',
      category: 'governance',
      icon: Shield,
      action: 'policycortex.compliance.scan --all --detailed',
      shortcut: '⌘⇧C',
      usage: 342,
      aiPowered: true
    },
    {
      id: 'fix-violations',
      title: 'Auto-Fix Violations',
      description: 'Automatically remediate policy violations',
      category: 'governance',
      icon: CheckCircle,
      action: 'policycortex.remediate --auto --safe-mode',
      shortcut: '⌘⇧F',
      usage: 189,
      aiPowered: true,
      isPremium: true
    },
    {
      id: 'generate-report',
      title: 'Generate Compliance Report',
      description: 'Create detailed compliance report for all subscriptions',
      category: 'governance',
      icon: FileText,
      action: 'policycortex.report.generate --type=compliance --format=pdf',
      usage: 267
    },

    // Cost Commands
    {
      id: 'optimize-costs',
      title: 'Optimize Costs',
      description: 'Find and apply cost optimization recommendations',
      category: 'cost',
      icon: DollarSign,
      action: 'policycortex.cost.optimize --apply-safe',
      shortcut: '⌘⇧O',
      usage: 523,
      aiPowered: true
    },
    {
      id: 'forecast-spend',
      title: 'Forecast Spending',
      description: 'Generate AI-powered spending forecast',
      category: 'cost',
      icon: Sparkles,
      action: 'policycortex.cost.forecast --period=90d --ml-model=advanced',
      usage: 156,
      aiPowered: true
    },
    {
      id: 'find-idle',
      title: 'Find Idle Resources',
      description: 'Identify and list all idle or underutilized resources',
      category: 'cost',
      icon: Search,
      action: 'policycortex.resources.find-idle --threshold=10%',
      usage: 412
    },

    // Resource Commands
    {
      id: 'scale-resources',
      title: 'Auto-Scale Resources',
      description: 'Configure auto-scaling for selected resource groups',
      category: 'resources',
      icon: Server,
      action: 'policycortex.scale.configure --mode=auto --optimize',
      shortcut: '⌘⇧S',
      usage: 234
    },
    {
      id: 'backup-all',
      title: 'Backup All Resources',
      description: 'Create backups for all critical resources',
      category: 'resources',
      icon: Shield,
      action: 'policycortex.backup.create --all --priority=critical',
      usage: 89
    },
    {
      id: 'tag-resources',
      title: 'Apply Smart Tags',
      description: 'Automatically tag resources based on AI recommendations',
      category: 'resources',
      icon: GitBranch,
      action: 'policycortex.tags.apply --smart --ml-suggestions',
      usage: 167,
      aiPowered: true
    },

    // Security Commands
    {
      id: 'security-scan',
      title: 'Security Scan',
      description: 'Run comprehensive security assessment',
      category: 'security',
      icon: Shield,
      action: 'policycortex.security.scan --deep --vulnerabilities',
      shortcut: '⌘⇧V',
      usage: 445
    },
    {
      id: 'rotate-keys',
      title: 'Rotate All Keys',
      description: 'Rotate all service principal and access keys',
      category: 'security',
      icon: Settings,
      action: 'policycortex.keys.rotate --all --notify',
      usage: 78
    },
    {
      id: 'threat-analysis',
      title: 'Threat Analysis',
      description: 'Analyze potential security threats using AI',
      category: 'security',
      icon: AlertTriangle,
      action: 'policycortex.threats.analyze --ai-powered --realtime',
      usage: 203,
      aiPowered: true,
      isPremium: true
    },

    // Automation Commands
    {
      id: 'deploy-pipeline',
      title: 'Deploy Pipeline',
      description: 'Deploy CI/CD pipeline with best practices',
      category: 'automation',
      icon: Zap,
      action: 'policycortex.pipeline.deploy --template=best-practice',
      usage: 134
    },
    {
      id: 'create-workflow',
      title: 'Create Workflow',
      description: 'Generate automated workflow from template',
      category: 'automation',
      icon: GitBranch,
      action: 'policycortex.workflow.create --interactive',
      usage: 298
    },
    {
      id: 'schedule-tasks',
      title: 'Schedule Tasks',
      description: 'Schedule recurring governance tasks',
      category: 'automation',
      icon: Clock,
      action: 'policycortex.schedule.create --recurring',
      usage: 87
    }
  ]

  useEffect(() => {
    // Focus search input on mount and when pressing '/'
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === '/' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        searchInputRef.current?.focus()
      }
    }
    
    window.addEventListener('keydown', handleKeyPress)
    searchInputRef.current?.focus()

    // Load recent commands
    setRecentCommands([
      {
        id: '1',
        command: 'policycortex.compliance.scan --all',
        timestamp: '2 minutes ago',
        status: 'success',
        result: 'Scan completed: 98% compliant'
      },
      {
        id: '2',
        command: 'policycortex.cost.optimize --dry-run',
        timestamp: '15 minutes ago',
        status: 'success',
        result: 'Found $12,450 in potential savings'
      },
      {
        id: '3',
        command: 'policycortex.backup.create --rg=production',
        timestamp: '1 hour ago',
        status: 'success',
        result: 'Backup completed successfully'
      }
    ])

    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [])

  const filteredCommands = commands.filter(cmd => {
    const matchesSearch = cmd.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          cmd.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          cmd.action.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesCategory = selectedCategory === 'all' || cmd.category === selectedCategory
    return matchesSearch && matchesCategory
  })

  const executeCommand = async (command: CommandItem) => {
    setIsExecuting(true)
    setSelectedCommand(command)
    
    // Simulate command execution
    setTimeout(() => {
      const newCommand: RecentCommand = {
        id: Date.now().toString(),
        command: command.action,
        timestamp: 'Just now',
        status: 'success',
        result: `${command.title} executed successfully`
      }
      setRecentCommands(prev => [newCommand, ...prev.slice(0, 9)])
      setIsExecuting(false)
    }, 2000)
  }

  const toggleFavorite = (commandId: string) => {
    setFavorites(prev => {
      const updated = new Set(prev)
      if (updated.has(commandId)) {
        updated.delete(commandId)
      } else {
        updated.add(commandId)
      }
      return updated
    })
  }

  const categories = [
    { id: 'all', label: 'All Commands', count: commands.length },
    { id: 'governance', label: 'Governance', count: commands.filter(c => c.category === 'governance').length },
    { id: 'cost', label: 'Cost', count: commands.filter(c => c.category === 'cost').length },
    { id: 'resources', label: 'Resources', count: commands.filter(c => c.category === 'resources').length },
    { id: 'security', label: 'Security', count: commands.filter(c => c.category === 'security').length },
    { id: 'automation', label: 'Automation', count: commands.filter(c => c.category === 'automation').length }
  ]

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-4 mb-2">
          <div className="p-3 bg-gray-800 rounded-xl">
            <Command className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Command Center</h1>
            <p className="text-gray-400 mt-1">Execute powerful commands with a single click</p>
          </div>
        </div>
      </motion.div>

      {/* Search Bar */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mb-6"
      >
        <div className="relative">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            ref={searchInputRef}
            type="text"
            placeholder="Search commands... (Press '/' to focus)"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-12 pr-12 py-4 bg-white/10 backdrop-blur-xl border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:border-purple-500 transition-all text-lg"
          />
          <kbd className="absolute right-4 top-1/2 transform -translate-y-1/2 px-2 py-1 bg-white/10 text-gray-400 rounded text-sm">
            /
          </kbd>
        </div>
      </motion.div>

      {/* Category Filters */}
      <div className="flex flex-wrap gap-3 mb-8">
        {categories.map((category) => (
          <button
            key={category.id}
            onClick={() => setSelectedCategory(category.id)}
            className={`px-4 py-2 rounded-lg transition-all ${
              selectedCategory === category.id
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            {category.label}
            <span className="ml-2 px-2 py-0.5 bg-black/20 rounded text-xs">
              {category.count}
            </span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Commands Grid */}
        <div className="lg:col-span-2">
          <h2 className="text-xl font-semibold text-white mb-4">Available Commands</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {filteredCommands.map((command, index) => {
              const Icon = command.icon
              return (
                <motion.div
                  key={command.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-4 hover:bg-white/15 transition-all cursor-pointer group"
                  onClick={() => executeCommand(command)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-purple-500/20 rounded-lg group-hover:bg-purple-500/30 transition-colors">
                        <Icon className="w-5 h-5 text-purple-400" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white flex items-center gap-2">
                          {command.title}
                          {command.aiPowered && <Sparkles className="w-4 h-4 text-purple-400" />}
                          {command.isPremium && <Star className="w-4 h-4 text-yellow-400" />}
                        </h3>
                        <p className="text-xs text-gray-400 mt-1">{command.description}</p>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        toggleFavorite(command.id)
                      }}
                      className="opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <Star className={`w-4 h-4 ${
                        favorites.has(command.id) ? 'text-yellow-400 fill-yellow-400' : 'text-gray-400'
                      }`} />
                    </button>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <code className="text-xs text-purple-400 bg-black/30 px-2 py-1 rounded">
                      {command.action.split(' ')[0]}
                    </code>
                    <div className="flex items-center gap-3">
                      {command.shortcut && (
                        <kbd className="text-xs px-2 py-1 bg-white/10 text-gray-400 rounded">
                          {command.shortcut}
                        </kbd>
                      )}
                      <span className="text-xs text-gray-500">{command.usage} uses</span>
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>

        {/* Recent Commands & Terminal */}
        <div className="space-y-6">
          {/* Terminal Preview */}
          {selectedCommand && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-black/50 backdrop-blur-xl rounded-xl border border-white/20 p-4"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-purple-400" />
                  <h3 className="text-sm font-medium text-white">Terminal</h3>
                </div>
                <button className="p-1 hover:bg-white/10 rounded transition-colors">
                  <Copy className="w-4 h-4 text-gray-400" />
                </button>
              </div>
              <div className="font-mono text-xs">
                <p className="text-green-400 mb-2">$ {selectedCommand.action}</p>
                {isExecuting ? (
                  <div className="flex items-center gap-2 text-yellow-400">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                    Executing...
                  </div>
                ) : (
                  <p className="text-gray-400">Press Enter to execute or ESC to cancel</p>
                )}
              </div>
            </motion.div>
          )}

          {/* Recent Commands */}
          <div className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <History className="w-5 h-5 text-purple-400" />
                Recent Commands
              </h3>
              <button className="text-xs text-purple-400 hover:text-purple-300 transition-colors">
                Clear History
              </button>
            </div>
            
            <div className="space-y-3">
              {recentCommands.map((cmd) => (
                <div key={cmd.id} className="bg-black/20 rounded-lg p-3">
                  <div className="flex items-start justify-between mb-2">
                    <code className="text-xs text-purple-400 break-all">{cmd.command}</code>
                    {cmd.status === 'success' && <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />}
                    {cmd.status === 'error' && <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />}
                  </div>
                  {cmd.result && (
                    <p className="text-xs text-gray-400 mb-1">{cmd.result}</p>
                  )}
                  <p className="text-xs text-gray-500">{cmd.timestamp}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Keyboard Shortcuts */}
          <div className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-4">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Keyboard className="w-5 h-5 text-purple-400" />
              Keyboard Shortcuts
            </h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Search</span>
                <kbd className="px-2 py-1 bg-white/10 text-gray-300 rounded text-xs">/</kbd>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Execute</span>
                <kbd className="px-2 py-1 bg-white/10 text-gray-300 rounded text-xs">Enter</kbd>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Cancel</span>
                <kbd className="px-2 py-1 bg-white/10 text-gray-300 rounded text-xs">ESC</kbd>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Navigate</span>
                <div className="flex gap-1">
                  <kbd className="px-2 py-1 bg-white/10 text-gray-300 rounded text-xs">↑</kbd>
                  <kbd className="px-2 py-1 bg-white/10 text-gray-300 rounded text-xs">↓</kbd>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}