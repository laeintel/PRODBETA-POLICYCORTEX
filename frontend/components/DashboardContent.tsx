'use client'

import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { motion } from 'framer-motion'
import { useGovernanceData } from '../lib/api'
import { 
  Shield, 
  AlertCircle, 
  CheckCircle, 
  XCircle,
  TrendingUp,
  TrendingDown,
  Activity,
  Server,
  DollarSign,
  Users,
  Settings,
  LogOut,
  Home,
  FileText,
  BarChart3,
  Bell,
  Network,
  Brain,
  Sparkles,
  Zap,
  Lock,
  Database,
  Cloud,
  Cpu,
  HardDrive,
  GitBranch,
  Key,
  CreditCard,
  AlertTriangle,
  CheckSquare,
  Target,
  Gauge
} from 'lucide-react'

export default function DashboardContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [selectedModule, setSelectedModule] = useState('overview')
  const { metrics, recommendations, loading, error } = useGovernanceData()

  // Handle URL query parameters to pre-select module
  useEffect(() => {
    const moduleFromUrl = searchParams.get('module')
    if (moduleFromUrl && ['policies', 'rbac', 'costs', 'network', 'resources', 'ai'].includes(moduleFromUrl)) {
      setSelectedModule(moduleFromUrl)
    }
  }, [searchParams])

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-400 mb-4">Connection Error</h1>
          <p className="text-white mb-4">{error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    )
  }

  const modules = [
    {
      id: 'overview',
      title: 'Unified Dashboard',
      icon: Home,
      description: 'Complete governance overview',
      color: 'purple'
    },
    {
      id: 'chat',
      title: 'AI Assistant',
      icon: Brain,
      description: 'Conversational intelligence',
      color: 'indigo'
    },
    {
      id: 'policies',
      title: 'Policy & Compliance',
      icon: Shield,
      description: 'Automated policy management',
      color: 'blue',
      metrics: [
        { label: 'Active Policies', value: metrics?.policies.active || 0, total: metrics?.policies.total || 0 },
        { label: 'Automated', value: metrics ? `${Math.round((metrics.policies.automated / metrics.policies.active) * 100)}%` : '0%' },
        { label: 'Violations', value: metrics?.policies.violations || 0, trend: 'down' },
        { label: 'Compliance Rate', value: `${metrics?.policies.compliance_rate || 0}%`, trend: 'up' }
      ]
    },
    {
      id: 'rbac',
      title: 'RBAC & Permissions',
      icon: Users,
      description: 'Intelligent access control',
      color: 'green',
      metrics: [
        { label: 'Total Users', value: metrics?.rbac.users || 0 },
        { label: 'Active Roles', value: metrics?.rbac.roles || 0 },
        { label: 'Risk Score', value: `${metrics?.rbac.risk_score || 0}%`, trend: 'down' },
        { label: 'Violations', value: metrics?.rbac.violations || 0, trend: 'down' }
      ]
    },
    {
      id: 'costs',
      title: 'Cost Management',
      icon: DollarSign,
      description: 'FinOps automation',
      color: 'yellow',
      metrics: [
        { label: 'Current Spend', value: `$${metrics?.costs.current_spend.toLocaleString() || '0'}` },
        { label: 'Predicted', value: `$${metrics?.costs.predicted_spend.toLocaleString() || '0'}` },
        { label: 'Savings', value: `$${metrics?.costs.savings_identified.toLocaleString() || '0'}`, trend: 'up' },
        { label: 'Optimized', value: `${metrics?.costs.optimization_rate || 0}%`, trend: 'up' }
      ]
    },
    {
      id: 'network',
      title: 'Network Security',
      icon: Network,
      description: 'Zero-trust governance',
      color: 'red',
      metrics: [
        { label: 'Endpoints', value: metrics?.network.endpoints || 0 },
        { label: 'Active Threats', value: metrics?.network.active_threats || 0, trend: 'down' },
        { label: 'Blocked', value: metrics?.network.blocked_attempts || 0 },
        { label: 'Latency', value: `${metrics?.network.latency_ms || 0}ms` }
      ]
    },
    {
      id: 'resources',
      title: 'Resource Management',
      icon: Server,
      description: 'Lifecycle automation',
      color: 'indigo',
      metrics: [
        { label: 'Total Resources', value: metrics?.resources.total || 0 },
        { label: 'Optimized', value: metrics?.resources.optimized || 0 },
        { label: 'Idle', value: metrics?.resources.idle || 0, trend: 'down' },
        { label: 'Over-provisioned', value: metrics?.resources.overprovisioned || 0, trend: 'down' }
      ]
    },
    {
      id: 'ai',
      title: 'AI Domain Expert',
      icon: Brain,
      description: 'Custom-trained intelligence',
      color: 'pink',
      metrics: [
        { label: 'Accuracy', value: `${metrics?.ai.accuracy || 0}%`, trend: 'up' },
        { label: 'Predictions', value: metrics?.ai.predictions_made.toLocaleString() || '0' },
        { label: 'Automations', value: metrics?.ai.automations_executed.toLocaleString() || '0' },
        { label: 'Learning', value: `${metrics?.ai.learning_progress.toFixed(1) || 0}%`, trend: 'up' }
      ]
    }
  ]

  const proactiveActions = recommendations.map(rec => ({
    type: rec.recommendation_type,
    severity: rec.severity,
    title: rec.title,
    description: rec.description,
    action: rec.automation_available ? 'Auto-Remediate' : 'Manual Review',
    icon: rec.recommendation_type === 'cost_optimization' ? DollarSign :
          rec.recommendation_type === 'security' ? Lock :
          rec.recommendation_type === 'rbac' ? Users : Server,
    savings: rec.potential_savings,
    risk_reduction: rec.risk_reduction,
    confidence: rec.confidence
  }))

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <motion.div className="text-center">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            className="w-16 h-16 border-4 border-purple-400 border-t-transparent rounded-full mx-auto mb-4"
          />
          <p className="text-white">AI is analyzing your environment...</p>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex">
      {/* Sidebar */}
      <motion.div
        initial={{ x: -300 }}
        animate={{ x: 0 }}
        className="w-72 bg-black/30 backdrop-blur-md border-r border-white/10"
      >
        <div className="p-6">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">PolicyCortex</h2>
              <p className="text-xs text-gray-400">AI Governance Suite</p>
            </div>
          </div>
          
          <div className="mb-6">
            <div className="bg-purple-600/20 rounded-lg p-3 border border-purple-500/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-purple-300">AI Learning Progress</span>
                <Brain className={`w-4 h-4 ${(metrics?.ai.learning_progress || 0) >= 100 ? 'text-green-400' : 'text-purple-400 animate-pulse'}`} />
              </div>
              <div className="w-full bg-purple-900/50 rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${metrics?.ai.learning_progress || 0}%` }}
                  className={`h-2 rounded-full ${
                    (metrics?.ai.learning_progress || 0) >= 100 
                      ? 'bg-gradient-to-r from-green-400 to-emerald-400' 
                      : (metrics?.ai.learning_progress || 0) >= 95
                      ? 'bg-gradient-to-r from-yellow-400 to-orange-400'
                      : 'bg-gradient-to-r from-purple-400 to-pink-400'
                  }`}
                />
              </div>
              <p className="text-xs text-purple-300 mt-1">
                {(metrics?.ai.learning_progress || 0) >= 100 
                  ? '🎉 AI Training Complete - Expert Level Achieved'
                  : (metrics?.ai.learning_progress || 0) >= 95
                  ? `🚀 ${metrics?.ai.learning_progress.toFixed(1)}% - Finalizing Domain Expertise`
                  : `🧠 ${metrics?.ai.learning_progress.toFixed(1)}% Environment Learned`
                }
              </p>
            </div>
          </div>

          <nav className="space-y-1">
            {modules.map((module, index) => (
              <motion.button
                key={module.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                onClick={() => module.id === 'chat' ? router.push('/chat') : setSelectedModule(module.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                  selectedModule === module.id
                    ? 'bg-purple-600/30 text-white border-l-4 border-purple-400'
                    : 'text-gray-300 hover:text-white hover:bg-white/10'
                }`}
              >
                <module.icon className="w-5 h-5" />
                <div className="text-left">
                  <p className="font-medium">{module.title}</p>
                  <p className="text-xs opacity-70">{module.description}</p>
                </div>
              </motion.button>
            ))}
          </nav>
          
          <button
            onClick={() => router.push('/')}
            className="w-full flex items-center gap-3 px-4 py-3 text-red-400 hover:text-red-300 hover:bg-red-900/20 rounded-lg transition-all mt-8"
          >
            <LogOut className="w-5 h-5" />
            <span>Logout</span>
          </button>
        </div>
      </motion.div>

      {/* Main Content */}
      <div className="flex-1 p-8 overflow-auto">
        {selectedModule === 'overview' ? (
          <>
            {/* Header */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-8"
            >
              <h1 className="text-4xl font-bold text-white mb-2">Unified Cloud Governance</h1>
              <p className="text-gray-300">Complete visibility and control across all Azure resources</p>
            </motion.div>

            {/* Integrated Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4 mb-8">
              {modules.slice(1).map((module, index) => {
                const Icon = module.icon
                return (
                  <motion.div
                    key={module.id}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    whileHover={{ scale: 1.02 }}
                    onClick={() => setSelectedModule(module.id)}
                    className="p-4 rounded-xl bg-white/10 backdrop-blur-md border border-white/20 cursor-pointer"
                  >
                    <Icon className={`w-8 h-8 text-${module.color}-400 mb-2`} />
                    <p className="text-sm text-gray-300">{module.title}</p>
                    <p className="text-2xl font-bold text-white">
                      {module.metrics?.[0].value}
                    </p>
                  </motion.div>
                )
              })}
            </div>

            {/* Proactive Actions */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mb-8"
            >
              <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                <Zap className="w-6 h-6 text-yellow-400" />
                Proactive AI Recommendations
              </h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {proactiveActions.map((action, index) => {
                  const Icon = action.icon
                  return (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.4 + index * 0.05 }}
                      className={`p-4 rounded-xl bg-white/10 backdrop-blur-md border ${
                        action.severity === 'critical' ? 'border-red-500/50' :
                        action.severity === 'high' ? 'border-orange-500/50' :
                        action.severity === 'medium' ? 'border-yellow-500/50' :
                        'border-white/20'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-3">
                          <Icon className={`w-6 h-6 ${
                            action.severity === 'critical' ? 'text-red-400' :
                            action.severity === 'high' ? 'text-orange-400' :
                            action.severity === 'medium' ? 'text-yellow-400' :
                            'text-blue-400'
                          }`} />
                          <div>
                            <h3 className="text-white font-semibold">{action.title}</h3>
                            <p className="text-sm text-gray-300">{action.description}</p>
                          </div>
                        </div>
                      </div>
                      <button className="mt-3 px-4 py-2 bg-purple-600/30 text-purple-300 rounded-lg text-sm hover:bg-purple-600/50 transition-colors">
                        {action.action}
                      </button>
                    </motion.div>
                  )
                })}
              </div>
            </motion.div>

            {/* Real-time Activity */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-white">Real-time Governance Activity</h2>
                <Activity className="w-6 h-6 text-purple-400 animate-pulse" />
              </div>
              <div className="grid grid-cols-5 gap-4">
                <div className="text-center">
                  <p className="text-3xl font-bold text-purple-400">{metrics?.policies.automated || 0}</p>
                  <p className="text-xs text-gray-400">Policies Automated</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-green-400">{metrics?.rbac.users || 0}</p>
                  <p className="text-xs text-gray-400">Users Managed</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-yellow-400">${((metrics?.costs.savings_identified || 0) / 1000).toFixed(1)}k</p>
                  <p className="text-xs text-gray-400">Costs Saved</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-red-400">{metrics?.network.blocked_attempts || 0}</p>
                  <p className="text-xs text-gray-400">Threats Blocked</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-indigo-400">{metrics?.resources.optimized || 0}</p>
                  <p className="text-xs text-gray-400">Resources Optimized</p>
                </div>
              </div>
            </motion.div>
          </>
        ) : (
          /* Module-specific view */
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            {(() => {
              const module = modules.find(m => m.id === selectedModule)
              const Icon = module?.icon || Shield
              return (
                <>
                  <div className="mb-8">
                    <div className="flex items-center gap-3 mb-2">
                      <Icon className={`w-10 h-10 text-${module?.color}-400`} />
                      <h1 className="text-4xl font-bold text-white">{module?.title}</h1>
                    </div>
                    <p className="text-gray-300">{module?.description}</p>
                  </div>
                  
                  {module?.metrics && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                      {module.metrics.map((metric, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.1 }}
                          className="p-6 rounded-xl bg-white/10 backdrop-blur-md border border-white/20"
                        >
                          <p className="text-sm text-gray-300 mb-2">{metric.label}</p>
                          <p className="text-3xl font-bold text-white">{metric.value}</p>
                          {metric.trend && (
                            <div className={`flex items-center gap-1 mt-2 text-sm ${
                              metric.trend === 'up' ? 'text-green-400' : 'text-red-400'
                            }`}>
                              {metric.trend === 'up' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                              <span>vs last month</span>
                            </div>
                          )}
                        </motion.div>
                      ))}
                    </div>
                  )}
                  
                  <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6">
                    <h2 className="text-xl font-semibold text-white mb-4">Module Details</h2>
                    <p className="text-gray-300">
                      This module provides comprehensive {module?.title?.toLowerCase()} capabilities with AI-driven automation
                      and proactive recommendations based on your specific environment.
                    </p>
                  </div>
                </>
              )
            })()}
          </motion.div>
        )}
      </div>
    </div>
  )
}