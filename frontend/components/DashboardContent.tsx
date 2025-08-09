'use client'

import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { motion } from 'framer-motion'
import { useGovernanceData } from '../lib/api'
import {
  useAzurePolicies,
  useAzureResources,
  useRbacAssignments,
  useCostBreakdown,
  type AzurePolicy,
  type AzureResource,
  type RbacAssignment,
  type CostBreakdown
} from '../lib/azure-api'
import PoliciesDeepView from './PoliciesDeepView'
import AppLayout from './AppLayout'
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
      <AppLayout>
        <div className="min-h-screen flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-red-400 mb-4">Connection Error</h1>
            <p className="text-white mb-4">{error}</p>
            <button 
              onClick={() => {
                if (typeof window !== 'undefined') {
                  window.location.reload()
                }
              }} 
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              Retry Connection
            </button>
          </div>
        </div>
      </AppLayout>
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

  // Transform recommendations to include appropriate icons and actions
  const proactiveActions = recommendations.map(rec => ({
    type: rec.recommendation_type,
    severity: rec.severity,
    title: rec.title,
    description: rec.description,
    action: rec.automation_available ? 'Auto-Remediate' : 'Manual Review',
    icon: rec.recommendation_type === 'cost_optimization' ? DollarSign :
          rec.recommendation_type === 'resource_optimization' ? Server :
          rec.recommendation_type === 'rightsizing' ? Gauge :
          rec.recommendation_type === 'compliance' ? Shield :
          rec.recommendation_type === 'security' ? Lock :
          rec.recommendation_type === 'rbac' ? Users : 
          rec.recommendation_type === 'network' ? Network : Server,
    savings: rec.potential_savings,
    risk_reduction: rec.risk_reduction,
    confidence: rec.confidence,
    id: rec.id
  }))

  if (loading) {
    return (
      <AppLayout>
        <div className="min-h-screen flex items-center justify-center">
          <motion.div className="text-center">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              className="w-16 h-16 border-4 border-purple-400 border-t-transparent rounded-full mx-auto mb-4"
            />
            <p className="text-white">AI is analyzing your environment...</p>
          </motion.div>
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout>
      {/* Main Content */}
      <div className="p-8 overflow-auto">
        {selectedModule === 'overview' ? (
          <>
            {/* Header with AI Learning Progress */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-8"
            >
              <h1 className="text-4xl font-bold text-white mb-2">Unified Cloud Governance</h1>
              <p className="text-gray-300 mb-4">Complete visibility and control across all Azure resources</p>
              
              {/* AI Learning Progress Bar */}
              <div className="bg-purple-600/20 rounded-lg p-3 border border-purple-500/30 max-w-md">
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

                  {/* Rich module content */}
                  {selectedModule === 'policies' && (
                    <div className="space-y-6">
                      <div className="flex items-center justify-between mb-2">
                        <h2 className="text-2xl font-semibold text-white">Policy Compliance</h2>
                        <button
                          onClick={() => router.push('/policies')}
                          className="px-3 py-2 rounded-lg bg-white/10 border border-white/20 text-sm text-white hover:bg-white/20"
                        >
                          Open full policies view
                        </button>
                      </div>
                      <PoliciesDeepView />
                    </div>
                  )}

                  {selectedModule === 'resources' && (
                    <ResourcesModule onOpenAll={() => router.push('/resources')} />
                  )}

                  {selectedModule === 'rbac' && (
                    <RbacModule />)
                  }

                  {selectedModule === 'costs' && (
                    <CostsModule />)
                  }
                </>
              )
            })()}
          </motion.div>
        )}
      </div>
    </AppLayout>
  )
}

// Module subviews (kept in this file for simplicity)

function ResourcesModule({ onOpenAll }: { onOpenAll: () => void }) {
  const { resources, loading } = useAzureResources()
  const top = (resources || []).slice(0, 10)
  return (
    <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20">
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
        <h2 className="text-lg font-semibold text-white">Top Resources</h2>
        <button onClick={onOpenAll} className="text-sm text-purple-300 hover:text-white">View all</button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-white/5">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Name</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Type</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Group</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Compliance</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Monthly</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/10">
            {(loading ? [] : top).map((r) => (
              <tr key={r.id} className="hover:bg-white/5">
                <td className="px-6 py-3 text-white text-sm">{r.name}</td>
                <td className="px-6 py-3 text-gray-300 text-sm">{r.type.split('/')[1]}</td>
                <td className="px-6 py-3 text-gray-300 text-sm">{r.resourceGroup}</td>
                <td className="px-6 py-3 text-gray-300 text-sm">{r.status}</td>
                <td className="px-6 py-3 text-gray-300 text-sm">{r.compliance}</td>
                <td className="px-6 py-3 text-gray-300 text-sm">${(r.monthlyCost || 0).toFixed(2)}</td>
              </tr>
            ))}
            {(!loading && top.length === 0) && (
              <tr><td className="px-6 py-4 text-sm text-gray-400" colSpan={6}>No resources found</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function RbacModule() {
  const { assignments, loading } = useRbacAssignments()
  const rows = (assignments || []).slice(0, 15)
  return (
    <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20">
      <div className="px-6 py-4 border-b border-white/10">
        <h2 className="text-lg font-semibold text-white">Recent RBAC Assignments</h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-white/5">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Principal</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Role</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Scope</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Last Used</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/10">
            {(loading ? [] : rows).map(a => (
              <tr key={a.id} className="hover:bg-white/5">
                <td className="px-6 py-3 text-white text-sm">{a.principalName}</td>
                <td className="px-6 py-3 text-gray-300 text-sm">{a.roleName}</td>
                <td className="px-6 py-3 text-gray-300 text-sm">{a.scope.split('/').slice(0,3).join('/')}</td>
                <td className="px-6 py-3 text-gray-300 text-sm">{a.lastUsed || '-'}</td>
              </tr>
            ))}
            {(!loading && rows.length === 0) && (
              <tr><td className="px-6 py-4 text-sm text-gray-400" colSpan={4}>No assignments found</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function CostsModule() {
  const { breakdown, loading } = useCostBreakdown()
  const rows = (breakdown || []).slice(0, 12)
  return (
    <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20">
      <div className="px-6 py-4 border-b border-white/10">
        <h2 className="text-lg font-semibold text-white">Cost Breakdown</h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-white/5">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Resource Type</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Daily</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Monthly</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Trend</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/10">
            {(loading ? [] : rows).map(r => (
              <tr key={r.resourceId} className="hover:bg-white/5">
                <td className="px-6 py-3 text-white text-sm">{r.resourceName}</td>
                <td className="px-6 py-3 text-gray-300 text-sm">${r.dailyCost.toFixed(2)}</td>
                <td className="px-6 py-3 text-gray-300 text-sm">${r.monthlyCost.toFixed(2)}</td>
                <td className={`px-6 py-3 text-sm ${r.trend > 0 ? 'text-red-300' : r.trend < 0 ? 'text-green-300' : 'text-gray-300'}`}>{r.trend}%</td>
              </tr>
            ))}
            {(!loading && rows.length === 0) && (
              <tr><td className="px-6 py-4 text-sm text-gray-400" colSpan={4}>No cost data found</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}