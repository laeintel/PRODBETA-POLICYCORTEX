'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X,
  Play,
  Square,
  RefreshCw,
  Eye,
  Settings,
  Maximize,
  Download,
  Trash2,
  Shield,
  DollarSign,
  Activity,
  AlertTriangle,
  CheckCircle,
  Info,
  Zap,
  Clock,
  TrendingUp,
  Users,
  Lock,
  Database,
  ChevronRight
} from 'lucide-react'
import { useResourceStore } from '@/stores/resourceStore'
import toast from 'react-hot-toast'

interface QuickActionsProps {
  resource: any
  onClose: () => void
}

const actionIcons = {
  Start: Play,
  Stop: Square,
  Restart: RefreshCw,
  Scale: Maximize,
  Configure: Settings,
  Optimize: Zap,
  Backup: Database,
  Delete: Trash2,
  ViewDetails: Eye,
  RunDiagnostics: Activity
}

export function QuickActions({ resource, onClose }: QuickActionsProps) {
  const { executeAction } = useResourceStore()
  const [activeTab, setActiveTab] = useState('overview')
  const [executingAction, setExecutingAction] = useState<string | null>(null)
  const [showConfirmation, setShowConfirmation] = useState<string | null>(null)

  const handleAction = async (actionId: string, requiresConfirmation: boolean) => {
    if (requiresConfirmation && showConfirmation !== actionId) {
      setShowConfirmation(actionId)
      return
    }

    setExecutingAction(actionId)
    setShowConfirmation(null)

    try {
      await executeAction(resource.id, actionId, true)
      toast.success(`Action "${actionId}" executed successfully`)
      
      // Close modal for certain actions
      if (['Delete', 'Stop'].includes(actionId)) {
        setTimeout(onClose, 1500)
      }
    } catch (error) {
      toast.error(`Failed to execute action: ${error}`)
    } finally {
      setExecutingAction(null)
    }
  }

  const getHealthIcon = () => {
    switch (resource.health.status) {
      case 'Healthy': return CheckCircle
      case 'Degraded': return AlertTriangle
      case 'Unhealthy': return X
      default: return Info
    }
  }

  const HealthIcon = getHealthIcon()

  return (
    <div className="relative">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              {resource.display_name}
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {resource.resource_type} • {resource.location}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Status Bar */}
        <div className="flex items-center gap-6 mt-4">
          <div className={`flex items-center gap-2 ${
            resource.health.status === 'Healthy' ? 'text-green-600' :
            resource.health.status === 'Degraded' ? 'text-yellow-600' :
            'text-red-600'
          }`}>
            <HealthIcon className="w-5 h-5" />
            <span className="font-medium">{resource.health.status}</span>
          </div>
          
          <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
            <Activity className="w-4 h-4" />
            <span className="text-sm">{resource.status.performance_score.toFixed(0)}% Performance</span>
          </div>
          
          <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
            <Shield className="w-4 h-4" />
            <span className="text-sm">{resource.compliance_status.compliance_score.toFixed(0)}% Compliant</span>
          </div>
          
          {resource.cost_data && (
            <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
              <DollarSign className="w-4 h-4" />
              <span className="text-sm">${resource.cost_data.daily_cost.toFixed(2)}/day</span>
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <div className="flex space-x-8 px-6">
          {['overview', 'actions', 'insights', 'details'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-3 border-b-2 transition-colors capitalize ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        <AnimatePresence mode="wait">
          {activeTab === 'overview' && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-6"
            >
              {/* Quick Actions Grid */}
              <div>
                <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
                <div className="grid grid-cols-3 gap-4">
                  {resource.quick_actions.map((action: any) => {
                    const Icon = actionIcons[action.action_type as keyof typeof actionIcons] || Settings
                    const isExecuting = executingAction === action.id
                    const needsConfirmation = showConfirmation === action.id

                    return (
                      <motion.button
                        key={action.id}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => handleAction(action.id, action.confirmation_required)}
                        disabled={isExecuting}
                        className={`relative p-4 rounded-xl border transition-all ${
                          needsConfirmation
                            ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                            : 'border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-400'
                        } ${isExecuting ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        <div className="flex flex-col items-center">
                          <div className={`p-3 rounded-lg mb-2 ${
                            needsConfirmation
                              ? 'bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-400'
                              : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                          }`}>
                            {isExecuting ? (
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                              >
                                <RefreshCw className="w-5 h-5" />
                              </motion.div>
                            ) : (
                              <Icon className="w-5 h-5" />
                            )}
                          </div>
                          <span className="text-sm font-medium">
                            {needsConfirmation ? 'Confirm?' : action.label}
                          </span>
                          {action.estimated_impact && !needsConfirmation && (
                            <span className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                              {action.estimated_impact}
                            </span>
                          )}
                        </div>
                      </motion.button>
                    )
                  })}
                </div>
              </div>

              {/* Health Issues */}
              {resource.health.issues.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-4">Health Issues</h3>
                  <div className="space-y-3">
                    {resource.health.issues.map((issue: any, idx: number) => (
                      <div
                        key={idx}
                        className={`p-4 rounded-lg border ${
                          issue.severity === 'Critical' ? 'border-red-500 bg-red-50 dark:bg-red-900/20' :
                          issue.severity === 'High' ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/20' :
                          'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <div>
                            <h4 className="font-medium">{issue.title}</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                              {issue.description}
                            </p>
                            {issue.mitigation && (
                              <p className="text-sm text-blue-600 dark:text-blue-400 mt-2">
                                <strong>Mitigation:</strong> {issue.mitigation}
                              </p>
                            )}
                          </div>
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            issue.severity === 'Critical' ? 'bg-red-200 text-red-800' :
                            issue.severity === 'High' ? 'bg-orange-200 text-orange-800' :
                            'bg-yellow-200 text-yellow-800'
                          }`}>
                            {issue.severity}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {resource.health.recommendations.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-4">Recommendations</h3>
                  <div className="space-y-2">
                    {resource.health.recommendations.map((rec: any, idx: number) => (
                      <div key={idx} className="flex items-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                        <Zap className="w-4 h-4 text-blue-500 mr-3 flex-shrink-0" />
                        <span className="text-sm">{rec}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'actions' && (
            <motion.div
              key="actions"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-4"
            >
              <h3 className="text-lg font-semibold mb-4">All Available Actions</h3>
              <div className="grid grid-cols-2 gap-4">
                {resource.quick_actions.map((action: any) => {
                  const Icon = actionIcons[action.action_type as keyof typeof actionIcons] || Settings
                  
                  return (
                    <div
                      key={action.id}
                      className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
                    >
                      <div className="flex items-start space-x-3">
                        <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
                          <Icon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                        </div>
                        <div className="flex-1">
                          <h4 className="font-medium">{action.label}</h4>
                          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                            {action.estimated_impact || 'Execute this action on the resource'}
                          </p>
                          {action.confirmation_required && (
                            <p className="text-xs text-orange-600 dark:text-orange-400 mt-2">
                              Requires confirmation
                            </p>
                          )}
                          <button
                            onClick={() => handleAction(action.id, action.confirmation_required)}
                            className="mt-3 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm"
                          >
                            Execute
                          </button>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </motion.div>
          )}

          {activeTab === 'insights' && (
            <motion.div
              key="insights"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-4"
            >
              <h3 className="text-lg font-semibold mb-4">AI-Powered Insights</h3>
              {resource.insights.length > 0 ? (
                <div className="space-y-4">
                  {resource.insights.map((insight: any, idx: number) => (
                    <div key={idx} className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-medium">{insight.title}</h4>
                        <div className="flex items-center">
                          <span className="text-sm text-gray-500 dark:text-gray-400 mr-2">Confidence</span>
                          <span className="font-medium">{(insight.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {insight.description}
                      </p>
                      <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Impact: {insight.impact}
                      </p>
                      {insight.recommendation && (
                        <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                          <p className="text-sm text-blue-700 dark:text-blue-400">
                            <strong>Recommendation:</strong> {insight.recommendation}
                          </p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 dark:text-gray-400">No insights available for this resource.</p>
              )}
            </motion.div>
          )}

          {activeTab === 'details' && (
            <motion.div
              key="details"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-6"
            >
              <div>
                <h3 className="text-lg font-semibold mb-4">Resource Details</h3>
                <dl className="grid grid-cols-2 gap-4">
                  <div>
                    <dt className="text-sm text-gray-500 dark:text-gray-400">Resource ID</dt>
                    <dd className="font-mono text-sm mt-1">{resource.id}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500 dark:text-gray-400">Resource Type</dt>
                    <dd className="text-sm mt-1">{resource.resource_type}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500 dark:text-gray-400">Category</dt>
                    <dd className="text-sm mt-1">{resource.category}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500 dark:text-gray-400">Location</dt>
                    <dd className="text-sm mt-1">{resource.location || 'Global'}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500 dark:text-gray-400">State</dt>
                    <dd className="text-sm mt-1">{resource.status.state}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500 dark:text-gray-400">Availability</dt>
                    <dd className="text-sm mt-1">{resource.status.availability}%</dd>
                  </div>
                </dl>
              </div>

              {/* Tags */}
              {Object.keys(resource.tags).length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-4">Tags</h3>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(resource.tags).map(([key, value]) => (
                      <span
                        key={key}
                        className="px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-sm"
                      >
                        {key}: {String(value)}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Compliance Violations */}
              {resource.compliance_status.violations.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-4">Compliance Violations</h3>
                  <div className="space-y-3">
                    {resource.compliance_status.violations.map((violation: any, idx: number) => (
                      <div key={idx} className="p-3 border border-gray-200 dark:border-gray-700 rounded-lg">
                        <div className="flex items-start justify-between">
                          <div>
                            <h4 className="font-medium">{violation.policy_name}</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                              {violation.description}
                            </p>
                            {violation.remediation && (
                              <p className="text-sm text-blue-600 dark:text-blue-400 mt-2">
                                {violation.remediation}
                              </p>
                            )}
                          </div>
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            violation.severity === 'Critical' ? 'bg-red-200 text-red-800' :
                            violation.severity === 'High' ? 'bg-orange-200 text-orange-800' :
                            'bg-yellow-200 text-yellow-800'
                          }`}>
                            {violation.severity}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}