'use client'

import React from 'react'
import { motion } from 'framer-motion'
import {
  CheckCircle,
  AlertCircle,
  XCircle,
  TrendingUp,
  TrendingDown,
  Minus,
  DollarSign,
  Shield,
  Activity,
  ChevronRight,
  Zap
} from 'lucide-react'

interface ResourceCardProps {
  resource: any
  onClick: () => void
}

const healthColors = {
  Healthy: 'bg-green-100 text-green-800 border-green-200',
  Degraded: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  Unhealthy: 'bg-red-100 text-red-800 border-red-200',
  Unknown: 'bg-gray-100 text-gray-800 border-gray-200'
}

const healthIcons = {
  Healthy: CheckCircle,
  Degraded: AlertCircle,
  Unhealthy: XCircle,
  Unknown: AlertCircle
}

export function ResourceCard({ resource, onClick }: ResourceCardProps) {
  const HealthIcon = healthIcons[resource.health.status as keyof typeof healthIcons]
  const hasCriticalIssues = resource.health.issues.some((i: any) => i.severity === 'Critical')
  const hasOptimization = resource.cost_data?.optimization_potential > 0

  const getTrendIcon = () => {
    if (!resource.cost_data?.cost_trend) return Minus
    if (resource.cost_data.cost_trend.type === 'Increasing') return TrendingUp
    if (resource.cost_data.cost_trend.type === 'Decreasing') return TrendingDown
    return Minus
  }

  const TrendIcon = getTrendIcon()

  return (
    <motion.div
      whileHover={{ scale: 1.02, y: -4 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className="bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-xl transition-all cursor-pointer overflow-hidden group"
    >
      {/* Status Bar */}
      <div className={`h-1 bg-gradient-to-r ${
        resource.health.status === 'Healthy' ? 'from-green-400 to-green-500' :
        resource.health.status === 'Degraded' ? 'from-yellow-400 to-yellow-500' :
        resource.health.status === 'Unhealthy' ? 'from-red-400 to-red-500' :
        'from-gray-400 to-gray-500'
      }`} />

      <div className="p-6">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <h3 className="font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
              {resource.display_name}
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {resource.resource_type.split('/').pop()}
            </p>
          </div>
          <div className={`px-2 py-1 rounded-full text-xs font-medium ${healthColors[resource.health.status as keyof typeof healthColors]}`}>
            <HealthIcon className="w-4 h-4 inline mr-1" />
            {resource.health.status}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-4 mb-4">
          {/* Cost */}
          <div className="text-center">
            <div className="flex items-center justify-center mb-1">
              <DollarSign className="w-4 h-4 text-gray-400 mr-1" />
              <TrendIcon className={`w-3 h-3 ${
                resource.cost_data?.cost_trend?.type === 'Increasing' ? 'text-red-500' :
                resource.cost_data?.cost_trend?.type === 'Decreasing' ? 'text-green-500' :
                'text-gray-400'
              }`} />
            </div>
            <p className="text-sm font-semibold text-gray-900 dark:text-white">
              ${resource.cost_data?.daily_cost?.toFixed(0) || '0'}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">Daily</p>
          </div>

          {/* Performance */}
          <div className="text-center">
            <Activity className="w-4 h-4 text-gray-400 mx-auto mb-1" />
            <p className="text-sm font-semibold text-gray-900 dark:text-white">
              {resource.status.performance_score.toFixed(0)}%
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">Perf</p>
          </div>

          {/* Compliance */}
          <div className="text-center">
            <Shield className="w-4 h-4 text-gray-400 mx-auto mb-1" />
            <p className="text-sm font-semibold text-gray-900 dark:text-white">
              {resource.compliance_status.compliance_score.toFixed(0)}%
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">Comply</p>
          </div>
        </div>

        {/* Alerts & Insights */}
        <div className="space-y-2">
          {hasCriticalIssues && (
            <div className="flex items-center text-xs bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 rounded-lg px-3 py-2">
              <XCircle className="w-3 h-3 mr-2 flex-shrink-0" />
              <span className="truncate">Critical issues detected</span>
            </div>
          )}
          
          {hasOptimization && (
            <div className="flex items-center text-xs bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 rounded-lg px-3 py-2">
              <Zap className="w-3 h-3 mr-2 flex-shrink-0" />
              <span className="truncate">
                Save ${resource.cost_data.optimization_potential.toFixed(0)}/mo
              </span>
            </div>
          )}

          {resource.insights.length > 0 && (
            <div className="flex items-center text-xs bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 rounded-lg px-3 py-2">
              <AlertCircle className="w-3 h-3 mr-2 flex-shrink-0" />
              <span className="truncate">{resource.insights[0].title}</span>
            </div>
          )}
        </div>

        {/* Actions Hint */}
        <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700 flex items-center justify-between">
          <div className="flex -space-x-2">
            {resource.quick_actions.slice(0, 3).map((action: any, idx: number) => (
              <div
                key={idx}
                className="w-8 h-8 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center border-2 border-white dark:border-gray-800"
                title={action.label}
              >
                <span className="text-xs">
                  {action.icon === 'play' ? '▶' :
                   action.icon === 'square' ? '◼' :
                   action.icon === 'refresh-cw' ? '↻' :
                   action.icon === 'eye' ? '👁' :
                   '⚙'}
                </span>
              </div>
            ))}
            {resource.quick_actions.length > 3 && (
              <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center border-2 border-white dark:border-gray-800">
                <span className="text-xs text-blue-600 dark:text-blue-400">
                  +{resource.quick_actions.length - 3}
                </span>
              </div>
            )}
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-blue-500 transition-colors" />
        </div>
      </div>
    </motion.div>
  )
}