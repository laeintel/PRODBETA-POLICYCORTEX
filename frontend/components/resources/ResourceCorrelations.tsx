'use client'

import React from 'react'
import { motion } from 'framer-motion'
import {
  GitBranch,
  Link,
  AlertTriangle,
  TrendingUp,
  Shield,
  DollarSign,
  Activity,
  ArrowRight,
  Zap
} from 'lucide-react'

interface Correlation {
  id: string
  source_resource: string
  target_resource: string
  correlation_type: string
  strength: number
  impact: string
  insights: Array<{
    title: string
    description: string
    evidence: string[]
    confidence: number
  }>
  recommended_actions: Array<{
    action: string
    priority: number
    expected_outcome: string
    effort_level: string
  }>
}

interface ResourceCorrelationsProps {
  correlations: Correlation[]
  resources: any[]
}

const correlationIcons = {
  CostDependency: DollarSign,
  SecurityRelationship: Shield,
  PerformanceImpact: Activity,
  ComplianceLink: AlertTriangle,
  NetworkConnectivity: GitBranch,
  DataFlow: Link,
  PolicyInheritance: Shield,
  AvailabilityDependency: TrendingUp
}

const correlationColors = {
  CostDependency: 'from-green-500 to-green-600',
  SecurityRelationship: 'from-red-500 to-red-600',
  PerformanceImpact: 'from-blue-500 to-blue-600',
  ComplianceLink: 'from-yellow-500 to-yellow-600',
  NetworkConnectivity: 'from-purple-500 to-purple-600',
  DataFlow: 'from-indigo-500 to-indigo-600',
  PolicyInheritance: 'from-pink-500 to-pink-600',
  AvailabilityDependency: 'from-orange-500 to-orange-600'
}

const effortColors = {
  Trivial: 'bg-green-100 text-green-800',
  Low: 'bg-blue-100 text-blue-800',
  Medium: 'bg-yellow-100 text-yellow-800',
  High: 'bg-orange-100 text-orange-800',
  Complex: 'bg-red-100 text-red-800'
}

export function ResourceCorrelations({ correlations, resources }: ResourceCorrelationsProps) {
  const getResourceName = (resourceId: string) => {
    const resource = resources.find(r => r.id === resourceId)
    return resource?.display_name || resourceId
  }

  const getStrengthColor = (strength: number) => {
    if (strength > 0.9) return 'text-red-600'
    if (strength > 0.7) return 'text-yellow-600'
    return 'text-green-600'
  }

  const getImpactBadgeColor = (impact: string) => {
    switch (impact) {
      case 'Critical': return 'bg-red-100 text-red-800'
      case 'High': return 'bg-orange-100 text-orange-800'
      case 'Medium': return 'bg-yellow-100 text-yellow-800'
      case 'Low': return 'bg-blue-100 text-blue-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  // Group correlations by type
  const groupedCorrelations = correlations.reduce((acc, corr) => {
    if (!acc[corr.correlation_type]) {
      acc[corr.correlation_type] = []
    }
    acc[corr.correlation_type].push(corr)
    return acc
  }, {} as Record<string, Correlation[]>)

  return (
    <div className="space-y-6">
      {/* Summary Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-2xl p-6 text-white"
      >
        <div className="flex items-center mb-4">
          <GitBranch className="w-6 h-6 mr-2" />
          <h2 className="text-xl font-bold">Cross-Domain Correlations</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p className="text-purple-100 text-sm">Total Correlations</p>
            <p className="text-3xl font-bold">{correlations.length}</p>
          </div>
          <div>
            <p className="text-purple-100 text-sm">Strong Correlations</p>
            <p className="text-3xl font-bold">
              {correlations.filter(c => c.strength > 0.8).length}
            </p>
          </div>
          <div>
            <p className="text-purple-100 text-sm">Action Items</p>
            <p className="text-3xl font-bold">
              {correlations.reduce((sum, c) => sum + c.recommended_actions.length, 0)}
            </p>
          </div>
        </div>
      </motion.div>

      {/* Correlation Graph Visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-4">Correlation Network</h3>
        <div className="relative h-64 bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden">
          <svg className="absolute inset-0 w-full h-full">
            {/* Draw correlation lines */}
            {correlations.slice(0, 10).map((corr, idx) => {
              const x1 = 50 + (idx % 5) * 150
              const y1 = 50 + Math.floor(idx / 5) * 100
              const x2 = 100 + (idx % 5) * 150
              const y2 = 100 + Math.floor(idx / 5) * 100
              
              return (
                <g key={corr.id}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={corr.strength > 0.8 ? '#ef4444' : corr.strength > 0.6 ? '#f59e0b' : '#10b981'}
                    strokeWidth={corr.strength * 3}
                    opacity={0.6}
                  />
                  <circle cx={x1} cy={y1} r="8" fill="#6366f1" />
                  <circle cx={x2} cy={y2} r="8" fill="#8b5cf6" />
                </g>
              )
            })}
          </svg>
          <div className="absolute bottom-4 left-4 flex items-center space-x-4 text-xs">
            <div className="flex items-center">
              <div className="w-3 h-0.5 bg-green-500 mr-1" />
              <span className="text-gray-600 dark:text-gray-400">Weak</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-1 bg-yellow-500 mr-1" />
              <span className="text-gray-600 dark:text-gray-400">Medium</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-1.5 bg-red-500 mr-1" />
              <span className="text-gray-600 dark:text-gray-400">Strong</span>
            </div>
          </div>
        </div>
      </div>

      {/* Grouped Correlations */}
      {Object.entries(groupedCorrelations).map(([type, typeCorrelations]) => {
        const Icon = correlationIcons[type as keyof typeof correlationIcons] || GitBranch
        const gradient = correlationColors[type as keyof typeof correlationColors] || 'from-gray-500 to-gray-600'
        
        return (
          <div key={type} className="space-y-4">
            <div className="flex items-center">
              <div className={`p-2 rounded-lg bg-gradient-to-r ${gradient} mr-3`}>
                <Icon className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-lg font-semibold">
                {type.replace(/([A-Z])/g, ' $1').trim()}
              </h3>
              <span className="ml-2 px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-xs">
                {typeCorrelations.length} correlations
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {typeCorrelations.slice(0, 4).map((correlation) => (
                <motion.div
                  key={correlation.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 hover:shadow-md transition-all"
                >
                  {/* Correlation Header */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center text-sm">
                        <span className="font-medium truncate">
                          {getResourceName(correlation.source_resource)}
                        </span>
                        <ArrowRight className="w-4 h-4 mx-2 text-gray-400" />
                        <span className="font-medium truncate">
                          {getResourceName(correlation.target_resource)}
                        </span>
                      </div>
                      <div className="flex items-center mt-2 space-x-2">
                        <span className={`text-sm font-medium ${getStrengthColor(correlation.strength)}`}>
                          {(correlation.strength * 100).toFixed(0)}% correlation
                        </span>
                        <span className={`px-2 py-0.5 rounded text-xs ${getImpactBadgeColor(correlation.impact)}`}>
                          {correlation.impact}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Primary Insight */}
                  {correlation.insights[0] && (
                    <div className="mb-3 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                      <p className="text-sm font-medium mb-1">{correlation.insights[0].title}</p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {correlation.insights[0].description}
                      </p>
                      <div className="flex items-center mt-2">
                        <div className="flex-1">
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                            <div
                              className="bg-blue-500 h-1.5 rounded-full"
                              style={{ width: `${correlation.insights[0].confidence * 100}%` }}
                            />
                          </div>
                        </div>
                        <span className="ml-2 text-xs text-gray-500">
                          {(correlation.insights[0].confidence * 100).toFixed(0)}% confident
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Recommended Actions */}
                  {correlation.recommended_actions.length > 0 && (
                    <div className="space-y-2">
                      {correlation.recommended_actions.slice(0, 2).map((action, idx) => (
                        <div key={idx} className="flex items-start p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                          <Zap className="w-4 h-4 text-blue-500 mr-2 mt-0.5 flex-shrink-0" />
                          <div className="flex-1">
                            <p className="text-xs font-medium text-blue-700 dark:text-blue-400">
                              {action.action}
                            </p>
                            <p className="text-xs text-blue-600 dark:text-blue-300 mt-1">
                              {action.expected_outcome}
                            </p>
                            <div className="flex items-center mt-2 space-x-2">
                              <span className={`px-2 py-0.5 rounded text-xs ${effortColors[action.effort_level as keyof typeof effortColors]}`}>
                                {action.effort_level} effort
                              </span>
                              <span className="text-xs text-gray-500">
                                Priority {action.priority}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </motion.div>
              ))}
            </div>

            {typeCorrelations.length > 4 && (
              <button className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300">
                View {typeCorrelations.length - 4} more {type.toLowerCase()} correlations →
              </button>
            )}
          </div>
        )
      })}

      {/* No Correlations Message */}
      {correlations.length === 0 && (
        <div className="text-center py-12">
          <GitBranch className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 dark:text-gray-400">
            No correlations detected yet. Correlations will appear as resources interact.
          </p>
        </div>
      )}
    </div>
  )
}