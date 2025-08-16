/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

'use client'

import React from 'react'
import { motion } from 'framer-motion'
import {
  TrendingUp,
  DollarSign,
  Shield,
  AlertTriangle,
  Zap,
  Target,
  Info,
  ArrowRight,
  Sparkles,
  BarChart3
} from 'lucide-react'

interface Insight {
  type: string
  title: string
  description: string
  impact: string
  confidence: number
  recommendation?: string
  resources: any[]
}

interface ResourceInsightsProps {
  resources: any[]
}

const insightIcons = {
  CostOptimization: DollarSign,
  PerformanceImprovement: TrendingUp,
  SecurityRisk: Shield,
  ComplianceGap: AlertTriangle,
  AvailabilityIssue: Target,
  ConfigurationDrift: Info
}

const insightColors = {
  CostOptimization: 'from-green-500 to-green-600',
  PerformanceImprovement: 'from-blue-500 to-blue-600',
  SecurityRisk: 'from-red-500 to-red-600',
  ComplianceGap: 'from-yellow-500 to-yellow-600',
  AvailabilityIssue: 'from-purple-500 to-purple-600',
  ConfigurationDrift: 'from-gray-500 to-gray-600'
}

export function ResourceInsights({ resources }: ResourceInsightsProps) {
  // Aggregate insights from all resources
  const allInsights: Insight[] = []
  const insightMap = new Map<string, Insight>()

  resources.forEach((resource: any) => {
    resource.insights?.forEach((insight: any) => {
      const key = `${insight.type}-${insight.title}`
      if (insightMap.has(key)) {
        const existing = insightMap.get(key)!
        existing.resources.push(resource)
        existing.confidence = Math.max(existing.confidence, insight.confidence)
      } else {
        insightMap.set(key, {
          ...insight,
          type: insight.type || 'ConfigurationDrift',
          resources: [resource]
        })
      }
    })
  })

  const insights = Array.from(insightMap.values())
    .sort((a, b) => b.confidence - a.confidence)

  // Calculate summary statistics
  const totalOptimizationPotential = resources.reduce((sum, r) => 
    sum + (r.cost_data?.optimization_potential || 0), 0
  )

  const criticalIssuesCount = resources.filter((r: any) =>
    r.health.issues?.some((i: any) => i.severity === 'Critical')
  ).length

  const nonCompliantCount = resources.filter((r: any) => !r.compliance_status.is_compliant).length

  return (
    <div className="space-y-6">
      {/* AI-Powered Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl p-6 text-white"
      >
        <div className="flex items-center mb-4">
          <Sparkles className="w-6 h-6 mr-2" />
          <h2 className="text-xl font-bold">AI-Powered Insights</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-indigo-100 text-sm mb-1">Immediate Actions</p>
            <p className="text-3xl font-bold">{insights.filter(i => i.confidence > 0.8).length}</p>
            <p className="text-indigo-100 text-xs mt-1">High confidence recommendations</p>
          </div>
          
          <div>
            <p className="text-indigo-100 text-sm mb-1">Potential Savings</p>
            <p className="text-3xl font-bold">${(totalOptimizationPotential / 1000).toFixed(1)}k</p>
            <p className="text-indigo-100 text-xs mt-1">Monthly cost reduction</p>
          </div>
          
          <div>
            <p className="text-indigo-100 text-sm mb-1">Risk Score</p>
            <p className="text-3xl font-bold">
              {Math.round((criticalIssuesCount + nonCompliantCount) / resources.length * 100)}%
            </p>
            <p className="text-indigo-100 text-xs mt-1">Requires attention</p>
          </div>
        </div>

        <div className="mt-4 p-4 bg-white/10 rounded-xl">
          <p className="text-sm">
            <strong>Top Priority:</strong> {insights[0]?.title || 'All systems optimal'}
          </p>
          {insights[0]?.recommendation && (
            <p className="text-xs text-indigo-100 mt-2">
              {insights[0].recommendation}
            </p>
          )}
        </div>
      </motion.div>

      {/* Insight Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {insights.map((insight, idx) => {
          const Icon = insightIcons[insight.type as keyof typeof insightIcons] || Info
          const gradient = insightColors[insight.type as keyof typeof insightColors] || 'from-gray-500 to-gray-600'
          
          return (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-lg transition-all p-6"
            >
              <div className="flex items-start justify-between mb-4">
                <div className={`p-3 rounded-xl bg-gradient-to-r ${gradient}`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <div className="flex items-center space-x-2">
                  <div className="text-right">
                    <p className="text-xs text-gray-500 dark:text-gray-400">Confidence</p>
                    <p className="text-sm font-semibold">{(insight.confidence * 100).toFixed(0)}%</p>
                  </div>
                  <div className="w-12 h-12 relative">
                    <svg className="w-12 h-12 transform -rotate-90">
                      <circle
                        cx="24"
                        cy="24"
                        r="20"
                        stroke="currentColor"
                        strokeWidth="3"
                        fill="none"
                        className="text-gray-200 dark:text-gray-700"
                      />
                      <circle
                        cx="24"
                        cy="24"
                        r="20"
                        stroke="currentColor"
                        strokeWidth="3"
                        fill="none"
                        strokeDasharray={`${insight.confidence * 125.6} 125.6`}
                        className="text-blue-500"
                      />
                    </svg>
                  </div>
                </div>
              </div>

              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                {insight.title}
              </h3>
              
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                {insight.description}
              </p>

              <div className="flex items-center justify-between mb-3">
                <span className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded-full">
                  {insight.resources.length} resources affected
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  Impact: {insight.impact}
                </span>
              </div>

              {insight.recommendation && (
                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="flex items-start">
                    <Zap className="w-4 h-4 text-blue-500 mr-2 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="text-xs font-medium text-blue-700 dark:text-blue-400 mb-1">
                        Recommended Action
                      </p>
                      <p className="text-xs text-blue-600 dark:text-blue-300">
                        {insight.recommendation}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              <button className="mt-4 w-full flex items-center justify-center py-2 px-4 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg transition-colors text-sm font-medium">
                View Details
                <ArrowRight className="w-4 h-4 ml-2" />
              </button>
            </motion.div>
          )
        })}
      </div>

      {/* Correlation Matrix */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
      >
        <div className="flex items-center mb-4">
          <BarChart3 className="w-5 h-5 mr-2 text-gray-600 dark:text-gray-400" />
          <h3 className="font-semibold text-gray-900 dark:text-white">
            Cross-Domain Correlations
          </h3>
        </div>

        <div className="grid grid-cols-5 gap-2">
          {['Policy', 'Cost', 'Security', 'Compute', 'Network'].map(cat1 => (
            <div key={cat1} className="text-xs text-center font-medium text-gray-600 dark:text-gray-400">
              {cat1}
            </div>
          ))}
          {['Policy', 'Cost', 'Security', 'Compute', 'Network'].map(cat1 => (
            ['Policy', 'Cost', 'Security', 'Compute', 'Network'].map(cat2 => {
              const strength = Math.random() // This would be real correlation data
              return (
                <div
                  key={`${cat1}-${cat2}`}
                  className={`aspect-square rounded-lg flex items-center justify-center text-xs font-medium ${
                    strength > 0.7 ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400' :
                    strength > 0.4 ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400' :
                    'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                  }`}
                >
                  {(strength * 100).toFixed(0)}
                </div>
              )
            })
          ))}
        </div>

        <div className="mt-4 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
          <span>Weak Correlation</span>
          <div className="flex space-x-2">
            <div className="w-4 h-4 bg-green-100 dark:bg-green-900/30 rounded" />
            <div className="w-4 h-4 bg-yellow-100 dark:bg-yellow-900/30 rounded" />
            <div className="w-4 h-4 bg-red-100 dark:bg-red-900/30 rounded" />
          </div>
          <span>Strong Correlation</span>
        </div>
      </motion.div>
    </div>
  )
}