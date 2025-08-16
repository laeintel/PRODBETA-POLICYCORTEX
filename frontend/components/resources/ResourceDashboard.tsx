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

import React, { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search,
  Filter,
  ChevronDown,
  Activity,
  DollarSign,
  Shield,
  Server,
  Globe,
  AlertCircle,
  CheckCircle,
  XCircle,
  TrendingUp,
  TrendingDown,
  Zap,
  Play,
  Square,
  RefreshCw,
  Eye,
  Settings,
  Maximize,
  Download,
  BarChart3,
  Users,
  Lock,
  Database,
  GitBranch,
  Heart
} from 'lucide-react'
import { useResourceStore } from '@/stores/resourceStore'
import { ResourceCard } from './ResourceCard'
import { ResourceInsights } from './ResourceInsights'
import { QuickActions } from './QuickActions'
import { ResourceFilters } from './ResourceFilters'
import { ResourceCorrelations } from './ResourceCorrelations'

interface Resource {
  id: string
  name: string
  display_name: string
  resource_type: string
  category: string
  location?: string
  tags: Record<string, string>
  status: {
    state: string
    availability: number
    performance_score: number
  }
  health: {
    status: 'Healthy' | 'Degraded' | 'Unhealthy' | 'Unknown'
    issues: Array<{
      severity: string
      title: string
      description: string
    }>
    recommendations: string[]
  }
  cost_data?: {
    daily_cost: number
    monthly_cost: number
    cost_trend: { type: string; value?: number }
    optimization_potential: number
    currency: string
  }
  compliance_status: {
    is_compliant: boolean
    compliance_score: number
    violations: Array<{
      severity: string
      description: string
    }>
  }
  quick_actions: Array<{
    id: string
    label: string
    icon: string
    action_type: string
  }>
  insights: Array<{
    title: string
    description: string
    impact: string
    confidence: number
  }>
}

const categoryIcons = {
  Policy: Shield,
  CostManagement: DollarSign,
  SecurityControls: Lock,
  ComputeStorage: Server,
  NetworksFirewalls: Globe
}

const categoryColors = {
  Policy: 'from-purple-500 to-purple-600',
  CostManagement: 'from-green-500 to-green-600',
  SecurityControls: 'from-red-500 to-red-600',
  ComputeStorage: 'from-blue-500 to-blue-600',
  NetworksFirewalls: 'from-orange-500 to-orange-600'
}

const healthColors = {
  Healthy: 'text-green-500',
  Degraded: 'text-yellow-500',
  Unhealthy: 'text-red-500',
  Unknown: 'text-gray-400'
}

const healthIcons = {
  Healthy: CheckCircle,
  Degraded: AlertCircle,
  Unhealthy: XCircle,
  Unknown: AlertCircle
}

export function ResourceDashboard() {
  const { resources, loading, error, fetchResources, summary } = useResourceStore()
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedResource, setSelectedResource] = useState<Resource | null>(null)
  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'insights'>('grid')
  const [showFilters, setShowFilters] = useState(false)
  const [filters, setFilters] = useState({
    health: [] as string[],
    compliance: 'all',
    costRange: { min: 0, max: Infinity }
  })

  useEffect(() => {
    fetchResources()
    const interval = setInterval(fetchResources, 30000) // Auto-refresh every 30s
    return () => clearInterval(interval)
  }, [])

  const filteredResources = useMemo(() => {
    let filtered = resources

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(r => 
        r.display_name.toLowerCase().includes(query) ||
        r.resource_type.toLowerCase().includes(query) ||
        r.name.toLowerCase().includes(query)
      )
    }

    // Category filter
    if (selectedCategory) {
      filtered = filtered.filter(r => r.category === selectedCategory)
    }

    // Health filter
    if (filters.health.length > 0) {
      filtered = filtered.filter(r => filters.health.includes(r.health.status))
    }

    // Compliance filter
    if (filters.compliance === 'compliant') {
      filtered = filtered.filter(r => r.compliance_status.is_compliant)
    } else if (filters.compliance === 'non-compliant') {
      filtered = filtered.filter(r => !r.compliance_status.is_compliant)
    }

    // Cost filter
    if (filters.costRange.max < Infinity) {
      filtered = filtered.filter(r => {
        if (!r.cost_data) return true
        return r.cost_data.daily_cost >= filters.costRange.min && 
               r.cost_data.daily_cost <= filters.costRange.max
      })
    }

    return filtered
  }, [resources, searchQuery, selectedCategory, filters])

  const criticalIssues = resources.filter(r => 
    r.health.issues.some(i => i.severity === 'Critical')
  )

  const costOptimizationOpportunities = resources.filter(r =>
    r.cost_data && r.cost_data.optimization_potential > 0
  )

  const totalDailyCost = resources.reduce((sum, r) => 
    sum + (r.cost_data?.daily_cost || 0), 0
  )

  const totalOptimizationPotential = resources.reduce((sum, r) =>
    sum + (r.cost_data?.optimization_potential || 0), 0
  )

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      {/* Header with Smart Summary */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Azure Resources
              </h1>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                {resources.length} resources across 5 categories • Real-time monitoring
              </p>
            </div>
            <div className="flex items-center space-x-3">
              <button
                onClick={() => fetchResources()}
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                <RefreshCw className="w-5 h-5" />
              </button>
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                <Filter className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Smart Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
            <motion.div
              whileHover={{ scale: 1.02 }}
              className="bg-gradient-to-r from-green-500 to-green-600 rounded-xl p-4 text-white"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-green-100 text-sm">Daily Cost</p>
                  <p className="text-2xl font-bold">${totalDailyCost.toFixed(2)}</p>
                  <p className="text-green-100 text-xs mt-1">
                    Save ${totalOptimizationPotential.toFixed(0)}/mo
                  </p>
                </div>
                <DollarSign className="w-8 h-8 text-green-200" />
              </div>
            </motion.div>

            <motion.div
              whileHover={{ scale: 1.02 }}
              className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl p-4 text-white"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-blue-100 text-sm">Resource Health</p>
                  <p className="text-2xl font-bold">
                    {((summary?.by_health?.Healthy || 0) / resources.length * 100).toFixed(0)}%
                  </p>
                  <p className="text-blue-100 text-xs mt-1">
                    {summary?.critical_issues || 0} critical issues
                  </p>
                </div>
                <Heart className="w-8 h-8 text-blue-200" />
              </div>
            </motion.div>

            <motion.div
              whileHover={{ scale: 1.02 }}
              className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-xl p-4 text-white"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-purple-100 text-sm">Compliance</p>
                  <p className="text-2xl font-bold">
                    {summary?.compliance_score?.toFixed(0) || 0}%
                  </p>
                  <p className="text-purple-100 text-xs mt-1">
                    {resources.filter(r => !r.compliance_status.is_compliant).length} violations
                  </p>
                </div>
                <Shield className="w-8 h-8 text-purple-200" />
              </div>
            </motion.div>

            <motion.div
              whileHover={{ scale: 1.02 }}
              className="bg-gradient-to-r from-orange-500 to-orange-600 rounded-xl p-4 text-white"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-orange-100 text-sm">Optimizations</p>
                  <p className="text-2xl font-bold">
                    {summary?.optimization_opportunities || 0}
                  </p>
                  <p className="text-orange-100 text-xs mt-1">
                    Quick wins available
                  </p>
                </div>
                <Zap className="w-8 h-8 text-orange-200" />
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Search and Category Filter */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col md:flex-row gap-4">
          {/* Search Bar */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search resources by name, type, or tag..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
            />
          </div>

          {/* View Mode Toggle */}
          <div className="flex rounded-xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-1">
            {['grid', 'list', 'insights'].map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode as any)}
                className={`px-4 py-2 rounded-lg capitalize transition-all ${
                  viewMode === mode 
                    ? 'bg-blue-500 text-white' 
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>

        {/* Category Pills */}
        <div className="flex flex-wrap gap-2 mt-4">
          <button
            onClick={() => setSelectedCategory(null)}
            className={`px-4 py-2 rounded-full transition-all ${
              !selectedCategory 
                ? 'bg-blue-500 text-white' 
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
          >
            All Categories
          </button>
          {Object.entries(categoryIcons).map(([category, Icon]) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-full flex items-center gap-2 transition-all ${
                selectedCategory === category
                  ? 'bg-gradient-to-r text-white ' + categoryColors[category as keyof typeof categoryColors]
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <Icon className="w-4 h-4" />
              {category.replace(/([A-Z])/g, ' $1').trim()}
              <span className="ml-1 px-2 py-0.5 bg-white/20 rounded-full text-xs">
                {resources.filter(r => r.category === category).length}
              </span>
            </button>
          ))}
        </div>

        {/* Advanced Filters (Collapsible) */}
        <AnimatePresence>
          {showFilters && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <ResourceFilters filters={filters} setFilters={setFilters} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Main Content Area */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">
        {viewMode === 'grid' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence>
              {filteredResources.map((resource) => (
                <ResourceCard
                  key={resource.id}
                  resource={resource}
                  onClick={() => setSelectedResource(resource)}
                />
              ))}
            </AnimatePresence>
          </div>
        )}

        {viewMode === 'list' && (
          <div className="space-y-4">
            {filteredResources.map((resource) => (
              <motion.div
                key={resource.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm hover:shadow-md transition-all cursor-pointer"
                onClick={() => setSelectedResource(resource)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className={`p-3 rounded-xl bg-gradient-to-r ${categoryColors[resource.category]}`}>
                      {React.createElement(categoryIcons[resource.category], {
                        className: 'w-6 h-6 text-white'
                      })}
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-white">
                        {resource.display_name}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {resource.resource_type} • {resource.location}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-6">
                    <div className="text-right">
                      <p className="text-sm text-gray-500 dark:text-gray-400">Daily Cost</p>
                      <p className="font-semibold">
                        ${resource.cost_data?.daily_cost.toFixed(2) || '0.00'}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-500 dark:text-gray-400">Health</p>
                      <div className={`flex items-center ${healthColors[resource.health.status]}`}>
                        {React.createElement(healthIcons[resource.health.status], {
                          className: 'w-5 h-5'
                        })}
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-500 dark:text-gray-400">Compliance</p>
                      <p className="font-semibold">
                        {resource.compliance_status.compliance_score.toFixed(0)}%
                      </p>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}

        {viewMode === 'insights' && (
          <ResourceInsights resources={filteredResources} />
        )}
      </div>

      {/* Resource Detail Modal */}
      <AnimatePresence>
        {selectedResource && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedResource(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white dark:bg-gray-800 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <QuickActions 
                resource={selectedResource} 
                onClose={() => setSelectedResource(null)} 
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}