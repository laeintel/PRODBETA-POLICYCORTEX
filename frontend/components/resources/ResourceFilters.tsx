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
  Filter,
  DollarSign,
  Shield,
  Heart,
  MapPin,
  Tag,
  TrendingUp,
  AlertCircle,
  X
} from 'lucide-react'

interface ResourceFiltersProps {
  filters: any
  setFilters: (filters: any) => void
}

export function ResourceFilters({ filters, setFilters }: ResourceFiltersProps) {
  const healthOptions = ['Healthy', 'Degraded', 'Unhealthy', 'Unknown']
  const complianceOptions = [
    { value: 'all', label: 'All Resources' },
    { value: 'compliant', label: 'Compliant Only' },
    { value: 'non-compliant', label: 'Non-Compliant Only' }
  ]
  
  const handleHealthToggle = (status: string) => {
    const current = filters.health || []
    if (current.includes(status)) {
      setFilters({
        ...filters,
        health: current.filter((h: string) => h !== status)
      })
    } else {
      setFilters({
        ...filters,
        health: [...current, status]
      })
    }
  }

  const handleCostRangeChange = (field: 'min' | 'max', value: string) => {
    const numValue = value === '' ? (field === 'min' ? 0 : Infinity) : parseFloat(value)
    setFilters({
      ...filters,
      costRange: {
        ...filters.costRange,
        [field]: numValue
      }
    })
  }

  const clearFilters = () => {
    setFilters({
      health: [],
      compliance: 'all',
      costRange: { min: 0, max: Infinity }
    })
  }

  const activeFilterCount = 
    (filters.health?.length || 0) +
    (filters.compliance !== 'all' ? 1 : 0) +
    (filters.costRange?.max < Infinity ? 1 : 0)

  return (
    <motion.div
      initial={{ height: 0 }}
      animate={{ height: 'auto' }}
      className="py-6 border-t border-gray-200 dark:border-gray-700"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <Filter className="w-5 h-5 text-gray-600 dark:text-gray-400 mr-2" />
          <h3 className="font-semibold text-gray-900 dark:text-white">Advanced Filters</h3>
          {activeFilterCount > 0 && (
            <span className="ml-2 px-2 py-0.5 bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full text-xs">
              {activeFilterCount} active
            </span>
          )}
        </div>
        {activeFilterCount > 0 && (
          <button
            onClick={clearFilters}
            className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 flex items-center"
          >
            <X className="w-4 h-4 mr-1" />
            Clear all
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Health Status Filter */}
        <div>
          <label className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            <Heart className="w-4 h-4 mr-2" />
            Health Status
          </label>
          <div className="space-y-2">
            {healthOptions.map(status => (
              <label
                key={status}
                className="flex items-center cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 p-2 rounded-lg transition-colors"
              >
                <input
                  type="checkbox"
                  checked={filters.health?.includes(status) || false}
                  onChange={() => handleHealthToggle(status)}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                />
                <span className="ml-3 text-sm text-gray-700 dark:text-gray-300">
                  {status}
                </span>
                <span className={`ml-auto w-2 h-2 rounded-full ${
                  status === 'Healthy' ? 'bg-green-500' :
                  status === 'Degraded' ? 'bg-yellow-500' :
                  status === 'Unhealthy' ? 'bg-red-500' :
                  'bg-gray-400'
                }`} />
              </label>
            ))}
          </div>
        </div>

        {/* Compliance Filter */}
        <div>
          <label className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            <Shield className="w-4 h-4 mr-2" />
            Compliance Status
          </label>
          <div className="space-y-2">
            {complianceOptions.map(option => (
              <label
                key={option.value}
                className="flex items-center cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 p-2 rounded-lg transition-colors"
              >
                <input
                  type="radio"
                  name="compliance"
                  value={option.value}
                  checked={filters.compliance === option.value}
                  onChange={(e) => setFilters({ ...filters, compliance: e.target.value })}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                />
                <span className="ml-3 text-sm text-gray-700 dark:text-gray-300">
                  {option.label}
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Cost Range Filter */}
        <div>
          <label className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            <DollarSign className="w-4 h-4 mr-2" />
            Daily Cost Range
          </label>
          <div className="space-y-3">
            <div>
              <label className="text-xs text-gray-500 dark:text-gray-400">Minimum</label>
              <div className="relative mt-1">
                <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500">$</span>
                <input
                  type="number"
                  min="0"
                  value={filters.costRange?.min === 0 ? '' : filters.costRange?.min}
                  onChange={(e) => handleCostRangeChange('min', e.target.value)}
                  placeholder="0"
                  className="w-full pl-8 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
            <div>
              <label className="text-xs text-gray-500 dark:text-gray-400">Maximum</label>
              <div className="relative mt-1">
                <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500">$</span>
                <input
                  type="number"
                  min="0"
                  value={filters.costRange?.max === Infinity ? '' : filters.costRange?.max}
                  onChange={(e) => handleCostRangeChange('max', e.target.value)}
                  placeholder="No limit"
                  className="w-full pl-8 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Filter Presets */}
      <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
        <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Quick Filters</p>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setFilters({
              health: ['Unhealthy', 'Degraded'],
              compliance: 'all',
              costRange: { min: 0, max: Infinity }
            })}
            className="px-3 py-1.5 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded-lg text-sm hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors flex items-center"
          >
            <AlertCircle className="w-3 h-3 mr-1" />
            Issues Only
          </button>
          
          <button
            onClick={() => setFilters({
              health: [],
              compliance: 'non-compliant',
              costRange: { min: 0, max: Infinity }
            })}
            className="px-3 py-1.5 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 rounded-lg text-sm hover:bg-yellow-200 dark:hover:bg-yellow-900/50 transition-colors flex items-center"
          >
            <Shield className="w-3 h-3 mr-1" />
            Non-Compliant
          </button>
          
          <button
            onClick={() => setFilters({
              health: [],
              compliance: 'all',
              costRange: { min: 100, max: Infinity }
            })}
            className="px-3 py-1.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-lg text-sm hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors flex items-center"
          >
            <TrendingUp className="w-3 h-3 mr-1" />
            High Cost
          </button>
          
          <button
            onClick={() => setFilters({
              health: ['Healthy'],
              compliance: 'compliant',
              costRange: { min: 0, max: Infinity }
            })}
            className="px-3 py-1.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-lg text-sm hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors flex items-center"
          >
            <Heart className="w-3 h-3 mr-1" />
            Optimal Only
          </button>
        </div>
      </div>
    </motion.div>
  )
}