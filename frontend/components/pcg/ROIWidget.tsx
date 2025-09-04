'use client'

import React from 'react'
import { DollarSign, TrendingUp, TrendingDown, Calculator, Target, PiggyBank } from 'lucide-react'

interface ROIMetrics {
  totalSavings: number
  roi: number
  preventionRate: number
  mttp: string // Mean Time to Prevent
  savingsBreakdown: {
    category: string
    amount: number
    percentage: number
  }[]
  projections: {
    period: string
    amount: number
  }[]
}

interface ROIWidgetProps {
  metrics: ROIMetrics
  showProjections?: boolean
  onViewDetails?: () => void
}

export default function ROIWidget({
  metrics,
  showProjections = false,
  onViewDetails
}: ROIWidgetProps) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount)
  }

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`
  }

  return (
    <div className="bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-xl border border-emerald-200 dark:border-emerald-800 overflow-hidden">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-emerald-100 dark:bg-emerald-900/30 rounded-xl">
              <DollarSign className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">Governance P&L</h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">Real-time financial impact</p>
            </div>
          </div>
          <button
            onClick={onViewDetails}
            className="text-sm font-medium text-emerald-600 hover:text-emerald-700 dark:text-emerald-400"
          >
            View Details →
          </button>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Total Savings</span>
              <TrendingUp className="h-4 w-4 text-green-500" />
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {formatCurrency(metrics.totalSavings)}
            </p>
            <p className="text-xs text-green-600 dark:text-green-400 mt-1">
              +23% vs last month
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">ROI</span>
              <Target className="h-4 w-4 text-blue-500" />
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {formatPercentage(metrics.roi)}
            </p>
            <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
              Above target
            </p>
          </div>
        </div>

        {/* Prevention Metrics */}
        <div className="bg-white/70 dark:bg-gray-800/70 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">Prevention Metrics</h3>
            <Calculator className="h-4 w-4 text-gray-500" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Prevention Rate</p>
              <div className="flex items-baseline gap-2">
                <span className="text-xl font-bold text-gray-900 dark:text-gray-100">
                  {formatPercentage(metrics.preventionRate)}
                </span>
                <span className="text-xs text-green-600 dark:text-green-400">↑ 5%</span>
              </div>
            </div>
            <div>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Mean Time to Prevent</p>
              <div className="flex items-baseline gap-2">
                <span className="text-xl font-bold text-gray-900 dark:text-gray-100">
                  {metrics.mttp}
                </span>
                <span className="text-xs text-green-600 dark:text-green-400">↓ 2h</span>
              </div>
            </div>
          </div>
        </div>

        {/* Savings Breakdown */}
        <div className="space-y-2 mb-6">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 text-sm">Savings by Category</h3>
          {metrics.savingsBreakdown.map((item, index) => (
            <div key={index} className="flex items-center justify-between">
              <div className="flex items-center gap-3 flex-1">
                <span className="text-sm text-gray-700 dark:text-gray-300">{item.category}</span>
                <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-emerald-500 to-teal-500"
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>
              </div>
              <span className="text-sm font-semibold text-gray-900 dark:text-gray-100 ml-3">
                {formatCurrency(item.amount)}
              </span>
            </div>
          ))}
        </div>

        {/* Projections */}
        {showProjections && metrics.projections.length > 0 && (
          <div className="bg-gradient-to-r from-emerald-100/50 to-teal-100/50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <PiggyBank className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
              <h3 className="font-semibold text-gray-900 dark:text-gray-100 text-sm">Projections</h3>
            </div>
            <div className="grid grid-cols-3 gap-3">
              {metrics.projections.map((proj, index) => (
                <div key={index} className="text-center">
                  <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">{proj.period}</p>
                  <p className="font-bold text-gray-900 dark:text-gray-100">
                    {formatCurrency(proj.amount)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}