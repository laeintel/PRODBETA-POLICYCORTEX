'use client'

import React from 'react'
import { TrendingUp, AlertTriangle, Clock, ArrowRight, Shield } from 'lucide-react'

interface ForecastCardProps {
  id: string
  violationType: string
  resourceName: string
  eta: string
  confidence: number
  severity: 'critical' | 'high' | 'medium' | 'low'
  impact: string
  onCreateFix?: () => void
  onViewDetails?: () => void
}

export default function ForecastCard({
  violationType,
  resourceName,
  eta,
  confidence,
  severity,
  impact,
  onCreateFix,
  onViewDetails
}: ForecastCardProps) {
  const getSeverityColor = () => {
    switch (severity) {
      case 'critical':
        return 'border-red-500 bg-red-50 dark:bg-red-900/20'
      case 'high':
        return 'border-orange-500 bg-orange-50 dark:bg-orange-900/20'
      case 'medium':
        return 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
      case 'low':
        return 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
      default:
        return 'border-gray-300 bg-gray-50 dark:bg-gray-800'
    }
  }

  const getSeverityBadge = () => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300'
      case 'high':
        return 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-300'
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300'
      case 'low':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
    }
  }

  const getConfidenceColor = () => {
    if (confidence >= 90) return 'text-green-600'
    if (confidence >= 70) return 'text-yellow-600'
    return 'text-orange-600'
  }

  return (
    <div className={`rounded-xl border-2 p-5 transition-all hover:shadow-lg ${getSeverityColor()}`}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-gray-700 dark:text-gray-300" />
          <span className={`text-xs px-2 py-1 rounded-full font-semibold ${getSeverityBadge()}`}>
            {severity.toUpperCase()}
          </span>
        </div>
        <div className={`text-sm font-bold ${getConfidenceColor()}`}>
          {confidence}% confidence
        </div>
      </div>

      <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
        {violationType}
      </h3>
      
      <div className="space-y-2 mb-4">
        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <Shield className="h-4 w-4" />
          <span className="truncate">{resourceName}</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <Clock className="h-4 w-4" />
          <span>ETA: {eta}</span>
        </div>
      </div>

      <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg p-3 mb-4">
        <p className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">Expected Impact</p>
        <p className="text-sm text-gray-900 dark:text-gray-100">{impact}</p>
      </div>

      <div className="flex gap-2">
        <button
          onClick={onCreateFix}
          className="flex-1 bg-gradient-to-r from-blue-600 to-blue-700 text-white px-4 py-2 rounded-lg font-medium text-sm hover:from-blue-700 hover:to-blue-800 transition-all flex items-center justify-center gap-2 shadow-md"
        >
          <TrendingUp className="h-4 w-4" />
          Create Fix PR
        </button>
        <button
          onClick={onViewDetails}
          className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-white/50 dark:hover:bg-gray-800/50 transition-colors flex items-center gap-2"
        >
          Details
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}