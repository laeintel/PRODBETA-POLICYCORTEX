'use client'

import { TrendingUp, AlertTriangle, Clock, GitPullRequest } from 'lucide-react'

interface Prediction {
  id: string
  title?: string
  kind?: string
  confidence?: number
  explanation?: string
  eta?: string
  impact?: string
  recommendation?: string
  fixPrUrl?: string
  riskLevel?: 'HIGH' | 'MEDIUM' | 'LOW'
  category?: string
}

export function PredictionCard({ p }: { p: Prediction }) {
  const getRiskColor = (level: string | undefined) => {
    switch (level) {
      case 'HIGH':
        return 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20'
      case 'MEDIUM':
        return 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20'
      case 'LOW':
        return 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20'
      default:
        return 'text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900/20'
    }
  }

  const handleCreateFixPR = () => {
    if (p.fixPrUrl) {
      window.open(p.fixPrUrl, '_blank')
    } else {
      // Open DevSecOps pipelines as fallback
      window.location.href = '/devsecops/pipelines'
    }
  }

  return (
    <div className="rounded-xl border border-border dark:border-gray-700 bg-card dark:bg-gray-800 p-4 hover:shadow-lg transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="font-semibold text-foreground dark:text-white">
            {p.title || p.kind || 'Prediction'}
          </h3>
          {p.category && (
            <span className="text-xs text-muted-foreground dark:text-gray-400">
              {p.category}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {p.confidence !== undefined && (
            <span className="text-xs font-medium text-muted-foreground dark:text-gray-400">
              {Math.round(p.confidence * 100)}% confidence
            </span>
          )}
          {p.riskLevel && (
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getRiskColor(p.riskLevel)}`}>
              {p.riskLevel}
            </span>
          )}
        </div>
      </div>
      
      <p className="text-sm text-muted-foreground dark:text-gray-300 mb-3">
        {p.explanation || 'Analysis pending...'}
      </p>
      
      {p.impact && (
        <div className="flex items-center gap-2 mb-3 text-sm">
          <AlertTriangle className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
          <span className="text-muted-foreground dark:text-gray-300">Impact: {p.impact}</span>
        </div>
      )}
      
      {p.recommendation && (
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg mb-3">
          <p className="text-sm text-blue-700 dark:text-blue-300 font-medium">
            Recommendation
          </p>
          <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
            {p.recommendation}
          </p>
        </div>
      )}
      
      <div className="flex items-center justify-between pt-3 border-t border-border dark:border-gray-700">
        <div className="flex items-center gap-3 text-xs text-muted-foreground dark:text-gray-400">
          {p.eta && (
            <div className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              <span>ETA: {p.eta}</span>
            </div>
          )}
        </div>
        <button
          onClick={handleCreateFixPR}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-primary dark:bg-blue-600 text-primary-foreground dark:text-white rounded-md hover:bg-primary/90 dark:hover:bg-blue-700 transition-colors text-xs font-medium"
        >
          <GitPullRequest className="w-3 h-3" />
          Create Fix PR
        </button>
      </div>
    </div>
  )
}