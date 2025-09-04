'use client'

import React, { useState, useEffect } from 'react'
import { 
  TrendingUp, 
  AlertTriangle, 
  Shield, 
  DollarSign,
  Activity,
  Clock,
  CheckCircle,
  FileText,
  Brain,
  Zap,
  Target,
  ArrowUpRight,
  ArrowDownRight,
  BarChart3,
  ChevronRight,
  Hash,
  TrendingDown
} from 'lucide-react'
import { useRouter } from 'next/navigation'
import ForecastCard from '@/components/pcg/ForecastCard'
import ROIWidget from '@/components/pcg/ROIWidget'
import QuickActions from '@/components/pcg/QuickActions'
import { toast } from '@/hooks/useToast'

// Executive Dashboard - Main Landing Page
export default function ExecutiveDashboard() {
  const router = useRouter()
  const [loading, setLoading] = useState(true)
  const [timeRange, setTimeRange] = useState('7d')

  // Mock predicted violations data
  const predictedViolations = [
    {
      id: '1',
      violationType: 'Encryption Policy Violation',
      resourceName: 'storage-prod-westus2',
      eta: '2 days',
      confidence: 94,
      severity: 'critical' as const,
      impact: '15 storage accounts will lose encryption if current drift continues'
    },
    {
      id: '2',
      violationType: 'Cost Overrun Alert',
      resourceName: 'ml-training-cluster',
      eta: '5 days',
      confidence: 87,
      severity: 'high' as const,
      impact: '$45,000 budget overrun predicted by month end'
    },
    {
      id: '3',
      violationType: 'Access Review Expiration',
      resourceName: 'privileged-admin-group',
      eta: '7 days',
      confidence: 99,
      severity: 'high' as const,
      impact: '12 privileged accounts will have expired access reviews'
    }
  ]

  // ROI Metrics
  const roiMetrics = {
    totalSavings: 1250000,
    roi: 312,
    preventionRate: 94.5,
    mttp: '4.2h',
    savingsBreakdown: [
      { category: 'Prevented Incidents', amount: 650000, percentage: 52 },
      { category: 'Resource Optimization', amount: 350000, percentage: 28 },
      { category: 'Automated Remediation', amount: 250000, percentage: 20 }
    ],
    projections: [
      { period: '30 days', amount: 425000 },
      { period: '60 days', amount: 875000 },
      { period: '90 days', amount: 1350000 }
    ]
  }

  // Evidence Chain Status
  const evidenceStatus = {
    verified: 847,
    pending: 23,
    total: 870,
    integrityScore: 98.5
  }

  // Quick Actions
  const quickActionsList = [
    {
      id: 'generate-report',
      label: 'Generate Board Report',
      description: 'Export executive summary',
      icon: 'send' as const,
      variant: 'primary' as const,
      onClick: () => {
        toast({ title: 'Generating Report', description: 'Your executive board report is being prepared...' })
      }
    },
    {
      id: 'view-predictions',
      label: 'View All Predictions',
      icon: 'alert' as const,
      variant: 'secondary' as const,
      onClick: () => router.push('/prevent')
    },
    {
      id: 'verify-evidence',
      label: 'Verify Evidence Chain',
      icon: 'shield' as const,
      variant: 'secondary' as const,
      onClick: () => router.push('/prove')
    }
  ]

  // Key Metrics
  const keyMetrics = [
    {
      label: 'MTTP',
      value: '4.2h',
      change: '-2.1h',
      trend: 'down',
      subtitle: 'Mean Time to Prevent',
      color: 'green'
    },
    {
      label: 'Prevention Rate',
      value: '94.5%',
      change: '+5.2%',
      trend: 'up',
      subtitle: 'Violations Prevented',
      color: 'blue'
    },
    {
      label: 'ROI',
      value: '312%',
      change: '+47%',
      trend: 'up',
      subtitle: 'Return on Investment',
      color: 'purple'
    },
    {
      label: 'Evidence Chain',
      value: '98.5%',
      change: '+0.3%',
      trend: 'up',
      subtitle: 'Integrity Score',
      color: 'indigo'
    }
  ]

  useEffect(() => {
    setTimeout(() => setLoading(false), 1000)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
                Predictive Cloud Governance
              </h1>
              <p className="text-gray-500 dark:text-gray-400 mt-1">
                AI-powered prevention, evidence, and ROI tracking
              </p>
            </div>
            <div className="flex items-center gap-3">
              <select 
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
              <button
                onClick={() => toast({ title: 'Generating Report', description: 'Executive board report is being prepared...' })}
                className="px-6 py-2 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all text-sm font-medium shadow-md flex items-center gap-2"
              >
                <FileText className="h-4 w-4" />
                Generate Board Report
              </button>
            </div>
          </div>
        </div>

        {/* Key Metrics Row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {keyMetrics.map((metric, index) => (
            <div key={index} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  {metric.subtitle}
                </span>
                {metric.trend === 'up' ? (
                  <ArrowUpRight className="h-4 w-4 text-green-500" />
                ) : (
                  <ArrowDownRight className="h-4 w-4 text-green-500" />
                )}
              </div>
              <div className="flex items-baseline gap-2">
                <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {metric.value}
                </h3>
                <span className={`text-sm font-medium ${
                  metric.trend === 'up' ? 'text-green-600' : 'text-green-600'
                }`}>
                  {metric.change}
                </span>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                {metric.label}
              </p>
            </div>
          ))}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Left Column - Predicted Violations */}
          <div className="lg:col-span-2 space-y-6">
            {/* Top Predicted Violations */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-red-100 dark:bg-red-900/30 rounded-lg">
                    <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                      Top 3 Predicted Violations
                    </h2>
                    <p className="text-sm text-gray-500 dark:text-gray-400">7-day horizon</p>
                  </div>
                </div>
                <button
                  onClick={() => router.push('/prevent')}
                  className="text-sm font-medium text-blue-600 hover:text-blue-700"
                >
                  View All →
                </button>
              </div>

              <div className="space-y-4">
                {predictedViolations.map((violation) => (
                  <ForecastCard
                    key={violation.id}
                    {...violation}
                    onCreateFix={() => {
                      toast({ title: 'Creating Fix PR', description: `Generating automated fix for ${violation.violationType}...` })
                    }}
                    onViewDetails={() => router.push('/prevent')}
                  />
                ))}
              </div>
            </div>

            {/* Evidence Chain Status */}
            <div className="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl border border-purple-200 dark:border-purple-800 p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Hash className="h-6 w-6 text-purple-600 dark:text-purple-400" />
                  <div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                      Evidence Chain Status
                    </h2>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Cryptographic audit trail
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-sm font-medium text-green-600 dark:text-green-400">
                    Chain Verified
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="bg-white/70 dark:bg-gray-800/70 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                    {evidenceStatus.verified}
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">Verified</p>
                </div>
                <div className="bg-white/70 dark:bg-gray-800/70 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                    {evidenceStatus.pending}
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">Pending</p>
                </div>
                <div className="bg-white/70 dark:bg-gray-800/70 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {evidenceStatus.integrityScore}%
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">Integrity</p>
                </div>
              </div>

              <button
                onClick={() => router.push('/prove')}
                className="w-full bg-purple-600 hover:bg-purple-700 text-white py-2 rounded-lg font-medium text-sm transition-colors flex items-center justify-center gap-2"
              >
                <Shield className="h-4 w-4" />
                Verify Full Chain
              </button>
            </div>
          </div>

          {/* Right Column - ROI Summary */}
          <div className="space-y-6">
            <ROIWidget
              metrics={roiMetrics}
              showProjections={true}
              onViewDetails={() => router.push('/payback')}
            />

            <QuickActions
              actions={quickActionsList}
              title="Quick Actions"
              layout="list"
            />

            {/* Platform Metrics */}
            <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl border border-blue-200 dark:border-blue-800 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Brain className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                  AI Platform Status
                </h3>
              </div>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Models Active</span>
                  <span className="font-semibold text-gray-900 dark:text-gray-100">12</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Predictions/Day</span>
                  <span className="font-semibold text-gray-900 dark:text-gray-100">45.2K</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Accuracy</span>
                  <span className="font-semibold text-green-600 dark:text-green-400">99.2%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Response Time</span>
                  <span className="font-semibold text-gray-900 dark:text-gray-100">&lt;50ms</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            onClick={() => router.push('/prevent')}
            className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 hover:shadow-lg transition-all text-left group"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded-xl">
                <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
              </div>
              <ChevronRight className="h-5 w-5 text-gray-400 group-hover:translate-x-1 transition-transform" />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-gray-100 mb-1">PREVENT</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Predictive violations & one-click fixes
            </p>
          </button>

          <button
            onClick={() => router.push('/prove')}
            className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 hover:shadow-lg transition-all text-left group"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-xl">
                <Shield className="h-6 w-6 text-purple-600 dark:text-purple-400" />
              </div>
              <ChevronRight className="h-5 w-5 text-gray-400 group-hover:translate-x-1 transition-transform" />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-gray-100 mb-1">PROVE</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Evidence chain & audit trails
            </p>
          </button>

          <button
            onClick={() => router.push('/payback')}
            className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 hover:shadow-lg transition-all text-left group"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="p-3 bg-emerald-100 dark:bg-emerald-900/30 rounded-xl">
                <DollarSign className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
              </div>
              <ChevronRight className="h-5 w-5 text-gray-400 group-hover:translate-x-1 transition-transform" />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-gray-100 mb-1">PAYBACK</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              ROI tracking & P&L statements
            </p>
          </button>
        </div>
      </div>
    </div>
  )
}