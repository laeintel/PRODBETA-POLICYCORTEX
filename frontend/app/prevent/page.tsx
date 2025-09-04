'use client'

import React, { useState, useEffect } from 'react'
import { 
  AlertTriangle, 
  TrendingUp, 
  Clock, 
  Zap,
  GitBranch,
  Activity,
  Filter,
  ChevronRight,
  Target,
  Brain,
  Gauge,
  RefreshCw,
  Play,
  Settings
} from 'lucide-react'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'
import ForecastCard from '@/components/pcg/ForecastCard'
import QuickActions from '@/components/pcg/QuickActions'
import { toast } from '@/hooks/useToast'
import { useRouter } from 'next/navigation'

export default function PreventPage() {
  const router = useRouter()
  const [loading, setLoading] = useState(true)
  const [selectedTimeframe, setSelectedTimeframe] = useState('7d')
  const [selectedSeverity, setSelectedSeverity] = useState('all')
  const [simulationActive, setSimulationActive] = useState(false)

  // Extended predictions data
  const predictions = [
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
    },
    {
      id: '4',
      violationType: 'Network Security Rule Drift',
      resourceName: 'vnet-production',
      eta: '3 days',
      confidence: 82,
      severity: 'medium' as const,
      impact: 'Unauthorized ports may be exposed to internet traffic'
    },
    {
      id: '5',
      violationType: 'Certificate Expiration',
      resourceName: 'app-gateway-ssl',
      eta: '14 days',
      confidence: 100,
      severity: 'medium' as const,
      impact: 'SSL certificate expires causing service disruption'
    },
    {
      id: '6',
      violationType: 'Resource Tag Compliance',
      resourceName: 'multiple-resources',
      eta: '1 day',
      confidence: 76,
      severity: 'low' as const,
      impact: '23 resources missing required cost center tags'
    }
  ]

  // Drift velocity data
  const driftVelocityData = [
    { time: '00:00', velocity: 2.1, baseline: 3 },
    { time: '04:00', velocity: 2.8, baseline: 3 },
    { time: '08:00', velocity: 4.2, baseline: 3 },
    { time: '12:00', velocity: 5.1, baseline: 3 },
    { time: '16:00', velocity: 4.6, baseline: 3 },
    { time: '20:00', velocity: 3.2, baseline: 3 },
    { time: '24:00', velocity: 2.9, baseline: 3 }
  ]

  // Real-time feed data
  const realTimeFeed = [
    { time: '2 min ago', event: 'New drift detected in storage-prod-westus2', type: 'warning' },
    { time: '5 min ago', event: 'Predictive model updated with latest telemetry', type: 'info' },
    { time: '12 min ago', event: 'Auto-remediation completed for vm-scale-set', type: 'success' },
    { time: '18 min ago', event: 'High confidence prediction for cost overrun', type: 'danger' },
    { time: '25 min ago', event: 'Policy simulation completed successfully', type: 'info' }
  ]

  // Quick actions for this page
  const preventActions = [
    {
      id: 'bulk-fix',
      label: 'Bulk Create Fix PRs',
      description: 'Fix multiple issues',
      icon: 'git' as const,
      variant: 'primary' as const,
      onClick: () => toast({ title: 'Creating PRs', description: 'Generating fixes for 3 critical issues...' })
    },
    {
      id: 'run-simulation',
      label: 'Run Policy Simulation',
      description: 'Test policy changes',
      icon: 'fix' as const,
      variant: 'secondary' as const,
      onClick: () => setSimulationActive(true)
    },
    {
      id: 'export-predictions',
      label: 'Export Predictions',
      description: 'Download CSV report',
      icon: 'send' as const,
      variant: 'secondary' as const,
      onClick: () => toast({ title: 'Exporting', description: 'Preparing predictions export...' })
    }
  ]

  const filteredPredictions = predictions.filter(p => 
    selectedSeverity === 'all' || p.severity === selectedSeverity
  )

  useEffect(() => {
    setTimeout(() => setLoading(false), 800)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-600"></div>
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
              <div className="flex items-center gap-3 mb-2">
                <button 
                  onClick={() => router.push('/')}
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                >
                  ← Back
                </button>
              </div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
                PREVENT - Prediction Dashboard
              </h1>
              <p className="text-gray-500 dark:text-gray-400 mt-1">
                AI-powered violation predictions with one-click remediation
              </p>
            </div>
            <div className="flex items-center gap-3">
              <select 
                value={selectedTimeframe}
                onChange={(e) => setSelectedTimeframe(e.target.value)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-sm"
              >
                <option value="24h">Next 24 Hours</option>
                <option value="7d">Next 7 Days</option>
                <option value="30d">Next 30 Days</option>
              </select>
              <button
                onClick={() => window.location.reload()}
                className="p-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Total Predictions</span>
              <AlertTriangle className="h-4 w-4 text-red-500" />
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">47</p>
            <p className="text-xs text-red-600 mt-1">12 critical</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Avg Confidence</span>
              <Target className="h-4 w-4 text-blue-500" />
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">89.2%</p>
            <p className="text-xs text-green-600 mt-1">↑ 3.5% from last week</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Auto-Fixed</span>
              <Zap className="h-4 w-4 text-yellow-500" />
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">31</p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">This week</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Drift Velocity</span>
              <Activity className="h-4 w-4 text-purple-500" />
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">4.2x</p>
            <p className="text-xs text-orange-600 mt-1">Above baseline</p>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Predictions List */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                  Forecast Cards
                </h2>
                <div className="flex items-center gap-2">
                  <Filter className="h-4 w-4 text-gray-500" />
                  <select
                    value={selectedSeverity}
                    onChange={(e) => setSelectedSeverity(e.target.value)}
                    className="text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1 bg-white dark:bg-gray-700"
                  >
                    <option value="all">All Severities</option>
                    <option value="critical">Critical</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                </div>
              </div>

              <div className="space-y-4 max-h-[600px] overflow-y-auto">
                {filteredPredictions.map((prediction) => (
                  <ForecastCard
                    key={prediction.id}
                    {...prediction}
                    onCreateFix={() => {
                      toast({ 
                        title: 'Creating Fix PR', 
                        description: `Generating automated fix for ${prediction.violationType}...` 
                      })
                    }}
                    onViewDetails={() => {
                      toast({ 
                        title: 'Opening Details', 
                        description: `Loading detailed analysis...` 
                      })
                    }}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {/* Quick Actions */}
            <QuickActions
              actions={preventActions}
              title="Prevention Actions"
              layout="list"
            />

            {/* Drift Velocity Chart */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                  Drift Velocity
                </h3>
                <Gauge className="h-4 w-4 text-purple-500" />
              </div>
              
              <ResponsiveContainer width="100%" height={150}>
                <AreaChart data={driftVelocityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="time" fontSize={10} stroke="#6b7280" />
                  <YAxis fontSize={10} stroke="#6b7280" />
                  <Tooltip />
                  <Area 
                    type="monotone" 
                    dataKey="velocity" 
                    stroke="#8b5cf6" 
                    fill="#8b5cf6" 
                    fillOpacity={0.3}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="baseline" 
                    stroke="#10b981" 
                    strokeDasharray="5 5"
                    fill="transparent"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Real-time Feed */}
            <div className="bg-gradient-to-br from-gray-50 to-slate-50 dark:from-gray-800 dark:to-slate-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                  Real-time Prediction Feed
                </h3>
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              </div>
              
              <div className="space-y-3 max-h-[200px] overflow-y-auto">
                {realTimeFeed.map((item, index) => (
                  <div key={index} className="flex items-start gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ${
                      item.type === 'warning' ? 'bg-yellow-500' :
                      item.type === 'danger' ? 'bg-red-500' :
                      item.type === 'success' ? 'bg-green-500' :
                      'bg-blue-500'
                    }`} />
                    <div className="flex-1">
                      <p className="text-sm text-gray-700 dark:text-gray-300">{item.event}</p>
                      <p className="text-xs text-gray-500 dark:text-gray-500">{item.time}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Policy Simulator */}
        <div className={`bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl border ${
          simulationActive ? 'border-indigo-500' : 'border-indigo-200 dark:border-indigo-800'
        } p-6 transition-all`}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Brain className="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
              <div>
                <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                  Policy Simulator
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Test policy changes before deployment
                </p>
              </div>
            </div>
            <button
              onClick={() => setSimulationActive(!simulationActive)}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition-colors flex items-center gap-2 ${
                simulationActive 
                  ? 'bg-indigo-600 text-white hover:bg-indigo-700' 
                  : 'bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 hover:bg-gray-50'
              }`}
            >
              {simulationActive ? (
                <>
                  <Activity className="h-4 w-4 animate-pulse" />
                  Simulation Active
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Start Simulation
                </>
              )}
            </button>
          </div>

          {simulationActive && (
            <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700 dark:text-gray-300">Simulating encryption policy change...</span>
                <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">12 resources affected</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700 dark:text-gray-300">Calculating cost impact...</span>
                <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">-$2,450/month</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700 dark:text-gray-300">Risk assessment...</span>
                <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">Low risk</span>
              </div>
              
              <div className="pt-3 border-t border-gray-200 dark:border-gray-700 flex gap-2">
                <button 
                  onClick={() => {
                    setSimulationActive(false)
                    toast({ title: 'Policy Applied', description: 'New policy rules have been deployed' })
                  }}
                  className="flex-1 bg-green-600 text-white py-2 rounded-lg font-medium text-sm hover:bg-green-700"
                >
                  Apply Changes
                </button>
                <button 
                  onClick={() => setSimulationActive(false)}
                  className="flex-1 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 py-2 rounded-lg font-medium text-sm hover:bg-gray-300 dark:hover:bg-gray-600"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}