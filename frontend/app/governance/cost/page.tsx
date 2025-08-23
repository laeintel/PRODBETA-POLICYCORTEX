'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import {
  DollarSign, TrendingUp, TrendingDown, AlertTriangle,
  BarChart3, PieChart as PieChartIcon, Download, RefreshCw,
  Search, Filter, Calendar, Clock, Target, Zap,
  ChevronUp, ChevronDown, ArrowUpRight, ArrowDownRight,
  Activity, CreditCard, Package, AlertCircle, CheckCircle
} from 'lucide-react'
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Treemap,
  ScatterChart, Scatter, RadialBarChart, RadialBar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

// TypeScript Types
interface CostMetrics {
  currentSpend: number
  budget: number
  forecast: number
  lastMonth: number
  yearToDate: number
  savingsIdentified: number
  unusedResources: number
}

interface ServiceCost {
  service: string
  cost: number
  change: number
  percentage: number
  trend: 'up' | 'down' | 'stable'
}

interface CostAnomaly {
  id: string
  resource: string
  service: string
  anomalyType: string
  amount: number
  severity: 'critical' | 'high' | 'medium' | 'low'
  detectedAt: string
  status: 'new' | 'investigating' | 'resolved'
}

interface OptimizationRecommendation {
  id: string
  type: string
  title: string
  description: string
  estimatedSavings: number
  effort: 'low' | 'medium' | 'high'
  impact: 'high' | 'medium' | 'low'
  resources: string[]
  automatable: boolean
}

interface Budget {
  id: string
  name: string
  allocated: number
  spent: number
  remaining: number
  threshold: number
  owner: string
  period: string
}

interface ReservedInstance {
  id: string
  type: string
  region: string
  quantity: number
  utilization: number
  expiry: string
  monthlySavings: number
}

export default function CostManagementPage() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedTimeRange, setSelectedTimeRange] = useState('30d')
  const [selectedService, setSelectedService] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  // Mock metrics
  const metrics: CostMetrics = {
    currentSpend: 127543,
    budget: 150000,
    forecast: 135000,
    lastMonth: 118234,
    yearToDate: 1453234,
    savingsIdentified: 45000,
    unusedResources: 12300
  }

  // Cost trend data
  const costTrend = [
    { date: 'Jan', actual: 115000, budget: 125000, forecast: 118000 },
    { date: 'Feb', actual: 118000, budget: 125000, forecast: 120000 },
    { date: 'Mar', actual: 122000, budget: 125000, forecast: 123000 },
    { date: 'Apr', actual: 119000, budget: 130000, forecast: 125000 },
    { date: 'May', actual: 124000, budget: 130000, forecast: 128000 },
    { date: 'Jun', actual: 127543, budget: 150000, forecast: 135000 }
  ]

  // Service breakdown
  const serviceCosts: ServiceCost[] = [
    { service: 'Compute', cost: 45234, change: 5.2, percentage: 35.5, trend: 'up' },
    { service: 'Storage', cost: 23456, change: -2.1, percentage: 18.4, trend: 'down' },
    { service: 'Network', cost: 18234, change: 3.5, percentage: 14.3, trend: 'up' },
    { service: 'Database', cost: 21234, change: -1.2, percentage: 16.6, trend: 'down' },
    { service: 'AI Services', cost: 12345, change: 12.5, percentage: 9.7, trend: 'up' },
    { service: 'Other', cost: 7074, change: 0.5, percentage: 5.5, trend: 'stable' }
  ]

  // Resource group costs for treemap
  const resourceGroupCosts = [
    {
      name: 'Production',
      size: 65000,
      children: [
        { name: 'prod-web', size: 25000 },
        { name: 'prod-api', size: 20000 },
        { name: 'prod-db', size: 20000 }
      ]
    },
    {
      name: 'Development',
      size: 35000,
      children: [
        { name: 'dev-env1', size: 15000 },
        { name: 'dev-env2', size: 12000 },
        { name: 'dev-test', size: 8000 }
      ]
    },
    {
      name: 'Staging',
      size: 27543,
      children: [
        { name: 'stage-app', size: 15000 },
        { name: 'stage-data', size: 12543 }
      ]
    }
  ]

  // Cost anomalies
  const anomalies: CostAnomaly[] = [
    {
      id: '1',
      resource: 'prod-vm-cluster',
      service: 'Compute',
      anomalyType: 'Spike in usage',
      amount: 3500,
      severity: 'high',
      detectedAt: '2024-03-01T10:30:00Z',
      status: 'new'
    },
    {
      id: '2',
      resource: 'backup-storage-01',
      service: 'Storage',
      anomalyType: 'Unexpected data transfer',
      amount: 1200,
      severity: 'medium',
      detectedAt: '2024-03-01T08:15:00Z',
      status: 'investigating'
    },
    {
      id: '3',
      resource: 'ml-training-gpu',
      service: 'AI Services',
      anomalyType: 'Continuous high utilization',
      amount: 5000,
      severity: 'critical',
      detectedAt: '2024-02-28T22:00:00Z',
      status: 'new'
    }
  ]

  // Optimization recommendations
  const recommendations: OptimizationRecommendation[] = [
    {
      id: '1',
      type: 'rightsizing',
      title: 'Rightsize underutilized VMs',
      description: '15 VMs are consistently using less than 20% CPU',
      estimatedSavings: 12000,
      effort: 'low',
      impact: 'high',
      resources: ['prod-vm-01', 'prod-vm-02', 'dev-vm-cluster'],
      automatable: true
    },
    {
      id: '2',
      type: 'reserved-instances',
      title: 'Purchase reserved instances',
      description: 'Save 40% on 24 always-on production VMs',
      estimatedSavings: 8000,
      effort: 'medium',
      impact: 'high',
      resources: ['prod-cluster'],
      automatable: false
    },
    {
      id: '3',
      type: 'unused-resources',
      title: 'Delete unattached disks',
      description: '32 unattached disks found across subscriptions',
      estimatedSavings: 3000,
      effort: 'low',
      impact: 'medium',
      resources: ['Various'],
      automatable: true
    },
    {
      id: '4',
      type: 'auto-shutdown',
      title: 'Enable auto-shutdown for dev/test',
      description: 'Automatically shut down non-production resources after hours',
      estimatedSavings: 5000,
      effort: 'low',
      impact: 'high',
      resources: ['dev-*', 'test-*'],
      automatable: true
    }
  ]

  // Budgets
  const budgets: Budget[] = [
    {
      id: '1',
      name: 'Production Environment',
      allocated: 80000,
      spent: 72543,
      remaining: 7457,
      threshold: 90,
      owner: 'ops-team@company.com',
      period: 'Monthly'
    },
    {
      id: '2',
      name: 'Development Team',
      allocated: 40000,
      spent: 35000,
      remaining: 5000,
      threshold: 80,
      owner: 'dev-lead@company.com',
      period: 'Monthly'
    },
    {
      id: '3',
      name: 'Marketing Campaign',
      allocated: 30000,
      spent: 20000,
      remaining: 10000,
      threshold: 75,
      owner: 'marketing@company.com',
      period: 'Quarterly'
    }
  ]

  // Reserved instances
  const reservedInstances: ReservedInstance[] = [
    {
      id: '1',
      type: 'D4s_v3',
      region: 'East US',
      quantity: 10,
      utilization: 95,
      expiry: '2024-12-31',
      monthlySavings: 3500
    },
    {
      id: '2',
      type: 'B2ms',
      region: 'West Europe',
      quantity: 25,
      utilization: 78,
      expiry: '2024-09-30',
      monthlySavings: 2100
    }
  ]

  // Tag distribution for pie chart
  const tagDistribution = [
    { name: 'Production', value: 65000, color: '#3b82f6' },
    { name: 'Development', value: 35000, color: '#10b981' },
    { name: 'Staging', value: 27543, color: '#f59e0b' },
    { name: 'Untagged', value: 12000, color: '#ef4444' }
  ]

  const handleOptimize = (recommendationId: string, automated: boolean) => {
    if (automated) {
      alert(`Starting automated optimization for recommendation ${recommendationId}`)
    } else {
      router.push(`/governance/cost/optimize/${recommendationId}`)
    }
  }

  const exportReport = (format: 'csv' | 'pdf') => {
    alert(`Exporting cost report as ${format.toUpperCase()}`)
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-white">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-gray-900/50 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Cost Management</h1>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Monitor spending, track budgets, and optimize cloud costs
              </p>
            </div>
            <div className="flex items-center gap-3">
              <select
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value)}
                className="px-3 py-2 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-sm focus:outline-none focus:border-blue-500">
                <option value="7d">Last 7 days</option>
                <option value="30d">Last 30 days</option>
                <option value="90d">Last 90 days</option>
                <option value="12m">Last 12 months</option>
              </select>
              <button type="button"
                onClick={() => exportReport('csv')}
                className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                <Download className="w-4 h-4" />
                Export
              </button>
              <button type="button"
                onClick={() => router.push('/governance/cost/optimize')}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
              >
                <Zap className="w-4 h-4" />
                Optimize Now
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-800 bg-gray-100/30 dark:bg-gray-900/30">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-6">
            {['overview', 'breakdown', 'anomalies', 'optimization', 'budgets', 'reserved', 'forecast'].map((tab) => (
              <button type="button"
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-3 border-b-2 transition-colors capitalize ${
                  activeTab === tab
                    ? 'border-blue-500 text-gray-900 dark:text-white'
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm text-gray-600 dark:text-gray-400">Current Spend</h3>
                  <DollarSign className="w-5 h-5 text-green-400" />
                </div>
                <p className="text-3xl font-bold">${(metrics.currentSpend / 1000).toFixed(1)}K</p>
                <div className="flex items-center gap-2 mt-2">
                  <ArrowUpRight className="w-4 h-4 text-red-400" />
                  <span className="text-sm text-red-400">+7.8% from last month</span>
                </div>
              </div>

              <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm text-gray-600 dark:text-gray-400">Budget</h3>
                  <Target className="w-5 h-5 text-blue-400" />
                </div>
                <p className="text-3xl font-bold">${(metrics.budget / 1000).toFixed(0)}K</p>
                <div className="mt-2">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-600 dark:text-gray-400">Used</span>
                    <span>{((metrics.currentSpend / metrics.budget) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full"
                      style={{ width: `${(metrics.currentSpend / metrics.budget) * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm text-gray-600 dark:text-gray-400">Forecast</h3>
                  <TrendingUp className="w-5 h-5 text-yellow-400" />
                </div>
                <p className="text-3xl font-bold">${(metrics.forecast / 1000).toFixed(0)}K</p>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">End of month projection</p>
              </div>

              <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm text-gray-600 dark:text-gray-400">Savings Available</h3>
                  <Zap className="w-5 h-5 text-purple-400" />
                </div>
                <p className="text-3xl font-bold">${(metrics.savingsIdentified / 1000).toFixed(0)}K</p>
                <p className="text-sm text-green-400 mt-2">Monthly potential</p>
              </div>
            </div>

            {/* Cost Trend Chart */}
            <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Cost Trend Analysis</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={costTrend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" tickFormatter={(value) => `$${value / 1000}K`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    formatter={(value: number) => [`$${(value / 1000).toFixed(1)}K`, '']}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="budget"
                    stroke="#6b7280"
                    fill="none"
                    strokeDasharray="5 5"
                    name="Budget"
                  />
                  <Area
                    type="monotone"
                    dataKey="actual"
                    stroke="#3b82f6"
                    fill="url(#colorActual)"
                    name="Actual Spend"
                  />
                  <Area
                    type="monotone"
                    dataKey="forecast"
                    stroke="#f59e0b"
                    fill="none"
                    strokeDasharray="3 3"
                    name="Forecast"
                  />
                  <defs>
                    <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1} />
                    </linearGradient>
                  </defs>
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Service Costs */}
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
                <h3 className="text-lg font-semibold mb-4">Cost by Service</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={serviceCosts}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="service" stroke="#9ca3af" angle={-45} textAnchor="end" height={80} />
                    <YAxis stroke="#9ca3af" tickFormatter={(value) => `$${value / 1000}K`} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      formatter={(value: number) => [`$${(value / 1000).toFixed(1)}K`, 'Cost']}
                    />
                    <Bar dataKey="cost" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
                <h3 className="text-lg font-semibold mb-4">Tag Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={tagDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {tagDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      formatter={(value: number) => [`$${(value / 1000).toFixed(1)}K`, '']}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex justify-center gap-4 mt-4">
                  {tagDistribution.map((item) => (
                    <div key={item.name} className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-xs text-gray-600 dark:text-gray-400">
                        {item.name} (${(item.value / 1000).toFixed(0)}K)
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'anomalies' && (
          <div className="space-y-6">
            {/* Anomaly Detection */}
            <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Cost Anomalies Detected</h3>
                <span className="px-3 py-1 bg-red-900/50 text-red-400 rounded-full text-sm">
                  {anomalies.filter(a => a.status === 'new').length} new
                </span>
              </div>
              <div className="space-y-3">
                {anomalies.map((anomaly) => (
                  <div key={anomaly.id} className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <AlertTriangle className={`w-5 h-5 ${
                          anomaly.severity === 'critical' ? 'text-red-400' :
                          anomaly.severity === 'high' ? 'text-orange-400' :
                          anomaly.severity === 'medium' ? 'text-yellow-400' :
                          'text-blue-400'
                        }`} />
                        <div>
                          <p className="font-medium">{anomaly.anomalyType}</p>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            {anomaly.resource} • {anomaly.service}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <p className="text-lg font-bold text-red-400">+${anomaly.amount}</p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            {new Date(anomaly.detectedAt).toLocaleDateString()}
                          </p>
                        </div>
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          anomaly.status === 'new' ? 'bg-yellow-900/50 text-yellow-400' :
                          anomaly.status === 'investigating' ? 'bg-blue-900/50 text-blue-400' :
                          'bg-green-900/50 text-green-400'
                        }`}>
                          {anomaly.status}
                        </span>
                        <button type="button" className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                          Investigate
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'optimization' && (
          <div className="space-y-6">
            {/* Optimization Recommendations */}
            <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Cost Optimization Opportunities</h3>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Total potential savings:</span>
                  <span className="text-2xl font-bold text-green-400">
                    ${(recommendations.reduce((sum, r) => sum + r.estimatedSavings, 0) / 1000).toFixed(0)}K/mo
                  </span>
                </div>
              </div>
              <div className="space-y-3">
                {recommendations.map((rec) => (
                  <div key={rec.id} className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <Package className="w-5 h-5 text-blue-400" />
                          <p className="font-medium">{rec.title}</p>
                          {rec.automatable && (
                            <span className="px-2 py-1 bg-green-900/50 text-green-400 rounded text-xs">
                              Automatable
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{rec.description}</p>
                        <div className="flex items-center gap-4 text-sm">
                          <span className="text-gray-600 dark:text-gray-400">
                            Effort: <span className={`${
                              rec.effort === 'low' ? 'text-green-400' :
                              rec.effort === 'medium' ? 'text-yellow-400' :
                              'text-red-400'
                            }`}>{rec.effort}</span>
                          </span>
                          <span className="text-gray-600 dark:text-gray-400">
                            Impact: <span className={`${
                              rec.impact === 'high' ? 'text-green-400' :
                              rec.impact === 'medium' ? 'text-yellow-400' :
                              'text-blue-400'
                            }`}>{rec.impact}</span>
                          </span>
                          <span className="text-gray-600 dark:text-gray-400">
                            Resources: {rec.resources.length}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <p className="text-2xl font-bold text-green-400">
                            ${(rec.estimatedSavings / 1000).toFixed(1)}K
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">per month</p>
                        </div>
                        <div className="flex gap-2">
                          {rec.automatable && (
                            <button type="button"
                              onClick={() => handleOptimize(rec.id, true)}
                              className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm"
                            >
                              Auto-fix
                            </button>
                          )}
                          <button type="button"
                            onClick={() => handleOptimize(rec.id, false)}
                            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                          >
                            Review
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'budgets' && (
          <div className="space-y-6">
            {/* Budget Tracking */}
            <div className="grid grid-cols-1 gap-4">
              {budgets.map((budget) => (
                <div key={budget.id} className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="font-medium">{budget.name}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Owner: {budget.owner} • Period: {budget.period}
                      </p>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-sm ${
                      (budget.spent / budget.allocated) * 100 > budget.threshold
                        ? 'bg-red-900/50 text-red-400'
                        : 'bg-green-900/50 text-green-400'
                    }`}>
                      {((budget.spent / budget.allocated) * 100).toFixed(0)}% used
                    </span>
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Budget</span>
                      <span>${(budget.allocated / 1000).toFixed(0)}K</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Spent</span>
                      <span className="text-yellow-400">${(budget.spent / 1000).toFixed(1)}K</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Remaining</span>
                      <span className="text-green-400">${(budget.remaining / 1000).toFixed(1)}K</span>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-400">Threshold: {budget.threshold}%</span>
                        <span>{((budget.spent / budget.allocated) * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-3 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                        <div className="h-full flex">
                          <div
                            className="bg-blue-500"
                            style={{ width: `${(budget.spent / budget.allocated) * 100}%` }}
                          />
                          <div
                            className="bg-red-500"
                            style={{
                              width: `${Math.max(0, ((budget.spent / budget.allocated) * 100) - budget.threshold)}%`
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}