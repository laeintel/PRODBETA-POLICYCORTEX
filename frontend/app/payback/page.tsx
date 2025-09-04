'use client'

import React, { useState, useEffect } from 'react'
import { 
  DollarSign, 
  TrendingUp, 
  TrendingDown, 
  Calculator,
  Target,
  PiggyBank,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  Sliders,
  FileText,
  Download,
  RefreshCw,
  ChevronRight,
  Percent
} from 'lucide-react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts'
import ROIWidget from '@/components/pcg/ROIWidget'
import { toast } from '@/hooks/useToast'
import { useRouter } from 'next/navigation'

export default function PaybackPage() {
  const router = useRouter()
  const [loading, setLoading] = useState(true)
  const [selectedPeriod, setSelectedPeriod] = useState('30d')
  const [simulationValues, setSimulationValues] = useState({
    preventionRate: 94,
    automationLevel: 75,
    incidentReduction: 82
  })

  // P&L Statement Data
  const plStatement = {
    revenue: {
      preventedIncidents: 650000,
      resourceOptimization: 350000,
      automatedRemediation: 250000,
      compliancePenaltiesAvoided: 180000,
      total: 1430000
    },
    costs: {
      platformLicense: 150000,
      implementation: 50000,
      training: 25000,
      maintenance: 35000,
      total: 260000
    },
    netSavings: 1170000,
    roi: 350,
    paybackPeriod: '2.3 months'
  }

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

  // Projection Chart Data
  const projectionData = [
    { month: 'Jan', actual: 320000, projected: 310000, savings: 10000 },
    { month: 'Feb', actual: 380000, projected: 350000, savings: 30000 },
    { month: 'Mar', actual: 420000, projected: 380000, savings: 40000 },
    { month: 'Apr', actual: 410000, projected: 360000, savings: 50000 },
    { month: 'May', actual: 450000, projected: 385000, savings: 65000 },
    { month: 'Jun', actual: 480000, projected: 400000, savings: 80000 }
  ]

  // Per-Policy Savings
  const policyBreakdown = [
    { policy: 'Encryption Standards', violations: 142, prevented: 138, savings: 125000, roi: 420 },
    { policy: 'Access Controls', violations: 89, prevented: 85, savings: 98000, roi: 380 },
    { policy: 'Cost Optimization', violations: 67, prevented: 61, savings: 156000, roi: 510 },
    { policy: 'Network Security', violations: 45, prevented: 43, savings: 72000, roi: 290 },
    { policy: 'Data Retention', violations: 34, prevented: 31, savings: 45000, roi: 240 },
    { policy: 'Compliance Tags', violations: 28, prevented: 26, savings: 32000, roi: 180 }
  ]

  // What-if Simulation
  const calculateSimulation = () => {
    const baseAmount = 1250000
    const preventionFactor = simulationValues.preventionRate / 100
    const automationFactor = simulationValues.automationLevel / 100
    const incidentFactor = simulationValues.incidentReduction / 100
    
    return Math.round(baseAmount * preventionFactor * automationFactor * incidentFactor)
  }

  const COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#6b7280']

  useEffect(() => {
    setTimeout(() => setLoading(false), 800)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-600"></div>
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
                PAYBACK - ROI Dashboard
              </h1>
              <p className="text-gray-500 dark:text-gray-400 mt-1">
                Financial impact analysis and P&L tracking
              </p>
            </div>
            <div className="flex items-center gap-3">
              <select 
                value={selectedPeriod}
                onChange={(e) => setSelectedPeriod(e.target.value)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-sm"
              >
                <option value="30d">Last 30 Days</option>
                <option value="90d">Last Quarter</option>
                <option value="1y">Last Year</option>
              </select>
              <button
                onClick={() => toast({ title: 'Exporting P&L', description: 'Generating financial report...' })}
                className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors text-sm font-medium flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                Export P&L
              </button>
            </div>
          </div>
        </div>

        {/* Main ROI Widget */}
        <div className="mb-8">
          <ROIWidget
            metrics={roiMetrics}
            showProjections={true}
            onViewDetails={() => toast({ title: 'Opening Details', description: 'Loading detailed ROI analysis...' })}
          />
        </div>

        {/* P&L Statement */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                Governance P&L Statement
              </h2>
              <FileText className="h-5 w-5 text-gray-500" />
            </div>

            <div className="space-y-4">
              {/* Revenue Section */}
              <div>
                <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-3">Revenue (Savings)</h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Prevented Incidents</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      ${plStatement.revenue.preventedIncidents.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Resource Optimization</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      ${plStatement.revenue.resourceOptimization.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Automated Remediation</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      ${plStatement.revenue.automatedRemediation.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Compliance Penalties Avoided</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      ${plStatement.revenue.compliancePenaltiesAvoided.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between pt-2 border-t border-gray-200 dark:border-gray-700">
                    <span className="font-semibold text-gray-700 dark:text-gray-300">Total Revenue</span>
                    <span className="font-bold text-green-600 dark:text-green-400">
                      ${plStatement.revenue.total.toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              {/* Costs Section */}
              <div>
                <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-3">Costs</h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Platform License</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      ${plStatement.costs.platformLicense.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Implementation</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      ${plStatement.costs.implementation.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Training</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      ${plStatement.costs.training.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Maintenance</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      ${plStatement.costs.maintenance.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between pt-2 border-t border-gray-200 dark:border-gray-700">
                    <span className="font-semibold text-gray-700 dark:text-gray-300">Total Costs</span>
                    <span className="font-bold text-red-600 dark:text-red-400">
                      ${plStatement.costs.total.toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              {/* Net Savings */}
              <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-gray-900 dark:text-gray-100">Net Savings</span>
                  <span className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
                    ${plStatement.netSavings.toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center justify-between mt-2 text-sm">
                  <span className="text-gray-600 dark:text-gray-400">ROI</span>
                  <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                    {plStatement.roi}%
                  </span>
                </div>
                <div className="flex items-center justify-between mt-1 text-sm">
                  <span className="text-gray-600 dark:text-gray-400">Payback Period</span>
                  <span className="font-semibold text-gray-900 dark:text-gray-100">
                    {plStatement.paybackPeriod}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* What-if Simulator */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-200 dark:border-blue-800 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                Interactive What-If Simulator
              </h2>
              <Sliders className="h-5 w-5 text-blue-500" />
            </div>

            <div className="space-y-6">
              {/* Prevention Rate Slider */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Prevention Rate
                  </label>
                  <span className="text-sm font-bold text-gray-900 dark:text-gray-100">
                    {simulationValues.preventionRate}%
                  </span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="100"
                  value={simulationValues.preventionRate}
                  onChange={(e) => setSimulationValues({
                    ...simulationValues,
                    preventionRate: parseInt(e.target.value)
                  })}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                />
              </div>

              {/* Automation Level Slider */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Automation Level
                  </label>
                  <span className="text-sm font-bold text-gray-900 dark:text-gray-100">
                    {simulationValues.automationLevel}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={simulationValues.automationLevel}
                  onChange={(e) => setSimulationValues({
                    ...simulationValues,
                    automationLevel: parseInt(e.target.value)
                  })}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                />
              </div>

              {/* Incident Reduction Slider */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Incident Reduction
                  </label>
                  <span className="text-sm font-bold text-gray-900 dark:text-gray-100">
                    {simulationValues.incidentReduction}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={simulationValues.incidentReduction}
                  onChange={(e) => setSimulationValues({
                    ...simulationValues,
                    incidentReduction: parseInt(e.target.value)
                  })}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                />
              </div>

              {/* Simulation Result */}
              <div className="bg-white/70 dark:bg-gray-800/70 rounded-lg p-4">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Projected Annual Savings</p>
                <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                  ${calculateSimulation().toLocaleString()}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
                  Based on current simulation parameters
                </p>
              </div>

              <button
                onClick={() => toast({ title: 'Simulation Applied', description: 'New projections calculated' })}
                className="w-full bg-blue-600 text-white py-2 rounded-lg font-medium text-sm hover:bg-blue-700 transition-colors"
              >
                Apply Simulation
              </button>
            </div>
          </div>
        </div>

        {/* Projections Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
              30/60/90-Day Projections
            </h2>
            <TrendingUp className="h-5 w-5 text-green-500" />
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={projectionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="month" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} tickFormatter={(value) => `$${value/1000}K`} />
              <Tooltip formatter={(value: any) => `$${(value/1000).toFixed(0)}K`} />
              <Legend />
              <Area 
                type="monotone" 
                dataKey="actual" 
                stackId="1"
                stroke="#ef4444" 
                fill="#ef4444"
                fillOpacity={0.3}
                name="Actual Costs"
              />
              <Area 
                type="monotone" 
                dataKey="projected" 
                stackId="2"
                stroke="#3b82f6" 
                fill="#3b82f6"
                fillOpacity={0.3}
                name="Projected w/ PolicyCortex"
              />
              <Area 
                type="monotone" 
                dataKey="savings" 
                stackId="3"
                stroke="#10b981" 
                fill="#10b981"
                fillOpacity={0.5}
                name="Savings"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Per-Policy Savings Breakdown */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
              Per-Policy Savings Breakdown
            </h2>
            <BarChart3 className="h-5 w-5 text-purple-500" />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">Policy</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">Violations</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">Prevented</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">Savings</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">ROI %</th>
                </tr>
              </thead>
              <tbody>
                {policyBreakdown.map((policy, index) => (
                  <tr key={index} className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                    <td className="py-3 px-4">
                      <span className="font-medium text-gray-900 dark:text-gray-100">{policy.policy}</span>
                    </td>
                    <td className="text-center py-3 px-4 text-gray-600 dark:text-gray-400">
                      {policy.violations}
                    </td>
                    <td className="text-center py-3 px-4">
                      <span className="text-green-600 dark:text-green-400 font-medium">
                        {policy.prevented}
                      </span>
                      <span className="text-xs text-gray-500 ml-1">
                        ({Math.round((policy.prevented / policy.violations) * 100)}%)
                      </span>
                    </td>
                    <td className="text-right py-3 px-4 font-semibold text-gray-900 dark:text-gray-100">
                      ${policy.savings.toLocaleString()}
                    </td>
                    <td className="text-center py-3 px-4">
                      <div className="flex items-center justify-center gap-1">
                        <Percent className="h-3 w-3 text-gray-400" />
                        <span className={`font-bold ${
                          policy.roi >= 400 ? 'text-green-600' :
                          policy.roi >= 300 ? 'text-blue-600' :
                          policy.roi >= 200 ? 'text-yellow-600' :
                          'text-gray-600'
                        }`}>
                          {policy.roi}
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="bg-gray-50 dark:bg-gray-700/50">
                  <td className="py-3 px-4 font-bold text-gray-900 dark:text-gray-100">Total</td>
                  <td className="text-center py-3 px-4 font-bold text-gray-900 dark:text-gray-100">
                    {policyBreakdown.reduce((sum, p) => sum + p.violations, 0)}
                  </td>
                  <td className="text-center py-3 px-4 font-bold text-green-600 dark:text-green-400">
                    {policyBreakdown.reduce((sum, p) => sum + p.prevented, 0)}
                  </td>
                  <td className="text-right py-3 px-4 font-bold text-gray-900 dark:text-gray-100">
                    ${policyBreakdown.reduce((sum, p) => sum + p.savings, 0).toLocaleString()}
                  </td>
                  <td className="text-center py-3 px-4 font-bold text-blue-600 dark:text-blue-400">
                    {Math.round(policyBreakdown.reduce((sum, p) => sum + p.roi, 0) / policyBreakdown.length)}%
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>

        {/* ROI Gauge */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl border border-green-200 dark:border-green-800 p-6 text-center">
            <Target className="h-8 w-8 text-green-600 mx-auto mb-3" />
            <p className="text-3xl font-bold text-gray-900 dark:text-gray-100">{plStatement.roi}%</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">Total ROI</p>
          </div>

          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-200 dark:border-blue-800 p-6 text-center">
            <Calculator className="h-8 w-8 text-blue-600 mx-auto mb-3" />
            <p className="text-3xl font-bold text-gray-900 dark:text-gray-100">{plStatement.paybackPeriod}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">Payback Period</p>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl border border-purple-200 dark:border-purple-800 p-6 text-center">
            <PiggyBank className="h-8 w-8 text-purple-600 mx-auto mb-3" />
            <p className="text-3xl font-bold text-gray-900 dark:text-gray-100">
              ${(plStatement.netSavings / 1000).toFixed(0)}K
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">Monthly Savings</p>
          </div>
        </div>
      </div>
    </div>
  )
}