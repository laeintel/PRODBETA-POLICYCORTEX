'use client'

import { useState } from 'react'
import { DollarSign, Users, TrendingUp, AlertTriangle, PieChart, BarChart3, ArrowUpRight, ArrowDownRight } from 'lucide-react'
import MetricCard from '@/components/MetricCard'
import ChartContainer from '@/components/ChartContainer'

export default function ChargebackPage() {
  const [selectedDepartment, setSelectedDepartment] = useState<string | null>(null)
  const [billingPeriod, setBillingPeriod] = useState('current')

  const departments = [
    {
      id: 'engineering',
      name: 'Engineering',
      budget: 150000,
      spent: 142000,
      forecast: 148000,
      utilization: 94.7,
      trend: 'up',
      resources: {
        compute: 78000,
        storage: 34000,
        network: 12000,
        database: 18000
      },
      tags: { compliance: 98, coverage: 92 }
    },
    {
      id: 'marketing',
      name: 'Marketing',
      budget: 50000,
      spent: 52000,
      forecast: 54000,
      utilization: 104,
      trend: 'up',
      resources: {
        compute: 28000,
        storage: 12000,
        network: 8000,
        database: 4000
      },
      tags: { compliance: 85, coverage: 78 }
    },
    {
      id: 'sales',
      name: 'Sales',
      budget: 75000,
      spent: 68000,
      forecast: 70000,
      utilization: 90.7,
      trend: 'down',
      resources: {
        compute: 35000,
        storage: 18000,
        network: 10000,
        database: 5000
      },
      tags: { compliance: 92, coverage: 88 }
    },
    {
      id: 'operations',
      name: 'Operations',
      budget: 200000,
      spent: 185000,
      forecast: 190000,
      utilization: 92.5,
      trend: 'stable',
      resources: {
        compute: 95000,
        storage: 45000,
        network: 25000,
        database: 20000
      },
      tags: { compliance: 96, coverage: 94 }
    }
  ]

  const untaggedResources = [
    { resource: 'vm-prod-234', type: 'Compute', cost: 2400, department: 'Unknown' },
    { resource: 'storage-backup-01', type: 'Storage', cost: 1800, department: 'Unknown' },
    { resource: 'lb-west-02', type: 'Network', cost: 950, department: 'Unknown' },
    { resource: 'db-analytics-03', type: 'Database', cost: 3200, department: 'Unknown' }
  ]

  const chargebackRules = [
    { rule: 'Compute Usage', method: 'Actual Usage', allocation: 'CPU Hours' },
    { rule: 'Storage Consumption', method: 'Tiered Pricing', allocation: 'GB/Month' },
    { rule: 'Network Transfer', method: 'Proportional', allocation: 'GB Transferred' },
    { rule: 'Shared Services', method: 'Even Split', allocation: 'Department Count' }
  ]

  const totalBudget = departments.reduce((sum, dept) => sum + dept.budget, 0)
  const totalSpent = departments.reduce((sum, dept) => sum + dept.spent, 0)
  const untaggedCost = untaggedResources.reduce((sum, res) => sum + res.cost, 0)

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-3 bg-gradient-to-r from-green-600 to-teal-600 rounded-lg">
              <Users className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Department Chargeback
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Automated cost allocation and department billing
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <MetricCard
            title="Total Budget"
            value={`$${(totalBudget / 1000).toFixed(0)}K`}
            icon={<DollarSign className="w-5 h-5 text-blue-500" />}
          />
          <MetricCard
            title="Current Spend"
            value={`$${(totalSpent / 1000).toFixed(0)}K`}
            trend={totalSpent < totalBudget ? 'up' : 'down'}
            icon={<TrendingUp className="w-5 h-5 text-green-500" />}
          />
          <MetricCard
            title="Untagged Costs"
            value={`$${(untaggedCost / 1000).toFixed(1)}K`}
            alert="Action needed"
            icon={<AlertTriangle className="w-5 h-5 text-orange-500" />}
          />
          <MetricCard
            title="Tag Compliance"
            value="89%"
            icon={<PieChart className="w-5 h-5 text-purple-500" />}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Department Breakdown</h3>
                <select 
                  value={billingPeriod}
                  onChange={(e) => setBillingPeriod(e.target.value)}
                  className="px-3 py-1 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg text-sm"
                >
                  <option value="current">Current Month</option>
                  <option value="last">Last Month</option>
                  <option value="quarter">This Quarter</option>
                </select>
              </div>
              
              <div className="space-y-4">
                {departments.map((dept) => (
                  <div
                    key={dept.id}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      selectedDepartment === dept.id
                        ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-green-300'
                    }`}
                    onClick={() => setSelectedDepartment(dept.id)}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h4 className="font-semibold">{dept.name}</h4>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          Budget: ${(dept.budget / 1000).toFixed(0)}K • 
                          Spent: ${(dept.spent / 1000).toFixed(0)}K
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-2xl font-bold ${
                          dept.utilization > 100 ? 'text-red-600' :
                          dept.utilization > 90 ? 'text-yellow-600' : 'text-green-600'
                        }`}>
                          {dept.utilization.toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500">Utilization</div>
                      </div>
                    </div>
                    
                    <div className="mb-3">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-600 dark:text-gray-400">Budget Usage</span>
                        <span className={dept.utilization > 100 ? 'text-red-600' : 'text-gray-600 dark:text-gray-400'}>
                          ${((dept.spent - dept.budget) / 1000).toFixed(1)}K {dept.utilization > 100 ? 'over' : 'remaining'}
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full ${
                            dept.utilization > 100 ? 'bg-red-500' :
                            dept.utilization > 90 ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${Math.min(dept.utilization, 100)}%` }}
                        />
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-4 gap-2 text-xs">
                      <div>
                        <div className="text-gray-600 dark:text-gray-400">Compute</div>
                        <div className="font-medium">${(dept.resources.compute / 1000).toFixed(0)}K</div>
                      </div>
                      <div>
                        <div className="text-gray-600 dark:text-gray-400">Storage</div>
                        <div className="font-medium">${(dept.resources.storage / 1000).toFixed(0)}K</div>
                      </div>
                      <div>
                        <div className="text-gray-600 dark:text-gray-400">Network</div>
                        <div className="font-medium">${(dept.resources.network / 1000).toFixed(0)}K</div>
                      </div>
                      <div>
                        <div className="text-gray-600 dark:text-gray-400">Database</div>
                        <div className="font-medium">${(dept.resources.database / 1000).toFixed(0)}K</div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-4 mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                      <div className="flex items-center gap-1">
                        {dept.trend === 'up' ? (
                          <ArrowUpRight className="w-4 h-4 text-red-500" />
                        ) : dept.trend === 'down' ? (
                          <ArrowDownRight className="w-4 h-4 text-green-500" />
                        ) : (
                          <BarChart3 className="w-4 h-4 text-gray-500" />
                        )}
                        <span className="text-xs text-gray-600 dark:text-gray-400">
                          Forecast: ${(dept.forecast / 1000).toFixed(0)}K
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-xs">
                        <span className="text-gray-600 dark:text-gray-400">Tags:</span>
                        <span className={`font-medium ${
                          dept.tags.compliance > 90 ? 'text-green-600' : 'text-orange-600'
                        }`}>
                          {dept.tags.compliance}% compliant
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Untagged Resources</h3>
              <div className="space-y-3">
                {untaggedResources.map((resource, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                    <div>
                      <div className="font-medium">{resource.resource}</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {resource.type} • ${(resource.cost / 1000).toFixed(1)}K/month
                      </div>
                    </div>
                    <button className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors">
                      Assign Dept
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Chargeback Rules</h3>
              <div className="space-y-3">
                {chargebackRules.map((rule, idx) => (
                  <div key={idx} className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                    <div className="font-medium text-sm">{rule.rule}</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      {rule.method} • {rule.allocation}
                    </div>
                  </div>
                ))}
              </div>
              <button className="w-full mt-4 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">
                Configure Rules
              </button>
            </div>

            <ChartContainer title="Cost Distribution">
              <div className="h-48 flex items-center justify-center text-gray-500">
                Department cost pie chart
              </div>
            </ChartContainer>

            <div className="bg-gradient-to-r from-green-600 to-teal-600 rounded-lg p-6 text-white">
              <h3 className="text-lg font-semibold mb-2">Automated Billing</h3>
              <p className="text-sm opacity-90 mb-4">
                Generate department invoices and integrate with your billing system
              </p>
              <button className="w-full px-4 py-2 bg-white text-green-600 rounded-lg hover:bg-gray-100 transition-colors">
                Generate Invoices
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}