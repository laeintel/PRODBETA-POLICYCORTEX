'use client'

import { useState } from 'react'
import { Calculator, TrendingUp, DollarSign, Clock, BarChart3, ArrowUpRight, ArrowDownRight, Target } from 'lucide-react'
import MetricCard from '@/components/MetricCard'
import ChartContainer from '@/components/ChartContainer'

export default function ROICalculatorPage() {
  const [inputs, setInputs] = useState({
    cloudSpend: 500000,
    teamSize: 25,
    avgSalary: 120000,
    complianceViolations: 12,
    securityIncidents: 3,
    deploymentFrequency: 20
  })

  const calculateROI = () => {
    const annualLaborCost = inputs.teamSize * inputs.avgSalary
    const violationCost = inputs.complianceViolations * 50000
    const incidentCost = inputs.securityIncidents * 250000
    
    const savings = {
      cloudOptimization: inputs.cloudSpend * 0.35,
      laborEfficiency: annualLaborCost * 0.28,
      complianceAutomation: violationCost * 0.85,
      securityImprovement: incidentCost * 0.72,
      deploymentAcceleration: inputs.deploymentFrequency * 15000
    }
    
    const totalSavings = Object.values(savings).reduce((a, b) => a + b, 0)
    const investmentCost = 280000 // PolicyCortex annual cost
    const netBenefit = totalSavings - investmentCost
    const roi = ((netBenefit / investmentCost) * 100).toFixed(0)
    const paybackMonths = Math.ceil((investmentCost / (totalSavings / 12)))
    
    return {
      savings,
      totalSavings,
      investmentCost,
      netBenefit,
      roi,
      paybackMonths
    }
  }

  const results = calculateROI()

  const benefitCategories = [
    {
      category: 'Cloud Cost Optimization',
      value: results.savings.cloudOptimization,
      percentage: 35,
      description: 'AI-driven rightsizing and waste elimination'
    },
    {
      category: 'Labor Efficiency',
      value: results.savings.laborEfficiency,
      percentage: 28,
      description: 'Automation reduces manual governance tasks'
    },
    {
      category: 'Compliance Automation',
      value: results.savings.complianceAutomation,
      percentage: 85,
      description: 'Prevent violations with predictive compliance'
    },
    {
      category: 'Security Enhancement',
      value: results.savings.securityImprovement,
      percentage: 72,
      description: 'Reduce incidents with AI threat detection'
    },
    {
      category: 'Deployment Velocity',
      value: results.savings.deploymentAcceleration,
      percentage: 75,
      description: 'Faster time-to-market with automated governance'
    }
  ]

  const timeToValue = [
    { milestone: 'Initial Setup', weeks: 1, value: 'Platform deployment' },
    { milestone: 'Quick Wins', weeks: 2, value: 'First cost savings identified' },
    { milestone: 'Process Integration', weeks: 4, value: 'Workflows automated' },
    { milestone: 'Full Automation', weeks: 8, value: 'AI models trained' },
    { milestone: 'ROI Positive', weeks: results.paybackMonths * 4, value: 'Break-even achieved' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-3 bg-gradient-to-r from-green-600 to-blue-600 rounded-lg">
              <Calculator className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                ROI Calculator
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Calculate your return on investment with PolicyCortex
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Your Organization</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Annual Cloud Spend
                  </label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">$</span>
                    <input
                      type="number"
                      value={inputs.cloudSpend}
                      onChange={(e) => setInputs({...inputs, cloudSpend: parseInt(e.target.value) || 0})}
                      className="w-full pl-8 pr-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
                    />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Cloud Team Size
                  </label>
                  <input
                    type="number"
                    value={inputs.teamSize}
                    onChange={(e) => setInputs({...inputs, teamSize: parseInt(e.target.value) || 0})}
                    className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Average Salary
                  </label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">$</span>
                    <input
                      type="number"
                      value={inputs.avgSalary}
                      onChange={(e) => setInputs({...inputs, avgSalary: parseInt(e.target.value) || 0})}
                      className="w-full pl-8 pr-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
                    />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Annual Compliance Violations
                  </label>
                  <input
                    type="number"
                    value={inputs.complianceViolations}
                    onChange={(e) => setInputs({...inputs, complianceViolations: parseInt(e.target.value) || 0})}
                    className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Security Incidents/Year
                  </label>
                  <input
                    type="number"
                    value={inputs.securityIncidents}
                    onChange={(e) => setInputs({...inputs, securityIncidents: parseInt(e.target.value) || 0})}
                    className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Deployments/Month
                  </label>
                  <input
                    type="number"
                    value={inputs.deploymentFrequency}
                    onChange={(e) => setInputs({...inputs, deploymentFrequency: parseInt(e.target.value) || 0})}
                    className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gradient-to-r from-green-600 to-emerald-600 rounded-lg p-6 text-white">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium opacity-90">ROI</h3>
                  <TrendingUp className="w-5 h-5" />
                </div>
                <div className="text-3xl font-bold">{results.roi}%</div>
                <div className="text-xs opacity-75 mt-1">Return on Investment</div>
              </div>
              
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg p-6 text-white">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium opacity-90">Net Benefit</h3>
                  <DollarSign className="w-5 h-5" />
                </div>
                <div className="text-3xl font-bold">${(results.netBenefit / 1000).toFixed(0)}K</div>
                <div className="text-xs opacity-75 mt-1">Annual Savings</div>
              </div>
              
              <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg p-6 text-white">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium opacity-90">Payback</h3>
                  <Clock className="w-5 h-5" />
                </div>
                <div className="text-3xl font-bold">{results.paybackMonths}</div>
                <div className="text-xs opacity-75 mt-1">Months to Break-Even</div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Value Breakdown</h3>
              <div className="space-y-4">
                {benefitCategories.map((benefit, idx) => (
                  <div key={idx} className="space-y-2">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-medium">{benefit.category}</div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">{benefit.description}</div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold text-green-600">${(benefit.value / 1000).toFixed(0)}K</div>
                        <div className="text-xs text-gray-500">{benefit.percentage}% improvement</div>
                      </div>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                        style={{ width: `${(benefit.value / results.totalSavings) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                <div className="flex justify-between items-center">
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Total Annual Savings</div>
                    <div className="text-2xl font-bold text-green-600">${(results.totalSavings / 1000).toFixed(0)}K</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Investment Cost</div>
                    <div className="text-2xl font-bold text-red-600">-${(results.investmentCost / 1000).toFixed(0)}K</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Net Annual Benefit</div>
                    <div className="text-2xl font-bold text-blue-600">${(results.netBenefit / 1000).toFixed(0)}K</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Time to Value</h3>
              <div className="space-y-3">
                {timeToValue.map((milestone, idx) => (
                  <div key={idx} className="flex items-center gap-4">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium ${
                      idx === 0 ? 'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400' :
                      idx < 3 ? 'bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400' :
                      'bg-purple-100 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400'
                    }`}>
                      W{milestone.weeks}
                    </div>
                    <div className="flex-1">
                      <div className="font-medium">{milestone.milestone}</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">{milestone.value}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gradient-to-r from-green-600 to-blue-600 rounded-lg p-6 text-white">
              <h3 className="text-lg font-semibold mb-2">Executive Summary</h3>
              <p className="text-sm opacity-90 mb-4">
                Based on your inputs, PolicyCortex will deliver a {results.roi}% ROI with payback in {results.paybackMonths} months. 
                The platform will save your organization ${(results.netBenefit / 1000).toFixed(0)}K annually through cloud optimization, 
                automation, and risk reduction.
              </p>
              <button className="px-4 py-2 bg-white text-green-600 rounded-lg hover:bg-gray-100 transition-colors">
                Download ROI Report
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}