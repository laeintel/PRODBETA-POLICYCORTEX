'use client'

import { useState } from 'react'
import { AlertTriangle, TrendingDown, DollarSign, Shield, BarChart3, Activity, Target, Zap } from 'lucide-react'
import MetricCard from '@/components/MetricCard'
import ChartContainer from '@/components/ChartContainer'

export default function RiskMapPage() {
  const [selectedRisk, setSelectedRisk] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'impact' | 'likelihood'>('impact')

  const riskCategories = [
    {
      id: 'compliance',
      name: 'Compliance Violations',
      impact: 8500000,
      likelihood: 65,
      revenueImpact: 12.3,
      mitigation: 'Predictive compliance engine',
      status: 'mitigating',
      trend: 'decreasing',
      incidents: [
        { date: '2024-01-15', cost: 250000, description: 'GDPR violation - data residency' },
        { date: '2024-02-03', cost: 180000, description: 'SOC2 audit failure' }
      ]
    },
    {
      id: 'security',
      name: 'Security Breaches',
      impact: 15000000,
      likelihood: 35,
      revenueImpact: 21.7,
      mitigation: 'AI threat detection',
      status: 'monitoring',
      trend: 'stable',
      incidents: [
        { date: '2024-01-28', cost: 450000, description: 'Ransomware attempt blocked' }
      ]
    },
    {
      id: 'availability',
      name: 'Service Downtime',
      impact: 6200000,
      likelihood: 45,
      revenueImpact: 8.9,
      mitigation: 'Auto-healing infrastructure',
      status: 'resolved',
      trend: 'decreasing',
      incidents: [
        { date: '2024-02-10', cost: 120000, description: '2-hour outage - Region failure' },
        { date: '2024-02-18', cost: 85000, description: 'API gateway degradation' }
      ]
    },
    {
      id: 'cost',
      name: 'Cloud Cost Overruns',
      impact: 3400000,
      likelihood: 78,
      revenueImpact: 4.9,
      mitigation: 'FinOps automation',
      status: 'active',
      trend: 'decreasing',
      incidents: [
        { date: '2024-01-05', cost: 95000, description: 'Untagged resources' },
        { date: '2024-01-20', cost: 110000, description: 'Orphaned volumes' }
      ]
    },
    {
      id: 'reputation',
      name: 'Reputation Damage',
      impact: 12000000,
      likelihood: 25,
      revenueImpact: 17.3,
      mitigation: 'Proactive monitoring',
      status: 'monitoring',
      trend: 'stable',
      incidents: []
    }
  ]

  const revenueStreams = [
    { name: 'Enterprise SaaS', revenue: 45000000, atRisk: 8.2 },
    { name: 'Professional Services', revenue: 18000000, atRisk: 3.5 },
    { name: 'Marketplace Sales', revenue: 12000000, atRisk: 5.1 },
    { name: 'Support Contracts', revenue: 8000000, atRisk: 2.3 }
  ]

  const riskMatrix = [
    { impact: 'Critical', likelihood: 'Very High', color: 'bg-red-600', risks: [] },
    { impact: 'Critical', likelihood: 'High', color: 'bg-red-500', risks: ['security'] },
    { impact: 'Critical', likelihood: 'Medium', color: 'bg-orange-500', risks: [] },
    { impact: 'Critical', likelihood: 'Low', color: 'bg-yellow-500', risks: ['reputation'] },
    { impact: 'High', likelihood: 'Very High', color: 'bg-red-500', risks: ['compliance'] },
    { impact: 'High', likelihood: 'High', color: 'bg-orange-500', risks: [] },
    { impact: 'High', likelihood: 'Medium', color: 'bg-orange-400', risks: ['availability'] },
    { impact: 'High', likelihood: 'Low', color: 'bg-yellow-400', risks: [] },
    { impact: 'Medium', likelihood: 'Very High', color: 'bg-orange-400', risks: ['cost'] },
    { impact: 'Medium', likelihood: 'High', color: 'bg-yellow-500', risks: [] },
    { impact: 'Medium', likelihood: 'Medium', color: 'bg-yellow-400', risks: [] },
    { impact: 'Medium', likelihood: 'Low', color: 'bg-green-400', risks: [] },
    { impact: 'Low', likelihood: 'Very High', color: 'bg-yellow-400', risks: [] },
    { impact: 'Low', likelihood: 'High', color: 'bg-yellow-300', risks: [] },
    { impact: 'Low', likelihood: 'Medium', color: 'bg-green-400', risks: [] },
    { impact: 'Low', likelihood: 'Low', color: 'bg-green-500', risks: [] }
  ]

  const totalRevenue = revenueStreams.reduce((sum, stream) => sum + stream.revenue, 0)
  const totalAtRisk = riskCategories.reduce((sum, risk) => sum + (risk.impact * risk.likelihood / 100), 0)

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-3 bg-gradient-to-r from-red-600 to-orange-600 rounded-lg">
              <AlertTriangle className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Risk-to-Revenue Map
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Visualize how risks impact your revenue streams
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <MetricCard
            title="Total Revenue"
            value={`$${(totalRevenue / 1000000).toFixed(1)}M`}
            subtitle="Annual recurring"
            icon={<DollarSign className="w-5 h-5 text-green-500" />}
          />
          <MetricCard
            title="Revenue at Risk"
            value={`$${(totalAtRisk / 1000000).toFixed(1)}M`}
            subtitle={`${((totalAtRisk / totalRevenue) * 100).toFixed(1)}% of total`}
            alert="High exposure"
            icon={<TrendingDown className="w-5 h-5 text-red-500" />}
          />
          <MetricCard
            title="Active Risks"
            value={riskCategories.filter(r => r.status !== 'resolved').length}
            subtitle="Requiring attention"
            icon={<AlertTriangle className="w-5 h-5 text-orange-500" />}
          />
          <MetricCard
            title="Risk Reduction"
            value="67%"
            subtitle="With PolicyCortex"
            trend="up"
            icon={<Shield className="w-5 h-5 text-blue-500" />}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Risk Heat Map</h3>
                <div className="flex gap-2">
                  <button
                    onClick={() => setViewMode('impact')}
                    className={`px-3 py-1 rounded text-sm ${
                      viewMode === 'impact' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    Impact View
                  </button>
                  <button
                    onClick={() => setViewMode('likelihood')}
                    className={`px-3 py-1 rounded text-sm ${
                      viewMode === 'likelihood' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    Likelihood View
                  </button>
                </div>
              </div>
              
              <div className="grid grid-cols-4 gap-1">
                <div className="text-xs text-gray-600 dark:text-gray-400 text-center"></div>
                <div className="text-xs text-gray-600 dark:text-gray-400 text-center">Low</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 text-center">Medium</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 text-center">High</div>
                
                <div className="text-xs text-gray-600 dark:text-gray-400 pr-2 text-right">Critical</div>
                <div className="aspect-square bg-yellow-500 rounded flex items-center justify-center">
                  {riskMatrix.find(m => m.impact === 'Critical' && m.likelihood === 'Low')?.risks.length || 0}
                </div>
                <div className="aspect-square bg-orange-500 rounded flex items-center justify-center"></div>
                <div className="aspect-square bg-red-500 rounded flex items-center justify-center text-white">
                  1
                </div>
                
                <div className="text-xs text-gray-600 dark:text-gray-400 pr-2 text-right">High</div>
                <div className="aspect-square bg-yellow-400 rounded flex items-center justify-center"></div>
                <div className="aspect-square bg-orange-400 rounded flex items-center justify-center text-white">
                  1
                </div>
                <div className="aspect-square bg-red-500 rounded flex items-center justify-center text-white">
                  1
                </div>
                
                <div className="text-xs text-gray-600 dark:text-gray-400 pr-2 text-right">Medium</div>
                <div className="aspect-square bg-green-400 rounded flex items-center justify-center"></div>
                <div className="aspect-square bg-yellow-400 rounded flex items-center justify-center"></div>
                <div className="aspect-square bg-orange-400 rounded flex items-center justify-center text-white">
                  1
                </div>
                
                <div className="text-xs text-gray-600 dark:text-gray-400 pr-2 text-right">Low</div>
                <div className="aspect-square bg-green-500 rounded flex items-center justify-center"></div>
                <div className="aspect-square bg-green-400 rounded flex items-center justify-center"></div>
                <div className="aspect-square bg-yellow-300 rounded flex items-center justify-center"></div>
              </div>
              
              <div className="mt-4 text-xs text-center text-gray-600 dark:text-gray-400">
                Likelihood →
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Revenue Stream Exposure</h3>
              <div className="space-y-4">
                {revenueStreams.map((stream, idx) => (
                  <div key={idx} className="space-y-2">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-medium">{stream.name}</div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          ${(stream.revenue / 1000000).toFixed(1)}M annual revenue
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`font-semibold ${
                          stream.atRisk > 5 ? 'text-red-600' : stream.atRisk > 3 ? 'text-orange-600' : 'text-green-600'
                        }`}>
                          {stream.atRisk}% at risk
                        </div>
                        <div className="text-xs text-gray-500">
                          ${((stream.revenue * stream.atRisk / 100) / 1000000).toFixed(1)}M exposed
                        </div>
                      </div>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          stream.atRisk > 5 ? 'bg-red-500' : stream.atRisk > 3 ? 'bg-orange-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${stream.atRisk * 10}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Risk Categories</h3>
              <div className="space-y-3">
                {riskCategories.map((risk) => (
                  <div
                    key={risk.id}
                    className={`p-3 rounded-lg border-2 cursor-pointer transition-all ${
                      selectedRisk === risk.id
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'
                    }`}
                    onClick={() => setSelectedRisk(risk.id)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <div className="font-medium">{risk.name}</div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {risk.mitigation}
                        </div>
                      </div>
                      <div className={`px-2 py-1 rounded text-xs font-medium ${
                        risk.status === 'resolved' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                        risk.status === 'mitigating' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400' :
                        risk.status === 'active' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' :
                        'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400'
                      }`}>
                        {risk.status.toUpperCase()}
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Impact: </span>
                        <span className="font-medium text-red-600">
                          ${(risk.impact / 1000000).toFixed(1)}M
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Likelihood: </span>
                        <span className="font-medium">{risk.likelihood}%</span>
                      </div>
                    </div>
                    
                    <div className="mt-2">
                      <div className="text-xs text-gray-600 dark:text-gray-400">Revenue Impact</div>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div 
                            className="bg-gradient-to-r from-red-500 to-orange-500 h-2 rounded-full"
                            style={{ width: `${risk.revenueImpact * 5}%` }}
                          />
                        </div>
                        <span className="text-xs font-medium">{risk.revenueImpact}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {selectedRisk && (
              <div className="bg-gradient-to-r from-red-600 to-orange-600 rounded-lg p-6 text-white">
                <h3 className="text-lg font-semibold mb-2">Mitigation Strategy</h3>
                <p className="text-sm opacity-90 mb-4">
                  PolicyCortex reduces {riskCategories.find(r => r.id === selectedRisk)?.name} risk by 67% through:
                </p>
                <ul className="text-sm space-y-1 opacity-90">
                  <li>• AI-powered predictive analytics</li>
                  <li>• Automated policy enforcement</li>
                  <li>• Real-time threat detection</li>
                  <li>• Continuous compliance monitoring</li>
                </ul>
              </div>
            )}
          </div>
        </div>

        <div className="mt-6 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold">Executive Risk Summary</h3>
              <p className="text-sm opacity-90 mt-1">
                Current revenue exposure: ${(totalAtRisk / 1000000).toFixed(1)}M • 
                With PolicyCortex: ${(totalAtRisk * 0.33 / 1000000).toFixed(1)}M • 
                Risk reduction: ${(totalAtRisk * 0.67 / 1000000).toFixed(1)}M annually
              </p>
            </div>
            <button className="px-4 py-2 bg-white text-blue-600 rounded-lg hover:bg-gray-100 transition-colors">
              Download Risk Report
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}