/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2026 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Brain,
  Sparkles,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Info,
  Lightbulb,
  Target,
  Zap,
  DollarSign,
  Shield,
  GitBranch,
  BarChart3,
  ArrowRight,
  Clock,
  Filter,
  Download,
  RefreshCw,
  ChevronRight
} from 'lucide-react'

interface PolicyInsight {
  id: string
  title: string
  category: string
  type: 'Optimization' | 'Security' | 'Cost' | 'Compliance' | 'Performance'
  impact: 'High' | 'Medium' | 'Low'
  confidence: number
  description: string
  recommendations: string[]
  affectedResources: string[]
  potentialSavings?: number
  riskReduction?: number
  implementationEffort: 'Low' | 'Medium' | 'High'
  automationAvailable: boolean
  correlatedPatterns: string[]
  detectedAt: string
  status: 'New' | 'Reviewed' | 'In Progress' | 'Implemented' | 'Dismissed'
}

export default function PolicyInsightsPage() {
  const [insights, setInsights] = useState<PolicyInsight[]>([])
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedImpact, setSelectedImpact] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => {
      setInsights([
        {
          id: 'ins-001',
          title: 'Optimize VM sizing across production environment',
          category: 'Resource Optimization',
          type: 'Optimization',
          impact: 'High',
          confidence: 92,
          description: 'AI analysis detected 23 VMs consistently using less than 30% of allocated resources. Right-sizing these instances could reduce costs without impacting performance.',
          recommendations: [
            'Resize 8 D4s_v3 VMs to D2s_v3',
            'Convert 5 B8ms VMs to B4ms',
            'Enable auto-scaling for variable workloads',
            'Implement resource tagging for better tracking'
          ],
          affectedResources: ['prod-web-*', 'api-server-*', 'worker-node-*'],
          potentialSavings: 4500,
          implementationEffort: 'Low',
          automationAvailable: true,
          correlatedPatterns: ['CPU usage pattern', 'Memory utilization trend', 'Network throughput'],
          detectedAt: '2 hours ago',
          status: 'New'
        },
        {
          id: 'ins-002',
          title: 'Security vulnerability pattern detected in NSG rules',
          category: 'Security',
          type: 'Security',
          impact: 'High',
          confidence: 88,
          description: 'Cross-domain analysis revealed a pattern of overly permissive NSG rules that could be exploited. 15 NSGs allow broader access than required.',
          recommendations: [
            'Implement least-privilege access model',
            'Use service tags instead of IP ranges',
            'Enable Azure Firewall for centralized control',
            'Configure NSG flow logs for monitoring'
          ],
          affectedResources: ['public-nsg', 'dmz-nsg', 'app-tier-nsg'],
          riskReduction: 78,
          implementationEffort: 'Medium',
          automationAvailable: true,
          correlatedPatterns: ['Access patterns', 'Traffic anomalies', 'Failed authentication attempts'],
          detectedAt: '1 day ago',
          status: 'Reviewed'
        },
        {
          id: 'ins-003',
          title: 'Predictive storage capacity planning required',
          category: 'Capacity Management',
          type: 'Performance',
          impact: 'Medium',
          confidence: 85,
          description: 'ML models predict storage accounts will reach 90% capacity within 30 days based on current growth patterns. Proactive expansion recommended.',
          recommendations: [
            'Enable auto-expansion for critical storage accounts',
            'Implement lifecycle policies for old data',
            'Archive inactive data to cool/archive tiers',
            'Set up capacity alerts at 75% threshold'
          ],
          affectedResources: ['prod-storage-east', 'backup-storage-west'],
          implementationEffort: 'Low',
          automationAvailable: false,
          correlatedPatterns: ['Data growth rate', 'Backup patterns', 'User activity trends'],
          detectedAt: '3 days ago',
          status: 'In Progress'
        },
        {
          id: 'ins-004',
          title: 'Compliance drift prevention opportunity',
          category: 'Compliance',
          type: 'Compliance',
          impact: 'High',
          confidence: 95,
          description: 'Pattern analysis shows 12 resources likely to become non-compliant with SOC 2 requirements within next audit cycle due to configuration drift.',
          recommendations: [
            'Implement Azure Policy for automatic remediation',
            'Enable continuous compliance monitoring',
            'Configure drift detection alerts',
            'Create compliance dashboard for stakeholders'
          ],
          affectedResources: ['sql-databases', 'storage-accounts', 'key-vaults'],
          riskReduction: 65,
          implementationEffort: 'Medium',
          automationAvailable: true,
          correlatedPatterns: ['Configuration changes', 'Policy violations', 'Audit findings'],
          detectedAt: '1 week ago',
          status: 'Reviewed'
        },
        {
          id: 'ins-005',
          title: 'Cost anomaly detected in data transfer charges',
          category: 'Cost Management',
          type: 'Cost',
          impact: 'Medium',
          confidence: 78,
          description: 'Unusual spike in cross-region data transfer costs detected. Analysis suggests inefficient data routing between services.',
          recommendations: [
            'Colocate frequently communicating services',
            'Implement Azure Front Door for optimal routing',
            'Use private endpoints to reduce costs',
            'Cache frequently accessed data locally'
          ],
          affectedResources: ['cdn-endpoint', 'api-gateway', 'storage-accounts'],
          potentialSavings: 2800,
          implementationEffort: 'High',
          automationAvailable: false,
          correlatedPatterns: ['Traffic patterns', 'Geographic distribution', 'Peak usage times'],
          detectedAt: '5 days ago',
          status: 'New'
        },
        {
          id: 'ins-006',
          title: 'Database performance optimization using AI insights',
          category: 'Performance',
          type: 'Performance',
          impact: 'Medium',
          confidence: 82,
          description: 'AI analysis of query patterns suggests index optimization and query restructuring could improve database performance by 40%.',
          recommendations: [
            'Create missing indexes on frequently queried columns',
            'Optimize top 10 slow queries',
            'Implement query result caching',
            'Consider read replicas for heavy read workloads'
          ],
          affectedResources: ['prod-sql-primary', 'analytics-postgres'],
          implementationEffort: 'Medium',
          automationAvailable: false,
          correlatedPatterns: ['Query patterns', 'Lock contention', 'I/O bottlenecks'],
          detectedAt: '2 weeks ago',
          status: 'Implemented'
        }
      ])
      setLoading(false)
    }, 1000)
  }, [])

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'High': return 'text-red-400 bg-red-500/20'
      case 'Medium': return 'text-yellow-400 bg-yellow-500/20'
      case 'Low': return 'text-green-400 bg-green-500/20'
      default: return 'text-gray-400 bg-gray-500/20'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'Optimization': return TrendingUp
      case 'Security': return Shield
      case 'Cost': return DollarSign
      case 'Compliance': return CheckCircle
      case 'Performance': return Zap
      default: return Info
    }
  }

  const filteredInsights = insights.filter(insight => {
    const matchesSearch = insight.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          insight.description.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesCategory = selectedCategory === 'all' || insight.type === selectedCategory
    const matchesImpact = selectedImpact === 'all' || insight.impact === selectedImpact
    return matchesSearch && matchesCategory && matchesImpact
  })

  const totalSavings = insights.reduce((sum, i) => sum + (i.potentialSavings || 0), 0)
  const newInsights = insights.filter(i => i.status === 'New').length

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-4 mb-2">
          <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">AI Policy Insights</h1>
            <p className="text-gray-400 mt-1">AI-powered recommendations and pattern analysis</p>
          </div>
        </div>
      </motion.div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <Lightbulb className="w-8 h-8 text-yellow-400" />
            <span className="text-2xl font-bold text-white">{insights.length}</span>
          </div>
          <p className="text-gray-400 text-sm">Total Insights</p>
          <p className="text-xs text-yellow-400 mt-1">{newInsights} new</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <DollarSign className="w-8 h-8 text-green-400" />
            <span className="text-2xl font-bold text-white">${totalSavings}</span>
          </div>
          <p className="text-gray-400 text-sm">Potential Savings</p>
          <p className="text-xs text-green-400 mt-1">Per month</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <Brain className="w-8 h-8 text-purple-400" />
            <span className="text-2xl font-bold text-white">87%</span>
          </div>
          <p className="text-gray-400 text-sm">Avg Confidence</p>
          <p className="text-xs text-purple-400 mt-1">ML accuracy</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <Zap className="w-8 h-8 text-orange-400" />
            <span className="text-2xl font-bold text-white">67%</span>
          </div>
          <p className="text-gray-400 text-sm">Automation Ready</p>
          <p className="text-xs text-orange-400 mt-1">Can be auto-applied</p>
        </motion.div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 mb-6">
        <input
          type="text"
          placeholder="Search insights..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
        />
        
        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="all">All Categories</option>
          <option value="Optimization">Optimization</option>
          <option value="Security">Security</option>
          <option value="Cost">Cost</option>
          <option value="Compliance">Compliance</option>
          <option value="Performance">Performance</option>
        </select>

        <select
          value={selectedImpact}
          onChange={(e) => setSelectedImpact(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="all">All Impact Levels</option>
          <option value="High">High Impact</option>
          <option value="Medium">Medium Impact</option>
          <option value="Low">Low Impact</option>
        </select>

        <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors flex items-center gap-2">
          <RefreshCw className="w-4 h-4" />
          Refresh Insights
        </button>

        <button className="px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white transition-colors flex items-center gap-2">
          <Download className="w-4 h-4" />
          Export Report
        </button>
      </div>

      {/* Insights Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {loading ? (
          <div className="col-span-2 flex items-center justify-center py-12">
            <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          filteredInsights.map((insight, index) => {
            const Icon = getTypeIcon(insight.type)
            return (
              <motion.div
                key={insight.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 overflow-hidden hover:bg-white/15 transition-colors"
              >
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-start gap-4">
                      <div className="p-3 bg-purple-500/20 rounded-lg">
                        <Icon className="w-6 h-6 text-purple-400" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-white mb-1 flex items-center gap-2">
                          {insight.title}
                          {insight.automationAvailable && (
                            <Sparkles className="w-4 h-4 text-purple-400" />
                          )}
                        </h3>
                        <p className="text-sm text-gray-400 mb-2">{insight.category}</p>
                        <p className="text-gray-300">{insight.description}</p>
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getImpactColor(insight.impact)}`}>
                        {insight.impact} Impact
                      </span>
                      <div className="flex items-center gap-1">
                        <Brain className="w-3 h-3 text-purple-400" />
                        <span className="text-xs text-purple-400">{insight.confidence}% confidence</span>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 mb-4">
                    {insight.potentialSavings && (
                      <div className="bg-black/20 rounded-lg p-2">
                        <p className="text-xs text-gray-400">Potential Savings</p>
                        <p className="text-sm font-semibold text-green-400">${insight.potentialSavings}/mo</p>
                      </div>
                    )}
                    {insight.riskReduction && (
                      <div className="bg-black/20 rounded-lg p-2">
                        <p className="text-xs text-gray-400">Risk Reduction</p>
                        <p className="text-sm font-semibold text-blue-400">{insight.riskReduction}%</p>
                      </div>
                    )}
                    <div className="bg-black/20 rounded-lg p-2">
                      <p className="text-xs text-gray-400">Implementation</p>
                      <p className="text-sm font-semibold text-white">{insight.implementationEffort} effort</p>
                    </div>
                    <div className="bg-black/20 rounded-lg p-2">
                      <p className="text-xs text-gray-400">Status</p>
                      <p className="text-sm font-semibold text-white">{insight.status}</p>
                    </div>
                  </div>

                  <div className="mb-4">
                    <h4 className="text-sm font-medium text-white mb-2">Recommendations</h4>
                    <div className="space-y-1">
                      {insight.recommendations.slice(0, 2).map((rec, idx) => (
                        <div key={idx} className="flex items-start gap-2">
                          <ChevronRight className="w-4 h-4 text-purple-400 mt-0.5" />
                          <p className="text-sm text-gray-300">{rec}</p>
                        </div>
                      ))}
                      {insight.recommendations.length > 2 && (
                        <p className="text-xs text-gray-400 ml-6">
                          +{insight.recommendations.length - 2} more recommendations
                        </p>
                      )}
                    </div>
                  </div>

                  {insight.correlatedPatterns.length > 0 && (
                    <div className="mb-4">
                      <p className="text-xs text-gray-400 mb-2">Correlated Patterns</p>
                      <div className="flex flex-wrap gap-2">
                        {insight.correlatedPatterns.map((pattern, idx) => (
                          <span key={idx} className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-xs">
                            {pattern}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex items-center justify-between pt-4 border-t border-white/10">
                    <div className="flex items-center gap-2 text-xs text-gray-400">
                      <Clock className="w-3 h-3" />
                      <span>Detected {insight.detectedAt}</span>
                    </div>
                    <div className="flex gap-2">
                      {insight.automationAvailable && insight.status === 'New' && (
                        <button className="px-3 py-1.5 bg-purple-600 hover:bg-purple-700 rounded-lg text-white text-sm transition-colors flex items-center gap-1">
                          <Zap className="w-3 h-3" />
                          Apply
                        </button>
                      )}
                      <button className="px-3 py-1.5 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white text-sm transition-colors">
                        View Details
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            )
          })
        )}
      </div>
    </div>
  )
}