/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2026 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  BarChart3,
  Activity,
  Target,
  Brain,
  Sparkles,
  Filter,
  Download,
  RefreshCw,
  ChevronRight,
  ArrowUp,
  ArrowDown,
  Minus
} from 'lucide-react'

interface Prediction {
  id: string
  resource: string
  type: string
  prediction: string
  confidence: number
  impact: 'High' | 'Medium' | 'Low'
  timeline: string
  currentValue: number
  predictedValue: number
  trend: 'up' | 'down' | 'stable'
  recommendations: string[]
  status: 'active' | 'monitoring' | 'resolved'
}

export default function PredictionsPage() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedImpact, setSelectedImpact] = useState('all')
  const [timeRange, setTimeRange] = useState('7d')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate loading predictions
    setTimeout(() => {
      setPredictions([
        {
          id: '1',
          resource: 'Production VMs',
          type: 'Cost Overrun',
          prediction: 'Expected 32% cost increase in next billing cycle',
          confidence: 92,
          impact: 'High',
          timeline: '5 days',
          currentValue: 12500,
          predictedValue: 16500,
          trend: 'up',
          recommendations: [
            'Enable auto-scaling policies',
            'Implement reserved instances',
            'Review VM sizing recommendations'
          ],
          status: 'active'
        },
        {
          id: '2',
          resource: 'Storage Account - West US',
          type: 'Capacity Threshold',
          prediction: 'Storage capacity will reach 90% in 3 days',
          confidence: 88,
          impact: 'Medium',
          timeline: '3 days',
          currentValue: 78,
          predictedValue: 90,
          trend: 'up',
          recommendations: [
            'Enable auto-expansion',
            'Archive old data to cool storage',
            'Implement data lifecycle policies'
          ],
          status: 'active'
        },
        {
          id: '3',
          resource: 'SQL Database Cluster',
          type: 'Performance Degradation',
          prediction: 'Query performance will degrade by 45% during peak hours',
          confidence: 76,
          impact: 'High',
          timeline: '12 hours',
          currentValue: 120,
          predictedValue: 180,
          trend: 'up',
          recommendations: [
            'Scale up compute tier',
            'Optimize query patterns',
            'Enable read replicas'
          ],
          status: 'monitoring'
        },
        {
          id: '4',
          resource: 'Network Security Groups',
          type: 'Compliance Drift',
          prediction: '12 NSG rules will become non-compliant',
          confidence: 95,
          impact: 'High',
          timeline: '2 days',
          currentValue: 98,
          predictedValue: 86,
          trend: 'down',
          recommendations: [
            'Review pending configuration changes',
            'Update compliance policies',
            'Enable drift detection alerts'
          ],
          status: 'active'
        },
        {
          id: '5',
          resource: 'AKS Cluster - Production',
          type: 'Resource Exhaustion',
          prediction: 'Node pool will reach capacity limits',
          confidence: 81,
          impact: 'Medium',
          timeline: '7 days',
          currentValue: 72,
          predictedValue: 95,
          trend: 'up',
          recommendations: [
            'Enable cluster autoscaler',
            'Add additional node pools',
            'Optimize pod resource requests'
          ],
          status: 'monitoring'
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

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <ArrowUp className="w-4 h-4 text-red-400" />
      case 'down': return <ArrowDown className="w-4 h-4 text-green-400" />
      default: return <Minus className="w-4 h-4 text-gray-400" />
    }
  }

  const filteredPredictions = predictions.filter(p => {
    if (selectedCategory !== 'all' && !p.type.toLowerCase().includes(selectedCategory)) return false
    if (selectedImpact !== 'all' && p.impact !== selectedImpact) return false
    return true
  })

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
            <TrendingUp className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Predictive Analytics</h1>
            <p className="text-gray-400 mt-1">AI-powered predictions and trend analysis</p>
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
            <AlertTriangle className="w-8 h-8 text-red-400" />
            <span className="text-2xl font-bold text-white">23</span>
          </div>
          <p className="text-gray-400 text-sm">Active Predictions</p>
          <p className="text-xs text-red-400 mt-1">↑ 15% from last week</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <Brain className="w-8 h-8 text-purple-400" />
            <span className="text-2xl font-bold text-white">87%</span>
          </div>
          <p className="text-gray-400 text-sm">Avg Confidence</p>
          <p className="text-xs text-green-400 mt-1">↑ 3% improvement</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <CheckCircle className="w-8 h-8 text-green-400" />
            <span className="text-2xl font-bold text-white">156</span>
          </div>
          <p className="text-gray-400 text-sm">Prevented Issues</p>
          <p className="text-xs text-green-400 mt-1">This month</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <Activity className="w-8 h-8 text-blue-400" />
            <span className="text-2xl font-bold text-white">$45K</span>
          </div>
          <p className="text-gray-400 text-sm">Costs Avoided</p>
          <p className="text-xs text-blue-400 mt-1">Based on predictions</p>
        </motion.div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 mb-6">
        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="all">All Categories</option>
          <option value="cost">Cost</option>
          <option value="performance">Performance</option>
          <option value="compliance">Compliance</option>
          <option value="capacity">Capacity</option>
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

        <select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="24h">Next 24 Hours</option>
          <option value="7d">Next 7 Days</option>
          <option value="30d">Next 30 Days</option>
          <option value="90d">Next 90 Days</option>
        </select>

        <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors flex items-center gap-2">
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>

        <button className="px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white transition-colors flex items-center gap-2">
          <Download className="w-4 h-4" />
          Export
        </button>
      </div>

      {/* Predictions List */}
      <div className="space-y-4">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          filteredPredictions.map((prediction, index) => (
            <motion.div
              key={prediction.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 overflow-hidden hover:bg-white/15 transition-colors"
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-purple-500/20 rounded-lg">
                      <Sparkles className="w-6 h-6 text-purple-400" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-1">{prediction.resource}</h3>
                      <p className="text-purple-400 text-sm mb-2">{prediction.type}</p>
                      <p className="text-gray-300">{prediction.prediction}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${getImpactColor(prediction.impact)}`}>
                      {prediction.impact} Impact
                    </span>
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-400">{prediction.timeline}</span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="bg-black/20 rounded-lg p-3">
                    <p className="text-xs text-gray-400 mb-1">Confidence</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-black/30 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
                          style={{ width: `${prediction.confidence}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-white">{prediction.confidence}%</span>
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-3">
                    <p className="text-xs text-gray-400 mb-1">Current → Predicted</p>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-white">
                        {typeof prediction.currentValue === 'number' && prediction.currentValue > 1000
                          ? `$${(prediction.currentValue / 1000).toFixed(1)}K`
                          : `${prediction.currentValue}%`}
                      </span>
                      {getTrendIcon(prediction.trend)}
                      <span className="text-sm font-medium text-white">
                        {typeof prediction.predictedValue === 'number' && prediction.predictedValue > 1000
                          ? `$${(prediction.predictedValue / 1000).toFixed(1)}K`
                          : `${prediction.predictedValue}%`}
                      </span>
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-3">
                    <p className="text-xs text-gray-400 mb-1">Status</p>
                    <div className="flex items-center gap-2">
                      {prediction.status === 'active' && <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse" />}
                      {prediction.status === 'monitoring' && <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />}
                      {prediction.status === 'resolved' && <div className="w-2 h-2 bg-green-400 rounded-full" />}
                      <span className="text-sm font-medium text-white capitalize">{prediction.status}</span>
                    </div>
                  </div>
                </div>

                <div className="border-t border-white/10 pt-4">
                  <h4 className="text-sm font-medium text-white mb-2">AI Recommendations</h4>
                  <div className="space-y-2">
                    {prediction.recommendations.map((rec, idx) => (
                      <div key={idx} className="flex items-start gap-2">
                        <ChevronRight className="w-4 h-4 text-purple-400 mt-0.5" />
                        <p className="text-sm text-gray-300">{rec}</p>
                      </div>
                    ))}
                  </div>
                  <div className="flex gap-3 mt-4">
                    <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white text-sm transition-colors">
                      Apply Recommendations
                    </button>
                    <button className="px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white text-sm transition-colors">
                      View Details
                    </button>
                    <button className="px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white text-sm transition-colors">
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          ))
        )}
      </div>
    </div>
  )
}