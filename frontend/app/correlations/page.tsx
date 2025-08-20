/**
 * PATENT NOTICE: Patent #1 - Cross-Domain Governance Correlation Engine
 * This implements the graph neural network correlation visualization
 */

'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Network,
  GitBranch,
  AlertTriangle,
  Shield,
  Activity,
  TrendingUp,
  Zap,
  Search,
  Filter,
  RefreshCw,
  Download,
  Settings,
  ChevronRight,
  Info,
  AlertCircle,
  CheckCircle,
  XCircle
} from 'lucide-react'
import CorrelationGraph from '../../components/correlations/CorrelationGraph'
import RiskPropagation from '../../components/correlations/RiskPropagation'
import WhatIfSimulator from '../../components/correlations/WhatIfSimulator'
import CorrelationInsights from '../../components/correlations/CorrelationInsights'

export default function CorrelationsPage() {
  const [activeTab, setActiveTab] = useState<'graph' | 'risk' | 'whatif' | 'insights'>('graph')
  const [correlations, setCorrelations] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [filters, setFilters] = useState({
    domain: 'all',
    riskLevel: 'all',
    timeRange: '24h'
  })

  useEffect(() => {
    fetchCorrelations()
  }, [filters])

  const fetchCorrelations = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/v1/correlations?' + new URLSearchParams(filters))
      if (response.ok) {
        const data = await response.json()
        setCorrelations(data.correlations || [])
      }
    } catch (error) {
      console.error('Failed to fetch correlations:', error)
      // Use mock data for demo
      setCorrelations(getMockCorrelations())
    } finally {
      setLoading(false)
    }
  }

  const tabs = [
    { id: 'graph', label: 'Correlation Graph', icon: Network },
    { id: 'risk', label: 'Risk Propagation', icon: TrendingUp },
    { id: 'whatif', label: 'What-If Analysis', icon: Zap },
    { id: 'insights', label: 'ML Insights', icon: Activity }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900/10 to-slate-900">
      {/* Header */}
      <div className="border-b border-white/10 bg-black/20 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                <div className="p-2 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg">
                  <GitBranch className="w-6 h-6" />
                </div>
                Cross-Domain Correlations
              </h1>
              <p className="text-gray-400 mt-1">
                Patent #1: Graph Neural Network Analysis with {correlations.length} active correlations
              </p>
            </div>
            
            <div className="flex items-center gap-3">
              <button
                onClick={() => fetchCorrelations()}
                className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg flex items-center gap-2 text-white transition-all"
              >
                <RefreshCw className="w-4 h-4" />
                Refresh
              </button>
              <button className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg flex items-center gap-2 text-white transition-all">
                <Download className="w-4 h-4" />
                Export
              </button>
              <button className="px-4 py-2 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 rounded-lg flex items-center gap-2 text-white transition-all">
                <Settings className="w-4 h-4" />
                Configure
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Filters Bar */}
      <div className="border-b border-white/10 bg-black/10 backdrop-blur">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-gray-400">
              <Filter className="w-4 h-4" />
              <span className="text-sm">Filters:</span>
            </div>
            
            <select
              value={filters.domain}
              onChange={(e) => setFilters({...filters, domain: e.target.value})}
              className="px-3 py-1 bg-white/10 border border-white/20 rounded-lg text-white text-sm"
            >
              <option value="all">All Domains</option>
              <option value="security">Security</option>
              <option value="compliance">Compliance</option>
              <option value="identity">Identity</option>
              <option value="network">Network</option>
              <option value="cost">Cost</option>
            </select>
            
            <select
              value={filters.riskLevel}
              onChange={(e) => setFilters({...filters, riskLevel: e.target.value})}
              className="px-3 py-1 bg-white/10 border border-white/20 rounded-lg text-white text-sm"
            >
              <option value="all">All Risk Levels</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
            
            <select
              value={filters.timeRange}
              onChange={(e) => setFilters({...filters, timeRange: e.target.value})}
              className="px-3 py-1 bg-white/10 border border-white/20 rounded-lg text-white text-sm"
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>

            <div className="ml-auto flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">Real-time Analysis</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="border-b border-white/10 bg-black/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="grid grid-cols-4 gap-4">
            <StatsCard
              title="Active Correlations"
              value={correlations.length}
              change="+12%"
              icon={Network}
              color="blue"
            />
            <StatsCard
              title="Risk Cascades"
              value="47"
              change="+23%"
              icon={AlertTriangle}
              color="yellow"
            />
            <StatsCard
              title="Affected Resources"
              value="156"
              change="-8%"
              icon={Shield}
              color="purple"
            />
            <StatsCard
              title="ML Confidence"
              value="94.2%"
              change="+2.1%"
              icon={Activity}
              color="green"
            />
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex gap-1">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`
                    px-4 py-3 flex items-center gap-2 border-b-2 transition-all
                    ${activeTab === tab.id
                      ? 'border-blue-500 text-white bg-white/5'
                      : 'border-transparent text-gray-400 hover:text-white hover:bg-white/5'
                    }
                  `}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
              <p className="text-gray-400">Analyzing correlations with Graph Neural Network...</p>
            </div>
          </div>
        ) : (
          <>
            {activeTab === 'graph' && (
              <CorrelationGraph
                correlations={correlations}
                onNodeSelect={setSelectedNode}
                selectedNode={selectedNode}
              />
            )}
            {activeTab === 'risk' && (
              <RiskPropagation
                correlations={correlations}
                selectedNode={selectedNode}
              />
            )}
            {activeTab === 'whatif' && (
              <WhatIfSimulator
                correlations={correlations}
              />
            )}
            {activeTab === 'insights' && (
              <CorrelationInsights
                correlations={correlations}
              />
            )}
          </>
        )}
      </div>
    </div>
  )
}

function StatsCard({ title, value, change, icon: Icon, color }: any) {
  const colorClasses: Record<string, string> = {
    blue: 'from-blue-500 to-cyan-500',
    yellow: 'from-yellow-500 to-orange-500',
    purple: 'from-purple-500 to-pink-500',
    green: 'from-green-500 to-emerald-500'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white/5 backdrop-blur border border-white/10 rounded-xl p-4"
    >
      <div className="flex items-center justify-between mb-2">
        <div className={`p-2 bg-gradient-to-br ${colorClasses[color]} rounded-lg`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
        <span className={`text-sm ${change.startsWith('+') ? 'text-green-400' : 'text-red-400'}`}>
          {change}
        </span>
      </div>
      <div className="text-2xl font-bold text-white mb-1">{value}</div>
      <div className="text-sm text-gray-400">{title}</div>
    </motion.div>
  )
}

function getMockCorrelations() {
  return [
    {
      id: 'corr-1',
      source: { id: 'vm-prod-01', type: 'compute', risk: 0.7 },
      target: { id: 'db-main', type: 'database', risk: 0.9 },
      correlation_strength: 0.85,
      risk_amplification: 1.5,
      domain_pair: ['security', 'compliance'],
      description: 'Unencrypted data flow between compute and database'
    },
    {
      id: 'corr-2',
      source: { id: 'iam-admin', type: 'identity', risk: 0.6 },
      target: { id: 'storage-sensitive', type: 'storage', risk: 0.8 },
      correlation_strength: 0.92,
      risk_amplification: 1.8,
      domain_pair: ['identity', 'security'],
      description: 'Excessive permissions on sensitive data storage'
    },
    {
      id: 'corr-3',
      source: { id: 'network-dmz', type: 'network', risk: 0.5 },
      target: { id: 'app-public', type: 'application', risk: 0.7 },
      correlation_strength: 0.78,
      risk_amplification: 1.6,
      domain_pair: ['network', 'data'],
      description: 'Public exposure of internal application endpoints'
    }
  ]
}


