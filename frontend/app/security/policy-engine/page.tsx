/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2026 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Settings,
  Shield,
  FileText,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Play,
  Pause,
  RefreshCw,
  Plus,
  Edit,
  Trash2,
  Copy,
  GitBranch,
  Zap,
  Clock,
  Target,
  ChevronRight,
  Search,
  Filter,
  Download,
  Eye,
  MoreVertical,
  Code,
  Activity,
  TrendingUp,
  TrendingDown,
  Calendar,
  Users,
  Bell,
  History,
  Database,
  Globe,
  Server,
  ChevronDown,
  X,
  Check,
  Info,
  ExternalLink,
  BookOpen,
  Layers,
  BarChart3,
  PieChart,
  Gauge
} from 'lucide-react'
import { Line, Bar, Doughnut } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
)

interface Policy {
  id: string
  name: string
  description: string
  type: 'preventive' | 'detective' | 'corrective'
  category: string
  scope: string
  enabled: boolean
  enforcement: 'enforce' | 'audit' | 'disabled'
  rules: number
  resources: number
  violations: number
  lastModified: string
  createdBy: string
  automationEnabled: boolean
  aiOptimized: boolean
}

interface PolicyExecution {
  id: string
  policyId: string
  policyName: string
  status: 'running' | 'completed' | 'failed' | 'scheduled'
  startTime: string
  duration: string
  resourcesScanned: number
  violationsFound: number
  remediations: number
  progress?: number
  logs: string[]
  triggerType: 'manual' | 'scheduled' | 'event'
  executedBy: string
}

interface PolicyRule {
  id: string
  name: string
  description: string
  condition: string
  action: string
  parameters: Record<string, any>
  enabled: boolean
}

interface PolicyTemplate {
  id: string
  name: string
  description: string
  category: string
  type: string
  rules: PolicyRule[]
  usageCount: number
}

interface PolicyMetrics {
  totalPolicies: number
  activePolicies: number
  totalViolations: number
  resolvedViolations: number
  totalExecutions: number
  successfulExecutions: number
  averageExecutionTime: number
  resourcesCovered: number
  complianceScore: number
  trendsData: Array<{
    date: string
    violations: number
    executions: number
    complianceScore: number
  }>
}

export default function PolicyEnginePage() {
  const [policies, setPolicies] = useState<Policy[]>([])
  const [executions, setExecutions] = useState<PolicyExecution[]>([])
  const [templates, setTemplates] = useState<PolicyTemplate[]>([])
  const [metrics, setMetrics] = useState<PolicyMetrics | null>(null)
  const [selectedTab, setSelectedTab] = useState<'overview' | 'policies' | 'executions' | 'templates' | 'editor' | 'analytics'>('overview')
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedPolicy, setSelectedPolicy] = useState<Policy | null>(null)
  const [showPolicyDetail, setShowPolicyDetail] = useState(false)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [realTimeData, setRealTimeData] = useState<any[]>([])

  useEffect(() => {
    setTimeout(() => {
      setPolicies([
        {
          id: 'pol-001',
          name: 'Require HTTPS for All Storage',
          description: 'Enforce HTTPS-only access for all storage accounts',
          type: 'preventive',
          category: 'Security',
          scope: 'All Subscriptions',
          enabled: true,
          enforcement: 'enforce',
          rules: 3,
          resources: 156,
          violations: 2,
          lastModified: '2 days ago',
          createdBy: 'security-team',
          automationEnabled: true,
          aiOptimized: true
        },
        {
          id: 'pol-002',
          name: 'Tag Compliance Policy',
          description: 'Ensure all resources have required tags',
          type: 'detective',
          category: 'Governance',
          scope: 'Production',
          enabled: true,
          enforcement: 'audit',
          rules: 5,
          resources: 423,
          violations: 34,
          lastModified: '1 week ago',
          createdBy: 'governance-team',
          automationEnabled: false,
          aiOptimized: false
        },
        {
          id: 'pol-003',
          name: 'VM Backup Policy',
          description: 'Automatically configure backup for all VMs',
          type: 'corrective',
          category: 'Operations',
          scope: 'Critical Resources',
          enabled: true,
          enforcement: 'enforce',
          rules: 2,
          resources: 89,
          violations: 5,
          lastModified: '3 days ago',
          createdBy: 'ops-team',
          automationEnabled: true,
          aiOptimized: true
        },
        {
          id: 'pol-004',
          name: 'Network Security Rules',
          description: 'Block public access to databases and key vaults',
          type: 'preventive',
          category: 'Network',
          scope: 'All Subscriptions',
          enabled: true,
          enforcement: 'enforce',
          rules: 8,
          resources: 234,
          violations: 0,
          lastModified: '1 month ago',
          createdBy: 'security-team',
          automationEnabled: true,
          aiOptimized: false
        },
        {
          id: 'pol-005',
          name: 'Cost Optimization Policy',
          description: 'Identify and remove idle resources',
          type: 'detective',
          category: 'Cost',
          scope: 'Development',
          enabled: false,
          enforcement: 'disabled',
          rules: 4,
          resources: 0,
          violations: 0,
          lastModified: '2 weeks ago',
          createdBy: 'finance-team',
          automationEnabled: true,
          aiOptimized: true
        }
      ])

      setExecutions([
        {
          id: 'exec-001',
          policyId: 'pol-001',
          policyName: 'Require HTTPS for All Storage',
          status: 'completed',
          startTime: '10 minutes ago',
          duration: '2m 34s',
          resourcesScanned: 156,
          violationsFound: 2,
          remediations: 2
        },
        {
          id: 'exec-002',
          policyId: 'pol-002',
          policyName: 'Tag Compliance Policy',
          status: 'running',
          startTime: '5 minutes ago',
          duration: '5m 12s',
          resourcesScanned: 312,
          violationsFound: 28,
          remediations: 0
        },
        {
          id: 'exec-003',
          policyId: 'pol-003',
          policyName: 'VM Backup Policy',
          status: 'failed',
          startTime: '1 hour ago',
          duration: '45s',
          resourcesScanned: 45,
          violationsFound: 5,
          remediations: 3
        }
      ])

      setLoading(false)
    }, 1000)
  }, [])

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'preventive': return 'bg-blue-500/20 text-blue-400'
      case 'detective': return 'bg-yellow-500/20 text-yellow-400'
      case 'corrective': return 'bg-green-500/20 text-green-400'
      default: return 'bg-gray-500/20 text-gray-400'
    }
  }

  const getEnforcementColor = (enforcement: string) => {
    switch (enforcement) {
      case 'enforce': return 'text-red-400'
      case 'audit': return 'text-yellow-400'
      case 'disabled': return 'text-gray-400'
      default: return 'text-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-400" />
      case 'failed': return <XCircle className="w-4 h-4 text-red-400" />
      default: return null
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': case 'success': return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'failed': case 'error': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'running': case 'in-progress': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      case 'scheduled': case 'pending': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  const filteredPolicies = policies.filter(policy =>
    searchTerm === '' ||
    policy.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    policy.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
    policy.category.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const filteredExecutions = executions.filter(execution =>
    searchTerm === '' ||
    execution.policyName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    execution.executedBy.toLowerCase().includes(searchTerm.toLowerCase())
  )

  // Chart data
  const complianceTrendData = {
    labels: realTimeData.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [
      {
        label: 'Compliance Score',
        data: realTimeData.map(d => d.complianceScore),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4
      }
    ]
  }

  const violationsTrendData = {
    labels: realTimeData.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [
      {
        label: 'Violations',
        data: realTimeData.map(d => d.violations),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4
      },
      {
        label: 'Executions',
        data: realTimeData.map(d => d.executions),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4
      }
    ]
  }

  const policyDistributionData = {
    labels: ['Security', 'Governance', 'Operations', 'Cost', 'Network', 'Reliability'],
    datasets: [{
      data: [
        policies.filter(p => p.category === 'Security').length,
        policies.filter(p => p.category === 'Governance').length,
        policies.filter(p => p.category === 'Operations').length,
        policies.filter(p => p.category === 'Cost').length,
        policies.filter(p => p.category === 'Network').length,
        policies.filter(p => p.category === 'Reliability').length
      ],
      backgroundColor: [
        'rgba(239, 68, 68, 0.8)',
        'rgba(251, 146, 60, 0.8)',
        'rgba(251, 191, 36, 0.8)',
        'rgba(34, 197, 94, 0.8)',
        'rgba(59, 130, 246, 0.8)',
        'rgba(168, 85, 247, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const totalPolicies = policies.length
  const enabledPolicies = policies.filter(p => p.enabled).length
  const totalViolations = policies.reduce((sum, p) => sum + p.violations, 0)
  const aiOptimized = policies.filter(p => p.aiOptimized).length

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading Policy Engine...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="bg-gray-950 border-b border-gray-800 sticky top-0 z-50">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Settings className="w-8 h-8 text-purple-500" />
              <div>
                <h1 className="text-2xl font-bold">Policy Engine</h1>
                <p className="text-sm text-gray-500">Automated policy management and enforcement</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">ENGINE OPERATIONAL</span>
              </div>
              <button 
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`p-2 rounded ${autoRefresh ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400'}`}
              >
                <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              </button>
              <select 
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value)}
                className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="1h">Last Hour</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
              <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded transition-colors flex items-center space-x-2">
                <Plus className="w-4 h-4" />
                <span>Create Policy</span>
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'policies', 'executions', 'templates', 'editor', 'analytics'].map((tab) => (
            <button
              key={tab}
              onClick={() => setSelectedTab(tab as any)}
              className={`py-3 px-1 border-b-2 transition-colors capitalize ${selectedTab === tab
                  ? 'border-purple-500 text-purple-500'
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </header>

      <div className="p-6">
        {selectedTab === 'overview' && metrics && (
          <>
            {/* Overview Metrics */}
            <div className="grid grid-cols-6 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <FileText className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Total Policies</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.totalPolicies}</p>
                <p className="text-xs text-gray-500 mt-1">{metrics.activePolicies} active</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-red-500" />
                  <span className="text-xs text-gray-500">Violations</span>
                </div>
                <p className="text-2xl font-bold font-mono text-red-500">{metrics.totalViolations}</p>
                <p className="text-xs text-gray-500 mt-1">{metrics.resolvedViolations} resolved</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Activity className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Executions</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.totalExecutions}</p>
                <p className="text-xs text-gray-500 mt-1">{Math.round((metrics.successfulExecutions / metrics.totalExecutions) * 100)}% success</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Clock className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Avg Time</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.averageExecutionTime}s</p>
                <p className="text-xs text-gray-500 mt-1">execution time</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Server className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Resources</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.resourcesCovered}</p>
                <p className="text-xs text-gray-500 mt-1">covered</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Gauge className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Compliance</span>
                </div>
                <p className="text-2xl font-bold font-mono text-green-500">{metrics.complianceScore}%</p>
                <div className="mt-2 h-1 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-green-500 rounded-full" style={{ width: `${metrics.complianceScore}%` }} />
                </div>
              </motion.div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {/* Compliance Trend */}
              <div className="col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">POLICY COMPLIANCE TREND</h3>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      <span className="text-xs text-gray-500">Compliance Score</span>
                    </div>
                  </div>
                </div>
                <div className="h-64">
                  <Line data={complianceTrendData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false }
                    },
                    scales: {
                      x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                      },
                      y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' },
                        min: 85,
                        max: 100
                      }
                    }
                  }} />
                </div>
              </div>

              {/* Policy Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">POLICY DISTRIBUTION</h3>
                <div className="h-64 flex items-center justify-center">
                  <Doughnut data={policyDistributionData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: { 
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 10 }
                        }
                      }
                    }
                  }} />
                </div>
              </div>
            </div>

            {/* Quick Stats Grid */}
            <div className="grid grid-cols-4 gap-6 mb-6">
              {/* AI Optimization */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">AI OPTIMIZATION</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">AI Optimized</span>
                    <span className="font-mono text-purple-500">{policies.filter(p => p.aiOptimized).length}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Automated</span>
                    <span className="font-mono text-blue-500">{policies.filter(p => p.automationEnabled).length}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Manual</span>
                    <span className="font-mono text-gray-400">{policies.filter(p => !p.automationEnabled).length}</span>
                  </div>
                </div>
              </div>

              {/* Recent Activity */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">RECENT ACTIVITY</h3>
                </div>
                <div className="p-4 space-y-2">
                  {executions.slice(0, 4).map((exec, index) => (
                    <div key={exec.id} className="flex items-center justify-between text-xs">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(exec.status)}
                        <span className="text-gray-400 truncate">{exec.policyName.substring(0, 15)}...</span>
                      </div>
                      <span className="text-gray-500">{exec.startTime}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Violation Trends */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">VIOLATION TRENDS</h3>
                </div>
                <div className="p-4">
                  <div className="h-24">
                    <Line data={violationsTrendData} options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: { legend: { display: false } },
                      scales: {
                        x: { display: false },
                        y: { display: false }
                      },
                      elements: {
                        point: { radius: 0 }
                      }
                    }} />
                  </div>
                </div>
              </div>

              {/* Templates */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">POLICY TEMPLATES</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Available</span>
                    <span className="font-mono text-white">{templates.length}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Most Used</span>
                    <span className="text-xs text-blue-400">Security Baseline</span>
                  </div>
                  <div className="pt-2 border-t border-gray-800">
                    <button className="w-full px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm">
                      Browse Templates
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {selectedTab === 'policies' && (
          <>
            {/* Search and Filters */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <input
                    type="text"
                    placeholder="Search policies..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                <select className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                  <option value="">All Categories</option>
                  <option value="security">Security</option>
                  <option value="governance">Governance</option>
                  <option value="cost">Cost</option>
                  <option value="compliance">Compliance</option>
                </select>
                <select className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                  <option value="">All Status</option>
                  <option value="enabled">Enabled</option>
                  <option value="disabled">Disabled</option>
                </select>
                <button className="p-2 bg-gray-800 border border-gray-700 rounded-lg hover:bg-gray-700">
                  <Filter className="w-4 h-4 text-gray-400" />
                </button>
              </div>
              <div className="flex items-center space-x-2">
                <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg flex items-center space-x-2">
                  <Download className="w-4 h-4" />
                  <span>Export</span>
                </button>
                <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg flex items-center space-x-2">
                  <Plus className="w-4 h-4" />
                  <span>Create Policy</span>
                </button>
              </div>
            </div>

            {/* Policies Grid */}
            <div className="space-y-4">
              {filteredPolicies.map((policy, index) => (
                <motion.div
                  key={policy.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-gray-900 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors"
                >
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-start gap-4">
                        <div className={`p-3 rounded-lg ${
                          policy.enabled ? 'bg-green-500/20' : 'bg-gray-500/20'
                        }`}>
                          <Shield className={`w-6 h-6 ${
                            policy.enabled ? 'text-green-400' : 'text-gray-400'
                          }`} />
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                            {policy.name}
                            {policy.aiOptimized && (
                              <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded border border-purple-500/30">
                                AI
                              </span>
                            )}
                            {policy.automationEnabled && (
                              <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 text-xs rounded border border-blue-500/30">
                                AUTO
                              </span>
                            )}
                          </h3>
                          <p className="text-sm text-gray-400 mt-1">{policy.description}</p>
                          <div className="flex items-center gap-4 mt-2">
                            <span className={`px-2 py-1 rounded text-xs ${getTypeColor(policy.type)}`}>
                              {policy.type}
                            </span>
                            <span className="text-xs text-gray-400">{policy.category}</span>
                            <span className="text-xs text-gray-400">Scope: {policy.scope}</span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className={`text-sm font-medium ${getEnforcementColor(policy.enforcement)}`}>
                          {policy.enforcement.toUpperCase()}
                        </span>
                        <button
                          className={`p-2 rounded-lg transition-colors ${
                            policy.enabled
                              ? 'bg-green-500/20 hover:bg-green-500/30 text-green-400'
                              : 'bg-gray-500/20 hover:bg-gray-500/30 text-gray-400'
                          }`}
                        >
                          {policy.enabled ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-4 mb-4">
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Rules</p>
                        <p className="text-lg font-semibold text-white">{policy.rules}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Resources</p>
                        <p className="text-lg font-semibold text-white">{policy.resources}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Violations</p>
                        <p className={`text-lg font-semibold ${
                          policy.violations > 0 ? 'text-red-400' : 'text-green-400'
                        }`}>
                          {policy.violations}
                        </p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Last Run</p>
                        <p className="text-sm text-white">{policy.lastModified}</p>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-4 border-t border-gray-700">
                      <div className="flex items-center space-x-4">
                        <span className="text-xs text-gray-400">
                          Modified {policy.lastModified} by {policy.createdBy}
                        </span>
                        <span className={`px-2 py-1 text-xs rounded border ${policy.enabled ? 'bg-green-500/20 text-green-400 border-green-500/30' : 'bg-gray-500/20 text-gray-400 border-gray-500/30'}`}>
                          {policy.enabled ? 'ACTIVE' : 'INACTIVE'}
                        </span>
                      </div>
                      <div className="flex gap-2">
                        <button className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white">
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white">
                          <Edit className="w-4 h-4" />
                        </button>
                        <button className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white">
                          <Copy className="w-4 h-4" />
                        </button>
                        <button className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white">
                          <Play className="w-4 h-4" />
                        </button>
                        <button className="p-1.5 hover:bg-gray-700 rounded">
                          <MoreVertical className="w-4 h-4 text-gray-400" />
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        )}

        {selectedTab === 'executions' && (
          <>
            {/* Execution Overview */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <Activity className="w-5 h-5 text-blue-500" />
                  <span className="text-2xl font-bold text-blue-500">{executions.filter(e => e.status === 'running').length}</span>
                </div>
                <p className="text-gray-400 text-sm">Running</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-2xl font-bold text-green-500">{executions.filter(e => e.status === 'completed').length}</span>
                </div>
                <p className="text-gray-400 text-sm">Completed</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <XCircle className="w-5 h-5 text-red-500" />
                  <span className="text-2xl font-bold text-red-500">{executions.filter(e => e.status === 'failed').length}</span>
                </div>
                <p className="text-gray-400 text-sm">Failed</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <Clock className="w-5 h-5 text-yellow-500" />
                  <span className="text-2xl font-bold text-yellow-500">{executions.filter(e => e.status === 'scheduled').length}</span>
                </div>
                <p className="text-gray-400 text-sm">Scheduled</p>
              </div>
            </div>

            {/* Executions Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                <h3 className="text-sm font-bold text-gray-400 uppercase">POLICY EXECUTIONS</h3>
                <div className="flex items-center space-x-2">
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                    <span className="text-xs text-gray-500">Live Updates</span>
                  </div>
                  <button className="p-1.5 hover:bg-gray-800 rounded">
                    <Download className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Policy</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Started</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Duration</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Resources</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Violations</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Fixed</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Triggered By</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {filteredExecutions.map((execution) => (
                      <motion.tr
                        key={execution.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div className="font-medium text-white">{execution.policyName}</div>
                          <div className="text-xs text-gray-500">ID: {execution.policyId}</div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-2">
                            {getStatusIcon(execution.status)}
                            <span className={`px-2 py-1 text-xs rounded border ${getStatusColor(execution.status)}`}>
                              {execution.status.toUpperCase()}
                            </span>
                          </div>
                          {execution.progress !== undefined && execution.progress < 100 && (
                            <div className="mt-1 w-full bg-gray-700 rounded-full h-1">
                              <div className="bg-blue-600 h-1 rounded-full" style={{ width: `${execution.progress}%` }}></div>
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm text-white">{execution.startTime}</div>
                          <div className="text-xs text-gray-500">{execution.triggerType}</div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm text-white font-mono">{execution.duration}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm text-white font-mono">{execution.resourcesScanned}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`text-sm font-mono ${execution.violationsFound > 0 ? 'text-red-400' : 'text-green-400'}`}>
                            {execution.violationsFound}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono text-green-400">{execution.remediations}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm text-white">{execution.executedBy}</div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button className="p-1 hover:bg-gray-700 rounded">
                              <Eye className="w-4 h-4 text-gray-400" />
                            </button>
                            {execution.status === 'running' && (
                              <button className="p-1 hover:bg-red-700 rounded text-red-400">
                                <X className="w-4 h-4" />
                              </button>
                            )}
                            {execution.status === 'failed' && (
                              <button className="p-1 hover:bg-blue-700 rounded text-blue-400">
                                <RefreshCw className="w-4 h-4" />
                              </button>
                            )}
                            <button className="p-1 hover:bg-gray-700 rounded">
                              <MoreVertical className="w-4 h-4 text-gray-400" />
                            </button>
                          </div>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {selectedTab === 'templates' && (
          <>
            {/* Templates Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {templates.map((template, index) => (
                <motion.div
                  key={template.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-gray-900 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors"
                >
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-white">{template.name}</h3>
                        <p className="text-sm text-gray-400 mt-1">{template.description}</p>
                      </div>
                      <span className={`px-2 py-1 text-xs rounded border ${getTypeColor(template.type)}`}>
                        {template.category}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400">Rules</p>
                        <p className="text-xl font-semibold text-white">{template.rules.length}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400">Usage</p>
                        <p className="text-xl font-semibold text-blue-400">{template.usageCount}</p>
                      </div>
                    </div>

                    <div className="mb-4">
                      <p className="text-xs text-gray-400 mb-2">Sample Rules</p>
                      {template.rules.slice(0, 2).map((rule) => (
                        <div key={rule.id} className="bg-gray-800 rounded p-2 mb-2">
                          <p className="text-xs font-medium text-white">{rule.name}</p>
                          <p className="text-xs text-gray-500">{rule.description}</p>
                        </div>
                      ))}
                    </div>

                    <div className="flex gap-2">
                      <button className="flex-1 px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded text-white text-sm">
                        Use Template
                      </button>
                      <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-white text-sm">
                        <Eye className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        )}

        {selectedTab === 'analytics' && (
          <>
            {/* Analytics Dashboard */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              {/* Violations Trend */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">VIOLATIONS & EXECUTIONS TREND</h3>
                <div className="h-64">
                  <Line data={violationsTrendData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: { 
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12 }
                        }
                      }
                    },
                    scales: {
                      x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                      },
                      y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                      }
                    }
                  }} />
                </div>
              </div>

              {/* Policy Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">POLICY DISTRIBUTION BY CATEGORY</h3>
                <div className="h-64 flex items-center justify-center">
                  <Doughnut data={policyDistributionData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'right',
                        labels: { 
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12 }
                        }
                      }
                    }
                  }} />
                </div>
              </div>
            </div>
          </>
        )}

        {selectedTab === 'editor' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="text-center py-12">
              <Code className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">Policy Editor</h3>
              <p className="text-gray-400 mb-4">Visual policy editor and code interface</p>
              <button className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium">
                Launch Editor
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}