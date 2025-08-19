/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2026 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  Lock,
  Eye,
  Activity,
  TrendingUp,
  Users,
  Key,
  FileCheck,
  AlertCircle,
  ShieldCheck,
  ShieldOff,
  Zap,
  BarChart3,
  Clock,
  Globe,
  Server,
  Database,
  TrendingDown,
  AlertOctagon,
  UserCheck,
  FileWarning,
  Fingerprint,
  Network,
  Cpu,
  HardDrive,
  Wifi,
  Cloud,
  GitBranch,
  Terminal,
  Code,
  Bug,
  Search,
  Filter,
  Download,
  Upload,
  RefreshCw,
  Settings,
  MoreVertical,
  ChevronRight,
  ExternalLink,
  Play,
  Pause,
  StopCircle
} from 'lucide-react'
import { Line, Bar, Doughnut, Radar } from 'react-chartjs-2'
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
  ArcElement,
  RadialLinearScale,
  Filler
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
  ArcElement,
  RadialLinearScale,
  Filler
)

export default function SecurityOverviewPage() {
  const [loading, setLoading] = useState(true)
  const [securityScore, setSecurityScore] = useState(0)
  const [threats, setThreats] = useState<any[]>([])
  const [metrics, setMetrics] = useState<any>(null)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [activeTab, setActiveTab] = useState('overview')
  const [realTimeData, setRealTimeData] = useState<any[]>([])
  const [complianceData, setComplianceData] = useState<any>(null)
  const [riskMatrix, setRiskMatrix] = useState<any[]>([])
  const [autoRefresh, setAutoRefresh] = useState(true)

  useEffect(() => {
    loadAllData()
    const interval = autoRefresh ? setInterval(loadRealTimeData, 5000) : null
    return () => { if (interval) clearInterval(interval) }
  }, [autoRefresh])

  const loadAllData = () => {
    setTimeout(() => {
      setSecurityScore(92)
      setMetrics({
        activeThreats: 7,
        vulnerabilities: { critical: 2, high: 5, medium: 16, low: 34 },
        blockedAttempts: 1247,
        compliance: 94,
        incidents: 3,
        users: { total: 2456, active: 1834, privileged: 89, external: 234 },
        devices: { managed: 3421, unmanaged: 156, compliant: 3265, nonCompliant: 156 },
        dataProtection: { encrypted: 98.5, classified: 87.2, backed: 99.9 },
        networkSecurity: { firewallRules: 234, securityGroups: 67, openPorts: 12 }
      })
      
      setThreats([
        { id: 1, type: 'SQL Injection', source: '185.220.101.45', target: 'API Gateway', severity: 'critical', status: 'active', timestamp: new Date(), confidence: 95 },
        { id: 2, type: 'Brute Force', source: '45.142.182.112', target: 'SSH Service', severity: 'high', status: 'mitigating', timestamp: new Date(), confidence: 87 },
        { id: 3, type: 'DDoS Attempt', source: 'Multiple', target: 'Load Balancer', severity: 'medium', status: 'blocked', timestamp: new Date(), confidence: 92 },
        { id: 4, type: 'Privilege Escalation', source: 'Internal', target: 'AD Controller', severity: 'critical', status: 'investigating', timestamp: new Date(), confidence: 78 },
        { id: 5, type: 'Data Exfiltration', source: '203.0.113.42', target: 'Storage Account', severity: 'high', status: 'contained', timestamp: new Date(), confidence: 83 }
      ])

      setComplianceData({
        frameworks: [
          { name: 'SOC 2', score: 94, controls: { passed: 112, failed: 7, total: 119 } },
          { name: 'ISO 27001', score: 91, controls: { passed: 89, failed: 9, total: 98 } },
          { name: 'HIPAA', score: 96, controls: { passed: 64, failed: 3, total: 67 } },
          { name: 'PCI DSS', score: 88, controls: { passed: 45, failed: 6, total: 51 } },
          { name: 'GDPR', score: 93, controls: { passed: 78, failed: 6, total: 84 } }
        ]
      })

      setRiskMatrix([
        { category: 'Data Breach', likelihood: 'Medium', impact: 'Critical', score: 75 },
        { category: 'Insider Threat', likelihood: 'Low', impact: 'High', score: 45 },
        { category: 'Ransomware', likelihood: 'High', impact: 'Critical', score: 90 },
        { category: 'Supply Chain', likelihood: 'Medium', impact: 'High', score: 60 },
        { category: 'Cloud Misconfiguration', likelihood: 'High', impact: 'Medium', score: 65 }
      ])

      setRealTimeData(generateRealTimeData())
      setLoading(false)
    }, 500)
  }

  const loadRealTimeData = () => {
    setRealTimeData(prev => {
      const newData = [...prev, {
        timestamp: new Date(),
        threats: Math.floor(Math.random() * 10),
        blocked: Math.floor(Math.random() * 50),
        allowed: Math.floor(Math.random() * 200)
      }]
      return newData.slice(-20)
    })
  }

  const generateRealTimeData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 60000),
      threats: Math.floor(Math.random() * 10),
      blocked: Math.floor(Math.random() * 50),
      allowed: Math.floor(Math.random() * 200)
    }))
  }

  const threatTrendData = {
    labels: realTimeData.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [
      {
        label: 'Threats Detected',
        data: realTimeData.map(d => d.threats),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4
      },
      {
        label: 'Blocked Attempts',
        data: realTimeData.map(d => d.blocked),
        borderColor: 'rgb(251, 191, 36)',
        backgroundColor: 'rgba(251, 191, 36, 0.1)',
        tension: 0.4
      }
    ]
  }

  const vulnerabilityData = {
    labels: ['Critical', 'High', 'Medium', 'Low'],
    datasets: [{
      data: [
        metrics?.vulnerabilities.critical || 0,
        metrics?.vulnerabilities.high || 0,
        metrics?.vulnerabilities.medium || 0,
        metrics?.vulnerabilities.low || 0
      ],
      backgroundColor: [
        'rgba(239, 68, 68, 0.8)',
        'rgba(251, 146, 60, 0.8)',
        'rgba(251, 191, 36, 0.8)',
        'rgba(163, 163, 163, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const securityPostureData = {
    labels: ['Access Control', 'Data Protection', 'Network Security', 'Compliance', 'Incident Response', 'Monitoring'],
    datasets: [{
      label: 'Current State',
      data: [88, 92, 85, 94, 78, 90],
      backgroundColor: 'rgba(59, 130, 246, 0.2)',
      borderColor: 'rgb(59, 130, 246)',
      pointBackgroundColor: 'rgb(59, 130, 246)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgb(59, 130, 246)'
    }]
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Initializing Security Operations Center...</p>
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
              <Shield className="w-8 h-8 text-green-500" />
              <div>
                <h1 className="text-2xl font-bold">Security Operations Center</h1>
                <p className="text-sm text-gray-500">Real-time threat monitoring and response</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">SYSTEMS OPERATIONAL</span>
              </div>
              <button 
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`p-2 rounded ${autoRefresh ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400'}`}
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
              <button className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded transition-colors">
                INCIDENT RESPONSE
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'threats', 'compliance', 'risk', 'incidents'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-3 px-1 border-b-2 transition-colors capitalize ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-500'
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </header>

      <div className="p-6">
        {activeTab === 'overview' && (
          <>
            {/* Security Score Card */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 mb-6">
              <div className="grid grid-cols-3 gap-6">
                <div>
                  <h2 className="text-xl font-bold mb-4">Overall Security Posture</h2>
                  <div className="flex items-center space-x-4">
                    <div className="relative w-32 h-32">
                      <svg className="w-full h-full transform -rotate-90">
                        <circle cx="64" cy="64" r="56" stroke="rgba(255,255,255,0.1)" strokeWidth="8" fill="none" />
                        <circle
                          cx="64" cy="64" r="56"
                          stroke="url(#scoreGradient)"
                          strokeWidth="8"
                          fill="none"
                          strokeDasharray={`${2 * Math.PI * 56}`}
                          strokeDashoffset={`${2 * Math.PI * 56 * (1 - securityScore / 100)}`}
                          className="transition-all duration-1000"
                        />
                        <defs>
                          <linearGradient id="scoreGradient">
                            <stop offset="0%" stopColor="#10b981" />
                            <stop offset="100%" stopColor="#3b82f6" />
                          </linearGradient>
                        </defs>
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="text-3xl font-bold">{securityScore}%</div>
                          <div className="text-xs text-gray-500">Score</div>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <TrendingUp className="w-4 h-4 text-green-500" />
                        <span className="text-sm text-green-500">+3% from last week</span>
                      </div>
                      <div className="text-sm text-gray-400">
                        <div>Last Assessment: 2 hours ago</div>
                        <div>Next Review: Tomorrow 9:00 AM</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="border-l border-gray-800 pl-6">
                  <h3 className="text-sm font-semibold text-gray-400 mb-3">THREAT LANDSCAPE</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Active Threats</span>
                      <span className="text-sm font-mono text-red-500">{metrics.activeThreats}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Blocked Today</span>
                      <span className="text-sm font-mono text-green-500">{metrics.blockedAttempts}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Open Incidents</span>
                      <span className="text-sm font-mono text-yellow-500">{metrics.incidents}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Mean Time to Respond</span>
                      <span className="text-sm font-mono">12m 34s</span>
                    </div>
                  </div>
                </div>

                <div className="border-l border-gray-800 pl-6">
                  <h3 className="text-sm font-semibold text-gray-400 mb-3">QUICK ACTIONS</h3>
                  <div className="space-y-2">
                    <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group">
                      <span>Run Security Scan</span>
                      <Play className="w-4 h-4 text-gray-500 group-hover:text-white" />
                    </button>
                    <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group">
                      <span>Update Firewall Rules</span>
                      <Shield className="w-4 h-4 text-gray-500 group-hover:text-white" />
                    </button>
                    <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group">
                      <span>Review Access Logs</span>
                      <FileCheck className="w-4 h-4 text-gray-500 group-hover:text-white" />
                    </button>
                    <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group">
                      <span>Generate Report</span>
                      <Download className="w-4 h-4 text-gray-500 group-hover:text-white" />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-6 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-red-500" />
                  <span className="text-xs text-gray-500">Threats</span>
                </div>
                <p className="text-2xl font-bold font-mono text-red-500">{metrics.activeThreats}</p>
                <p className="text-xs text-red-400 mt-1">2 critical</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Bug className="w-5 h-5 text-orange-500" />
                  <span className="text-xs text-gray-500">Vulnerabilities</span>
                </div>
                <div className="flex items-center space-x-1 text-sm">
                  <span className="text-red-500 font-bold">{metrics.vulnerabilities.critical}C</span>
                  <span className="text-orange-500 font-bold">{metrics.vulnerabilities.high}H</span>
                  <span className="text-yellow-500">{metrics.vulnerabilities.medium}M</span>
                </div>
                <p className="text-xs text-gray-500 mt-1">57 total</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <ShieldCheck className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Blocked</span>
                </div>
                <p className="text-2xl font-bold font-mono text-green-500">{metrics.blockedAttempts}</p>
                <p className="text-xs text-gray-500 mt-1">attacks today</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <FileCheck className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Compliance</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.compliance}%</p>
                <div className="mt-2 h-1 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500 rounded-full" style={{ width: `${metrics.compliance}%` }} />
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <AlertCircle className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Incidents</span>
                </div>
                <p className="text-2xl font-bold font-mono text-yellow-500">{metrics.incidents}</p>
                <p className="text-xs text-gray-500 mt-1">under review</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Users className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Users</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.users.active}</p>
                <p className="text-xs text-gray-500 mt-1">active now</p>
              </motion.div>
            </div>

            {/* Charts and Threats Grid */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {/* Real-time Threat Monitoring */}
              <div className="col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">THREAT ACTIVITY</h3>
                  <div className="flex items-center space-x-2">
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-red-500 rounded-full" />
                      <span className="text-xs text-gray-500">Threats</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-yellow-500 rounded-full" />
                      <span className="text-xs text-gray-500">Blocked</span>
                    </div>
                  </div>
                </div>
                <div className="h-64">
                  <Line data={threatTrendData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false },
                      tooltip: {
                        mode: 'index',
                        intersect: false,
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

              {/* Vulnerability Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">VULNERABILITY BREAKDOWN</h3>
                <div className="h-64 flex items-center justify-center">
                  <Doughnut data={vulnerabilityData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: { color: 'rgba(255, 255, 255, 0.7)' }
                      }
                    }
                  }} />
                </div>
              </div>
            </div>

            {/* Active Threats Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                <h3 className="text-sm font-bold text-gray-400 uppercase">ACTIVE THREAT DETECTION</h3>
                <div className="flex items-center space-x-2">
                  <button className="p-1.5 hover:bg-gray-800 rounded">
                    <Filter className="w-4 h-4 text-gray-500" />
                  </button>
                  <button className="p-1.5 hover:bg-gray-800 rounded">
                    <Search className="w-4 h-4 text-gray-500" />
                  </button>
                  <button className="p-1.5 hover:bg-gray-800 rounded">
                    <Download className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Severity</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Threat Type</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Source</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Target</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Confidence</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Time</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {threats.map((threat) => (
                      <motion.tr
                        key={threat.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            threat.severity === 'critical' ? 'text-red-500' :
                            threat.severity === 'high' ? 'text-orange-500' :
                            threat.severity === 'medium' ? 'text-yellow-500' :
                            'text-gray-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              threat.severity === 'critical' ? 'bg-red-500 animate-pulse' :
                              threat.severity === 'high' ? 'bg-orange-500' :
                              threat.severity === 'medium' ? 'bg-yellow-500' :
                              'bg-gray-500'
                            }`} />
                            <span className="uppercase">{threat.severity}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="font-medium">{threat.type}</div>
                        </td>
                        <td className="px-4 py-3">
                          <code className="text-xs bg-gray-800 px-2 py-1 rounded">{threat.source}</code>
                        </td>
                        <td className="px-4 py-3 text-sm">{threat.target}</td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex px-2 py-1 text-xs rounded ${
                            threat.status === 'active' ? 'bg-red-900/30 text-red-500' :
                            threat.status === 'mitigating' ? 'bg-yellow-900/30 text-yellow-500' :
                            threat.status === 'investigating' ? 'bg-blue-900/30 text-blue-500' :
                            threat.status === 'contained' ? 'bg-purple-900/30 text-purple-500' :
                            'bg-green-900/30 text-green-500'
                          }`}>
                            {threat.status.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <div className="w-16 bg-gray-800 rounded-full h-2">
                              <div 
                                className={`h-2 rounded-full ${
                                  threat.confidence > 80 ? 'bg-green-500' :
                                  threat.confidence > 60 ? 'bg-yellow-500' :
                                  'bg-red-500'
                                }`}
                                style={{ width: `${threat.confidence}%` }}
                              />
                            </div>
                            <span className="text-xs text-gray-500">{threat.confidence}%</span>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-xs text-gray-500">
                          {threat.timestamp.toLocaleTimeString()}
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button className="p-1 hover:bg-gray-700 rounded">
                              <Eye className="w-4 h-4 text-gray-400" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded">
                              <Shield className="w-4 h-4 text-gray-400" />
                            </button>
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

            {/* Security Services and Additional Info */}
            <div className="grid grid-cols-4 gap-6 mt-6">
              {/* Security Services Status */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">SECURITY SERVICES</h3>
                </div>
                <div className="p-4 space-y-3">
                  {[
                    { name: 'Azure Sentinel', status: 'active', health: 100, icon: Eye },
                    { name: 'Azure Firewall', status: 'active', health: 100, icon: Shield },
                    { name: 'DDoS Protection', status: 'active', health: 98, icon: Network },
                    { name: 'Key Vault', status: 'active', health: 100, icon: Key },
                    { name: 'Security Center', status: 'active', health: 99, icon: Lock },
                    { name: 'WAF', status: 'active', health: 97, icon: Globe },
                    { name: 'Defender', status: 'active', health: 100, icon: ShieldCheck }
                  ].map((service, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 bg-gray-800 rounded hover:bg-gray-700 transition-colors">
                      <div className="flex items-center space-x-2">
                        <service.icon className="w-4 h-4 text-gray-500" />
                        <span className="text-sm">{service.name}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${
                          service.health === 100 ? 'bg-green-500' : 
                          service.health > 95 ? 'bg-yellow-500' : 
                          'bg-red-500'
                        }`} />
                        <span className="text-xs text-gray-500">{service.health}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Identity & Access */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">IDENTITY & ACCESS</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Total Users</span>
                    <span className="font-mono">{metrics.users.total}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Active Now</span>
                    <span className="font-mono text-green-500">{metrics.users.active}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Privileged</span>
                    <span className="font-mono text-yellow-500">{metrics.users.privileged}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">External</span>
                    <span className="font-mono">{metrics.users.external}</span>
                  </div>
                  <div className="pt-2 border-t border-gray-800">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-400">MFA Enabled</span>
                      <span className="text-sm text-green-500">87%</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full">
                      <div className="h-2 bg-green-500 rounded-full" style={{ width: '87%' }} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Device Compliance */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">DEVICE COMPLIANCE</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Managed</span>
                    <span className="font-mono">{metrics.devices.managed}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Compliant</span>
                    <span className="font-mono text-green-500">{metrics.devices.compliant}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Non-Compliant</span>
                    <span className="font-mono text-red-500">{metrics.devices.nonCompliant}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Unmanaged</span>
                    <span className="font-mono text-yellow-500">{metrics.devices.unmanaged}</span>
                  </div>
                  <div className="pt-2 border-t border-gray-800">
                    <button className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                      Enforce Compliance
                    </button>
                  </div>
                </div>
              </div>

              {/* Data Protection */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">DATA PROTECTION</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-400">Encrypted</span>
                      <span className="text-sm">{metrics.dataProtection.encrypted}%</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full">
                      <div className="h-2 bg-blue-500 rounded-full" style={{ width: `${metrics.dataProtection.encrypted}%` }} />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-400">Classified</span>
                      <span className="text-sm">{metrics.dataProtection.classified}%</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full">
                      <div className="h-2 bg-purple-500 rounded-full" style={{ width: `${metrics.dataProtection.classified}%` }} />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-400">Backed Up</span>
                      <span className="text-sm">{metrics.dataProtection.backed}%</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full">
                      <div className="h-2 bg-green-500 rounded-full" style={{ width: `${metrics.dataProtection.backed}%` }} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'threats' && (
          <div className="space-y-6">
            {/* Threat Intelligence Dashboard */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h2 className="text-xl font-bold mb-4">Threat Intelligence Dashboard</h2>
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-gray-800 rounded-lg p-4">
                  <h3 className="text-sm text-gray-400 mb-2">Attack Vectors</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Web Application</span>
                      <span className="text-red-500">45%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Email/Phishing</span>
                      <span className="text-orange-500">28%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Network</span>
                      <span className="text-yellow-500">18%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Insider</span>
                      <span className="text-gray-500">9%</span>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <h3 className="text-sm text-gray-400 mb-2">Top Threat Actors</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>APT28</span>
                      <span className="text-red-500">High</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Lazarus</span>
                      <span className="text-orange-500">Medium</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>FIN7</span>
                      <span className="text-yellow-500">Medium</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Unknown</span>
                      <span className="text-gray-500">Low</span>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <h3 className="text-sm text-gray-400 mb-2">Geographic Origins</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Russia</span>
                      <span>234</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>China</span>
                      <span>189</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>North Korea</span>
                      <span>67</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Iran</span>
                      <span>45</span>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <h3 className="text-sm text-gray-400 mb-2">MITRE ATT&CK</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>T1566</span>
                      <span className="text-red-500">Phishing</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>T1059</span>
                      <span className="text-orange-500">Command</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>T1055</span>
                      <span className="text-yellow-500">Injection</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>T1003</span>
                      <span className="text-gray-500">Credential</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Threat Map */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-bold mb-4">Global Threat Map</h3>
              <div className="h-96 bg-gray-800 rounded-lg flex items-center justify-center">
                <p className="text-gray-500">Interactive threat map visualization</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'compliance' && (
          <div className="space-y-6">
            {/* Compliance Frameworks */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h2 className="text-xl font-bold mb-4">Compliance Frameworks</h2>
              <div className="grid grid-cols-5 gap-4">
                {complianceData?.frameworks.map((framework: any) => (
                  <div key={framework.name} className="bg-gray-800 rounded-lg p-4">
                    <h3 className="font-semibold mb-2">{framework.name}</h3>
                    <div className="text-3xl font-bold mb-2">{framework.score}%</div>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Passed</span>
                        <span className="text-green-500">{framework.controls.passed}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Failed</span>
                        <span className="text-red-500">{framework.controls.failed}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Total</span>
                        <span>{framework.controls.total}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Compliance Posture */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-bold mb-4">Security Posture Analysis</h3>
              <div className="h-80">
                <Radar data={securityPostureData} options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { display: false }
                  },
                  scales: {
                    r: {
                      beginAtZero: true,
                      max: 100,
                      ticks: { color: 'rgba(255, 255, 255, 0.5)' },
                      grid: { color: 'rgba(255, 255, 255, 0.1)' },
                      pointLabels: { color: 'rgba(255, 255, 255, 0.7)' }
                    }
                  }
                }} />
              </div>
            </div>
          </div>
        )}

        {activeTab === 'risk' && (
          <div className="space-y-6">
            {/* Risk Matrix */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h2 className="text-xl font-bold mb-4">Risk Assessment Matrix</h2>
              <div className="grid grid-cols-5 gap-4 mb-6">
                <div className="col-span-3">
                  <div className="grid grid-cols-4 gap-2">
                    <div className="text-center text-xs text-gray-500">Impact →</div>
                    <div className="text-center text-xs text-gray-500">Low</div>
                    <div className="text-center text-xs text-gray-500">Medium</div>
                    <div className="text-center text-xs text-gray-500">High</div>
                    
                    <div className="text-center text-xs text-gray-500">High ↑</div>
                    <div className="h-24 bg-yellow-900/30 rounded"></div>
                    <div className="h-24 bg-orange-900/30 rounded"></div>
                    <div className="h-24 bg-red-900/30 rounded flex items-center justify-center">
                      <span className="text-xs">3 risks</span>
                    </div>
                    
                    <div className="text-center text-xs text-gray-500">Medium</div>
                    <div className="h-24 bg-green-900/30 rounded"></div>
                    <div className="h-24 bg-yellow-900/30 rounded flex items-center justify-center">
                      <span className="text-xs">2 risks</span>
                    </div>
                    <div className="h-24 bg-orange-900/30 rounded"></div>
                    
                    <div className="text-center text-xs text-gray-500">Low</div>
                    <div className="h-24 bg-green-900/30 rounded"></div>
                    <div className="h-24 bg-green-900/30 rounded"></div>
                    <div className="h-24 bg-yellow-900/30 rounded"></div>
                  </div>
                </div>
                <div className="col-span-2">
                  <h3 className="text-sm font-semibold text-gray-400 mb-3">TOP RISKS</h3>
                  <div className="space-y-2">
                    {riskMatrix.map((risk, idx) => (
                      <div key={idx} className="bg-gray-800 rounded p-3">
                        <div className="flex justify-between items-start">
                          <div>
                            <p className="font-medium text-sm">{risk.category}</p>
                            <p className="text-xs text-gray-500">
                              {risk.likelihood} likelihood × {risk.impact} impact
                            </p>
                          </div>
                          <span className={`text-lg font-bold ${
                            risk.score > 80 ? 'text-red-500' :
                            risk.score > 60 ? 'text-orange-500' :
                            risk.score > 40 ? 'text-yellow-500' :
                            'text-green-500'
                          }`}>
                            {risk.score}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'incidents' && (
          <div className="space-y-6">
            {/* Incident Response */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                <h2 className="text-xl font-bold">Incident Management</h2>
                <button className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm">
                  Create Incident
                </button>
              </div>
              <div className="p-6">
                <div className="grid grid-cols-4 gap-4 mb-6">
                  <div className="bg-gray-800 rounded-lg p-4">
                    <p className="text-xs text-gray-400 mb-1">Open Incidents</p>
                    <p className="text-2xl font-bold">12</p>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-4">
                    <p className="text-xs text-gray-400 mb-1">In Progress</p>
                    <p className="text-2xl font-bold text-yellow-500">5</p>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-4">
                    <p className="text-xs text-gray-400 mb-1">Resolved Today</p>
                    <p className="text-2xl font-bold text-green-500">8</p>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-4">
                    <p className="text-xs text-gray-400 mb-1">Avg Resolution</p>
                    <p className="text-2xl font-bold">2.4h</p>
                  </div>
                </div>

                <div className="space-y-3">
                  {[
                    { id: 'INC-2024-001', title: 'Unauthorized Access Attempt', severity: 'critical', status: 'investigating', assigned: 'Security Team', time: '15 mins ago' },
                    { id: 'INC-2024-002', title: 'Data Exfiltration Alert', severity: 'high', status: 'contained', assigned: 'SOC Analyst', time: '1 hour ago' },
                    { id: 'INC-2024-003', title: 'Malware Detection', severity: 'medium', status: 'remediation', assigned: 'IR Team', time: '3 hours ago' }
                  ].map((incident) => (
                    <div key={incident.id} className="bg-gray-800 rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <span className={`w-2 h-8 rounded-full ${
                            incident.severity === 'critical' ? 'bg-red-500' :
                            incident.severity === 'high' ? 'bg-orange-500' :
                            'bg-yellow-500'
                          }`} />
                          <div>
                            <p className="font-semibold">{incident.id}: {incident.title}</p>
                            <p className="text-sm text-gray-400">
                              Assigned to {incident.assigned} • {incident.time}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 text-xs rounded ${
                            incident.status === 'investigating' ? 'bg-blue-900/30 text-blue-500' :
                            incident.status === 'contained' ? 'bg-purple-900/30 text-purple-500' :
                            'bg-yellow-900/30 text-yellow-500'
                          }`}>
                            {incident.status.toUpperCase()}
                          </span>
                          <button className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm">
                            View Details
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}