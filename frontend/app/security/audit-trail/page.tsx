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
  FileText,
  User,
  Clock,
  Activity,
  Shield,
  AlertTriangle,
  CheckCircle,
  X,
  XCircle,
  Info,
  Search,
  Filter,
  Download,
  RefreshCw,
  Eye,
  ChevronRight,
  Calendar,
  MapPin,
  Monitor,
  TrendingUp,
  BarChart3,
  PieChart,
  ExternalLink,
  Settings,
  Bell,
  HelpCircle,
  ChevronDown,
  Globe,
  Lock,
  Database,
  Network,
  Server,
  Zap,
  History
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

interface AuditLog {
  id: string
  timestamp: string
  user: string
  userEmail: string
  action: string
  resource: string
  resourceType: string
  result: 'success' | 'failure' | 'warning'
  ipAddress: string
  location: string
  details: string
  category: string
  severity: 'info' | 'warning' | 'error' | 'critical'
  userAgent?: string
  sessionId?: string
  correlationId?: string
  additionalContext?: Record<string, any>
}

interface AuditMetrics {
  totalEvents: number
  successfulEvents: number
  failedEvents: number
  criticalEvents: number
  uniqueUsers: number
  uniqueResources: number
  topActions: Array<{ action: string; count: number }>
  topUsers: Array<{ user: string; count: number }>
  timelineData: Array<{ time: string; events: number; failures: number }>
  categoryDistribution: Array<{ category: string; count: number }>
  locationData: Array<{ location: string; count: number }>
}

export default function AuditTrailPage() {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [metrics, setMetrics] = useState<AuditMetrics | null>(null)
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedResult, setSelectedResult] = useState('all')
  const [selectedSeverity, setSelectedSeverity] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [dateRange, setDateRange] = useState('24h')
  const [selectedLog, setSelectedLog] = useState<AuditLog | null>(null)
  const [showLogDetail, setShowLogDetail] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage] = useState(20)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadAuditData()
    const interval = autoRefresh ? setInterval(loadAuditData, 30000) : null
    return () => { if (interval) clearInterval(interval) }
  }, [autoRefresh, dateRange])

  const loadAuditData = () => {
    setLoading(true)
    setTimeout(() => {
      // Enhanced audit logs with more comprehensive data
      setLogs([
        {
          id: 'log-001',
          timestamp: '2024-01-15 14:32:15',
          user: 'admin',
          userEmail: 'admin@company.com',
          action: 'DELETE',
          resource: 'vm-prod-01',
          resourceType: 'Virtual Machine',
          result: 'success',
          ipAddress: '192.168.1.100',
          location: 'New York, US',
          details: 'Deleted virtual machine vm-prod-01 from resource group production-rg',
          category: 'Resource Management',
          severity: 'warning',
          userAgent: 'Azure Portal/1.0',
          sessionId: 'sess-789abc',
          correlationId: 'corr-123def',
          additionalContext: {
            resourceGroup: 'production-rg',
            subscriptionId: 'sub-123',
            cost: '$245.50/month'
          }
        },
        ...Array.from({ length: 45 }, (_, i) => ({
          id: `log-${String(i + 2).padStart(3, '0')}`,
          timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString().replace('T', ' ').substring(0, 19),
          user: ['john.doe', 'jane.smith', 'bob.wilson', 'alice.johnson', 'admin', 'system'][Math.floor(Math.random() * 6)],
          userEmail: ['john.doe@company.com', 'jane.smith@company.com', 'bob.wilson@company.com', 'alice.johnson@company.com', 'admin@company.com', 'system@policycortex'][Math.floor(Math.random() * 6)],
          action: ['LOGIN', 'LOGOUT', 'CREATE', 'DELETE', 'MODIFY', 'ACCESS_DENIED', 'POLICY_VIOLATION', 'BACKUP_CREATED'][Math.floor(Math.random() * 8)],
          resource: `resource-${Math.floor(Math.random() * 100).toString().padStart(3, '0')}`,
          resourceType: ['Virtual Machine', 'Storage Account', 'Key Vault', 'Network Security Group', 'Database', 'App Service'][Math.floor(Math.random() * 6)],
          result: ['success', 'failure', 'warning'][Math.floor(Math.random() * 3)] as 'success' | 'failure' | 'warning',
          ipAddress: `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
          location: ['New York, US', 'London, UK', 'Tokyo, JP', 'Sydney, AU', 'San Francisco, US', 'Singapore, SG'][Math.floor(Math.random() * 6)],
          details: 'Auto-generated audit log entry for demonstration purposes',
          category: ['Authentication', 'Resource Management', 'Security', 'Network', 'Compliance', 'Backup'][Math.floor(Math.random() * 6)],
          severity: ['info', 'warning', 'error', 'critical'][Math.floor(Math.random() * 4)] as 'info' | 'warning' | 'error' | 'critical',
          userAgent: ['Azure Portal/1.0', 'Azure CLI/2.0', 'REST API', 'PowerShell/7.0'][Math.floor(Math.random() * 4)],
          sessionId: `sess-${Math.random().toString(36).substring(2, 8)}`,
          correlationId: `corr-${Math.random().toString(36).substring(2, 8)}`,
          additionalContext: {
            requestId: `req-${Math.random().toString(36).substring(2, 8)}`,
            duration: `${Math.floor(Math.random() * 5000)}ms`,
            dataSize: `${Math.floor(Math.random() * 1000)}KB`
          }
        }))
      ])

      // Calculate metrics
      const logsData = logs.length ? logs : []
      setMetrics({
        totalEvents: logsData.length,
        successfulEvents: logsData.filter(l => l.result === 'success').length,
        failedEvents: logsData.filter(l => l.result === 'failure').length,
        criticalEvents: logsData.filter(l => l.severity === 'critical').length,
        uniqueUsers: new Set(logsData.map(l => l.user)).size,
        uniqueResources: new Set(logsData.map(l => l.resource)).size,
        topActions: Object.entries(
          logsData.reduce((acc, log) => {
            acc[log.action] = (acc[log.action] || 0) + 1
            return acc
          }, {} as Record<string, number>)
        ).map(([action, count]) => ({ action, count })).sort((a, b) => b.count - a.count).slice(0, 5),
        topUsers: Object.entries(
          logsData.reduce((acc, log) => {
            acc[log.user] = (acc[log.user] || 0) + 1
            return acc
          }, {} as Record<string, number>)
        ).map(([user, count]) => ({ user, count })).sort((a, b) => b.count - a.count).slice(0, 5),
        timelineData: Array.from({ length: 24 }, (_, i) => {
          const hour = String(i).padStart(2, '0')
          const hourLogs = logsData.filter(l => l.timestamp.includes(`${hour}:`))
          return {
            time: `${hour}:00`,
            events: hourLogs.length,
            failures: hourLogs.filter(l => l.result === 'failure').length
          }
        }),
        categoryDistribution: Object.entries(
          logsData.reduce((acc, log) => {
            acc[log.category] = (acc[log.category] || 0) + 1
            return acc
          }, {} as Record<string, number>)
        ).map(([category, count]) => ({ category, count })),
        locationData: Object.entries(
          logsData.reduce((acc, log) => {
            acc[log.location] = (acc[log.location] || 0) + 1
            return acc
          }, {} as Record<string, number>)
        ).map(([location, count]) => ({ location, count })).sort((a, b) => b.count - a.count).slice(0, 10)
      })

      setLoading(false)
    }, 1000)
  }

  const getResultColor = (result: string) => {
    switch (result) {
      case 'success': return 'text-green-400'
      case 'failure': return 'text-red-400'
      case 'warning': return 'text-yellow-400'
      default: return 'text-gray-400'
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'error': return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
      case 'warning': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'info': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  const getResultIcon = (result: string) => {
    switch (result) {
      case 'success': return <CheckCircle className="w-4 h-4 text-green-400" />
      case 'failure': return <XCircle className="w-4 h-4 text-red-400" />
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-400" />
      default: return <Info className="w-4 h-4 text-gray-400" />
    }
  }

  const filteredLogs = logs.filter(log => {
    const matchesSearch = log.user.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          log.action.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          log.resource.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          log.details.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          log.resourceType.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          log.location.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesCategory = selectedCategory === 'all' || log.category === selectedCategory
    const matchesResult = selectedResult === 'all' || log.result === selectedResult
    const matchesSeverity = selectedSeverity === 'all' || log.severity === selectedSeverity
    return matchesSearch && matchesCategory && matchesResult && matchesSeverity
  })

  // Pagination
  const totalPages = Math.ceil(filteredLogs.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const paginatedLogs = filteredLogs.slice(startIndex, startIndex + itemsPerPage)

  // Chart data
  const timelineChartData = metrics ? {
    labels: metrics.timelineData.map(d => d.time),
    datasets: [
      {
        label: 'Total Events',
        data: metrics.timelineData.map(d => d.events),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4
      },
      {
        label: 'Failed Events',
        data: metrics.timelineData.map(d => d.failures),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4
      }
    ]
  } : null

  const categoryChartData = metrics ? {
    labels: metrics.categoryDistribution.map(d => d.category),
    datasets: [{
      data: metrics.categoryDistribution.map(d => d.count),
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
  } : null

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading audit trail...</p>
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
              <FileText className="w-8 h-8 text-blue-500" />
              <div>
                <h1 className="text-2xl font-bold">Audit Trail</h1>
                <p className="text-sm text-gray-500">Complete activity log and security audit history</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">LIVE MONITORING</span>
              </div>
              <button 
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`p-2 rounded ${autoRefresh ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400'}`}
              >
                <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              </button>
              <select 
                value={dateRange}
                onChange={(e) => setDateRange(e.target.value)}
                className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="1h">Last Hour</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
              <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors flex items-center space-x-2">
                <Download className="w-4 h-4" />
                <span>Export</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="p-6">
        {/* Enhanced Metrics Dashboard */}
        {metrics && (
          <>
            {/* Primary Metrics */}
            <div className="grid grid-cols-6 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Activity className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Total Events</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.totalEvents}</p>
                <p className="text-xs text-gray-500 mt-1">Last {dateRange}</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Successful</span>
                </div>
                <p className="text-2xl font-bold font-mono text-green-500">{metrics.successfulEvents}</p>
                <p className="text-xs text-gray-500 mt-1">{Math.round((metrics.successfulEvents / metrics.totalEvents) * 100)}% success rate</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <XCircle className="w-5 h-5 text-red-500" />
                  <span className="text-xs text-gray-500">Failed</span>
                </div>
                <p className="text-2xl font-bold font-mono text-red-500">{metrics.failedEvents}</p>
                <p className="text-xs text-gray-500 mt-1">{Math.round((metrics.failedEvents / metrics.totalEvents) * 100)}% failure rate</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Critical</span>
                </div>
                <p className="text-2xl font-bold font-mono text-yellow-500">{metrics.criticalEvents}</p>
                <p className="text-xs text-gray-500 mt-1">High priority events</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <User className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Users</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.uniqueUsers}</p>
                <p className="text-xs text-gray-500 mt-1">Active users</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Database className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Resources</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.uniqueResources}</p>
                <p className="text-xs text-gray-500 mt-1">Affected resources</p>
              </motion.div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {/* Timeline Chart */}
              <div className="col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">EVENT TIMELINE (24H)</h3>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full" />
                      <span className="text-xs text-gray-500">Total Events</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-red-500 rounded-full" />
                      <span className="text-xs text-gray-500">Failed Events</span>
                    </div>
                  </div>
                </div>
                <div className="h-64">
                  {timelineChartData && (
                    <Line data={timelineChartData} options={{
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
                          ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                        }
                      }
                    }} />
                  )}
                </div>
              </div>

              {/* Category Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">EVENT CATEGORIES</h3>
                <div className="h-64 flex items-center justify-center">
                  {categoryChartData && (
                    <Doughnut data={categoryChartData} options={{
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
                  )}
                </div>
              </div>
            </div>

            {/* Quick Stats Grid */}
            <div className="grid grid-cols-4 gap-6 mb-6">
              {/* Top Actions */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">TOP ACTIONS</h3>
                </div>
                <div className="p-4 space-y-3">
                  {metrics.topActions.slice(0, 5).map((action, index) => (
                    <div key={action.action} className="flex justify-between items-center">
                      <span className="text-sm text-gray-300">{action.action}</span>
                      <span className="font-mono text-blue-400">{action.count}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Top Users */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">MOST ACTIVE USERS</h3>
                </div>
                <div className="p-4 space-y-3">
                  {metrics.topUsers.slice(0, 5).map((user, index) => (
                    <div key={user.user} className="flex justify-between items-center">
                      <span className="text-sm text-gray-300">{user.user}</span>
                      <span className="font-mono text-purple-400">{user.count}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Top Locations */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">TOP LOCATIONS</h3>
                </div>
                <div className="p-4 space-y-3">
                  {metrics.locationData.slice(0, 5).map((location, index) => (
                    <div key={location.location} className="flex justify-between items-center">
                      <span className="text-sm text-gray-300">{location.location}</span>
                      <span className="font-mono text-green-400">{location.count}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* System Health */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">SYSTEM HEALTH</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Success Rate</span>
                    <span className="font-mono text-green-400">{Math.round((metrics.successfulEvents / metrics.totalEvents) * 100)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Error Rate</span>
                    <span className="font-mono text-red-400">{Math.round((metrics.failedEvents / metrics.totalEvents) * 100)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Critical Rate</span>
                    <span className="font-mono text-yellow-400">{Math.round((metrics.criticalEvents / metrics.totalEvents) * 100)}%</span>
                  </div>
                  <div className="pt-2 border-t border-gray-800">
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div className="bg-green-500 h-2 rounded-full" style={{ width: `${Math.round((metrics.successfulEvents / metrics.totalEvents) * 100)}%` }}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Enhanced Filters */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
              <input
                type="text"
                placeholder="Search logs, users, actions, resources..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent w-80"
              />
            </div>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white"
            >
              <option value="all">All Categories</option>
              <option value="Authentication">Authentication</option>
              <option value="Resource Management">Resource Management</option>
              <option value="Network">Network</option>
              <option value="Security">Security</option>
              <option value="Compliance">Compliance</option>
              <option value="Backup">Backup</option>
            </select>
            <select
              value={selectedResult}
              onChange={(e) => setSelectedResult(e.target.value)}
              className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white"
            >
              <option value="all">All Results</option>
              <option value="success">Success</option>
              <option value="failure">Failure</option>
              <option value="warning">Warning</option>
            </select>
            <select
              value={selectedSeverity}
              onChange={(e) => setSelectedSeverity(e.target.value)}
              className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white"
            >
              <option value="all">All Severities</option>
              <option value="info">Info</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
              <option value="critical">Critical</option>
            </select>
            <button className="p-2 bg-gray-800 border border-gray-700 rounded-lg hover:bg-gray-700">
              <Filter className="w-4 h-4 text-gray-400" />
            </button>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-500">
              Showing {startIndex + 1}-{Math.min(startIndex + itemsPerPage, filteredLogs.length)} of {filteredLogs.length}
            </span>
            <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export CSV</span>
            </button>
          </div>
        </div>

        {/* Enhanced Logs Table */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
          <div className="p-4 border-b border-gray-800 flex items-center justify-between">
            <h3 className="text-sm font-bold text-gray-400 uppercase">AUDIT LOG ENTRIES</h3>
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                <span className="text-xs text-gray-500">Real-time Updates</span>
              </div>
              <button className="p-1.5 hover:bg-gray-800 rounded">
                <Settings className="w-4 h-4 text-gray-500" />
              </button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-800/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Timestamp</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">User</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Action</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Resource</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Result</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Severity</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Location</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {paginatedLogs.map((log, index) => (
                  <motion.tr
                    key={log.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: index * 0.02 }}
                    className="hover:bg-gray-800/30 transition-colors cursor-pointer"
                    onClick={() => {
                      setSelectedLog(log)
                      setShowLogDetail(true)
                    }}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <Clock className="w-4 h-4 text-gray-500" />
                        <span className="text-sm text-white font-mono">{log.timestamp}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div>
                        <p className="text-sm font-medium text-white">{log.user}</p>
                        <p className="text-xs text-gray-500">{log.userEmail}</p>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs rounded border border-blue-500/30">
                        {log.action}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div>
                        <p className="text-sm text-white">{log.resource}</p>
                        <p className="text-xs text-gray-500">{log.resourceType}</p>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        {getResultIcon(log.result)}
                        <span className={`text-sm ${getResultColor(log.result)}`}>
                          {log.result.toUpperCase()}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs font-medium border ${getSeverityColor(log.severity)}`}>
                        {log.severity.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1">
                        <MapPin className="w-3 h-3 text-gray-500" />
                        <span className="text-xs text-gray-400">{log.location}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center space-x-1">
                        <button 
                          className="p-1 hover:bg-gray-700 rounded"
                          onClick={(e) => {
                            e.stopPropagation()
                            setSelectedLog(log)
                            setShowLogDetail(true)
                          }}
                        >
                          <Eye className="w-4 h-4 text-gray-400" />
                        </button>
                        <button className="p-1 hover:bg-gray-700 rounded">
                          <ExternalLink className="w-4 h-4 text-gray-400" />
                        </button>
                        <button className="p-1 hover:bg-gray-700 rounded">
                          <History className="w-4 h-4 text-gray-400" />
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="px-4 py-3 border-t border-gray-800 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-500">
                Page {currentPage} of {totalPages}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 bg-gray-800 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded text-sm"
              >
                Previous
              </button>
              <div className="flex space-x-1">
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  const pageNum = Math.max(1, Math.min(totalPages - 4, currentPage - 2)) + i
                  return pageNum <= totalPages ? (
                    <button
                      key={pageNum}
                      onClick={() => setCurrentPage(pageNum)}
                      className={`px-3 py-1 rounded text-sm ${
                        currentPage === pageNum
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-800 hover:bg-gray-700 text-gray-300'
                      }`}
                    >
                      {pageNum}
                    </button>
                  ) : null
                })}
              </div>
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 bg-gray-800 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded text-sm"
              >
                Next
              </button>
            </div>
          </div>
        </div>

        {/* Log Detail Modal */}
        <AnimatePresence>
          {showLogDetail && selectedLog && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
              onClick={() => setShowLogDetail(false)}
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="bg-gray-900 border border-gray-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-auto"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="p-6 border-b border-gray-800 flex items-center justify-between">
                  <h3 className="text-lg font-bold text-white">Audit Log Details</h3>
                  <button
                    onClick={() => setShowLogDetail(false)}
                    className="p-2 hover:bg-gray-800 rounded"
                  >
                    <X className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
                <div className="p-6">
                  <div className="grid grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div>
                        <h4 className="text-sm font-bold text-gray-400 uppercase mb-2">Event Information</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Event ID:</span>
                            <span className="font-mono text-white">{selectedLog.id}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Timestamp:</span>
                            <span className="font-mono text-white">{selectedLog.timestamp}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Action:</span>
                            <span className="text-blue-400">{selectedLog.action}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Category:</span>
                            <span className="text-white">{selectedLog.category}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Severity:</span>
                            <span className={`px-2 py-1 rounded text-xs ${getSeverityColor(selectedLog.severity)}`}>
                              {selectedLog.severity.toUpperCase()}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div>
                        <h4 className="text-sm font-bold text-gray-400 uppercase mb-2">User Information</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-gray-400">User:</span>
                            <span className="text-white">{selectedLog.user}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Email:</span>
                            <span className="text-white">{selectedLog.userEmail}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">IP Address:</span>
                            <span className="font-mono text-white">{selectedLog.ipAddress}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Location:</span>
                            <span className="text-white">{selectedLog.location}</span>
                          </div>
                          {selectedLog.userAgent && (
                            <div className="flex justify-between">
                              <span className="text-gray-400">User Agent:</span>
                              <span className="text-white">{selectedLog.userAgent}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div>
                        <h4 className="text-sm font-bold text-gray-400 uppercase mb-2">Resource Information</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Resource:</span>
                            <span className="text-white">{selectedLog.resource}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Type:</span>
                            <span className="text-white">{selectedLog.resourceType}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Result:</span>
                            <span className={`${getResultColor(selectedLog.result)}`}>
                              {selectedLog.result.toUpperCase()}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div>
                        <h4 className="text-sm font-bold text-gray-400 uppercase mb-2">Technical Details</h4>
                        <div className="space-y-2">
                          {selectedLog.sessionId && (
                            <div className="flex justify-between">
                              <span className="text-gray-400">Session ID:</span>
                              <span className="font-mono text-white">{selectedLog.sessionId}</span>
                            </div>
                          )}
                          {selectedLog.correlationId && (
                            <div className="flex justify-between">
                              <span className="text-gray-400">Correlation ID:</span>
                              <span className="font-mono text-white">{selectedLog.correlationId}</span>
                            </div>
                          )}
                          {selectedLog.additionalContext && Object.entries(selectedLog.additionalContext).map(([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span className="text-gray-400">{key}:</span>
                              <span className="font-mono text-white">{String(value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="mt-6">
                    <h4 className="text-sm font-bold text-gray-400 uppercase mb-2">Event Details</h4>
                    <div className="bg-gray-800 rounded-lg p-4">
                      <p className="text-white">{selectedLog.details}</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}