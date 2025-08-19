/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2026 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  FileText,
  User,
  Clock,
  Activity,
  Shield,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Info,
  Search,
  Filter,
  Download,
  RefreshCw,
  Eye,
  ChevronRight
} from 'lucide-react'

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
}

export default function AuditTrailPage() {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedResult, setSelectedResult] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => {
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
          severity: 'warning'
        },
        {
          id: 'log-002',
          timestamp: '2024-01-15 14:28:45',
          user: 'john.doe',
          userEmail: 'john.doe@company.com',
          action: 'LOGIN',
          resource: 'Portal',
          resourceType: 'Authentication',
          result: 'success',
          ipAddress: '10.0.0.45',
          location: 'San Francisco, US',
          details: 'User successfully authenticated with MFA',
          category: 'Authentication',
          severity: 'info'
        },
        {
          id: 'log-003',
          timestamp: '2024-01-15 14:15:30',
          user: 'system',
          userEmail: 'system@policycortex',
          action: 'POLICY_VIOLATION',
          resource: 'storage-account-01',
          resourceType: 'Storage Account',
          result: 'failure',
          ipAddress: 'N/A',
          location: 'N/A',
          details: 'Storage account missing required encryption settings',
          category: 'Compliance',
          severity: 'error'
        },
        {
          id: 'log-004',
          timestamp: '2024-01-15 13:45:22',
          user: 'jane.smith',
          userEmail: 'jane.smith@company.com',
          action: 'MODIFY',
          resource: 'nsg-web-tier',
          resourceType: 'Network Security Group',
          result: 'success',
          ipAddress: '172.16.0.100',
          location: 'London, UK',
          details: 'Added new inbound rule allowing HTTPS traffic',
          category: 'Network',
          severity: 'info'
        },
        {
          id: 'log-005',
          timestamp: '2024-01-15 13:20:10',
          user: 'bob.wilson',
          userEmail: 'bob.wilson@company.com',
          action: 'ACCESS_DENIED',
          resource: 'key-vault-prod',
          resourceType: 'Key Vault',
          result: 'failure',
          ipAddress: '192.168.2.50',
          location: 'Tokyo, JP',
          details: 'Access denied: insufficient permissions to access key vault secrets',
          category: 'Security',
          severity: 'critical'
        },
        {
          id: 'log-006',
          timestamp: '2024-01-15 12:55:33',
          user: 'alice.johnson',
          userEmail: 'alice.johnson@company.com',
          action: 'CREATE',
          resource: 'backup-policy-01',
          resourceType: 'Backup Policy',
          result: 'success',
          ipAddress: '10.1.0.200',
          location: 'Sydney, AU',
          details: 'Created new backup policy for production databases',
          category: 'Backup',
          severity: 'info'
        }
      ])
      setLoading(false)
    }, 1000)
  }, [])

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
                          log.details.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesCategory = selectedCategory === 'all' || log.category === selectedCategory
    const matchesResult = selectedResult === 'all' || log.result === selectedResult
    return matchesSearch && matchesCategory && matchesResult
  })

  const totalLogs = logs.length
  const failedActions = logs.filter(l => l.result === 'failure').length
  const criticalEvents = logs.filter(l => l.severity === 'critical' || l.severity === 'error').length

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-4 mb-2">
          <div className="p-3 bg-gradient-to-br from-indigo-500 to-blue-500 rounded-xl">
            <FileText className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Audit Trail</h1>
            <p className="text-gray-400 mt-1">Complete activity log and audit history</p>
          </div>
        </div>
      </motion.div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-6 h-6 text-purple-400" />
            <span className="text-2xl font-bold text-white">{totalLogs}</span>
          </div>
          <p className="text-gray-400 text-sm">Total Events</p>
          <p className="text-xs text-gray-500">Last 24 hours</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <XCircle className="w-6 h-6 text-red-400" />
            <span className="text-2xl font-bold text-white">{failedActions}</span>
          </div>
          <p className="text-gray-400 text-sm">Failed Actions</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <AlertTriangle className="w-6 h-6 text-yellow-400" />
            <span className="text-2xl font-bold text-white">{criticalEvents}</span>
          </div>
          <p className="text-gray-400 text-sm">Critical Events</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <Users className="w-6 h-6 text-blue-400" />
            <span className="text-2xl font-bold text-white">
              {new Set(logs.map(l => l.user)).size}
            </span>
          </div>
          <p className="text-gray-400 text-sm">Active Users</p>
        </motion.div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div className="flex-1 min-w-[300px]">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search logs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
            />
          </div>
        </div>

        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
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
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="all">All Results</option>
          <option value="success">Success</option>
          <option value="failure">Failure</option>
          <option value="warning">Warning</option>
        </select>

        <button className="px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white transition-colors flex items-center gap-2">
          <Download className="w-4 h-4" />
          Export
        </button>
      </div>

      {/* Logs Table */}
      <div className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left p-4 text-sm font-medium text-gray-400">Timestamp</th>
                <th className="text-left p-4 text-sm font-medium text-gray-400">User</th>
                <th className="text-left p-4 text-sm font-medium text-gray-400">Action</th>
                <th className="text-left p-4 text-sm font-medium text-gray-400">Resource</th>
                <th className="text-left p-4 text-sm font-medium text-gray-400">Result</th>
                <th className="text-left p-4 text-sm font-medium text-gray-400">Severity</th>
                <th className="text-left p-4 text-sm font-medium text-gray-400">Details</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={7} className="text-center py-12">
                    <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin mx-auto" />
                  </td>
                </tr>
              ) : (
                filteredLogs.map((log, index) => (
                  <motion.tr
                    key={log.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="border-b border-white/5 hover:bg-white/5 transition-colors"
                  >
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        <Clock className="w-4 h-4 text-gray-400" />
                        <span className="text-sm text-white font-mono">{log.timestamp}</span>
                      </div>
                    </td>
                    <td className="p-4">
                      <div>
                        <p className="text-sm text-white">{log.user}</p>
                        <p className="text-xs text-gray-400">{log.userEmail}</p>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className="text-sm font-medium text-purple-400">{log.action}</span>
                    </td>
                    <td className="p-4">
                      <div>
                        <p className="text-sm text-white">{log.resource}</p>
                        <p className="text-xs text-gray-400">{log.resourceType}</p>
                      </div>
                    </td>
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        {getResultIcon(log.result)}
                        <span className={`text-sm ${getResultColor(log.result)}`}>
                          {log.result}
                        </span>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getSeverityColor(log.severity)}`}>
                        {log.severity}
                      </span>
                    </td>
                    <td className="p-4">
                      <button className="flex items-center gap-1 text-sm text-purple-400 hover:text-purple-300">
                        <Eye className="w-4 h-4" />
                        View
                      </button>
                    </td>
                  </motion.tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}