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
  AlertTriangle,
  Shield,
  Activity,
  Globe,
  Zap,
  Eye,
  TrendingUp,
  Clock,
  MapPin,
  Wifi,
  Server,
  Database,
  Lock,
  AlertCircle,
  CheckCircle,
  XCircle,
  ChevronRight,
  Filter,
  Download,
  RefreshCw,
  Play,
  Pause
} from 'lucide-react'

interface Threat {
  id: string
  name: string
  type: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  source: {
    ip: string
    location: string
    country: string
    asn: string
  }
  target: {
    resource: string
    type: string
    port: number
  }
  attackVector: string
  attempts: number
  blocked: boolean
  firstSeen: string
  lastSeen: string
  status: 'active' | 'mitigated' | 'monitoring' | 'blocked'
  aiConfidence: number
  relatedIncidents: number
}

interface ThreatStats {
  total: number
  critical: number
  high: number
  medium: number
  low: number
  blocked: number
  active: number
  trend: number
}

export default function ThreatDetectionPage() {
  const [threats, setThreats] = useState<Threat[]>([])
  const [stats, setStats] = useState<ThreatStats | null>(null)
  const [selectedSeverity, setSelectedSeverity] = useState('all')
  const [selectedStatus, setSelectedStatus] = useState('all')
  const [isRealtime, setIsRealtime] = useState(true)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate loading threat data
    setTimeout(() => {
      setStats({
        total: 247,
        critical: 8,
        high: 34,
        medium: 89,
        low: 116,
        blocked: 198,
        active: 12,
        trend: -15
      })

      setThreats([
        {
          id: 'THR-001',
          name: 'Distributed Brute Force Attack',
          type: 'Brute Force',
          severity: 'critical',
          source: {
            ip: '185.220.101.45',
            location: 'Moscow, Russia',
            country: 'RU',
            asn: 'AS13335'
          },
          target: {
            resource: 'SSH Gateway',
            type: 'Network Service',
            port: 22
          },
          attackVector: 'Network',
          attempts: 15234,
          blocked: false,
          firstSeen: '2 hours ago',
          lastSeen: '5 minutes ago',
          status: 'active',
          aiConfidence: 98,
          relatedIncidents: 7
        },
        {
          id: 'THR-002',
          name: 'SQL Injection Campaign',
          type: 'Web Application Attack',
          severity: 'high',
          source: {
            ip: '45.142.182.112',
            location: 'Shanghai, China',
            country: 'CN',
            asn: 'AS4134'
          },
          target: {
            resource: 'API Gateway',
            type: 'Web Service',
            port: 443
          },
          attackVector: 'Application',
          attempts: 892,
          blocked: true,
          firstSeen: '1 day ago',
          lastSeen: '30 minutes ago',
          status: 'blocked',
          aiConfidence: 95,
          relatedIncidents: 3
        },
        {
          id: 'THR-003',
          name: 'Suspicious Data Exfiltration',
          type: 'Data Breach Attempt',
          severity: 'critical',
          source: {
            ip: '10.0.15.234',
            location: 'Internal Network',
            country: 'Local',
            asn: 'Private'
          },
          target: {
            resource: 'Database Cluster',
            type: 'Database',
            port: 5432
          },
          attackVector: 'Insider',
          attempts: 45,
          blocked: false,
          firstSeen: '15 minutes ago',
          lastSeen: 'Just now',
          status: 'active',
          aiConfidence: 87,
          relatedIncidents: 0
        },
        {
          id: 'THR-004',
          name: 'DDoS Attack - Layer 7',
          type: 'Denial of Service',
          severity: 'high',
          source: {
            ip: 'Multiple',
            location: 'Global Botnet',
            country: 'Multiple',
            asn: 'Various'
          },
          target: {
            resource: 'Load Balancer',
            type: 'Infrastructure',
            port: 443
          },
          attackVector: 'Network',
          attempts: 450000,
          blocked: true,
          firstSeen: '3 hours ago',
          lastSeen: '1 hour ago',
          status: 'mitigated',
          aiConfidence: 99,
          relatedIncidents: 12
        },
        {
          id: 'THR-005',
          name: 'Privilege Escalation Attempt',
          type: 'Access Control',
          severity: 'medium',
          source: {
            ip: '172.16.5.102',
            location: 'Corporate Network',
            country: 'Local',
            asn: 'Private'
          },
          target: {
            resource: 'Active Directory',
            type: 'Identity Service',
            port: 389
          },
          attackVector: 'Identity',
          attempts: 23,
          blocked: true,
          firstSeen: '6 hours ago',
          lastSeen: '4 hours ago',
          status: 'blocked',
          aiConfidence: 76,
          relatedIncidents: 1
        },
        {
          id: 'THR-006',
          name: 'Cryptomining Malware',
          type: 'Malware',
          severity: 'medium',
          source: {
            ip: '192.168.10.45',
            location: 'Guest Network',
            country: 'Local',
            asn: 'Private'
          },
          target: {
            resource: 'Compute Instances',
            type: 'Virtual Machine',
            port: 0
          },
          attackVector: 'Malware',
          attempts: 1,
          blocked: false,
          firstSeen: '2 days ago',
          lastSeen: '1 hour ago',
          status: 'monitoring',
          aiConfidence: 82,
          relatedIncidents: 2
        }
      ])

      setLoading(false)
    }, 1000)
  }, [])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'low': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <AlertCircle className="w-4 h-4 text-red-400 animate-pulse" />
      case 'blocked': return <XCircle className="w-4 h-4 text-gray-400" />
      case 'mitigated': return <CheckCircle className="w-4 h-4 text-green-400" />
      case 'monitoring': return <Eye className="w-4 h-4 text-yellow-400" />
      default: return null
    }
  }

  const filteredThreats = threats.filter(threat => {
    const matchesSeverity = selectedSeverity === 'all' || threat.severity === selectedSeverity
    const matchesStatus = selectedStatus === 'all' || threat.status === selectedStatus
    return matchesSeverity && matchesStatus
  })

  return (
    <div className="min-h-screen bg-black p-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-4 mb-2">
          <div className="p-3 bg-gradient-to-br from-red-500 to-orange-500 rounded-xl">
            <AlertTriangle className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Threat Detection</h1>
            <p className="text-gray-400 mt-1">Real-time threat monitoring and analysis</p>
          </div>
        </div>
      </motion.div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
          >
            <div className="flex items-center justify-between mb-4">
              <AlertTriangle className="w-8 h-8 text-red-400" />
              <span className="text-2xl font-bold text-white">{stats.total}</span>
            </div>
            <p className="text-gray-400 text-sm">Total Threats</p>
            <p className="text-xs text-green-400 mt-1">↓ {Math.abs(stats.trend)}% from yesterday</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
          >
            <div className="flex items-center justify-between mb-4">
              <AlertCircle className="w-8 h-8 text-orange-400" />
              <span className="text-2xl font-bold text-white">{stats.critical}</span>
            </div>
            <p className="text-gray-400 text-sm">Critical Threats</p>
            <p className="text-xs text-red-400 mt-1">Immediate action required</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
          >
            <div className="flex items-center justify-between mb-4">
              <Shield className="w-8 h-8 text-green-400" />
              <span className="text-2xl font-bold text-white">{stats.blocked}</span>
            </div>
            <p className="text-gray-400 text-sm">Blocked</p>
            <p className="text-xs text-green-400 mt-1">{Math.round((stats.blocked / stats.total) * 100)}% success rate</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4 }}
            className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
          >
            <div className="flex items-center justify-between mb-4">
              <Activity className="w-8 h-8 text-purple-400" />
              <span className="text-2xl font-bold text-white">{stats.active}</span>
            </div>
            <p className="text-gray-400 text-sm">Active Threats</p>
            <div className="flex items-center gap-2 mt-1">
              {isRealtime && <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />}
              <p className="text-xs text-gray-400">Real-time monitoring</p>
            </div>
          </motion.div>
        </div>
      )}

      {/* Filters and Controls */}
      <div className="flex flex-wrap gap-4 mb-6">
        <select
          value={selectedSeverity}
          onChange={(e) => setSelectedSeverity(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="all">All Severities</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>

        <select
          value={selectedStatus}
          onChange={(e) => setSelectedStatus(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="all">All Status</option>
          <option value="active">Active</option>
          <option value="blocked">Blocked</option>
          <option value="mitigated">Mitigated</option>
          <option value="monitoring">Monitoring</option>
        </select>

        <button
          onClick={() => setIsRealtime(!isRealtime)}
          className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
            isRealtime 
              ? 'bg-green-600 hover:bg-green-700 text-white' 
              : 'bg-white/10 hover:bg-white/20 border border-white/20 text-white'
          }`}
        >
          {isRealtime ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {isRealtime ? 'Pause' : 'Resume'} Real-time
        </button>

        <button className="px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white transition-colors flex items-center gap-2">
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>

        <button className="px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white transition-colors flex items-center gap-2">
          <Download className="w-4 h-4" />
          Export
        </button>
      </div>

      {/* Threats List */}
      <div className="space-y-4">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          filteredThreats.map((threat, index) => (
            <motion.div
              key={threat.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 overflow-hidden hover:bg-white/15 transition-colors"
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-lg ${
                      threat.severity === 'critical' ? 'bg-red-500/20' :
                      threat.severity === 'high' ? 'bg-orange-500/20' :
                      threat.severity === 'medium' ? 'bg-yellow-500/20' :
                      'bg-blue-500/20'
                    }`}>
                      <AlertTriangle className={`w-6 h-6 ${
                        threat.severity === 'critical' ? 'text-red-400' :
                        threat.severity === 'high' ? 'text-orange-400' :
                        threat.severity === 'medium' ? 'text-yellow-400' :
                        'text-blue-400'
                      }`} />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-white mb-1">{threat.name}</h3>
                      <p className="text-sm text-gray-400 mb-2">
                        {threat.type} • {threat.attackVector} Vector
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getSeverityColor(threat.severity)}`}>
                      {threat.severity.toUpperCase()}
                    </span>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(threat.status)}
                      <span className="text-sm text-gray-400 capitalize">{threat.status}</span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="bg-black/20 rounded-lg p-3">
                    <p className="text-xs text-gray-400 mb-2">Source</p>
                    <div className="space-y-1">
                      <p className="text-sm text-white font-mono">{threat.source.ip}</p>
                      <p className="text-xs text-gray-400 flex items-center gap-1">
                        <MapPin className="w-3 h-3" />
                        {threat.source.location}
                      </p>
                      <p className="text-xs text-gray-500">{threat.source.asn}</p>
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-3">
                    <p className="text-xs text-gray-400 mb-2">Target</p>
                    <div className="space-y-1">
                      <p className="text-sm text-white">{threat.target.resource}</p>
                      <p className="text-xs text-gray-400">{threat.target.type}</p>
                      {threat.target.port > 0 && (
                        <p className="text-xs text-gray-500">Port {threat.target.port}</p>
                      )}
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-3">
                    <p className="text-xs text-gray-400 mb-2">Statistics</p>
                    <div className="space-y-1">
                      <p className="text-sm text-white">{threat.attempts.toLocaleString()} attempts</p>
                      <p className="text-xs text-gray-400">First: {threat.firstSeen}</p>
                      <p className="text-xs text-gray-400">Last: {threat.lastSeen}</p>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-white/10">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <Zap className="w-4 h-4 text-purple-400" />
                      <span className="text-sm text-purple-400">AI Confidence: {threat.aiConfidence}%</span>
                    </div>
                    {threat.relatedIncidents > 0 && (
                      <span className="text-sm text-gray-400">
                        {threat.relatedIncidents} related incidents
                      </span>
                    )}
                    <span className={`text-sm ${threat.blocked ? 'text-green-400' : 'text-red-400'}`}>
                      {threat.blocked ? 'Blocked' : 'Not Blocked'}
                    </span>
                  </div>
                  <div className="flex gap-2">
                    {threat.status === 'active' && (
                      <button className="px-3 py-1.5 bg-red-600 hover:bg-red-700 rounded-lg text-white text-sm transition-colors">
                        Block Threat
                      </button>
                    )}
                    <button className="px-3 py-1.5 bg-purple-600 hover:bg-purple-700 rounded-lg text-white text-sm transition-colors">
                      Investigate
                    </button>
                    <button className="px-3 py-1.5 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white text-sm transition-colors">
                      View Details
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