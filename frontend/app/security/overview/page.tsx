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
  Database
} from 'lucide-react'

interface SecurityMetric {
  label: string
  value: number | string
  change: number
  status: 'good' | 'warning' | 'critical'
}

interface ThreatInfo {
  id: string
  type: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  source: string
  target: string
  detectedAt: string
  status: 'active' | 'mitigated' | 'investigating'
}

interface ComplianceStatus {
  framework: string
  score: number
  issues: number
  lastAudit: string
}

export default function SecurityOverviewPage() {
  const [metrics, setMetrics] = useState<SecurityMetric[]>([])
  const [threats, setThreats] = useState<ThreatInfo[]>([])
  const [compliance, setCompliance] = useState<ComplianceStatus[]>([])
  const [loading, setLoading] = useState(true)
  const [securityScore, setSecurityScore] = useState(0)

  useEffect(() => {
    // Simulate loading data
    setTimeout(() => {
      setSecurityScore(87)
      
      setMetrics([
        { label: 'Security Score', value: 87, change: 3, status: 'good' },
        { label: 'Active Threats', value: 4, change: -2, status: 'warning' },
        { label: 'Vulnerabilities', value: 23, change: 5, status: 'warning' },
        { label: 'Failed Logins', value: 156, change: 12, status: 'warning' },
        { label: 'Encrypted Resources', value: '94%', change: 2, status: 'good' },
        { label: 'MFA Coverage', value: '78%', change: 5, status: 'warning' }
      ])

      setThreats([
        {
          id: 'T001',
          type: 'Brute Force Attack',
          severity: 'high',
          source: '185.220.101.45',
          target: 'SSH Service',
          detectedAt: '5 minutes ago',
          status: 'active'
        },
        {
          id: 'T002',
          type: 'SQL Injection Attempt',
          severity: 'critical',
          source: '45.142.182.112',
          target: 'Web Application',
          detectedAt: '15 minutes ago',
          status: 'mitigated'
        },
        {
          id: 'T003',
          type: 'Suspicious API Activity',
          severity: 'medium',
          source: 'Internal Network',
          target: 'API Gateway',
          detectedAt: '1 hour ago',
          status: 'investigating'
        },
        {
          id: 'T004',
          type: 'DDoS Attack',
          severity: 'high',
          source: 'Multiple IPs',
          target: 'Load Balancer',
          detectedAt: '2 hours ago',
          status: 'mitigated'
        }
      ])

      setCompliance([
        { framework: 'SOC 2', score: 92, issues: 3, lastAudit: '2 weeks ago' },
        { framework: 'ISO 27001', score: 88, issues: 7, lastAudit: '1 month ago' },
        { framework: 'HIPAA', score: 95, issues: 2, lastAudit: '3 weeks ago' },
        { framework: 'PCI DSS', score: 90, issues: 5, lastAudit: '1 month ago' },
        { framework: 'GDPR', score: 93, issues: 4, lastAudit: '2 weeks ago' }
      ])

      setLoading(false)
    }, 1000)
  }, [])

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-400'
    if (score >= 70) return 'text-yellow-400'
    return 'text-red-400'
  }

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
      case 'active': return <AlertCircle className="w-4 h-4 text-red-400" />
      case 'mitigated': return <CheckCircle className="w-4 h-4 text-green-400" />
      case 'investigating': return <Eye className="w-4 h-4 text-yellow-400" />
      default: return null
    }
  }

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
            <Shield className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Security Overview</h1>
            <p className="text-gray-400 mt-1">Comprehensive security posture and threat monitoring</p>
          </div>
        </div>
      </motion.div>

      {/* Security Score */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        className="mb-8"
      >
        <div className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-white mb-2">Overall Security Score</h2>
              <p className="text-gray-400">Based on 150+ security checks and compliance standards</p>
            </div>
            <div className="relative w-32 h-32">
              <svg className="w-full h-full transform -rotate-90">
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="rgba(255,255,255,0.1)"
                  strokeWidth="12"
                  fill="none"
                />
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="url(#gradient)"
                  strokeWidth="12"
                  fill="none"
                  strokeDasharray={`${2 * Math.PI * 56}`}
                  strokeDashoffset={`${2 * Math.PI * 56 * (1 - securityScore / 100)}`}
                  className="transition-all duration-1000"
                />
                <defs>
                  <linearGradient id="gradient">
                    <stop offset="0%" stopColor="#a855f7" />
                    <stop offset="100%" stopColor="#ec4899" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className={`text-3xl font-bold ${getScoreColor(securityScore)}`}>
                  {securityScore}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        {loading ? (
          <div className="col-span-6 flex items-center justify-center py-12">
            <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          metrics.map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-4"
            >
              <p className="text-xs text-gray-400 mb-2">{metric.label}</p>
              <p className="text-2xl font-bold text-white mb-1">{metric.value}</p>
              <div className="flex items-center gap-1">
                {metric.change > 0 ? (
                  <TrendingUp className="w-3 h-3 text-red-400" />
                ) : (
                  <TrendingUp className="w-3 h-3 text-green-400 rotate-180" />
                )}
                <span className={`text-xs ${metric.change > 0 ? 'text-red-400' : 'text-green-400'}`}>
                  {Math.abs(metric.change)}%
                </span>
              </div>
            </motion.div>
          ))
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Active Threats */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-400" />
                Active Threats
              </h3>
              <button className="text-sm text-purple-400 hover:text-purple-300">
                View All
              </button>
            </div>
            
            <div className="space-y-3">
              {threats.map((threat) => (
                <div key={threat.id} className="bg-black/20 rounded-lg p-3">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <p className="font-medium text-white">{threat.type}</p>
                      <p className="text-xs text-gray-400">
                        {threat.source} → {threat.target}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getSeverityColor(threat.severity)}`}>
                        {threat.severity}
                      </span>
                      {getStatusIcon(threat.status)}
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">{threat.detectedAt}</span>
                    <span className="text-xs text-gray-400 capitalize">{threat.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Compliance Status */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <FileCheck className="w-5 h-5 text-green-400" />
                Compliance Status
              </h3>
              <button className="text-sm text-purple-400 hover:text-purple-300">
                View Reports
              </button>
            </div>
            
            <div className="space-y-3">
              {compliance.map((framework) => (
                <div key={framework.framework} className="bg-black/20 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-white">{framework.framework}</span>
                    <span className={`text-lg font-bold ${getScoreColor(framework.score)}`}>
                      {framework.score}%
                    </span>
                  </div>
                  <div className="w-full bg-black/30 rounded-full h-2 mb-2">
                    <div
                      className={`h-2 rounded-full transition-all ${
                        framework.score >= 90 ? 'bg-green-400' :
                        framework.score >= 70 ? 'bg-yellow-400' : 'bg-red-400'
                      }`}
                      style={{ width: `${framework.score}%` }}
                    />
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-400">{framework.issues} issues</span>
                    <span className="text-gray-500">Audited {framework.lastAudit}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="mt-8"
      >
        <div className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <button className="p-4 bg-purple-600/20 hover:bg-purple-600/30 border border-purple-500/30 rounded-lg transition-colors">
              <Shield className="w-6 h-6 text-purple-400 mx-auto mb-2" />
              <p className="text-sm text-white">Run Security Scan</p>
            </button>
            <button className="p-4 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/30 rounded-lg transition-colors">
              <Lock className="w-6 h-6 text-blue-400 mx-auto mb-2" />
              <p className="text-sm text-white">Review Permissions</p>
            </button>
            <button className="p-4 bg-green-600/20 hover:bg-green-600/30 border border-green-500/30 rounded-lg transition-colors">
              <Key className="w-6 h-6 text-green-400 mx-auto mb-2" />
              <p className="text-sm text-white">Rotate Keys</p>
            </button>
            <button className="p-4 bg-red-600/20 hover:bg-red-600/30 border border-red-500/30 rounded-lg transition-colors">
              <AlertTriangle className="w-6 h-6 text-red-400 mx-auto mb-2" />
              <p className="text-sm text-white">Emergency Lockdown</p>
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  )
}