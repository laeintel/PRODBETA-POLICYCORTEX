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
  FileCheck,
  Shield,
  AlertTriangle,
  CheckCircle,
  XCircle,
  TrendingUp,
  Clock,
  Download,
  RefreshCw,
  BarChart3,
  Target,
  Award,
  FileText,
  AlertCircle,
  ChevronRight
} from 'lucide-react'

interface ComplianceFramework {
  id: string
  name: string
  acronym: string
  score: number
  status: 'compliant' | 'non-compliant' | 'partial'
  controls: {
    total: number
    passed: number
    failed: number
    notApplicable: number
  }
  lastAudit: string
  nextAudit: string
  criticalFindings: number
  recommendations: string[]
}

interface ComplianceControl {
  id: string
  controlId: string
  title: string
  framework: string
  category: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  status: 'pass' | 'fail' | 'partial' | 'n/a'
  evidence: number
  lastChecked: string
}

export default function ComplianceHubPage() {
  const [frameworks, setFrameworks] = useState<ComplianceFramework[]>([])
  const [controls, setControls] = useState<ComplianceControl[]>([])
  const [selectedFramework, setSelectedFramework] = useState('all')
  const [overallScore, setOverallScore] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => {
      setOverallScore(91)
      
      setFrameworks([
        {
          id: 'fw-001',
          name: 'System and Organization Controls 2',
          acronym: 'SOC 2',
          score: 94,
          status: 'compliant',
          controls: {
            total: 156,
            passed: 147,
            failed: 5,
            notApplicable: 4
          },
          lastAudit: '2 weeks ago',
          nextAudit: 'in 3 months',
          criticalFindings: 2,
          recommendations: [
            'Implement automated log monitoring',
            'Update incident response procedures',
            'Enhance data encryption at rest'
          ]
        },
        {
          id: 'fw-002',
          name: 'ISO/IEC 27001:2013',
          acronym: 'ISO 27001',
          score: 88,
          status: 'partial',
          controls: {
            total: 114,
            passed: 100,
            failed: 8,
            notApplicable: 6
          },
          lastAudit: '1 month ago',
          nextAudit: 'in 2 months',
          criticalFindings: 3,
          recommendations: [
            'Update risk assessment methodology',
            'Implement continuous monitoring',
            'Review access control policies'
          ]
        },
        {
          id: 'fw-003',
          name: 'Health Insurance Portability and Accountability Act',
          acronym: 'HIPAA',
          score: 96,
          status: 'compliant',
          controls: {
            total: 78,
            passed: 75,
            failed: 2,
            notApplicable: 1
          },
          lastAudit: '3 weeks ago',
          nextAudit: 'in 6 months',
          criticalFindings: 0,
          recommendations: [
            'Update BAA agreements',
            'Enhance PHI encryption'
          ]
        },
        {
          id: 'fw-004',
          name: 'Payment Card Industry Data Security Standard',
          acronym: 'PCI DSS',
          score: 92,
          status: 'compliant',
          controls: {
            total: 248,
            passed: 228,
            failed: 12,
            notApplicable: 8
          },
          lastAudit: '1 month ago',
          nextAudit: 'in 3 months',
          criticalFindings: 1,
          recommendations: [
            'Segment cardholder data environment',
            'Update firewall rules',
            'Implement file integrity monitoring'
          ]
        },
        {
          id: 'fw-005',
          name: 'General Data Protection Regulation',
          acronym: 'GDPR',
          score: 89,
          status: 'partial',
          controls: {
            total: 99,
            passed: 88,
            failed: 7,
            notApplicable: 4
          },
          lastAudit: '2 months ago',
          nextAudit: 'in 4 months',
          criticalFindings: 2,
          recommendations: [
            'Update privacy notices',
            'Implement data retention policies',
            'Enhance consent management'
          ]
        }
      ])

      setControls([
        {
          id: 'ctrl-001',
          controlId: 'CC6.1',
          title: 'Logical and Physical Access Controls',
          framework: 'SOC 2',
          category: 'Security',
          severity: 'critical',
          status: 'fail',
          evidence: 3,
          lastChecked: '1 day ago'
        },
        {
          id: 'ctrl-002',
          controlId: 'A.12.1.1',
          title: 'Documented Operating Procedures',
          framework: 'ISO 27001',
          category: 'Operations',
          severity: 'medium',
          status: 'partial',
          evidence: 8,
          lastChecked: '2 days ago'
        },
        {
          id: 'ctrl-003',
          controlId: '164.312(a)',
          title: 'Access Control',
          framework: 'HIPAA',
          category: 'Technical Safeguards',
          severity: 'high',
          status: 'pass',
          evidence: 12,
          lastChecked: '1 week ago'
        }
      ])

      setLoading(false)
    }, 1000)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'compliant':
      case 'pass':
        return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'non-compliant':
      case 'fail':
        return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'partial':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'n/a':
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-400'
    if (score >= 70) return 'text-yellow-400'
    return 'text-red-400'
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
          <div className="p-3 bg-gradient-to-br from-green-500 to-blue-500 rounded-xl">
            <FileCheck className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Compliance Hub</h1>
            <p className="text-gray-400 mt-1">Regulatory compliance and audit management</p>
          </div>
        </div>
      </motion.div>

      {/* Overall Compliance Score */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        className="mb-8"
      >
        <div className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-white mb-2">Overall Compliance Score</h2>
              <p className="text-gray-400">Across all regulatory frameworks</p>
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
                  stroke="url(#complianceGradient)"
                  strokeWidth="12"
                  fill="none"
                  strokeDasharray={`${2 * Math.PI * 56}`}
                  strokeDashoffset={`${2 * Math.PI * 56 * (1 - overallScore / 100)}`}
                  className="transition-all duration-1000"
                />
                <defs>
                  <linearGradient id="complianceGradient">
                    <stop offset="0%" stopColor="#10b981" />
                    <stop offset="100%" stopColor="#3b82f6" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className={`text-3xl font-bold ${getScoreColor(overallScore)}`}>
                  {overallScore}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Framework Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
        {loading ? (
          <div className="col-span-3 flex items-center justify-center py-12">
            <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          frameworks.map((framework, index) => (
            <motion.div
              key={framework.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-6 hover:bg-white/15 transition-colors"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-white">{framework.acronym}</h3>
                  <p className="text-xs text-gray-400 mt-1">{framework.name}</p>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getStatusColor(framework.status)}`}>
                  {framework.status === 'compliant' ? 'Compliant' : 
                   framework.status === 'partial' ? 'Partial' : 'Non-Compliant'}
                </span>
              </div>

              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Compliance Score</span>
                  <span className={`text-2xl font-bold ${getScoreColor(framework.score)}`}>
                    {framework.score}%
                  </span>
                </div>
                <div className="w-full bg-black/30 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all ${
                      framework.score >= 90 ? 'bg-green-400' :
                      framework.score >= 70 ? 'bg-yellow-400' : 'bg-red-400'
                    }`}
                    style={{ width: `${framework.score}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-black/20 rounded-lg p-2">
                  <p className="text-xs text-gray-400">Controls</p>
                  <p className="text-sm text-white">
                    {framework.controls.passed}/{framework.controls.total}
                  </p>
                </div>
                <div className="bg-black/20 rounded-lg p-2">
                  <p className="text-xs text-gray-400">Critical</p>
                  <p className={`text-sm font-semibold ${
                    framework.criticalFindings > 0 ? 'text-red-400' : 'text-green-400'
                  }`}>
                    {framework.criticalFindings} findings
                  </p>
                </div>
              </div>

              <div className="border-t border-white/10 pt-4">
                <div className="flex items-center justify-between text-xs text-gray-400 mb-3">
                  <span>Last audit: {framework.lastAudit}</span>
                  <span>Next: {framework.nextAudit}</span>
                </div>
                <div className="flex gap-2">
                  <button className="flex-1 px-3 py-1.5 bg-purple-600 hover:bg-purple-700 rounded text-white text-sm">
                    View Report
                  </button>
                  <button className="px-3 py-1.5 bg-white/10 hover:bg-white/20 border border-white/20 rounded text-white text-sm">
                    Run Audit
                  </button>
                </div>
              </div>
            </motion.div>
          ))
        )}
      </div>

      {/* Recent Control Failures */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <div className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-yellow-400" />
              Recent Control Failures
            </h3>
            <button className="text-sm text-purple-400 hover:text-purple-300">
              View All Controls
            </button>
          </div>

          <div className="space-y-3">
            {controls.filter(c => c.status !== 'pass').map((control) => (
              <div key={control.id} className="bg-black/20 rounded-lg p-4">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <p className="font-medium text-white">{control.title}</p>
                    <p className="text-xs text-gray-400 mt-1">
                      {control.framework} • {control.controlId} • {control.category}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getStatusColor(control.status)}`}>
                      {control.status.toUpperCase()}
                    </span>
                    <span className={`px-2 py-1 rounded text-xs ${
                      control.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                      control.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                      control.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-blue-500/20 text-blue-400'
                    }`}>
                      {control.severity}
                    </span>
                  </div>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">
                    {control.evidence} evidence items • Last checked {control.lastChecked}
                  </span>
                  <button className="text-purple-400 hover:text-purple-300">
                    Remediate →
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  )
}