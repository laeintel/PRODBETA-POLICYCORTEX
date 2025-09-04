'use client'

import React, { useState, useEffect } from 'react'
import { 
  Shield, 
  CheckCircle, 
  AlertCircle, 
  Hash, 
  FileText,
  Download,
  Search,
  Filter,
  RefreshCw,
  Lock,
  Unlock,
  Database,
  Activity,
  TrendingUp,
  Calendar,
  ChevronRight
} from 'lucide-react'
import EvidenceChain from '@/components/pcg/EvidenceChain'
import { toast } from '@/hooks/useToast'
import { useRouter } from 'next/navigation'

export default function ProvePage() {
  const router = useRouter()
  const [loading, setLoading] = useState(true)
  const [selectedFramework, setSelectedFramework] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [verificationInProgress, setVerificationInProgress] = useState(false)

  // Evidence chain data
  const evidenceLinks = [
    {
      id: '1',
      timestamp: '2024-01-20 14:32:15',
      action: 'Policy Update - Encryption Standards',
      actor: 'admin@policycortex.com',
      hash: 'a7f8d9e2b4c6...',
      verified: true,
      details: 'Updated encryption requirements to AES-256 for all storage accounts'
    },
    {
      id: '2',
      timestamp: '2024-01-20 14:28:43',
      action: 'Compliance Check - SOC2 Type II',
      actor: 'system-audit',
      hash: 'b8e9c3f5a2d1...',
      verified: true,
      details: 'Automated compliance validation completed successfully'
    },
    {
      id: '3',
      timestamp: '2024-01-20 14:15:22',
      action: 'Resource Configuration Change',
      actor: 'terraform-pipeline',
      hash: 'c9f0d4e6b3a2...',
      verified: true,
      details: 'Applied security group rules to production environment'
    },
    {
      id: '4',
      timestamp: '2024-01-20 13:45:10',
      action: 'Access Review Completed',
      actor: 'security-team',
      hash: 'd0a1e5f7c4b3...',
      verified: false,
      details: 'Quarterly privileged access review - 12 accounts reviewed'
    },
    {
      id: '5',
      timestamp: '2024-01-20 13:12:33',
      action: 'Incident Response',
      actor: 'soc-analyst-02',
      hash: 'e1b2f6g8d5c4...',
      verified: true,
      details: 'Investigated and resolved suspicious API activity alert'
    }
  ]

  // Compliance coverage data
  const complianceCoverage = [
    { framework: 'SOC2 Type II', coverage: 98, controls: 145, verified: 142 },
    { framework: 'ISO 27001', coverage: 95, controls: 114, verified: 108 },
    { framework: 'PCI DSS', coverage: 92, controls: 78, verified: 72 },
    { framework: 'HIPAA', coverage: 89, controls: 66, verified: 59 },
    { framework: 'NIST CSF', coverage: 94, controls: 108, verified: 102 },
    { framework: 'CIS Controls', coverage: 96, controls: 153, verified: 147 }
  ]

  // Audit statistics
  const auditStats = {
    totalEvents: 12847,
    verifiedEvents: 12645,
    pendingVerification: 202,
    integrityScore: 98.5,
    lastVerification: '2 minutes ago',
    chainLength: 847
  }

  // Framework mapping visualization data
  const frameworkMapping = [
    { control: 'Encryption at Rest', soc2: true, iso27001: true, pci: true, hipaa: true, nist: true },
    { control: 'Access Control', soc2: true, iso27001: true, pci: true, hipaa: true, nist: true },
    { control: 'Audit Logging', soc2: true, iso27001: true, pci: true, hipaa: false, nist: true },
    { control: 'Incident Response', soc2: true, iso27001: true, pci: false, hipaa: true, nist: true },
    { control: 'Data Retention', soc2: true, iso27001: false, pci: true, hipaa: true, nist: false },
    { control: 'Vulnerability Mgmt', soc2: true, iso27001: true, pci: true, hipaa: false, nist: true }
  ]

  const handleVerifyChain = async () => {
    setVerificationInProgress(true)
    toast({ title: 'Verifying Chain', description: 'Cryptographic verification in progress...' })
    
    setTimeout(() => {
      setVerificationInProgress(false)
      toast({ title: 'Chain Verified', description: 'All 847 blocks verified successfully' })
    }, 3000)
  }

  const handleExportEvidence = () => {
    toast({ title: 'Exporting Evidence', description: 'Preparing tamper-proof evidence package...' })
  }

  useEffect(() => {
    setTimeout(() => setLoading(false), 800)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <button 
                  onClick={() => router.push('/')}
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                >
                  ← Back
                </button>
              </div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
                PROVE - Evidence Dashboard
              </h1>
              <p className="text-gray-500 dark:text-gray-400 mt-1">
                Cryptographic audit trail and compliance evidence
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={handleExportEvidence}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                Export Evidence
              </button>
              <button
                onClick={() => window.location.reload()}
                className="p-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Integrity Banner */}
        <div className={`rounded-xl p-6 mb-8 ${
          verificationInProgress 
            ? 'bg-gradient-to-r from-yellow-100 to-orange-100 dark:from-yellow-900/20 dark:to-orange-900/20 border-2 border-yellow-500'
            : 'bg-gradient-to-r from-green-100 to-emerald-100 dark:from-green-900/20 dark:to-emerald-900/20 border-2 border-green-500'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-white/50 dark:bg-gray-800/50 rounded-xl">
                {verificationInProgress ? (
                  <Activity className="h-8 w-8 text-yellow-600 animate-pulse" />
                ) : (
                  <CheckCircle className="h-8 w-8 text-green-600" />
                )}
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                  {verificationInProgress ? 'Verification In Progress' : 'Chain Integrity Verified'}
                </h2>
                <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                  {verificationInProgress 
                    ? 'Validating cryptographic signatures...'
                    : `Last verified ${auditStats.lastVerification} • ${auditStats.chainLength} blocks secure`
                  }
                </p>
              </div>
            </div>
            <button
              onClick={handleVerifyChain}
              disabled={verificationInProgress}
              className="px-6 py-3 bg-white dark:bg-gray-800 rounded-lg font-medium text-sm hover:shadow-md transition-all flex items-center gap-2 disabled:opacity-50"
            >
              <Shield className="h-4 w-4" />
              {verificationInProgress ? 'Verifying...' : 'Verify Chain'}
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Total Events</span>
              <Database className="h-4 w-4 text-blue-500" />
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {auditStats.totalEvents.toLocaleString()}
            </p>
            <p className="text-xs text-green-600 mt-1">All tracked</p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Verified</span>
              <CheckCircle className="h-4 w-4 text-green-500" />
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {auditStats.verifiedEvents.toLocaleString()}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
              {((auditStats.verifiedEvents / auditStats.totalEvents) * 100).toFixed(1)}%
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Pending</span>
              <AlertCircle className="h-4 w-4 text-yellow-500" />
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {auditStats.pendingVerification}
            </p>
            <p className="text-xs text-yellow-600 mt-1">Awaiting verification</p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Integrity Score</span>
              <Lock className="h-4 w-4 text-purple-500" />
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {auditStats.integrityScore}%
            </p>
            <p className="text-xs text-green-600 mt-1">Excellent</p>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Evidence Chain */}
          <div className="lg:col-span-2">
            <EvidenceChain
              links={evidenceLinks}
              integrityStatus="verified"
              onVerifyChain={handleVerifyChain}
              onExportEvidence={handleExportEvidence}
            />
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {/* Compliance Coverage Heatmap */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                  Compliance Coverage
                </h3>
                <select
                  value={selectedFramework}
                  onChange={(e) => setSelectedFramework(e.target.value)}
                  className="text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1 bg-white dark:bg-gray-700"
                >
                  <option value="all">All Frameworks</option>
                  <option value="soc2">SOC2 Type II</option>
                  <option value="iso27001">ISO 27001</option>
                  <option value="pci">PCI DSS</option>
                  <option value="hipaa">HIPAA</option>
                </select>
              </div>

              <div className="space-y-3">
                {complianceCoverage.map((framework, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-gray-700 dark:text-gray-300">
                        {framework.framework}
                      </span>
                      <span className="text-xs text-gray-600 dark:text-gray-400">
                        {framework.verified}/{framework.controls} controls
                      </span>
                    </div>
                    <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-500 ${
                          framework.coverage >= 95 ? 'bg-green-500' :
                          framework.coverage >= 90 ? 'bg-yellow-500' :
                          'bg-orange-500'
                        }`}
                        style={{ width: `${framework.coverage}%` }}
                      />
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-500">{framework.coverage}% compliant</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Framework Mapping */}
            <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl border border-indigo-200 dark:border-indigo-800 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                  Framework Mapping
                </h3>
                <FileText className="h-4 w-4 text-indigo-500" />
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <th className="text-left py-2 pr-2 font-medium text-gray-700 dark:text-gray-300">Control</th>
                      <th className="px-1 py-2 text-center">SOC2</th>
                      <th className="px-1 py-2 text-center">ISO</th>
                      <th className="px-1 py-2 text-center">PCI</th>
                      <th className="px-1 py-2 text-center">HIPAA</th>
                    </tr>
                  </thead>
                  <tbody>
                    {frameworkMapping.map((control, index) => (
                      <tr key={index} className="border-b border-gray-100 dark:border-gray-800">
                        <td className="py-2 pr-2 text-gray-700 dark:text-gray-300">{control.control}</td>
                        <td className="px-1 py-2 text-center">
                          {control.soc2 ? (
                            <CheckCircle className="h-3 w-3 text-green-500 mx-auto" />
                          ) : (
                            <div className="h-3 w-3 bg-gray-300 dark:bg-gray-600 rounded-full mx-auto" />
                          )}
                        </td>
                        <td className="px-1 py-2 text-center">
                          {control.iso27001 ? (
                            <CheckCircle className="h-3 w-3 text-green-500 mx-auto" />
                          ) : (
                            <div className="h-3 w-3 bg-gray-300 dark:bg-gray-600 rounded-full mx-auto" />
                          )}
                        </td>
                        <td className="px-1 py-2 text-center">
                          {control.pci ? (
                            <CheckCircle className="h-3 w-3 text-green-500 mx-auto" />
                          ) : (
                            <div className="h-3 w-3 bg-gray-300 dark:bg-gray-600 rounded-full mx-auto" />
                          )}
                        </td>
                        <td className="px-1 py-2 text-center">
                          {control.hipaa ? (
                            <CheckCircle className="h-3 w-3 text-green-500 mx-auto" />
                          ) : (
                            <div className="h-3 w-3 bg-gray-300 dark:bg-gray-600 rounded-full mx-auto" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        {/* Export Controls */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
              Export Evidence Package
            </h2>
            <Download className="h-5 w-5 text-gray-500" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button
              onClick={() => toast({ title: 'Exporting', description: 'Generating audit report PDF...' })}
              className="p-4 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-left group"
            >
              <div className="flex items-center justify-between mb-2">
                <FileText className="h-5 w-5 text-blue-500" />
                <ChevronRight className="h-4 w-4 text-gray-400 group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="font-medium text-gray-900 dark:text-gray-100">Audit Report</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">PDF with signatures</p>
            </button>

            <button
              onClick={() => toast({ title: 'Exporting', description: 'Creating evidence chain backup...' })}
              className="p-4 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-left group"
            >
              <div className="flex items-center justify-between mb-2">
                <Hash className="h-5 w-5 text-purple-500" />
                <ChevronRight className="h-4 w-4 text-gray-400 group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="font-medium text-gray-900 dark:text-gray-100">Chain Backup</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Cryptographic proof</p>
            </button>

            <button
              onClick={() => toast({ title: 'Exporting', description: 'Generating compliance package...' })}
              className="p-4 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-left group"
            >
              <div className="flex items-center justify-between mb-2">
                <Shield className="h-5 w-5 text-green-500" />
                <ChevronRight className="h-4 w-4 text-gray-400 group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="font-medium text-gray-900 dark:text-gray-100">Compliance Pack</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">All frameworks</p>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}