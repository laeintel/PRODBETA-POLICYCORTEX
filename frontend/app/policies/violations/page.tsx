/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2026 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  AlertTriangle,
  Shield,
  XCircle,
  CheckCircle,
  Clock,
  TrendingUp,
  Filter,
  Download,
  RefreshCw,
  ChevronRight,
  AlertCircle,
  Info,
  Zap,
  FileText,
  GitBranch,
  Settings
} from 'lucide-react'

interface PolicyViolation {
  id: string
  policyName: string
  policyId: string
  resourceName: string
  resourceType: string
  resourceGroup: string
  severity: 'Critical' | 'High' | 'Medium' | 'Low'
  status: 'Active' | 'Remediated' | 'Exception' | 'Pending'
  violationType: string
  description: string
  detectedAt: string
  lastChecked: string
  remediationSteps: string[]
  automationAvailable: boolean
  impactedResources: number
  complianceFramework: string[]
  estimatedCost: number
}

export default function PolicyViolationsPage() {
  const [violations, setViolations] = useState<PolicyViolation[]>([])
  const [selectedSeverity, setSelectedSeverity] = useState('all')
  const [selectedStatus, setSelectedStatus] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => {
      setViolations([
        {
          id: 'vio-001',
          policyName: 'Require HTTPS for Storage Accounts',
          policyId: 'pol-sec-001',
          resourceName: 'devstorageaccount',
          resourceType: 'Storage Account',
          resourceGroup: 'development-rg',
          severity: 'Critical',
          status: 'Active',
          violationType: 'Security',
          description: 'Storage account is not configured to require HTTPS for all connections',
          detectedAt: '2 hours ago',
          lastChecked: '5 minutes ago',
          remediationSteps: [
            'Navigate to Storage Account settings',
            'Enable "Secure transfer required"',
            'Update all client connections to use HTTPS'
          ],
          automationAvailable: true,
          impactedResources: 1,
          complianceFramework: ['SOC 2', 'ISO 27001', 'HIPAA'],
          estimatedCost: 0
        },
        {
          id: 'vio-002',
          policyName: 'VM Backup Required',
          policyId: 'pol-backup-002',
          resourceName: 'prod-web-vm-01',
          resourceType: 'Virtual Machine',
          resourceGroup: 'production-rg',
          severity: 'High',
          status: 'Active',
          violationType: 'Data Protection',
          description: 'Virtual machine does not have backup configured',
          detectedAt: '1 day ago',
          lastChecked: '1 hour ago',
          remediationSteps: [
            'Create Recovery Services Vault',
            'Configure backup policy',
            'Enable backup for the VM'
          ],
          automationAvailable: true,
          impactedResources: 3,
          complianceFramework: ['SOC 2', 'PCI DSS'],
          estimatedCost: 25
        },
        {
          id: 'vio-003',
          policyName: 'Require Tags for Cost Center',
          policyId: 'pol-tag-003',
          resourceName: 'Multiple Resources',
          resourceType: 'Various',
          resourceGroup: 'analytics-rg',
          severity: 'Medium',
          status: 'Active',
          violationType: 'Governance',
          description: '12 resources missing required cost center tags',
          detectedAt: '3 days ago',
          lastChecked: '2 hours ago',
          remediationSteps: [
            'Identify resource owners',
            'Determine appropriate cost center',
            'Apply tags via portal or CLI'
          ],
          automationAvailable: false,
          impactedResources: 12,
          complianceFramework: ['Internal Policy'],
          estimatedCost: 0
        },
        {
          id: 'vio-004',
          policyName: 'Network Security Group Rules',
          policyId: 'pol-net-004',
          resourceName: 'public-nsg',
          resourceType: 'Network Security Group',
          resourceGroup: 'network-rg',
          severity: 'Critical',
          status: 'Pending',
          violationType: 'Security',
          description: 'NSG allows unrestricted inbound access on port 3389 (RDP)',
          detectedAt: '30 minutes ago',
          lastChecked: 'Just now',
          remediationSteps: [
            'Review NSG inbound rules',
            'Restrict RDP access to specific IP ranges',
            'Consider using Azure Bastion instead'
          ],
          automationAvailable: true,
          impactedResources: 5,
          complianceFramework: ['CIS Benchmark', 'Azure Security Benchmark'],
          estimatedCost: 0
        },
        {
          id: 'vio-005',
          policyName: 'SQL Database Encryption',
          policyId: 'pol-data-005',
          resourceName: 'analytics-sql-db',
          resourceType: 'SQL Database',
          resourceGroup: 'data-rg',
          severity: 'High',
          status: 'Remediated',
          violationType: 'Data Protection',
          description: 'Transparent Data Encryption (TDE) was not enabled',
          detectedAt: '1 week ago',
          lastChecked: '2 days ago',
          remediationSteps: [
            'Enable TDE on database',
            'Verify encryption status',
            'Update compliance documentation'
          ],
          automationAvailable: true,
          impactedResources: 1,
          complianceFramework: ['GDPR', 'HIPAA', 'SOC 2'],
          estimatedCost: 0
        },
        {
          id: 'vio-006',
          policyName: 'Resource Location Restriction',
          policyId: 'pol-loc-006',
          resourceName: 'test-storage',
          resourceType: 'Storage Account',
          resourceGroup: 'test-rg',
          severity: 'Low',
          status: 'Exception',
          violationType: 'Compliance',
          description: 'Resource deployed in non-approved region (West Europe)',
          detectedAt: '2 weeks ago',
          lastChecked: '1 day ago',
          remediationSteps: [
            'Request exception approval',
            'Or migrate resource to approved region',
            'Update deployment templates'
          ],
          automationAvailable: false,
          impactedResources: 1,
          complianceFramework: ['Data Residency Policy'],
          estimatedCost: 150
        }
      ])
      setLoading(false)
    }, 1000)
  }, [])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Critical': return 'text-red-400 bg-red-500/20 border-red-500/30'
      case 'High': return 'text-orange-400 bg-orange-500/20 border-orange-500/30'
      case 'Medium': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30'
      case 'Low': return 'text-blue-400 bg-blue-500/20 border-blue-500/30'
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Active': return 'text-red-400'
      case 'Remediated': return 'text-green-400'
      case 'Exception': return 'text-yellow-400'
      case 'Pending': return 'text-orange-400'
      default: return 'text-gray-400'
    }
  }

  const filteredViolations = violations.filter(v => {
    const matchesSearch = v.policyName.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          v.resourceName.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          v.description.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesSeverity = selectedSeverity === 'all' || v.severity === selectedSeverity
    const matchesStatus = selectedStatus === 'all' || v.status === selectedStatus
    return matchesSearch && matchesSeverity && matchesStatus
  })

  const activeViolations = violations.filter(v => v.status === 'Active' || v.status === 'Pending')
  const criticalCount = activeViolations.filter(v => v.severity === 'Critical').length
  const highCount = activeViolations.filter(v => v.severity === 'High').length

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
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
            <h1 className="text-4xl font-bold text-white">Policy Violations</h1>
            <p className="text-gray-400 mt-1">Monitor and remediate policy compliance issues</p>
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
            <XCircle className="w-8 h-8 text-red-400" />
            <span className="text-2xl font-bold text-white">{activeViolations.length}</span>
          </div>
          <p className="text-gray-400 text-sm">Active Violations</p>
          <p className="text-xs text-red-400 mt-1">↑ 15% from last week</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <AlertCircle className="w-8 h-8 text-orange-400" />
            <span className="text-2xl font-bold text-white">{criticalCount}</span>
          </div>
          <p className="text-gray-400 text-sm">Critical Issues</p>
          <p className="text-xs text-orange-400 mt-1">Immediate attention required</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <Zap className="w-8 h-8 text-purple-400" />
            <span className="text-2xl font-bold text-white">67%</span>
          </div>
          <p className="text-gray-400 text-sm">Auto-Remediation</p>
          <p className="text-xs text-purple-400 mt-1">Available for violations</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <CheckCircle className="w-8 h-8 text-green-400" />
            <span className="text-2xl font-bold text-white">24</span>
          </div>
          <p className="text-gray-400 text-sm">Remediated</p>
          <p className="text-xs text-green-400 mt-1">This month</p>
        </motion.div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 mb-6">
        <input
          type="text"
          placeholder="Search violations..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
        />
        
        <select
          value={selectedSeverity}
          onChange={(e) => setSelectedSeverity(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="all">All Severities</option>
          <option value="Critical">Critical</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>

        <select
          value={selectedStatus}
          onChange={(e) => setSelectedStatus(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="all">All Status</option>
          <option value="Active">Active</option>
          <option value="Pending">Pending</option>
          <option value="Remediated">Remediated</option>
          <option value="Exception">Exception</option>
        </select>

        <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors flex items-center gap-2">
          <Zap className="w-4 h-4" />
          Auto-Remediate All
        </button>

        <button className="px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white transition-colors flex items-center gap-2">
          <Download className="w-4 h-4" />
          Export Report
        </button>
      </div>

      {/* Violations List */}
      <div className="space-y-4">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          filteredViolations.map((violation, index) => (
            <motion.div
              key={violation.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 overflow-hidden hover:bg-white/15 transition-colors"
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-lg ${
                      violation.severity === 'Critical' ? 'bg-red-500/20' :
                      violation.severity === 'High' ? 'bg-orange-500/20' :
                      violation.severity === 'Medium' ? 'bg-yellow-500/20' :
                      'bg-blue-500/20'
                    }`}>
                      <AlertTriangle className={`w-6 h-6 ${
                        violation.severity === 'Critical' ? 'text-red-400' :
                        violation.severity === 'High' ? 'text-orange-400' :
                        violation.severity === 'Medium' ? 'text-yellow-400' :
                        'text-blue-400'
                      }`} />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-white mb-1">{violation.policyName}</h3>
                      <p className="text-sm text-gray-400 mb-2">
                        {violation.resourceName} • {violation.resourceType} • {violation.resourceGroup}
                      </p>
                      <p className="text-gray-300">{violation.description}</p>
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getSeverityColor(violation.severity)}`}>
                      {violation.severity}
                    </span>
                    <span className={`text-sm font-medium ${getStatusColor(violation.status)}`}>
                      {violation.status}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="bg-black/20 rounded-lg p-3">
                    <p className="text-xs text-gray-400 mb-1">Detected</p>
                    <p className="text-sm text-white">{violation.detectedAt}</p>
                  </div>
                  <div className="bg-black/20 rounded-lg p-3">
                    <p className="text-xs text-gray-400 mb-1">Impacted Resources</p>
                    <p className="text-sm text-white">{violation.impactedResources}</p>
                  </div>
                  <div className="bg-black/20 rounded-lg p-3">
                    <p className="text-xs text-gray-400 mb-1">Compliance</p>
                    <p className="text-sm text-white">{violation.complianceFramework.join(', ')}</p>
                  </div>
                </div>

                <div className="border-t border-white/10 pt-4">
                  <h4 className="text-sm font-medium text-white mb-2">Remediation Steps</h4>
                  <div className="space-y-1 mb-4">
                    {violation.remediationSteps.map((step, idx) => (
                      <div key={idx} className="flex items-start gap-2">
                        <ChevronRight className="w-4 h-4 text-purple-400 mt-0.5" />
                        <p className="text-sm text-gray-300">{step}</p>
                      </div>
                    ))}
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3 text-xs">
                      {violation.automationAvailable && (
                        <span className="flex items-center gap-1 text-purple-400">
                          <Zap className="w-3 h-3" />
                          Auto-remediation available
                        </span>
                      )}
                      {violation.estimatedCost > 0 && (
                        <span className="text-gray-400">
                          Est. cost: ${violation.estimatedCost}
                        </span>
                      )}
                      <span className="text-gray-400">
                        Last checked: {violation.lastChecked}
                      </span>
                    </div>
                    <div className="flex gap-2">
                      {violation.automationAvailable && violation.status === 'Active' && (
                        <button className="px-3 py-1.5 bg-purple-600 hover:bg-purple-700 rounded-lg text-white text-sm transition-colors flex items-center gap-1">
                          <Zap className="w-3 h-3" />
                          Auto-Fix
                        </button>
                      )}
                      <button className="px-3 py-1.5 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white text-sm transition-colors">
                        View Details
                      </button>
                      {violation.status === 'Active' && (
                        <button className="px-3 py-1.5 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-white text-sm transition-colors">
                          Request Exception
                        </button>
                      )}
                    </div>
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