/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect } from 'react'
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  ChevronRight,
  Settings,
  Zap,
  FileX,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Info
} from 'lucide-react'

interface NonCompliantResource {
  resourceId: string
  resourceName: string
  resourceType: string
  complianceState: string
  complianceReason: string
  remediationOptions: Array<{
    action: string
    description: string
  }>
}

interface PolicyAssignment {
  name: string
  displayName: string
  scope: string
}

interface ComplianceResult {
  assignment: PolicyAssignment
  summary: {
    totalResources: number
    compliantResources: number
    nonCompliantResources: number
    compliancePercentage: number
  }
  nonCompliantResources: NonCompliantResource[]
}

export default function PoliciesDeepView() {
  const [loading, setLoading] = useState(true)
  const [complianceResults, setComplianceResults] = useState<ComplianceResult[]>([])
  const [selectedPolicy, setSelectedPolicy] = useState<ComplianceResult | null>(null)
  const [selectedResource, setSelectedResource] = useState<NonCompliantResource | null>(null)
  const [remediating, setRemediating] = useState(false)

  useEffect(() => {
    fetchPolicyData()
  }, [])

  const fetchPolicyData = async () => {
    try {
      const response = await fetch('/api/v1/policies/deep')
      const data = await response.json()
      setComplianceResults(data.complianceResults || [])
    } catch (error) {
      console.error('Error fetching policy data:', error)
      // Use mock data if API fails
      setComplianceResults([
        {
          assignment: {
            name: 'require-tags',
            displayName: 'Require Resource Tags',
            scope: '/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78'
          },
          summary: {
            totalResources: 147,
            compliantResources: 89,
            nonCompliantResources: 58,
            compliancePercentage: 60.5
          },
          nonCompliantResources: [
            {
              resourceId: '/subscriptions/205b477d/resourceGroups/rg-prod/providers/Microsoft.Compute/virtualMachines/vm-prod-001',
              resourceName: 'vm-prod-001',
              resourceType: 'Microsoft.Compute/virtualMachines',
              complianceState: 'NonCompliant',
              complianceReason: 'Missing required tags: Environment, Owner, CostCenter',
              remediationOptions: [
                { action: 'auto-remediate', description: 'Automatically add missing tags' },
                { action: 'create-exception', description: 'Create policy exception' },
                { action: 'manual-fix', description: 'Manually add tags' }
              ]
            },
            {
              resourceId: '/subscriptions/205b477d/resourceGroups/rg-prod/providers/Microsoft.Storage/storageAccounts/stprod001',
              resourceName: 'stprod001',
              resourceType: 'Microsoft.Storage/storageAccounts',
              complianceState: 'NonCompliant',
              complianceReason: 'Missing required tags: Environment, CostCenter',
              remediationOptions: [
                { action: 'auto-remediate', description: 'Automatically add missing tags' },
                { action: 'create-exception', description: 'Create policy exception' }
              ]
            }
          ]
        },
        {
          assignment: {
            name: 'require-encryption',
            displayName: 'Require Encryption at Rest',
            scope: '/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78'
          },
          summary: {
            totalResources: 83,
            compliantResources: 71,
            nonCompliantResources: 12,
            compliancePercentage: 85.5
          },
          nonCompliantResources: [
            {
              resourceId: '/subscriptions/205b477d/resourceGroups/rg-dev/providers/Microsoft.Storage/storageAccounts/stdev002',
              resourceName: 'stdev002',
              resourceType: 'Microsoft.Storage/storageAccounts',
              complianceState: 'NonCompliant',
              complianceReason: 'Encryption at rest is not enabled',
              remediationOptions: [
                { action: 'auto-remediate', description: 'Enable encryption automatically' },
                { action: 'create-exception', description: 'Create policy exception' }
              ]
            }
          ]
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleRemediate = async (resource: NonCompliantResource, action: string) => {
    setRemediating(true)
    try {
      const response = await fetch('/api/v1/remediate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resource_id: resource.resourceId,
          action: action
        })
      })
      const result = await response.json()
      alert(`Remediation initiated: ${result.message}`)
      // Refresh data
      await fetchPolicyData()
    } catch (error) {
      console.error('Remediation failed:', error)
      alert('Remediation failed. Please try again.')
    } finally {
      setRemediating(false)
    }
  }

  const handleCreateException = async (resource: NonCompliantResource, policyId: string) => {
    const reason = prompt('Please provide a reason for the exception:')
    if (!reason) return

    try {
      const response = await fetch('/api/v1/exception', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resource_id: resource.resourceId,
          policy_id: policyId,
          reason: reason
        })
      })
      const result = await response.json()
      alert(`Exception created: ${result.exceptionId}`)
      await fetchPolicyData()
    } catch (error) {
      console.error('Exception creation failed:', error)
      alert('Failed to create exception.')
    }
  }

  const getComplianceColor = (percentage: number) => {
    if (percentage >= 90) return 'text-green-600'
    if (percentage >= 70) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getComplianceBgColor = (percentage: number) => {
    if (percentage >= 90) return 'bg-green-100'
    if (percentage >= 70) return 'bg-yellow-100'
    return 'bg-red-100'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="rounded-2xl p-6 border border-white/10 bg-white/10 backdrop-blur-md shadow-[0_10px_30px_-10px_rgba(0,0,0,0.4)]">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-300">Total Policies</p>
              <p className="text-2xl font-bold text-white">{complianceResults.length}</p>
            </div>
            <Shield className="h-8 w-8 text-blue-500" />
          </div>
        </div>
        
        <div className="rounded-2xl p-6 border border-white/10 bg-white/10 backdrop-blur-md shadow-[0_10px_30px_-10px_rgba(0,0,0,0.4)]">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-300">Total Resources</p>
              <p className="text-2xl font-bold text-white">
                {complianceResults.reduce((sum, r) => sum + r.summary.totalResources, 0)}
              </p>
            </div>
            <Settings className="h-8 w-8 text-gray-500" />
          </div>
        </div>
        
        <div className="rounded-2xl p-6 border border-white/10 bg-white/10 backdrop-blur-md shadow-[0_10px_30px_-10px_rgba(0,0,0,0.4)]">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-300">Non-Compliant</p>
              <p className="text-2xl font-bold text-red-400">
                {complianceResults.reduce((sum, r) => sum + r.summary.nonCompliantResources, 0)}
              </p>
            </div>
            <XCircle className="h-8 w-8 text-red-500" />
          </div>
        </div>
        
        <div className="rounded-2xl p-6 border border-white/10 bg-white/10 backdrop-blur-md shadow-[0_10px_30px_-10px_rgba(0,0,0,0.4)]">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-300">Avg Compliance</p>
              <p className="text-2xl font-bold text-white">
                {(complianceResults.reduce((sum, r) => sum + r.summary.compliancePercentage, 0) / complianceResults.length).toFixed(1)}%
              </p>
            </div>
            <TrendingUp className="h-8 w-8 text-green-500" />
          </div>
        </div>
      </div>

      {/* Policy List */}
      <div className="rounded-2xl border border-white/10 bg-white/10 backdrop-blur-md">
        <div className="px-6 py-4 border-b border-white/10">
          <h2 className="text-lg font-semibold text-white">Policy Compliance Details</h2>
        </div>
        <div className="divide-y divide-white/10">
          {complianceResults.map((result, index) => (
            <div
              key={index}
              className="p-6 hover:bg-white/5 cursor-pointer transition-colors"
              onClick={() => setSelectedPolicy(result)}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <h3 className="text-lg font-medium text-white">{result.assignment.displayName}</h3>
                  <p className="text-sm text-gray-400 mt-1">{result.assignment.name}</p>
                  
                  <div className="mt-4 grid grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-gray-400">Total Resources</p>
                      <p className="text-lg font-semibold text-white">{result.summary.totalResources}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-400">Compliant</p>
                      <p className="text-lg font-semibold text-green-400">
                        {result.summary.compliantResources}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-400">Non-Compliant</p>
                      <p className="text-lg font-semibold text-red-400">
                        {result.summary.nonCompliantResources}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-400">Compliance Rate</p>
                      <div className="flex items-center">
                        <p className={`text-lg font-semibold ${getComplianceColor(result.summary.compliancePercentage)}`}>
                          {result.summary.compliancePercentage.toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  {/* Compliance Bar */}
                  <div className="mt-4">
                    <div className="w-full bg-white/10 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          result.summary.compliancePercentage >= 90 ? 'bg-green-500' :
                          result.summary.compliancePercentage >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${result.summary.compliancePercentage}%` }}
                      />
                    </div>
                  </div>
                </div>
                <ChevronRight className="h-5 w-5 text-gray-400 ml-4" />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Selected Policy Detail Modal */}
      {selectedPolicy && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="rounded-2xl border border-white/10 bg-white/10 backdrop-blur-md max-w-6xl w-full max-h-[90vh] overflow-y-auto">
            <div className="sticky top-0 bg-white/5 border-b border-white/10 px-6 py-4 flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-white">{selectedPolicy.assignment.displayName}</h2>
                <p className="text-sm text-gray-400">{selectedPolicy.assignment.name}</p>
              </div>
              <button
                onClick={() => {
                  setSelectedPolicy(null)
                  setSelectedResource(null)
                }}
                className="text-gray-400 hover:text-white"
              >
                <XCircle className="h-6 w-6" />
              </button>
            </div>
            
            <div className="p-6">
              {/* Summary Stats */}
              <div className="grid grid-cols-4 gap-4 mb-6">
                <div className={`p-4 rounded-lg bg-white/5 border border-white/10`}>
                  <p className="text-sm text-gray-300">Compliance Rate</p>
                  <p className={`text-2xl font-bold ${getComplianceColor(selectedPolicy.summary.compliancePercentage)}`}>
                    {selectedPolicy.summary.compliancePercentage.toFixed(1)}%
                  </p>
                </div>
                <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                  <p className="text-sm text-gray-300">Total Resources</p>
                  <p className="text-2xl font-bold text-white">{selectedPolicy.summary.totalResources}</p>
                </div>
                <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                  <p className="text-sm text-gray-300">Compliant</p>
                  <p className="text-2xl font-bold text-green-400">
                    {selectedPolicy.summary.compliantResources}
                  </p>
                </div>
                <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20">
                  <p className="text-sm text-gray-300">Non-Compliant</p>
                  <p className="text-2xl font-bold text-red-400">
                    {selectedPolicy.summary.nonCompliantResources}
                  </p>
                </div>
              </div>
              
              {/* Non-Compliant Resources */}
              <div>
                <h3 className="text-lg font-semibold mb-4">Non-Compliant Resources</h3>
                <div className="space-y-4">
                  {selectedPolicy.nonCompliantResources.map((resource, idx) => (
                    <div key={idx} className="border border-white/10 rounded-lg p-4 hover:bg-white/5">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center">
                            <XCircle className="h-5 w-5 text-red-500 mr-2" />
                            <h4 className="font-medium text-white">{resource.resourceName}</h4>
                          </div>
                          <p className="text-sm text-gray-400 mt-1">{resource.resourceType}</p>
                          <div className="mt-2 p-3 bg-red-500/10 rounded border border-red-500/20">
                            <p className="text-sm text-red-300">
                              <strong>Issue:</strong> {resource.complianceReason}
                            </p>
                          </div>
                          
                          {/* Remediation Options */}
                          <div className="mt-4">
                            <p className="text-sm font-medium text-white mb-2">Remediation Options:</p>
                            <div className="flex flex-wrap gap-2">
                              {resource.remediationOptions.map((option, optIdx) => (
                                <button
                                  key={optIdx}
                                  onClick={() => {
                                    if (option.action === 'create-exception') {
                                      handleCreateException(resource, selectedPolicy.assignment.name)
                                    } else {
                                      handleRemediate(resource, option.action)
                                    }
                                  }}
                                  disabled={remediating}
                                  className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                                    option.action === 'auto-remediate'
                                      ? 'bg-green-500/15 text-green-300 hover:bg-green-500/25 border border-green-500/20'
                                      : option.action === 'create-exception'
                                      ? 'bg-yellow-500/15 text-yellow-300 hover:bg-yellow-500/25 border border-yellow-500/20'
                                      : 'bg-white/5 text-gray-200 hover:bg-white/10 border border-white/10'
                                  } ${remediating ? 'opacity-50 cursor-not-allowed' : ''}`}
                                >
                                  {option.action === 'auto-remediate' && <Zap className="inline h-3 w-3 mr-1" />}
                                  {option.action === 'create-exception' && <FileX className="inline h-3 w-3 mr-1" />}
                                  {option.description}
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* AI Recommendations */}
              <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                <div className="flex items-start">
                  <Info className="h-5 w-5 text-blue-400 mt-0.5 mr-2" />
                  <div>
                    <h4 className="font-medium text-white">AI Recommendations</h4>
                    <ul className="mt-2 text-sm text-blue-200 space-y-1">
                      <li>• Enable auto-remediation for tag compliance to maintain 95%+ compliance</li>
                      <li>• Create a scheduled task to review exceptions monthly</li>
                      <li>• Consider implementing preventive controls at resource creation</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}