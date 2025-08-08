'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import AppLayout from '../../components/AppLayout'
import { useAzurePolicies, type AzurePolicy } from '../../lib/azure-api'
import { 
  Shield,
  Search,
  Filter,
  CheckCircle,
  XCircle,
  AlertTriangle,
  FileText,
  Settings,
  Play,
  Pause,
  Edit,
  Trash2,
  RefreshCw,
  Download,
  Upload,
  ChevronRight,
  Info,
  TrendingUp,
  BarChart3,
  Clock,
  Zap,
  Code,
  GitBranch
} from 'lucide-react'

interface Policy {
  id: string
  name: string
  description: string
  category: string
  type: 'BuiltIn' | 'Custom'
  effect: 'Deny' | 'Audit' | 'AuditIfNotExists' | 'DeployIfNotExists' | 'Append'
  scope: string
  assignments: number
  compliance: {
    compliant: number
    nonCompliant: number
    exempt: number
    percentage: number
  }
  lastModified: string
  createdBy: string
  parameters: Record<string, any>
  resourceTypes: string[]
  status: 'Active' | 'Disabled' | 'Draft'
}

interface NonCompliantResource {
  id: string
  name: string
  type: string
  resourceGroup: string
  reason: string
  lastEvaluated: string
}

export default function PoliciesPage() {
  const [policies, setPolicies] = useState<Policy[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterCategory, setFilterCategory] = useState('all')
  const [filterStatus, setFilterStatus] = useState('all')
  const [selectedPolicy, setSelectedPolicy] = useState<Policy | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [nonCompliantResources, setNonCompliantResources] = useState<NonCompliantResource[]>([])

  useEffect(() => {
    fetchPolicies()
  }, [])

  const fetchPolicies = async () => {
    try {
      // Mock Azure Policy data that would come from the backend
      const mockPolicies: Policy[] = [
        {
          id: 'pol-001',
          name: 'Require encryption for storage accounts',
          description: 'Ensures all storage accounts have encryption enabled for data at rest',
          category: 'Security',
          type: 'BuiltIn',
          effect: 'Deny',
          scope: '/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78',
          assignments: 3,
          compliance: {
            compliant: 45,
            nonCompliant: 3,
            exempt: 2,
            percentage: 93.8
          },
          lastModified: '2025-01-07',
          createdBy: 'Microsoft',
          parameters: {
            effect: { type: 'String', defaultValue: 'Deny' }
          },
          resourceTypes: ['Microsoft.Storage/storageAccounts'],
          status: 'Active'
        },
        {
          id: 'pol-002',
          name: 'Require tag and its value',
          description: 'Enforces existence of a tag and its value on resources',
          category: 'Governance',
          type: 'Custom',
          effect: 'Audit',
          scope: '/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78',
          assignments: 5,
          compliance: {
            compliant: 72,
            nonCompliant: 13,
            exempt: 0,
            percentage: 84.7
          },
          lastModified: '2025-01-05',
          createdBy: 'PolicyCortex Admin',
          parameters: {
            tagName: { type: 'String', value: 'Environment' },
            tagValue: { type: 'String', value: 'Production' }
          },
          resourceTypes: ['*'],
          status: 'Active'
        },
        {
          id: 'pol-003',
          name: 'Allowed locations',
          description: 'Restricts the locations where resources can be deployed',
          category: 'Compliance',
          type: 'BuiltIn',
          effect: 'Deny',
          scope: '/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78',
          assignments: 2,
          compliance: {
            compliant: 85,
            nonCompliant: 0,
            exempt: 0,
            percentage: 100
          },
          lastModified: '2025-01-06',
          createdBy: 'Microsoft',
          parameters: {
            listOfAllowedLocations: { 
              type: 'Array', 
              value: ['East US', 'West US', 'Central US'] 
            }
          },
          resourceTypes: ['*'],
          status: 'Active'
        },
        {
          id: 'pol-004',
          name: 'VM backup should be enabled',
          description: 'Ensures Azure Backup service is enabled for Virtual Machines',
          category: 'Backup',
          type: 'BuiltIn',
          effect: 'AuditIfNotExists',
          scope: '/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78',
          assignments: 1,
          compliance: {
            compliant: 8,
            nonCompliant: 4,
            exempt: 1,
            percentage: 66.7
          },
          lastModified: '2025-01-04',
          createdBy: 'Microsoft',
          parameters: {},
          resourceTypes: ['Microsoft.Compute/virtualMachines'],
          status: 'Active'
        },
        {
          id: 'pol-005',
          name: 'Diagnostic logs in Key Vault should be enabled',
          description: 'Audit enabling of diagnostic logs to track Key Vault activity',
          category: 'Monitoring',
          type: 'BuiltIn',
          effect: 'Audit',
          scope: '/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78',
          assignments: 2,
          compliance: {
            compliant: 3,
            nonCompliant: 2,
            exempt: 0,
            percentage: 60.0
          },
          lastModified: '2025-01-03',
          createdBy: 'Microsoft',
          parameters: {
            requiredRetentionDays: { type: 'Integer', value: 90 }
          },
          resourceTypes: ['Microsoft.KeyVault/vaults'],
          status: 'Active'
        },
        {
          id: 'pol-006',
          name: 'SQL Server auditing should be enabled',
          description: 'Enables auditing on SQL servers to track database activities',
          category: 'Security',
          type: 'BuiltIn',
          effect: 'AuditIfNotExists',
          scope: '/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78',
          assignments: 1,
          compliance: {
            compliant: 2,
            nonCompliant: 1,
            exempt: 0,
            percentage: 66.7
          },
          lastModified: '2025-01-02',
          createdBy: 'Microsoft',
          parameters: {},
          resourceTypes: ['Microsoft.Sql/servers'],
          status: 'Active'
        },
        {
          id: 'pol-007',
          name: 'Network Watcher should be enabled',
          description: 'Ensures Network Watcher is enabled for all regions with virtual networks',
          category: 'Network',
          type: 'BuiltIn',
          effect: 'Audit',
          scope: '/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78',
          assignments: 1,
          compliance: {
            compliant: 2,
            nonCompliant: 1,
            exempt: 0,
            percentage: 66.7
          },
          lastModified: '2025-01-01',
          createdBy: 'Microsoft',
          parameters: {},
          resourceTypes: ['Microsoft.Network/networkWatchers'],
          status: 'Draft'
        }
      ]

      // Mock non-compliant resources
      const mockNonCompliant: NonCompliantResource[] = [
        {
          id: 'res-nc-001',
          name: 'stpolicycortexdev',
          type: 'Microsoft.Storage/storageAccounts',
          resourceGroup: 'rg-policycortex-dev',
          reason: 'Storage account does not have encryption enabled',
          lastEvaluated: '2025-01-08 10:30 AM'
        },
        {
          id: 'res-nc-002',
          name: 'vm-test-01',
          type: 'Microsoft.Compute/virtualMachines',
          resourceGroup: 'rg-test',
          reason: 'Missing required tag: Environment',
          lastEvaluated: '2025-01-08 09:15 AM'
        },
        {
          id: 'res-nc-003',
          name: 'vm-legacy-app',
          type: 'Microsoft.Compute/virtualMachines',
          resourceGroup: 'rg-legacy',
          reason: 'Azure Backup is not configured',
          lastEvaluated: '2025-01-08 08:45 AM'
        }
      ]
      
      setPolicies(mockPolicies)
      setNonCompliantResources(mockNonCompliant)
    } catch (error) {
      console.error('Error fetching policies:', error)
    } finally {
      setLoading(false)
    }
  }

  const filteredPolicies = policies.filter(policy => {
    const matchesSearch = policy.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         policy.description.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesCategory = filterCategory === 'all' || policy.category === filterCategory
    const matchesStatus = filterStatus === 'all' || policy.status === filterStatus
    
    return matchesSearch && matchesCategory && matchesStatus
  })

  const totalCompliant = filteredPolicies.reduce((sum, p) => sum + p.compliance.compliant, 0)
  const totalNonCompliant = filteredPolicies.reduce((sum, p) => sum + p.compliance.nonCompliant, 0)
  const overallCompliance = totalCompliant + totalNonCompliant > 0 
    ? ((totalCompliant / (totalCompliant + totalNonCompliant)) * 100).toFixed(1)
    : '0'

  return (
    <AppLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">Policy Management</h1>
            <p className="text-gray-400">Azure Policy compliance and governance control</p>
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20"
            >
              <div className="flex items-center justify-between mb-2">
                <Shield className="w-5 h-5 text-purple-400" />
                <span className="text-xs text-gray-400">Total</span>
              </div>
              <p className="text-2xl font-bold text-white">{filteredPolicies.length}</p>
              <p className="text-sm text-gray-300">Active Policies</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20"
            >
              <div className="flex items-center justify-between mb-2">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span className="text-xs text-gray-400">Compliant</span>
              </div>
              <p className="text-2xl font-bold text-white">{totalCompliant}</p>
              <p className="text-sm text-gray-300">Resources</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20"
            >
              <div className="flex items-center justify-between mb-2">
                <XCircle className="w-5 h-5 text-red-400" />
                <span className="text-xs text-gray-400">Non-Compliant</span>
              </div>
              <p className="text-2xl font-bold text-white">{totalNonCompliant}</p>
              <p className="text-sm text-gray-300">Resources</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20"
            >
              <div className="flex items-center justify-between mb-2">
                <TrendingUp className="w-5 h-5 text-blue-400" />
                <span className="text-xs text-gray-400">Overall</span>
              </div>
              <p className="text-2xl font-bold text-white">{overallCompliance}%</p>
              <p className="text-sm text-gray-300">Compliance</p>
            </motion.div>
          </div>

          {/* Search and Filters */}
          <div className="flex flex-col md:flex-row gap-4 mb-6">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search policies..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-400"
              />
            </div>
            
            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
              className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-400"
            >
              <option value="all">All Categories</option>
              <option value="Security">Security</option>
              <option value="Governance">Governance</option>
              <option value="Compliance">Compliance</option>
              <option value="Backup">Backup</option>
              <option value="Monitoring">Monitoring</option>
              <option value="Network">Network</option>
            </select>
            
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-400"
            >
              <option value="all">All Status</option>
              <option value="Active">Active</option>
              <option value="Disabled">Disabled</option>
              <option value="Draft">Draft</option>
            </select>

            <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2">
              <Upload className="w-4 h-4" />
              Import Policy
            </button>
          </div>

          {/* Policies Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
            {filteredPolicies.map((policy) => (
              <motion.div
                key={policy.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ scale: 1.02 }}
                onClick={() => {
                  setSelectedPolicy(policy)
                  setShowDetails(true)
                }}
                className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20 cursor-pointer hover:bg-white/15 transition-all"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Shield className={`w-5 h-5 ${
                      policy.status === 'Active' ? 'text-green-400' :
                      policy.status === 'Draft' ? 'text-yellow-400' :
                      'text-gray-400'
                    }`} />
                    <span className={`text-xs px-2 py-1 rounded ${
                      policy.type === 'BuiltIn' 
                        ? 'bg-blue-900/30 text-blue-400'
                        : 'bg-purple-900/30 text-purple-400'
                    }`}>
                      {policy.type}
                    </span>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded ${
                    policy.effect === 'Deny' ? 'bg-red-900/30 text-red-400' :
                    policy.effect === 'Audit' ? 'bg-yellow-900/30 text-yellow-400' :
                    'bg-green-900/30 text-green-400'
                  }`}>
                    {policy.effect}
                  </span>
                </div>

                <h3 className="text-white font-medium mb-1">{policy.name}</h3>
                <p className="text-xs text-gray-400 mb-3 line-clamp-2">{policy.description}</p>

                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs text-gray-400">{policy.category}</span>
                  <span className="text-xs text-gray-400">{policy.assignments} assignments</span>
                </div>

                {/* Compliance Bar */}
                <div className="mb-2">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-gray-400">Compliance</span>
                    <span className="text-white">{policy.compliance.percentage.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        policy.compliance.percentage >= 90 ? 'bg-green-400' :
                        policy.compliance.percentage >= 70 ? 'bg-yellow-400' :
                        'bg-red-400'
                      }`}
                      style={{ width: `${policy.compliance.percentage}%` }}
                    />
                  </div>
                </div>

                <div className="flex justify-between text-xs">
                  <span className="text-green-400">{policy.compliance.compliant} compliant</span>
                  <span className="text-red-400">{policy.compliance.nonCompliant} non-compliant</span>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Non-Compliant Resources */}
          {nonCompliantResources.length > 0 && (
            <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-yellow-400" />
                Non-Compliant Resources Requiring Action
              </h2>
              <div className="space-y-3">
                {nonCompliantResources.map((resource) => (
                  <div key={resource.id} className="flex items-center justify-between p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
                    <div>
                      <p className="text-sm font-medium text-white">{resource.name}</p>
                      <p className="text-xs text-gray-400">{resource.resourceGroup} • {resource.type}</p>
                      <p className="text-xs text-red-300 mt-1">{resource.reason}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-400">{resource.lastEvaluated}</span>
                      <button className="px-3 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700 transition-colors">
                        Remediate
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Policy Details Modal */}
          {showDetails && selectedPolicy && (
            <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-slate-900 rounded-xl border border-white/20 max-w-4xl w-full max-h-[90vh] overflow-y-auto"
              >
                <div className="p-6">
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h2 className="text-2xl font-bold text-white">{selectedPolicy.name}</h2>
                      <p className="text-sm text-gray-400 mt-1">{selectedPolicy.description}</p>
                    </div>
                    <button
                      onClick={() => setShowDetails(false)}
                      className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                    >
                      <XCircle className="w-5 h-5 text-gray-400" />
                    </button>
                  </div>

                  <div className="grid grid-cols-2 gap-6 mb-6">
                    <div>
                      <h3 className="text-sm font-medium text-gray-400 mb-3">Policy Information</h3>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Type:</span>
                          <span className="text-sm text-white">{selectedPolicy.type}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Effect:</span>
                          <span className={`text-sm px-2 py-1 rounded ${
                            selectedPolicy.effect === 'Deny' ? 'bg-red-900/30 text-red-400' :
                            selectedPolicy.effect === 'Audit' ? 'bg-yellow-900/30 text-yellow-400' :
                            'bg-green-900/30 text-green-400'
                          }`}>
                            {selectedPolicy.effect}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Category:</span>
                          <span className="text-sm text-white">{selectedPolicy.category}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Status:</span>
                          <span className="text-sm text-white">{selectedPolicy.status}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Created By:</span>
                          <span className="text-sm text-white">{selectedPolicy.createdBy}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Last Modified:</span>
                          <span className="text-sm text-white">{selectedPolicy.lastModified}</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-sm font-medium text-gray-400 mb-3">Compliance Summary</h3>
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm text-gray-400">Overall Compliance</span>
                            <span className="text-sm text-white">{selectedPolicy.compliance.percentage.toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-700 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full ${
                                selectedPolicy.compliance.percentage >= 90 ? 'bg-green-400' :
                                selectedPolicy.compliance.percentage >= 70 ? 'bg-yellow-400' :
                                'bg-red-400'
                              }`}
                              style={{ width: `${selectedPolicy.compliance.percentage}%` }}
                            />
                          </div>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-center">
                          <div className="p-2 bg-green-900/20 rounded">
                            <p className="text-lg font-bold text-green-400">{selectedPolicy.compliance.compliant}</p>
                            <p className="text-xs text-gray-400">Compliant</p>
                          </div>
                          <div className="p-2 bg-red-900/20 rounded">
                            <p className="text-lg font-bold text-red-400">{selectedPolicy.compliance.nonCompliant}</p>
                            <p className="text-xs text-gray-400">Non-Compliant</p>
                          </div>
                          <div className="p-2 bg-yellow-900/20 rounded">
                            <p className="text-lg font-bold text-yellow-400">{selectedPolicy.compliance.exempt}</p>
                            <p className="text-xs text-gray-400">Exempt</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Parameters */}
                  {Object.keys(selectedPolicy.parameters).length > 0 && (
                    <div className="mb-6">
                      <h3 className="text-sm font-medium text-gray-400 mb-3">Parameters</h3>
                      <div className="bg-black/30 rounded-lg p-3">
                        <pre className="text-xs text-gray-300">
                          {JSON.stringify(selectedPolicy.parameters, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* Resource Types */}
                  <div className="mb-6">
                    <h3 className="text-sm font-medium text-gray-400 mb-3">Applies To</h3>
                    <div className="flex flex-wrap gap-2">
                      {selectedPolicy.resourceTypes.map((type, index) => (
                        <span key={index} className="px-3 py-1 bg-purple-600/20 text-purple-300 rounded-lg text-sm">
                          {type}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="flex justify-end gap-3">
                    <button className="px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors">
                      View in Azure Portal
                    </button>
                    <button className="px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors flex items-center gap-2">
                      <Download className="w-4 h-4" />
                      Export Definition
                    </button>
                    <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2">
                      <Edit className="w-4 h-4" />
                      Edit Policy
                    </button>
                  </div>
                </div>
              </motion.div>
            </div>
          )}
        </div>
      </div>
    </AppLayout>
  )
}