'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import AppLayout from '../../components/AppLayout'
import { useAzureResources, type AzureResource } from '../../lib/azure-api'
import { 
  Server,
  Search,
  Filter,
  ChevronDown,
  AlertCircle,
  CheckCircle,
  XCircle,
  Clock,
  DollarSign,
  Cpu,
  HardDrive,
  Database,
  Globe,
  Shield,
  Activity,
  TrendingUp,
  TrendingDown,
  MoreVertical,
  Play,
  Pause,
  Trash2,
  RefreshCw,
  Settings,
  Tag,
  MapPin,
  Layers,
  Zap
} from 'lucide-react'
import ActionDrawer from '../../components/ActionDrawer'
import type { CreateActionRequest } from '../../lib/actions-api'

export default function ResourcesPage() {
  const { resources: azureResources, loading, error } = useAzureResources()
  // Read initial filters from query string
  const initialParams = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null
  const initialType = initialParams?.get('type') || 'all'
  const initialStatus = initialParams?.get('status') || 'all'
  // Enable deep-link support via hash (?sel=<id>) without Next dynamic route for now
  // If a hash or query is provided, preselect that resource in the modal
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState(initialType)
  const [filterStatus, setFilterStatus] = useState(initialStatus)
  const [selectedResource, setSelectedResource] = useState<AzureResource | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [drawerRequest, setDrawerRequest] = useState<CreateActionRequest | null>(null)

  // Transform Azure resources to match our UI needs
  const resources = azureResources?.map(r => ({
    ...r,
    cost: r.cost || 0,
    monthlyCost: r.monthlyCost || 0,
    savings: r.savings || 0,
    cpu: r.cpu || 0,
    memory: r.memory || 0,
    storage: r.storage || 0,
    createdDate: r.createdDate || '2024-01-01',
    lastModified: r.lastModified || '2025-01-08',
    recommendations: r.recommendations || []
  })) || []

  const filteredResources = resources.filter(resource => {
    const matchesSearch = resource.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         resource.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         resource.resourceGroup.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         resource.location.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         Object.entries(resource.tags || {}).some(([key, value]) => 
                           key.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           value.toLowerCase().includes(searchQuery.toLowerCase())
                         )
    
    const matchesType = filterType === 'all' || resource.type.includes(filterType)
    const matchesStatus = filterStatus === 'all' || resource.status === filterStatus
    
    return matchesSearch && matchesType && matchesStatus
  })

  // Keep URL in sync when filters change
  useEffect(() => {
    if (typeof window === 'undefined') return
    const url = new URL(window.location.href)
    if (filterType && filterType !== 'all') url.searchParams.set('type', filterType); else url.searchParams.delete('type')
    if (filterStatus && filterStatus !== 'all') url.searchParams.set('status', filterStatus); else url.searchParams.delete('status')
    window.history.replaceState(null, '', url.toString())
  }, [filterType, filterStatus])

  const totalCost = filteredResources.reduce((sum, r) => sum + r.monthlyCost, 0)
  const totalSavings = filteredResources.reduce((sum, r) => sum + r.savings, 0)
  const idleResources = filteredResources.filter(r => r.status === 'Idle')
  const nonCompliant = filteredResources.filter(r => r.compliance === 'Non-Compliant')

  const handleResourceAction = (resource: AzureResource, action: string) => {
    // Example: open action drawer to run a remediation/operation
    const req: CreateActionRequest = {
      action_type: action,
      resource_id: resource.id,
      params: { name: resource.name }
    }
    setDrawerRequest(req)
    setDrawerOpen(true)
  }

  return (
    <AppLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">Resource Management</h1>
            <p className="text-gray-400">Complete visibility and control over all Azure resources</p>
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20"
            >
              <div className="flex items-center justify-between mb-2">
                <Server className="w-5 h-5 text-blue-400" />
                <span className="text-xs text-gray-400">Total</span>
              </div>
              <p className="text-2xl font-bold text-white">{filteredResources.length}</p>
              <p className="text-sm text-gray-300">Resources</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20"
            >
              <div className="flex items-center justify-between mb-2">
                <DollarSign className="w-5 h-5 text-green-400" />
                <span className="text-xs text-gray-400">Monthly</span>
              </div>
              <p className="text-2xl font-bold text-white">${totalCost.toFixed(2)}</p>
              <p className="text-sm text-gray-300">Total Cost</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20"
            >
              <div className="flex items-center justify-between mb-2">
                <TrendingDown className="w-5 h-5 text-yellow-400" />
                <span className="text-xs text-gray-400">Potential</span>
              </div>
              <p className="text-2xl font-bold text-white">${totalSavings.toFixed(2)}</p>
              <p className="text-sm text-gray-300">Savings</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20"
            >
              <div className="flex items-center justify-between mb-2">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <span className="text-xs text-gray-400">Action</span>
              </div>
              <p className="text-2xl font-bold text-white">{idleResources.length}</p>
              <p className="text-sm text-gray-300">Idle Resources</p>
            </motion.div>
          </div>

          {/* Search and Filters */}
          <div className="flex flex-col md:flex-row gap-4 mb-6">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search by name, type, resource group, location, or tags..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-400"
              />
            </div>
            
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-400"
            >
              <option value="all">All Types</option>
              <option value="virtualMachines">Virtual Machines</option>
              <option value="storageAccounts">Storage</option>
              <option value="databases">Databases</option>
              <option value="managedClusters">Kubernetes</option>
              <option value="Web">Web Apps</option>
            </select>
            
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-400"
            >
              <option value="all">All Status</option>
              <option value="Running">Running</option>
              <option value="Stopped">Stopped</option>
              <option value="Idle">Idle</option>
              <option value="Optimized">Optimized</option>
              <option value="Over-provisioned">Over-provisioned</option>
            </select>
          </div>

          {/* Resources Table */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-white/5 border-b border-white/10">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                      Resource
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                      Location
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                      Compliance
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                      Cost/Month
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                      Utilization
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {filteredResources.map((resource) => (
                    <tr 
                      key={resource.id} 
                      className="hover:bg-white/5 cursor-pointer transition-colors"
                      onClick={() => {
                        // Navigate to deep drill-in page for this resource
                        if (typeof window !== 'undefined') {
                          window.location.href = `/resources/${encodeURIComponent(resource.id)}`
                        }
                      }}
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <p className="text-sm font-medium text-white">{resource.name}</p>
                          <p className="text-xs text-gray-400">{resource.resourceGroup}</p>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <p className="text-sm text-gray-300">{resource.type.split('/')[1]}</p>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center gap-1">
                          <MapPin className="w-3 h-3 text-gray-400" />
                          <span className="text-sm text-gray-300">{resource.location}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 text-xs rounded-lg ${
                          resource.status === 'Running' ? 'bg-green-900/30 text-green-400' :
                          resource.status === 'Idle' ? 'bg-yellow-900/30 text-yellow-400' :
                          resource.status === 'Stopped' ? 'bg-gray-900/30 text-gray-400' :
                          resource.status === 'Optimized' ? 'bg-blue-900/30 text-blue-400' :
                          'bg-orange-900/30 text-orange-400'
                        }`}>
                          {resource.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 text-xs rounded-lg ${
                          resource.compliance === 'Compliant' ? 'bg-green-900/30 text-green-400' :
                          resource.compliance === 'Warning' ? 'bg-yellow-900/30 text-yellow-400' :
                          'bg-red-900/30 text-red-400'
                        }`}>
                          {resource.compliance}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <p className="text-sm text-white">${resource.monthlyCost.toFixed(2)}</p>
                          {resource.savings > 0 && (
                            <p className="text-xs text-yellow-400">Save ${resource.savings.toFixed(2)}</p>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex gap-2">
                          {resource.cpu > 0 && (
                            <div className="text-xs">
                              <span className="text-gray-400">CPU:</span>
                              <span className={`ml-1 ${
                                resource.cpu > 80 ? 'text-red-400' :
                                resource.cpu > 50 ? 'text-yellow-400' :
                                'text-green-400'
                              }`}>{resource.cpu}%</span>
                            </div>
                          )}
                          {resource.memory > 0 && (
                            <div className="text-xs">
                              <span className="text-gray-400">RAM:</span>
                              <span className={`ml-1 ${
                                resource.memory > 80 ? 'text-red-400' :
                                resource.memory > 50 ? 'text-yellow-400' :
                                'text-green-400'
                              }`}>{resource.memory}%</span>
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          {resource.status === 'Running' ? (
                            <button
                              onClick={(e) => { e.stopPropagation(); handleResourceAction(resource, 'stop') }}
                              className="p-1 hover:bg-white/10 rounded transition-colors"
                              title="Stop"
                            >
                              <Pause className="w-4 h-4 text-gray-400" />
                            </button>
                          ) : (
                            <button
                              onClick={(e) => { e.stopPropagation(); handleResourceAction(resource, 'start') }}
                              className="p-1 hover:bg-white/10 rounded transition-colors"
                              title="Start"
                            >
                              <Play className="w-4 h-4 text-gray-400" />
                            </button>
                          )}
                          <button
                            onClick={(e) => { e.stopPropagation(); handleResourceAction(resource, 'restart') }}
                            className="p-1 hover:bg-white/10 rounded transition-colors"
                            title="Restart"
                          >
                            <RefreshCw className="w-4 h-4 text-gray-400" />
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); handleResourceAction(resource, 'delete') }}
                            className="p-1 hover:bg-white/10 rounded transition-colors"
                            title="Delete"
                          >
                            <Trash2 className="w-4 h-4 text-red-400" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Resource Details Modal */}
          {showDetails && selectedResource && (
            <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-slate-900 rounded-xl border border-white/20 max-w-4xl w-full max-h-[90vh] overflow-y-auto"
              >
                <div className="p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-2xl font-bold text-white">{selectedResource.name}</h2>
                    <button
                      onClick={() => setShowDetails(false)}
                      className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                    >
                      <XCircle className="w-5 h-5 text-gray-400" />
                    </button>
                  </div>

                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-sm font-medium text-gray-400 mb-3">Resource Information</h3>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Type:</span>
                          <span className="text-sm text-white">{selectedResource.type}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Resource Group:</span>
                          <span className="text-sm text-white">{selectedResource.resourceGroup}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Location:</span>
                          <span className="text-sm text-white">{selectedResource.location}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Created:</span>
                          <span className="text-sm text-white">{selectedResource.createdDate}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Last Modified:</span>
                          <span className="text-sm text-white">{selectedResource.lastModified}</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-sm font-medium text-gray-400 mb-3">Cost Analysis</h3>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Hourly Cost:</span>
                          <span className="text-sm text-white">${(selectedResource.cost || 0).toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Monthly Cost:</span>
                          <span className="text-sm text-white">${(selectedResource.monthlyCost || 0).toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Potential Savings:</span>
                          <span className="text-sm text-yellow-400">${(selectedResource.savings || 0).toFixed(2)}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-6">
                    <h3 className="text-sm font-medium text-gray-400 mb-3">Tags</h3>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(selectedResource.tags || {}).map(([key, value]) => (
                        <span key={key} className="px-3 py-1 bg-purple-600/20 text-purple-300 rounded-lg text-sm">
                          {key}: {value}
                        </span>
                      ))}
                    </div>
                  </div>

                  {selectedResource.recommendations && selectedResource.recommendations.length > 0 && (
                    <div className="mt-6">
                      <h3 className="text-sm font-medium text-gray-400 mb-3">AI Recommendations</h3>
                      <div className="space-y-2">
                        {selectedResource.recommendations.map((rec, index) => (
                          <div key={index} className="flex items-start gap-2 p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
                            <Zap className="w-4 h-4 text-yellow-400 mt-0.5" />
                            <p className="text-sm text-yellow-300">{rec}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="mt-6 flex justify-end gap-3">
                    <button className="px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors">
                      View in Azure Portal
                    </button>
                    <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                      Apply Recommendations
                    </button>
                  </div>
                </div>
              </motion.div>
            </div>
          )}
        </div>
      </div>
      <ActionDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} request={drawerRequest} />
    </AppLayout>
  )
}