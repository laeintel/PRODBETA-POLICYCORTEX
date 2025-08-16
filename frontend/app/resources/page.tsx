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

import { useState, useEffect, useMemo, useCallback } from 'react'
import { usePathname } from 'next/navigation'
import dynamic from 'next/dynamic'
import { motion } from 'framer-motion'
import AppLayout from '../../components/AppLayout'
import MockDataIndicator, { useMockDataStatus } from '@/components/MockDataIndicator'
const VirtualizedTable = dynamic(() => import('@/components/VirtualizedTable'), { ssr: false })
const ActionDrawer = dynamic(() => import('../../components/ActionDrawer'), { ssr: false })
import ServerPagination from '@/components/ServerPagination'
import { useServerPagination } from '@/hooks/useServerPagination'
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
// ActionDrawer dynamically imported above to reduce initial bundle size
import type { CreateActionRequest } from '../../lib/actions-api'
import Pagination from '../../components/Pagination'

export default function ResourcesPage() {
  const { resources: azureResources, loading, error, isUsingFallback } = useAzureResources()
  const pathname = usePathname()
  const { isMockData } = useMockDataStatus()
  // Read initial filters from query string
  const initialParams = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null
  const initialType = initialParams?.get('type') || 'all'
  const initialStatus = initialParams?.get('status') || 'all'
  // Enable deep-link support via hash (?sel=<id>) without Next dynamic route for now
  // If a hash or query is provided, preselect that resource in the modal
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState(initialType)
  // Apply subroute filters, e.g. /resources/vm, /resources/storage
  useEffect(() => {
    if (!pathname) return
    if (pathname.startsWith('/resources/')) {
      const seg = pathname.split('/')[2]
      const map: Record<string,string> = {
        vm: 'virtualMachines',
        storage: 'storageAccounts',
        db: 'databases',
        k8s: 'managedClusters',
        web: 'Web',
      }
      if (map[seg]) setFilterType(map[seg])
    }
  }, [pathname])
  const [filterStatus, setFilterStatus] = useState(initialStatus)
  const [selectedResource, setSelectedResource] = useState<AzureResource | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [drawerRequest, setDrawerRequest] = useState<CreateActionRequest | null>(null)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(25)

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

  const filteredResources = useMemo(() => resources.filter(resource => {
    const q = (searchQuery || '').toLowerCase()
    const name = (resource?.name || '').toLowerCase()
    const type = (resource?.type || '').toLowerCase()
    const rg = (resource?.resourceGroup || '').toLowerCase()
    const location = (resource?.location || '').toLowerCase()
    const matchesSearch = name.includes(q) ||
                         type.includes(q) ||
                         rg.includes(q) ||
                         location.includes(q) ||
                         Object.entries(resource.tags || {}).some(([key, value]) => 
                           (key || '').toLowerCase().includes(q) ||
                           String(value ?? '').toLowerCase().includes(q)
                         )
    
    const matchesType = filterType === 'all' || (resource.type || '').includes(filterType)
    const matchesStatus = filterStatus === 'all' || (resource.status || '') === filterStatus
    
    return matchesSearch && matchesType && matchesStatus
  }), [resources, searchQuery, filterType, filterStatus])

  const pagedResources = useMemo(() => {
    const start = (page - 1) * pageSize
    return filteredResources.slice(start, start + pageSize)
  }, [filteredResources, page, pageSize])

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
          {/* Mock Data Indicator */}
          {(isMockData || isUsingFallback || error) && (
            <MockDataIndicator 
              type="banner" 
              dataSource={error ? "Cached Resources (API Error)" : isUsingFallback ? "Sample Resources" : "Mock Resources"}
              className="mb-6"
            />
          )}
          
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">Resource Management</h1>
                <p className="text-gray-400">Complete visibility and control over all Azure resources</p>
              </div>
              {!isMockData && !isUsingFallback && !error && (
                <MockDataIndicator type="badge" />
              )}
            </div>
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

          {/* Resources Table (Virtualized) */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 overflow-hidden">
            <VirtualizedTable
              data={filteredResources}
              rowHeight={56}
              overscan={10}
              onRowClick={(resource) => {
                if (typeof window !== 'undefined') {
                  // @ts-ignore dynamic resource typing
                  window.location.href = `/resources/${encodeURIComponent(resource.id)}`
                }
              }}
              columns={[
                { key: 'name', label: 'Resource', render: (_: any, r: any) => (
                  <div>
                    <p className="text-sm font-medium text-white">{r.name}</p>
                    <p className="text-xs text-gray-400">{r.resourceGroup}</p>
                  </div>
                ) },
                { key: 'type', label: 'Type', render: (v: string) => {
                  const typeStr = v || ''
                  const seg = typeStr.includes('/') ? typeStr.split('/')[1] : typeStr
                  return <p className="text-sm text-gray-300">{seg || 'Unknown'}</p>
                } },
                { key: 'location', label: 'Location', render: (v: string) => (
                  <div className="flex items-center gap-1">
                    <MapPin className="w-3 h-3 text-gray-400" />
                    <span className="text-sm text-gray-300">{v}</span>
                  </div>
                ) },
                { key: 'status', label: 'Status', render: (v: string) => (
                  <span className={`px-2 py-1 text-xs rounded-lg ${
                    v === 'Running' ? 'bg-green-900/30 text-green-400' :
                    v === 'Idle' ? 'bg-yellow-900/30 text-yellow-400' :
                    v === 'Stopped' ? 'bg-gray-900/30 text-gray-400' :
                    v === 'Optimized' ? 'bg-blue-900/30 text-blue-400' :
                    'bg-orange-900/30 text-orange-400'
                  }`}>
                    {v}
                  </span>
                ) },
                { key: 'compliance', label: 'Compliance', render: (v: string) => (
                  <span className={`px-2 py-1 text-xs rounded-lg ${
                    v === 'Compliant' ? 'bg-green-900/30 text-green-400' :
                    v === 'Warning' ? 'bg-yellow-900/30 text-yellow-400' :
                    'bg-red-900/30 text-red-400'
                  }`}>
                    {v}
                  </span>
                ) },
                { key: 'monthlyCost', label: 'Cost/Month', render: (v: number, r: any) => (
                  <div>
                    <p className="text-sm text-white">${(v || 0).toFixed(2)}</p>
                    {r.savings > 0 && (
                      <p className="text-xs text-yellow-400">Save ${r.savings.toFixed(2)}</p>
                    )}
                  </div>
                ) },
                { key: 'cpu', label: 'Utilization', render: (_: any, r: any) => (
                  <div className="flex gap-2">
                    {r.cpu > 0 && (
                      <div className="text-xs">
                        <span className="text-gray-400">CPU:</span>
                        <span className={`ml-1 ${r.cpu > 80 ? 'text-red-400' : r.cpu > 50 ? 'text-yellow-400' : 'text-green-400'}`}>{r.cpu}%</span>
                      </div>
                    )}
                    {r.memory > 0 && (
                      <div className="text-xs">
                        <span className="text-gray-400">RAM:</span>
                        <span className={`ml-1 ${r.memory > 80 ? 'text-red-400' : r.memory > 50 ? 'text-yellow-400' : 'text-green-400'}`}>{r.memory}%</span>
                      </div>
                    )}
                  </div>
                ) },
                { key: 'actions', label: 'Actions', render: (_: any, r: any) => (
                  <div className="flex items-center gap-2">
                    {r.status === 'Running' ? (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleResourceAction(r, 'stop') }}
                        className="p-1 hover:bg-white/10 rounded transition-colors"
                        title="Stop"
                      >
                        <Pause className="w-4 h-4 text-gray-400" />
                      </button>
                    ) : (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleResourceAction(r, 'start') }}
                        className="p-1 hover:bg-white/10 rounded transition-colors"
                        title="Start"
                      >
                        <Play className="w-4 h-4 text-gray-400" />
                      </button>
                    )}
                    <button
                      onClick={(e) => { e.stopPropagation(); handleResourceAction(r, 'restart') }}
                      className="p-1 hover:bg-white/10 rounded transition-colors"
                      title="Restart"
                    >
                      <RefreshCw className="w-4 h-4 text-gray-400" />
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleResourceAction(r, 'delete') }}
                      className="p-1 hover:bg-white/10 rounded transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4 text-red-400" />
                    </button>
                  </div>
                ) },
              ]}
            />

            {/* Optional: keep pagination UI for UX familiarity */}
            <div className="sticky bottom-0">
              <Pagination
                page={page}
                pageSize={pageSize}
                total={filteredResources.length}
                onPageChange={setPage}
                onPageSizeChange={setPageSize}
              />
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