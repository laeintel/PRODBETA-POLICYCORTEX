'use client'

import { useState, useEffect, useMemo } from 'react'
import { useSearchParams } from 'next/navigation'
import { motion } from 'framer-motion'
import VirtualizedTable from '@/components/VirtualizedTable'
import ServerPagination from '@/components/ServerPagination'
import { useServerPagination } from '@/hooks/useServerPagination'
import { ChartCard, ComplianceTrend } from '../../components/ChartCards'
import FilterBar from '../../components/FilterBar'
import AppLayout from '../../components/AppLayout'
import { azurePolicies, getPolicyStatistics, getNonCompliantResources, type PolicyDefinition } from '../../lib/policies-data'
import MockDataIndicator, { DataWithIndicator, useMockDataStatus } from '@/components/MockDataIndicator'
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
import ActionDrawer from '../../components/ActionDrawer'
import type { CreateActionRequest } from '../../lib/actions-api'

interface NonCompliantResource {
  id: string
  name: string
  type: string
  resourceGroup: string
  subscription: string
  violations: { policy: string; reason: string }[]
  lastEvaluated: string
  riskLevel: string
}

export default function PoliciesPage() {
  const [policies, setPolicies] = useState<PolicyDefinition[]>([])
  const [loading, setLoading] = useState(true)
  const [isUsingFallback, setIsUsingFallback] = useState(false)
  const { isMockData } = useMockDataStatus()
  const searchParams = useSearchParams()
  const [searchQuery, setSearchQuery] = useState('')
  const [filterCategory, setFilterCategory] = useState('all')
  const [filterStatus, setFilterStatus] = useState('all')
  const [selectedPolicy, setSelectedPolicy] = useState<PolicyDefinition | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [nonCompliantResources, setNonCompliantResources] = useState<NonCompliantResource[]>([])
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [drawerRequest, setDrawerRequest] = useState<CreateActionRequest | null>(null)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const pagination = useServerPagination({ initialPageSize: 25 })
  const [showStatistics, setShowStatistics] = useState(true)

  useEffect(() => {
    // Try to fetch real policies first
    const fetchPolicies = async () => {
      try {
        const response = await fetch('/api/v1/policies/deep')
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        const data = await response.json()
        setPolicies(data.policies || [])
        setNonCompliantResources(data.nonCompliantResources || [])
        setIsUsingFallback(false)
      } catch (error) {
        console.warn('Failed to fetch policies, using fallback data:', error)
        // Use fallback data
        setPolicies(azurePolicies)
        setNonCompliantResources(getNonCompliantResources())
        setIsUsingFallback(true)
      } finally {
        setLoading(false)
      }
    }
    
    fetchPolicies()
  }, [])

  // Keep URL in sync with filters/search
  // Reflect UI changes into URL (one-way to avoid loops)
  useEffect(() => {
    if (typeof window === 'undefined') return
    const url = new URL(window.location.href)
    const q = url.searchParams.get('q') || ''
    const cat = url.searchParams.get('category') || 'all'
    const stat = url.searchParams.get('status') || 'all'
    if (q !== searchQuery) (searchQuery ? url.searchParams.set('q', searchQuery) : url.searchParams.delete('q'))
    if (cat !== filterCategory) (filterCategory !== 'all' ? url.searchParams.set('category', filterCategory) : url.searchParams.delete('category'))
    if (stat !== filterStatus) (filterStatus !== 'all' ? url.searchParams.set('status', filterStatus) : url.searchParams.delete('status'))
    window.history.replaceState(null, '', url.toString())
  }, [searchQuery, filterCategory, filterStatus])

  // Respond to URL param changes (e.g., submenu navigation)
  useEffect(() => {
    const q = searchParams.get('q') || ''
    const cat = searchParams.get('category') || (typeof window !== 'undefined' && window.location.pathname.startsWith('/policies/') ? decodeURIComponent(window.location.pathname.split('/')[2] || 'all') : 'all')
    const normalizedCat = (
      cat.toLowerCase() === 'non-compliant' ? 'Non-Compliant' :
      cat.charAt(0).toUpperCase() + cat.slice(1)
    )
    const stat = searchParams.get('status') || 'all'
    setSearchQuery(q)
    setFilterCategory(cat === 'all' ? 'all' : normalizedCat)
    setFilterStatus(stat)
  }, [searchParams])

  const filteredPolicies = policies.filter(policy => {
    const q = (searchQuery || '').toLowerCase()
    const displayName = (policy?.displayName || '').toLowerCase()
    const description = (policy?.description || '').toLowerCase()
    const name = (policy?.name || '').toLowerCase()
    const matchesSearch = displayName.includes(q) ||
                         description.includes(q) ||
                         name.includes(q)
    const matchesCategory = filterCategory === 'all' || policy.category === filterCategory
    const matchesStatus = filterStatus === 'all' || policy.status === filterStatus
    return matchesSearch && matchesCategory && matchesStatus
  })

  const stats = getPolicyStatistics()

  const complianceTrend = useMemo(() => (
    Array.from({ length: 12 }).map((_, i) => ({ name: `W${i + 1}`, value: 80 + (i % 4) }))
  ), [])

  const totalCompliant = filteredPolicies.reduce((sum, p) => sum + (p.compliance?.compliant || 0), 0)
  const totalNonCompliant = filteredPolicies.reduce((sum, p) => sum + (p.compliance?.nonCompliant || 0), 0)
  const overallCompliance = totalCompliant + totalNonCompliant > 0 
    ? ((totalCompliant / (totalCompliant + totalNonCompliant)) * 100).toFixed(1)
    : '0'

  const openRemediate = (resourceId: string, action: string) => {
    setDrawerRequest({ action_type: action, resource_id: resourceId })
    setDrawerOpen(true)
    // bring drawer to focus in next tick for accessibility
    setTimeout(() => {
      const el = document.getElementById('action-drawer-title')
      el?.focus()
    }, 0)
  }

  const tableColumns = [
    { key: 'displayName', label: 'Name', sortable: true },
    { key: 'category', label: 'Category', sortable: true },
    { key: 'type', label: 'Type', sortable: true },
    { key: 'effect', label: 'Effect', sortable: true },
    { key: 'assignments', label: 'Assignments', sortable: true },
  ] as const

  const paged = filteredPolicies.slice((pagination.page - 1) * pagination.pageSize, pagination.page * pagination.pageSize)

  return (
    <AppLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          {/* Mock Data Indicator */}
          {(isMockData || isUsingFallback) && (
            <MockDataIndicator 
              type="banner" 
              dataSource={isUsingFallback ? "Cached Policy Data" : "Mock Policy Data"}
              className="mb-6"
            />
          )}
          
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">Policy Management</h1>
                <p className="text-gray-400">Azure Policy compliance and governance control</p>
              </div>
              {!isMockData && !isUsingFallback && (
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
              <option value="Cost Management">Cost Management</option>
              <option value="SQL">SQL</option>
              <option value="Kubernetes">Kubernetes</option>
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

            <div className="flex gap-2">
              <button 
                onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
                className="px-3 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
              >
                {viewMode === 'grid' ? <BarChart3 className="w-4 h-4" /> : <Shield className="w-4 h-4" />}
              </button>
              <button 
                onClick={() => setShowStatistics(!showStatistics)}
                className="px-3 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
              >
                <Info className="w-4 h-4" />
              </button>
              <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2">
                <Upload className="w-4 h-4" />
                Import
              </button>
            </div>
          </div>

          {/* Statistics & Trend */}
          {showStatistics && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-8 p-6 bg-white/10 backdrop-blur-md rounded-xl border border-white/20"
            >
              <h2 className="text-xl font-semibold text-white mb-4">Policy Analytics</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <p className="text-3xl font-bold text-purple-400">{stats.total}</p>
                  <p className="text-sm text-gray-400">Total Policies</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-green-400">{stats.active}</p>
                  <p className="text-sm text-gray-400">Active</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-blue-400">{stats.totalAssignments}</p>
                  <p className="text-sm text-gray-400">Assignments</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-yellow-400">{stats.totalRemediationTasks}</p>
                  <p className="text-sm text-gray-400">Remediation Tasks</p>
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-white/10">
                <div className="flex flex-wrap gap-2">
                  {Object.entries(stats.byCategory).map(([category, count]) => (
                    <span key={category} className="px-3 py-1 bg-white/10 text-white rounded-lg text-sm">
                      {category}: {count}
                    </span>
                  ))}
                </div>
              </div>
              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                <ChartCard title="Compliance Trend" subtitle="Last 12 weeks">
                  <ComplianceTrend data={[{name:'W1',value:78},{name:'W2',value:80},{name:'W3',value:81},{name:'W4',value:83},{name:'W5',value:84},{name:'W6',value:85},{name:'W7',value:86},{name:'W8',value:88},{name:'W9',value:89},{name:'W10',value:90},{name:'W11',value:92},{name:'W12',value:93}]} />
                </ChartCard>
                <div className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20">
                  <h3 className="text-sm font-medium text-white mb-2">Quick Filters</h3>
                  <FilterBar
                    facets={[
                      { key: 'subscription', label: 'Subscription' },
                      { key: 'resourceGroup', label: 'Resource Group' },
                      { key: 'effect', label: 'Effect', options: [
                        { label: 'Deny', value: 'Deny' },
                        { label: 'Audit', value: 'Audit' },
                        { label: 'Append', value: 'Append' },
                        { label: 'Modify', value: 'Modify' },
                        { label: 'DeployIfNotExists', value: 'DeployIfNotExists' }
                      ]},
                    ]}
                  />
                </div>
              </div>
            </motion.div>
          )}

          {/* Policies Grid/Table */}
          {viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8 sticky top-2">
            {filteredPolicies.map((policy) => (
              <motion.div
                key={policy.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ scale: 1.02 }}
                onClick={() => {
                  if (typeof window !== 'undefined') {
                    window.location.href = `/policies/${encodeURIComponent(policy.id || policy.name)}`
                  }
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

                <h3 className="text-white font-medium mb-1 line-clamp-1">{policy.displayName}</h3>
                <p className="text-xs text-gray-400 mb-3 line-clamp-2">{policy.description}</p>

                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs text-gray-400">{policy.category}</span>
                  <span className="text-xs text-gray-400">{policy.assignments} assignment{policy.assignments !== 1 ? 's' : ''}</span>
                </div>

                {/* Compliance Bar */}
                <div className="mb-2">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-gray-400">Compliance</span>
                    <span className="text-white">{(policy.compliance?.percentage || 0).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        (policy.compliance?.percentage || 0) >= 90 ? 'bg-green-400' :
                        (policy.compliance?.percentage || 0) >= 70 ? 'bg-yellow-400' :
                        'bg-red-400'
                      }`}
                      style={{ width: `${policy.compliance?.percentage || 0}%` }}
                    />
                  </div>
                </div>

                <div className="flex justify-between text-xs">
                  <span className="text-green-400">{policy.compliance?.compliant || 0} compliant</span>
                  <span className="text-red-400">{policy.compliance?.nonCompliant || 0} non-compliant</span>
                </div>
              </motion.div>
            ))}
          </div>
          ) : (
            <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 overflow-hidden">
              <VirtualizedTable
                data={paged}
                columns={tableColumns as any}
                onRowClick={(p:any)=>{ if (typeof window !== 'undefined') window.location.href = `/policies/${encodeURIComponent(p.id || p.name)}` }}
                rowHeight={48}
                overscan={8}
                loading={false}
              />
              <ServerPagination
                page={pagination.page}
                pageSize={pagination.pageSize}
                total={filteredPolicies.length}
                totalPages={Math.ceil(filteredPolicies.length / pagination.pageSize) || 1}
                onPageChange={pagination.goToPage}
                onPageSizeChange={pagination.setPageSize}
                pageSizeOptions={[10,25,50,100]}
              />
            </div>
          )}

          {/* Non-Compliant Resources */}
          {nonCompliantResources.length > 0 && (
            <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-yellow-400" />
                Non-Compliant Resources Requiring Action
              </h2>
              <div className="space-y-3">
                {nonCompliantResources.slice(0, 5).map((resource) => (
                  <div key={resource.id} className="flex items-center justify-between p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium text-white">{resource.name}</p>
                        <span className={`text-xs px-2 py-0.5 rounded ${
                          resource.riskLevel === 'Critical' ? 'bg-red-900/50 text-red-400' :
                          resource.riskLevel === 'High' ? 'bg-orange-900/50 text-orange-400' :
                          'bg-yellow-900/50 text-yellow-400'
                        }`}>
                          {resource.riskLevel} Risk
                        </span>
                      </div>
                      <p className="text-xs text-gray-400">{resource.subscription} / {resource.resourceGroup}</p>
                      <p className="text-xs text-gray-500 mt-1">{resource.type}</p>
                      <div className="mt-2">
                        {resource.violations.map((v, idx) => (
                          <p key={idx} className="text-xs text-red-300">
                            • {v.policy}: {v.reason}
                          </p>
                        ))}
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      <span className="text-xs text-gray-400">
                        {new Date(resource.lastEvaluated).toLocaleString()}
                      </span>
                      <button onClick={() => openRemediate(resource.id, 'auto-remediate')} className="px-3 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700 transition-colors">
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
                      <h2 className="text-2xl font-bold text-white">{selectedPolicy.displayName}</h2>
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
                          <span className="text-sm text-gray-400">Version:</span>
                          <span className="text-sm text-white">{selectedPolicy.version}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Created By:</span>
                          <span className="text-sm text-white">{selectedPolicy.metadata.createdBy}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Last Modified:</span>
                          <span className="text-sm text-white">{new Date(selectedPolicy.metadata.updatedOn).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-sm font-medium text-gray-400 mb-3">Compliance Summary</h3>
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm text-gray-400">Overall Compliance</span>
                            <span className="text-sm text-white">{(selectedPolicy.compliance?.percentage || 0).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-700 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full ${
                                (selectedPolicy.compliance?.percentage || 0) >= 90 ? 'bg-green-400' :
                                (selectedPolicy.compliance?.percentage || 0) >= 70 ? 'bg-yellow-400' :
                                'bg-red-400'
                              }`}
                              style={{ width: `${selectedPolicy.compliance?.percentage || 0}%` }}
                            />
                          </div>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-center">
                          <div className="p-2 bg-green-900/20 rounded">
                            <p className="text-lg font-bold text-green-400">{selectedPolicy.compliance?.compliant || 0}</p>
                            <p className="text-xs text-gray-400">Compliant</p>
                          </div>
                          <div className="p-2 bg-red-900/20 rounded">
                            <p className="text-lg font-bold text-red-400">{selectedPolicy.compliance?.nonCompliant || 0}</p>
                            <p className="text-xs text-gray-400">Non-Compliant</p>
                          </div>
                          <div className="p-2 bg-yellow-900/20 rounded">
                            <p className="text-lg font-bold text-yellow-400">{selectedPolicy.compliance?.exempt || 0}</p>
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
                    <button disabled title="Disabled in demo" className="px-4 py-2 bg-gray-400 text-white rounded-lg disabled:opacity-60 cursor-not-allowed flex items-center gap-2">
                      <Download className="w-4 h-4" />
                      Export Definition (Disabled)
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
      <ActionDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} request={drawerRequest} />
    </AppLayout>
  )
}