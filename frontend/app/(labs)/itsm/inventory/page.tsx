'use client'

import { useState, useEffect, useCallback } from 'react'
import { 
  Search, Filter, Download, RefreshCw, ChevronDown, ChevronRight,
  Play, Square, Pause, Trash2, Tag, MoreVertical, 
  CheckCircle, XCircle, AlertTriangle, Clock, Wrench,
  Server, Database, Globe, Shield, HardDrive, Network,
  Cloud, Cpu, Package, DollarSign, Activity, AlertCircle
} from 'lucide-react'

interface Resource {
  id: string
  name: string
  type: string
  provider: 'Azure' | 'AWS' | 'GCP'
  status: 'running' | 'stopped' | 'idle' | 'orphaned' | 'degraded' | 'scheduled' | 'maintenance' | 'decommissioned'
  region: string
  resourceGroup?: string
  tags: Record<string, string>
  cost: number
  lastActivity: string
  created: string
  owner?: string
  application?: string
  environment: 'production' | 'staging' | 'development' | 'test'
  health?: number
  metrics?: {
    cpu?: number
    memory?: number
    disk?: number
    network?: number
  }
}

const mockResources: Resource[] = [
  {
    id: 'vm-001',
    name: 'web-server-01',
    type: 'Virtual Machine',
    provider: 'Azure',
    status: 'running',
    region: 'East US',
    resourceGroup: 'rg-production',
    tags: { environment: 'production', team: 'web', costCenter: 'CC100' },
    cost: 125.50,
    lastActivity: '2 minutes ago',
    created: '2023-06-15',
    owner: 'john.doe@company.com',
    application: 'E-Commerce Platform',
    environment: 'production',
    health: 98,
    metrics: { cpu: 45, memory: 62, disk: 38, network: 12 }
  },
  {
    id: 'db-002',
    name: 'postgres-prod-01',
    type: 'Database',
    provider: 'AWS',
    status: 'running',
    region: 'us-east-1',
    tags: { environment: 'production', service: 'database', backup: 'daily' },
    cost: 450.00,
    lastActivity: '5 minutes ago',
    created: '2023-01-20',
    owner: 'sarah.admin@company.com',
    application: 'Customer Database',
    environment: 'production',
    health: 100,
    metrics: { cpu: 22, memory: 78, disk: 65, network: 34 }
  },
  {
    id: 'vm-003',
    name: 'test-server-legacy',
    type: 'Virtual Machine',
    provider: 'Azure',
    status: 'idle',
    region: 'West Europe',
    resourceGroup: 'rg-test',
    tags: { environment: 'test', status: 'idle' },
    cost: 89.00,
    lastActivity: '3 days ago',
    created: '2022-11-10',
    application: 'Legacy App',
    environment: 'test',
    health: 75,
    metrics: { cpu: 2, memory: 15, disk: 45, network: 0 }
  },
  {
    id: 'storage-004',
    name: 'backup-storage-01',
    type: 'Storage',
    provider: 'GCP',
    status: 'running',
    region: 'us-central1',
    tags: { type: 'backup', retention: '30days' },
    cost: 35.75,
    lastActivity: '1 hour ago',
    created: '2023-03-05',
    owner: 'backup.team@company.com',
    application: 'Backup System',
    environment: 'production',
    health: 100
  },
  {
    id: 'vm-005',
    name: 'abandoned-dev-03',
    type: 'Virtual Machine',
    provider: 'AWS',
    status: 'orphaned',
    region: 'eu-west-1',
    tags: {},
    cost: 156.00,
    lastActivity: '2 weeks ago',
    created: '2023-02-28',
    environment: 'development',
    health: 0
  },
  {
    id: 'lb-006',
    name: 'api-loadbalancer',
    type: 'Load Balancer',
    provider: 'Azure',
    status: 'degraded',
    region: 'East US 2',
    resourceGroup: 'rg-api',
    tags: { service: 'api', tier: 'premium' },
    cost: 245.00,
    lastActivity: '30 seconds ago',
    created: '2023-05-12',
    owner: 'platform.team@company.com',
    application: 'API Gateway',
    environment: 'production',
    health: 65,
    metrics: { cpu: 89, memory: 45, network: 78 }
  },
  {
    id: 'k8s-007',
    name: 'microservices-cluster',
    type: 'Kubernetes Cluster',
    provider: 'GCP',
    status: 'running',
    region: 'europe-west1',
    tags: { orchestration: 'k8s', version: '1.27' },
    cost: 890.00,
    lastActivity: '1 minute ago',
    created: '2023-04-01',
    owner: 'devops@company.com',
    application: 'Microservices Platform',
    environment: 'production',
    health: 95,
    metrics: { cpu: 67, memory: 72, disk: 55, network: 45 }
  },
  {
    id: 'vm-008',
    name: 'batch-processor',
    type: 'Virtual Machine',
    provider: 'Azure',
    status: 'scheduled',
    region: 'Central US',
    resourceGroup: 'rg-batch',
    tags: { schedule: 'nightly', purpose: 'batch' },
    cost: 78.50,
    lastActivity: '8 hours ago',
    created: '2023-07-20',
    owner: 'data.team@company.com',
    application: 'Batch Processing',
    environment: 'production',
    health: 90
  },
  {
    id: 'db-009',
    name: 'analytics-warehouse',
    type: 'Database',
    provider: 'AWS',
    status: 'maintenance',
    region: 'us-west-2',
    tags: { type: 'warehouse', maintenance: 'scheduled' },
    cost: 1250.00,
    lastActivity: 'In maintenance',
    created: '2022-09-15',
    owner: 'analytics@company.com',
    application: 'Data Warehouse',
    environment: 'production',
    health: 100
  },
  {
    id: 'vm-010',
    name: 'deprecated-app-server',
    type: 'Virtual Machine',
    provider: 'Azure',
    status: 'decommissioned',
    region: 'North Europe',
    resourceGroup: 'rg-deprecated',
    tags: { status: 'decommissioned', deleteAfter: '2024-01-01' },
    cost: 0,
    lastActivity: '1 month ago',
    created: '2021-06-10',
    environment: 'development',
    health: 0
  }
]

export default function InventoryPage() {
  const [resources, setResources] = useState<Resource[]>(mockResources)
  const [filteredResources, setFilteredResources] = useState<Resource[]>(mockResources)
  const [selectedResources, setSelectedResources] = useState<Set<string>>(new Set())
  const [searchTerm, setSearchTerm] = useState('')
  const [filterOpen, setFilterOpen] = useState(false)
  const [selectedProvider, setSelectedProvider] = useState<string>('all')
  const [selectedStatus, setSelectedStatus] = useState<string>('all')
  const [selectedType, setSelectedType] = useState<string>('all')
  const [selectedEnvironment, setSelectedEnvironment] = useState<string>('all')
  const [sortBy, setSortBy] = useState<'name' | 'cost' | 'lastActivity' | 'health'>('lastActivity')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [isRefreshing, setIsRefreshing] = useState(false)

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      handleRefresh()
    }, 30000)
    return () => clearInterval(interval)
  }, [])

  // Filter resources based on criteria
  useEffect(() => {
    let filtered = [...resources]

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(r => 
        r.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        r.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        r.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
        r.application?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        r.owner?.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    // Provider filter
    if (selectedProvider !== 'all') {
      filtered = filtered.filter(r => r.provider === selectedProvider)
    }

    // Status filter
    if (selectedStatus !== 'all') {
      filtered = filtered.filter(r => r.status === selectedStatus)
    }

    // Type filter
    if (selectedType !== 'all') {
      filtered = filtered.filter(r => r.type === selectedType)
    }

    // Environment filter
    if (selectedEnvironment !== 'all') {
      filtered = filtered.filter(r => r.environment === selectedEnvironment)
    }

    // Sort
    filtered.sort((a, b) => {
      let comparison = 0
      switch (sortBy) {
        case 'name':
          comparison = a.name.localeCompare(b.name)
          break
        case 'cost':
          comparison = a.cost - b.cost
          break
        case 'health':
          comparison = (a.health || 0) - (b.health || 0)
          break
        case 'lastActivity':
          // This would normally compare actual timestamps
          comparison = 0
          break
      }
      return sortOrder === 'asc' ? comparison : -comparison
    })

    setFilteredResources(filtered)
  }, [resources, searchTerm, selectedProvider, selectedStatus, selectedType, selectedEnvironment, sortBy, sortOrder])

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true)
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    setIsRefreshing(false)
  }, [])

  const handleSelectAll = () => {
    if (selectedResources.size === filteredResources.length) {
      setSelectedResources(new Set())
    } else {
      setSelectedResources(new Set(filteredResources.map(r => r.id)))
    }
  }

  const handleSelectResource = (id: string) => {
    const newSelected = new Set(selectedResources)
    if (newSelected.has(id)) {
      newSelected.delete(id)
    } else {
      newSelected.add(id)
    }
    setSelectedResources(newSelected)
  }

  const handleBulkAction = (action: string) => {
    console.log(`Performing ${action} on ${selectedResources.size} resources`)
    // Implement bulk actions
  }

  const handleExport = () => {
    const data = filteredResources.map(r => ({
      ...r,
      tags: JSON.stringify(r.tags)
    }))
    const csv = [
      Object.keys(data[0]).join(','),
      ...data.map(row => Object.values(row).join(','))
    ].join('\n')
    
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'resource-inventory.csv'
    a.click()
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
      case 'stopped': return <XCircle className="w-4 h-4 text-gray-600 dark:text-gray-400" />
      case 'idle': return <Clock className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
      case 'orphaned': return <AlertTriangle className="w-4 h-4 text-orange-600 dark:text-orange-400" />
      case 'degraded': return <AlertCircle className="w-4 h-4 text-red-600 dark:text-red-400" />
      case 'scheduled': return <Clock className="w-4 h-4 text-blue-600 dark:text-blue-400" />
      case 'maintenance': return <Wrench className="w-4 h-4 text-purple-600 dark:text-purple-400" />
      case 'decommissioned': return <XCircle className="w-4 h-4 text-gray-500" />
      default: return <Activity className="w-4 h-4 text-gray-600 dark:text-gray-400" />
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'Virtual Machine': return <Server className="w-4 h-4" />
      case 'Database': return <Database className="w-4 h-4" />
      case 'Storage': return <HardDrive className="w-4 h-4" />
      case 'Load Balancer': return <Network className="w-4 h-4" />
      case 'Kubernetes Cluster': return <Package className="w-4 h-4" />
      default: return <Cloud className="w-4 h-4" />
    }
  }

  const getProviderColor = (provider: string) => {
    switch (provider) {
      case 'Azure': return 'text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/30'
      case 'AWS': return 'text-orange-600 dark:text-orange-400 bg-orange-100 dark:bg-orange-900/30'
      case 'GCP': return 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30'
      default: return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900/30'
    }
  }

  const resourceStats = {
    total: filteredResources.length,
    running: filteredResources.filter(r => r.status === 'running').length,
    stopped: filteredResources.filter(r => r.status === 'stopped').length,
    idle: filteredResources.filter(r => r.status === 'idle').length,
    orphaned: filteredResources.filter(r => r.status === 'orphaned').length,
    totalCost: filteredResources.reduce((sum, r) => sum + r.cost, 0)
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground dark:text-white">Resource Inventory</h1>
          <p className="text-muted-foreground dark:text-gray-400 mt-1">
            Real-time multi-cloud resource discovery and management
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="px-4 py-2 bg-muted text-foreground rounded-lg hover:bg-accent transition-colors flex items-center gap-2"
          >
            <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={handleExport}
            className="px-4 py-2 bg-muted text-foreground rounded-lg hover:bg-accent transition-colors flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-6 gap-4">
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-foreground dark:text-white">{resourceStats.total}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Total Resources</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">{resourceStats.running}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Running</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-gray-600 dark:text-gray-400">{resourceStats.stopped}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Stopped</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{resourceStats.idle}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Idle</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">{resourceStats.orphaned}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Orphaned</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            ${(resourceStats.totalCost / 1000).toFixed(1)}K
          </div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Monthly Cost</div>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
        <div className="flex gap-4 items-center">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground dark:text-gray-400" />
            <input
              type="text"
              placeholder="Search by name, ID, type, application, or owner..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          {/* Filter Button */}
          <button
            onClick={() => setFilterOpen(!filterOpen)}
            className="px-4 py-2 bg-muted text-foreground rounded-lg hover:bg-accent transition-colors flex items-center gap-2"
          >
            <Filter className="w-4 h-4" />
            Filters
            <ChevronDown className={`w-4 h-4 transition-transform ${filterOpen ? 'rotate-180' : ''}`} />
          </button>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-4 py-2 bg-muted text-foreground rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="lastActivity">Last Activity</option>
            <option value="name">Name</option>
            <option value="cost">Cost</option>
            <option value="health">Health</option>
          </select>

          <button
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            className="px-3 py-2 bg-muted text-foreground rounded-lg hover:bg-accent transition-colors"
          >
            {sortOrder === 'asc' ? '↑' : '↓'}
          </button>
        </div>

        {/* Expanded Filters */}
        {filterOpen && (
          <div className="mt-4 pt-4 border-t border-border dark:border-gray-700 grid grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-foreground dark:text-white mb-1">Provider</label>
              <select
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value)}
                className="w-full px-3 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="all">All Providers</option>
                <option value="Azure">Azure</option>
                <option value="AWS">AWS</option>
                <option value="GCP">GCP</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground dark:text-white mb-1">Status</label>
              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
                className="w-full px-3 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="all">All Status</option>
                <option value="running">Running</option>
                <option value="stopped">Stopped</option>
                <option value="idle">Idle</option>
                <option value="orphaned">Orphaned</option>
                <option value="degraded">Degraded</option>
                <option value="scheduled">Scheduled</option>
                <option value="maintenance">Maintenance</option>
                <option value="decommissioned">Decommissioned</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground dark:text-white mb-1">Type</label>
              <select
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value)}
                className="w-full px-3 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="all">All Types</option>
                <option value="Virtual Machine">Virtual Machine</option>
                <option value="Database">Database</option>
                <option value="Storage">Storage</option>
                <option value="Load Balancer">Load Balancer</option>
                <option value="Kubernetes Cluster">Kubernetes Cluster</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground dark:text-white mb-1">Environment</label>
              <select
                value={selectedEnvironment}
                onChange={(e) => setSelectedEnvironment(e.target.value)}
                className="w-full px-3 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="all">All Environments</option>
                <option value="production">Production</option>
                <option value="staging">Staging</option>
                <option value="development">Development</option>
                <option value="test">Test</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {/* Bulk Actions Bar */}
      {selectedResources.size > 0 && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 flex items-center justify-between">
          <span className="text-blue-700 dark:text-blue-300">
            {selectedResources.size} resource{selectedResources.size > 1 ? 's' : ''} selected
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => handleBulkAction('start')}
              className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 transition-colors flex items-center gap-1"
            >
              <Play className="w-4 h-4" />
              Start
            </button>
            <button
              onClick={() => handleBulkAction('stop')}
              className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 transition-colors flex items-center gap-1"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
            <button
              onClick={() => handleBulkAction('tag')}
              className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors flex items-center gap-1"
            >
              <Tag className="w-4 h-4" />
              Tag
            </button>
            <button
              onClick={() => handleBulkAction('delete')}
              className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 transition-colors flex items-center gap-1"
            >
              <Trash2 className="w-4 h-4" />
              Delete
            </button>
          </div>
        </div>
      )}

      {/* Resources Table */}
      <div className="bg-card dark:bg-gray-800 rounded-lg border border-border dark:border-gray-700 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-muted dark:bg-gray-900 border-b border-border dark:border-gray-700">
              <tr>
                <th className="p-4 text-left">
                  <input
                    type="checkbox"
                    checked={selectedResources.size === filteredResources.length && filteredResources.length > 0}
                    onChange={handleSelectAll}
                    className="rounded"
                  />
                </th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Resource</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Type</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Provider</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Status</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Environment</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Cost/Month</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Health</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Last Activity</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredResources.map((resource) => (
                <tr key={resource.id} className="border-b border-border dark:border-gray-700 hover:bg-muted dark:hover:bg-gray-900">
                  <td className="p-4">
                    <input
                      type="checkbox"
                      checked={selectedResources.has(resource.id)}
                      onChange={() => handleSelectResource(resource.id)}
                      className="rounded"
                    />
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-muted dark:bg-gray-900 rounded">
                        {getTypeIcon(resource.type)}
                      </div>
                      <div>
                        <div className="font-medium text-foreground dark:text-white">{resource.name}</div>
                        <div className="text-sm text-muted-foreground dark:text-gray-400">{resource.id}</div>
                        {resource.application && (
                          <div className="text-xs text-blue-600 dark:text-blue-400">{resource.application}</div>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="p-4 text-sm text-foreground dark:text-gray-300">{resource.type}</td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getProviderColor(resource.provider)}`}>
                      {resource.provider}
                    </span>
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(resource.status)}
                      <span className="text-sm text-foreground dark:text-gray-300 capitalize">{resource.status}</span>
                    </div>
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      resource.environment === 'production' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                      resource.environment === 'staging' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                      'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
                    }`}>
                      {resource.environment}
                    </span>
                  </td>
                  <td className="p-4">
                    <div className="text-sm font-medium text-foreground dark:text-white">
                      ${resource.cost.toFixed(2)}
                    </div>
                  </td>
                  <td className="p-4">
                    {resource.health !== undefined && (
                      <div className="flex items-center gap-2">
                        <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              resource.health >= 90 ? 'bg-green-500' :
                              resource.health >= 70 ? 'bg-yellow-500' :
                              resource.health >= 50 ? 'bg-orange-500' :
                              'bg-red-500'
                            }`}
                            style={{ width: `${resource.health}%` }}
                          />
                        </div>
                        <span className="text-xs text-muted-foreground dark:text-gray-400">{resource.health}%</span>
                      </div>
                    )}
                  </td>
                  <td className="p-4 text-sm text-muted-foreground dark:text-gray-400">{resource.lastActivity}</td>
                  <td className="p-4">
                    <div className="flex items-center gap-1">
                      {resource.status === 'stopped' && (
                        <button className="p-1 hover:bg-muted dark:hover:bg-gray-700 rounded" title="Start">
                          <Play className="w-4 h-4 text-green-600 dark:text-green-400" />
                        </button>
                      )}
                      {resource.status === 'running' && (
                        <button className="p-1 hover:bg-muted dark:hover:bg-gray-700 rounded" title="Stop">
                          <Square className="w-4 h-4 text-red-600 dark:text-red-400" />
                        </button>
                      )}
                      <button className="p-1 hover:bg-muted dark:hover:bg-gray-700 rounded" title="Tag">
                        <Tag className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                      </button>
                      <button className="p-1 hover:bg-muted dark:hover:bg-gray-700 rounded" title="More">
                        <MoreVertical className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      <div className="flex justify-between items-center">
        <div className="text-sm text-muted-foreground dark:text-gray-400">
          Showing {filteredResources.length} of {resources.length} resources
        </div>
        <div className="flex gap-2">
          <button className="px-3 py-1 bg-muted text-foreground rounded hover:bg-accent transition-colors">
            Previous
          </button>
          <button className="px-3 py-1 bg-primary text-primary-foreground rounded">1</button>
          <button className="px-3 py-1 bg-muted text-foreground rounded hover:bg-accent transition-colors">2</button>
          <button className="px-3 py-1 bg-muted text-foreground rounded hover:bg-accent transition-colors">3</button>
          <button className="px-3 py-1 bg-muted text-foreground rounded hover:bg-accent transition-colors">
            Next
          </button>
        </div>
      </div>
    </div>
  )
}