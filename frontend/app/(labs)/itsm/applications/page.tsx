'use client'

import { useState, useEffect } from 'react'
import { 
  Cpu, Activity, Clock, AlertTriangle, CheckCircle, XCircle,
  TrendingUp, TrendingDown, Users, Calendar, DollarSign,
  BarChart3, PieChart, GitBranch, Package, Database, Server,
  Play, Square, RefreshCw, Settings, MoreVertical, Trash2,
  AlertCircle, Wrench, Zap, Globe, Shield, Network
} from 'lucide-react'

interface Application {
  id: string
  name: string
  description: string
  status: 'running' | 'stopped' | 'idle' | 'orphaned' | 'degraded'
  health: number
  environment: 'production' | 'staging' | 'development' | 'test'
  version: string
  lastDeployed: string
  owner: string
  team: string
  dependencies: string[]
  resources: {
    vms: number
    databases: number
    storage: number
    containers: number
  }
  metrics: {
    cpu: number
    memory: number
    requests: number
    errors: number
    latency: number
  }
  cost: {
    monthly: number
    trend: 'up' | 'down' | 'stable'
    percentChange: number
  }
  lastActivity: string
  idleTime?: string
  sla: {
    target: number
    current: number
    status: 'met' | 'at-risk' | 'breached'
  }
  incidents: number
  changes: number
}

const mockApplications: Application[] = [
  {
    id: 'app-001',
    name: 'E-Commerce Platform',
    description: 'Main customer-facing shopping platform',
    status: 'running',
    health: 98,
    environment: 'production',
    version: 'v2.4.1',
    lastDeployed: '3 days ago',
    owner: 'john.doe@company.com',
    team: 'Platform Team',
    dependencies: ['Payment Gateway', 'Inventory Service', 'User Auth'],
    resources: { vms: 12, databases: 3, storage: 5, containers: 24 },
    metrics: { cpu: 45, memory: 62, requests: 15420, errors: 12, latency: 125 },
    cost: { monthly: 12500, trend: 'up', percentChange: 5.2 },
    lastActivity: '2 minutes ago',
    sla: { target: 99.9, current: 99.95, status: 'met' },
    incidents: 2,
    changes: 5
  },
  {
    id: 'app-002',
    name: 'Analytics Dashboard',
    description: 'Business intelligence and reporting platform',
    status: 'idle',
    health: 75,
    environment: 'production',
    version: 'v1.8.3',
    lastDeployed: '2 weeks ago',
    owner: 'sarah.admin@company.com',
    team: 'Data Team',
    dependencies: ['Data Warehouse', 'ETL Pipeline'],
    resources: { vms: 4, databases: 2, storage: 3, containers: 8 },
    metrics: { cpu: 12, memory: 35, requests: 230, errors: 0, latency: 450 },
    cost: { monthly: 3200, trend: 'stable', percentChange: 0.5 },
    lastActivity: '4 hours ago',
    idleTime: '3 hours 45 minutes',
    sla: { target: 95, current: 98.2, status: 'met' },
    incidents: 0,
    changes: 1
  },
  {
    id: 'app-003',
    name: 'Legacy CRM System',
    description: 'Customer relationship management (deprecated)',
    status: 'orphaned',
    health: 0,
    environment: 'development',
    version: 'v0.9.2',
    lastDeployed: '3 months ago',
    owner: 'unknown',
    team: 'Unknown',
    dependencies: [],
    resources: { vms: 2, databases: 1, storage: 1, containers: 0 },
    metrics: { cpu: 0, memory: 0, requests: 0, errors: 0, latency: 0 },
    cost: { monthly: 850, trend: 'stable', percentChange: 0 },
    lastActivity: '2 months ago',
    sla: { target: 90, current: 0, status: 'breached' },
    incidents: 0,
    changes: 0
  },
  {
    id: 'app-004',
    name: 'API Gateway',
    description: 'Central API management and routing',
    status: 'degraded',
    health: 65,
    environment: 'production',
    version: 'v3.2.0',
    lastDeployed: '1 week ago',
    owner: 'platform.team@company.com',
    team: 'Platform Team',
    dependencies: ['Auth Service', 'Rate Limiter', 'CDN'],
    resources: { vms: 8, databases: 1, storage: 2, containers: 16 },
    metrics: { cpu: 78, memory: 82, requests: 98500, errors: 245, latency: 280 },
    cost: { monthly: 8900, trend: 'up', percentChange: 12.3 },
    lastActivity: '30 seconds ago',
    sla: { target: 99.99, current: 98.5, status: 'at-risk' },
    incidents: 5,
    changes: 3
  },
  {
    id: 'app-005',
    name: 'Mobile Backend',
    description: 'Backend services for mobile applications',
    status: 'running',
    health: 92,
    environment: 'production',
    version: 'v4.1.2',
    lastDeployed: '5 days ago',
    owner: 'mobile.team@company.com',
    team: 'Mobile Team',
    dependencies: ['Push Notification Service', 'User Service', 'Content API'],
    resources: { vms: 6, databases: 2, storage: 3, containers: 18 },
    metrics: { cpu: 55, memory: 68, requests: 45200, errors: 34, latency: 95 },
    cost: { monthly: 6750, trend: 'down', percentChange: -3.8 },
    lastActivity: '1 minute ago',
    sla: { target: 99.5, current: 99.7, status: 'met' },
    incidents: 1,
    changes: 4
  },
  {
    id: 'app-006',
    name: 'Batch Processing System',
    description: 'Nightly batch jobs and data processing',
    status: 'stopped',
    health: 100,
    environment: 'production',
    version: 'v2.0.5',
    lastDeployed: '1 month ago',
    owner: 'data.team@company.com',
    team: 'Data Team',
    dependencies: ['Data Lake', 'Message Queue'],
    resources: { vms: 10, databases: 2, storage: 8, containers: 0 },
    metrics: { cpu: 0, memory: 0, requests: 0, errors: 0, latency: 0 },
    cost: { monthly: 2100, trend: 'down', percentChange: -15.2 },
    lastActivity: '8 hours ago',
    sla: { target: 95, current: 100, status: 'met' },
    incidents: 0,
    changes: 2
  }
]

export default function ApplicationsPage() {
  const [applications, setApplications] = useState<Application[]>(mockApplications)
  const [selectedApp, setSelectedApp] = useState<Application | null>(null)
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [filterEnvironment, setFilterEnvironment] = useState<string>('all')
  const [sortBy, setSortBy] = useState<'name' | 'health' | 'cost' | 'activity'>('activity')
  const [view, setView] = useState<'grid' | 'list'>('grid')

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30'
      case 'stopped': return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900/30'
      case 'idle': return 'text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30'
      case 'orphaned': return 'text-orange-600 dark:text-orange-400 bg-orange-100 dark:bg-orange-900/30'
      case 'degraded': return 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30'
      default: return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900/30'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <CheckCircle className="w-5 h-5" />
      case 'stopped': return <XCircle className="w-5 h-5" />
      case 'idle': return <Clock className="w-5 h-5" />
      case 'orphaned': return <AlertTriangle className="w-5 h-5" />
      case 'degraded': return <AlertCircle className="w-5 h-5" />
      default: return <Activity className="w-5 h-5" />
    }
  }

  const getHealthColor = (health: number) => {
    if (health >= 90) return 'text-green-600 dark:text-green-400'
    if (health >= 70) return 'text-yellow-600 dark:text-yellow-400'
    if (health >= 50) return 'text-orange-600 dark:text-orange-400'
    return 'text-red-600 dark:text-red-400'
  }

  const getSLAStatusColor = (status: string) => {
    switch (status) {
      case 'met': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
      case 'at-risk': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
      case 'breached': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
      default: return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
    }
  }

  const filteredApps = applications.filter(app => {
    if (filterStatus !== 'all' && app.status !== filterStatus) return false
    if (filterEnvironment !== 'all' && app.environment !== filterEnvironment) return false
    return true
  }).sort((a, b) => {
    switch (sortBy) {
      case 'name': return a.name.localeCompare(b.name)
      case 'health': return b.health - a.health
      case 'cost': return b.cost.monthly - a.cost.monthly
      case 'activity': return 0 // Would compare actual timestamps
      default: return 0
    }
  })

  const stats = {
    total: applications.length,
    running: applications.filter(a => a.status === 'running').length,
    idle: applications.filter(a => a.status === 'idle').length,
    orphaned: applications.filter(a => a.status === 'orphaned').length,
    degraded: applications.filter(a => a.status === 'degraded').length,
    totalCost: applications.reduce((sum, a) => sum + a.cost.monthly, 0),
    totalIncidents: applications.reduce((sum, a) => sum + a.incidents, 0)
  }

  const handleAction = (appId: string, action: string) => {
    console.log(`Performing ${action} on application ${appId}`)
    // Implement action logic
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground dark:text-white">Application Lifecycle</h1>
          <p className="text-muted-foreground dark:text-gray-400 mt-1">
            Monitor and manage application health, performance, and lifecycle
          </p>
        </div>
        <div className="flex gap-2">
          <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
            Deploy New App
          </button>
          <button className="px-4 py-2 bg-muted text-foreground rounded-lg hover:bg-accent transition-colors">
            Import Applications
          </button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-7 gap-4">
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-foreground dark:text-white">{stats.total}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Total Apps</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">{stats.running}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Running</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{stats.idle}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Idle</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">{stats.orphaned}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Orphaned</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-red-600 dark:text-red-400">{stats.degraded}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Degraded</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            ${(stats.totalCost / 1000).toFixed(1)}K
          </div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Monthly Cost</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">{stats.totalIncidents}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Incidents</div>
        </div>
      </div>

      {/* Filters and View Toggle */}
      <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
        <div className="flex justify-between items-center">
          <div className="flex gap-4">
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-3 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="all">All Status</option>
              <option value="running">Running</option>
              <option value="stopped">Stopped</option>
              <option value="idle">Idle</option>
              <option value="orphaned">Orphaned</option>
              <option value="degraded">Degraded</option>
            </select>

            <select
              value={filterEnvironment}
              onChange={(e) => setFilterEnvironment(e.target.value)}
              className="px-3 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="all">All Environments</option>
              <option value="production">Production</option>
              <option value="staging">Staging</option>
              <option value="development">Development</option>
              <option value="test">Test</option>
            </select>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="px-3 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="activity">Last Activity</option>
              <option value="name">Name</option>
              <option value="health">Health</option>
              <option value="cost">Cost</option>
            </select>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setView('grid')}
              className={`p-2 rounded ${view === 'grid' ? 'bg-primary text-primary-foreground' : 'bg-muted text-foreground hover:bg-accent'}`}
            >
              <Package className="w-4 h-4" />
            </button>
            <button
              onClick={() => setView('list')}
              className={`p-2 rounded ${view === 'list' ? 'bg-primary text-primary-foreground' : 'bg-muted text-foreground hover:bg-accent'}`}
            >
              <BarChart3 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Applications Grid/List */}
      {view === 'grid' ? (
        <div className="grid grid-cols-2 gap-6">
          {filteredApps.map((app) => (
            <div key={app.id} className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
              {/* App Header */}
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-foreground dark:text-white">{app.name}</h3>
                  <p className="text-sm text-muted-foreground dark:text-gray-400">{app.description}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(app.status)}`}>
                      {getStatusIcon(app.status)}
                      <span className="ml-1 capitalize">{app.status}</span>
                    </span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      app.environment === 'production' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                      app.environment === 'staging' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                      'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
                    }`}>
                      {app.environment}
                    </span>
                    <span className="text-xs text-muted-foreground dark:text-gray-400">{app.version}</span>
                  </div>
                </div>
                <div className="flex gap-1">
                  {app.status === 'stopped' && (
                    <button
                      onClick={() => handleAction(app.id, 'start')}
                      className="p-1 hover:bg-muted dark:hover:bg-gray-700 rounded"
                      title="Start"
                    >
                      <Play className="w-4 h-4 text-green-600 dark:text-green-400" />
                    </button>
                  )}
                  {app.status === 'running' && (
                    <button
                      onClick={() => handleAction(app.id, 'stop')}
                      className="p-1 hover:bg-muted dark:hover:bg-gray-700 rounded"
                      title="Stop"
                    >
                      <Square className="w-4 h-4 text-red-600 dark:text-red-400" />
                    </button>
                  )}
                  <button
                    onClick={() => handleAction(app.id, 'restart')}
                    className="p-1 hover:bg-muted dark:hover:bg-gray-700 rounded"
                    title="Restart"
                  >
                    <RefreshCw className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  </button>
                  <button
                    onClick={() => setSelectedApp(app)}
                    className="p-1 hover:bg-muted dark:hover:bg-gray-700 rounded"
                    title="Settings"
                  >
                    <Settings className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                  </button>
                </div>
              </div>

              {/* Health and SLA */}
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <div className="text-sm text-muted-foreground dark:text-gray-400 mb-1">Health Score</div>
                  <div className="flex items-center gap-2">
                    <div className={`text-2xl font-bold ${getHealthColor(app.health)}`}>
                      {app.health}%
                    </div>
                    <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          app.health >= 90 ? 'bg-green-500' :
                          app.health >= 70 ? 'bg-yellow-500' :
                          app.health >= 50 ? 'bg-orange-500' :
                          'bg-red-500'
                        }`}
                        style={{ width: `${app.health}%` }}
                      />
                    </div>
                  </div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground dark:text-gray-400 mb-1">SLA Status</div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getSLAStatusColor(app.sla.status)}`}>
                      {app.sla.status.toUpperCase()}
                    </span>
                    <span className="text-sm text-foreground dark:text-white">
                      {app.sla.current}% / {app.sla.target}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Resources */}
              <div className="grid grid-cols-4 gap-2 mb-4 p-3 bg-muted dark:bg-gray-900 rounded">
                <div className="text-center">
                  <Server className="w-4 h-4 mx-auto mb-1 text-blue-600 dark:text-blue-400" />
                  <div className="text-sm font-medium text-foreground dark:text-white">{app.resources.vms}</div>
                  <div className="text-xs text-muted-foreground dark:text-gray-400">VMs</div>
                </div>
                <div className="text-center">
                  <Database className="w-4 h-4 mx-auto mb-1 text-green-600 dark:text-green-400" />
                  <div className="text-sm font-medium text-foreground dark:text-white">{app.resources.databases}</div>
                  <div className="text-xs text-muted-foreground dark:text-gray-400">DBs</div>
                </div>
                <div className="text-center">
                  <Package className="w-4 h-4 mx-auto mb-1 text-purple-600 dark:text-purple-400" />
                  <div className="text-sm font-medium text-foreground dark:text-white">{app.resources.containers}</div>
                  <div className="text-xs text-muted-foreground dark:text-gray-400">Containers</div>
                </div>
                <div className="text-center">
                  <Database className="w-4 h-4 mx-auto mb-1 text-orange-600 dark:text-orange-400" />
                  <div className="text-sm font-medium text-foreground dark:text-white">{app.resources.storage}</div>
                  <div className="text-xs text-muted-foreground dark:text-gray-400">Storage</div>
                </div>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-3 gap-3 mb-4">
                <div>
                  <div className="text-xs text-muted-foreground dark:text-gray-400">CPU</div>
                  <div className="flex items-center gap-1">
                    <Cpu className="w-3 h-3 text-blue-600 dark:text-blue-400" />
                    <span className="text-sm font-medium text-foreground dark:text-white">{app.metrics.cpu}%</span>
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground dark:text-gray-400">Memory</div>
                  <div className="flex items-center gap-1">
                    <BarChart3 className="w-3 h-3 text-green-600 dark:text-green-400" />
                    <span className="text-sm font-medium text-foreground dark:text-white">{app.metrics.memory}%</span>
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground dark:text-gray-400">Requests</div>
                  <div className="flex items-center gap-1">
                    <Activity className="w-3 h-3 text-purple-600 dark:text-purple-400" />
                    <span className="text-sm font-medium text-foreground dark:text-white">
                      {app.metrics.requests > 1000 ? `${(app.metrics.requests / 1000).toFixed(1)}K` : app.metrics.requests}
                    </span>
                  </div>
                </div>
              </div>

              {/* Footer Info */}
              <div className="flex justify-between items-center pt-4 border-t border-border dark:border-gray-700">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1">
                    <DollarSign className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                    <span className="text-sm font-medium text-foreground dark:text-white">
                      ${app.cost.monthly.toLocaleString()}
                    </span>
                    {app.cost.trend === 'up' && (
                      <TrendingUp className="w-3 h-3 text-red-600 dark:text-red-400" />
                    )}
                    {app.cost.trend === 'down' && (
                      <TrendingDown className="w-3 h-3 text-green-600 dark:text-green-400" />
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    <AlertCircle className="w-4 h-4 text-orange-600 dark:text-orange-400" />
                    <span className="text-sm text-foreground dark:text-white">{app.incidents}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <GitBranch className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                    <span className="text-sm text-foreground dark:text-white">{app.changes}</span>
                  </div>
                </div>
                <div className="text-xs text-muted-foreground dark:text-gray-400">
                  {app.idleTime ? `Idle: ${app.idleTime}` : app.lastActivity}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-card dark:bg-gray-800 rounded-lg border border-border dark:border-gray-700 overflow-hidden">
          <table className="w-full">
            <thead className="bg-muted dark:bg-gray-900 border-b border-border dark:border-gray-700">
              <tr>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Application</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Status</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Health</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Environment</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Resources</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Cost</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">SLA</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredApps.map((app) => (
                <tr key={app.id} className="border-b border-border dark:border-gray-700 hover:bg-muted dark:hover:bg-gray-900">
                  <td className="p-4">
                    <div>
                      <div className="font-medium text-foreground dark:text-white">{app.name}</div>
                      <div className="text-sm text-muted-foreground dark:text-gray-400">{app.version}</div>
                    </div>
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(app.status)} inline-flex items-center gap-1`}>
                      {getStatusIcon(app.status)}
                      <span className="capitalize">{app.status}</span>
                    </span>
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      <div className={`font-medium ${getHealthColor(app.health)}`}>{app.health}%</div>
                      <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            app.health >= 90 ? 'bg-green-500' :
                            app.health >= 70 ? 'bg-yellow-500' :
                            app.health >= 50 ? 'bg-orange-500' :
                            'bg-red-500'
                          }`}
                          style={{ width: `${app.health}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      app.environment === 'production' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                      app.environment === 'staging' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                      'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
                    }`}>
                      {app.environment}
                    </span>
                  </td>
                  <td className="p-4 text-sm text-foreground dark:text-gray-300">
                    {app.resources.vms} VMs, {app.resources.databases} DBs
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-1">
                      <span className="text-sm font-medium text-foreground dark:text-white">
                        ${app.cost.monthly.toLocaleString()}
                      </span>
                      {app.cost.trend === 'up' && (
                        <TrendingUp className="w-3 h-3 text-red-600 dark:text-red-400" />
                      )}
                      {app.cost.trend === 'down' && (
                        <TrendingDown className="w-3 h-3 text-green-600 dark:text-green-400" />
                      )}
                    </div>
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getSLAStatusColor(app.sla.status)}`}>
                      {app.sla.status}
                    </span>
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-1">
                      {app.status === 'stopped' && (
                        <button className="p-1 hover:bg-muted dark:hover:bg-gray-700 rounded" title="Start">
                          <Play className="w-4 h-4 text-green-600 dark:text-green-400" />
                        </button>
                      )}
                      {app.status === 'running' && (
                        <button className="p-1 hover:bg-muted dark:hover:bg-gray-700 rounded" title="Stop">
                          <Square className="w-4 h-4 text-red-600 dark:text-red-400" />
                        </button>
                      )}
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
      )}

      {/* Orphaned Apps Alert */}
      {stats.orphaned > 0 && (
        <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-orange-600 dark:text-orange-400 mt-0.5" />
            <div>
              <h3 className="font-semibold text-orange-900 dark:text-orange-300">
                {stats.orphaned} Orphaned Application{stats.orphaned > 1 ? 's' : ''} Detected
              </h3>
              <p className="text-sm text-orange-700 dark:text-orange-400 mt-1">
                These applications have no owner and are consuming ${
                  applications.filter(a => a.status === 'orphaned')
                    .reduce((sum, a) => sum + a.cost.monthly, 0).toLocaleString()
                } per month. Consider decommissioning or reassigning ownership.
              </p>
              <button className="mt-2 px-3 py-1 bg-orange-600 text-white rounded hover:bg-orange-700 transition-colors text-sm">
                Review Orphaned Apps
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Idle Resources Warning */}
      {stats.idle > 0 && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <Clock className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
            <div>
              <h3 className="font-semibold text-yellow-900 dark:text-yellow-300">
                {stats.idle} Idle Application{stats.idle > 1 ? 's' : ''}
              </h3>
              <p className="text-sm text-yellow-700 dark:text-yellow-400 mt-1">
                Applications with minimal activity could be optimized or scheduled to reduce costs by ${
                  (applications.filter(a => a.status === 'idle')
                    .reduce((sum, a) => sum + a.cost.monthly, 0) * 0.3).toFixed(0)
                } per month.
              </p>
              <button className="mt-2 px-3 py-1 bg-yellow-600 text-white rounded hover:bg-yellow-700 transition-colors text-sm">
                Optimize Idle Apps
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}