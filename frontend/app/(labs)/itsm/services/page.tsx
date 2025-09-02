'use client'

import { useState } from 'react'
import { 
  Shield, Activity, CheckCircle, AlertCircle, XCircle,
  TrendingUp, TrendingDown, Clock, Users, Globe,
  Database, Server, Cloud, Network, Lock, Zap,
  BarChart3, AlertTriangle, Settings, RefreshCw
} from 'lucide-react'

interface Service {
  id: string
  name: string
  description: string
  category: 'infrastructure' | 'application' | 'data' | 'security' | 'network'
  status: 'operational' | 'degraded' | 'partial-outage' | 'major-outage' | 'maintenance'
  health: number
  availability: number
  sla: {
    target: number
    current: number
    status: 'met' | 'at-risk' | 'breached'
  }
  dependencies: string[]
  consumers: string[]
  incidents: {
    total: number
    active: number
    mttr: number // Mean Time To Resolution in minutes
  }
  performance: {
    responseTime: number
    throughput: number
    errorRate: number
  }
  lastIncident?: {
    time: string
    severity: 'low' | 'medium' | 'high' | 'critical'
    resolved: boolean
  }
}

const mockServices: Service[] = [
  {
    id: 'srv-001',
    name: 'Authentication Service',
    description: 'Central authentication and authorization service',
    category: 'security',
    status: 'operational',
    health: 99,
    availability: 99.99,
    sla: { target: 99.9, current: 99.99, status: 'met' },
    dependencies: ['Database Cluster', 'Redis Cache'],
    consumers: ['All Applications'],
    incidents: { total: 2, active: 0, mttr: 15 },
    performance: { responseTime: 45, throughput: 5000, errorRate: 0.01 }
  },
  {
    id: 'srv-002',
    name: 'API Gateway',
    description: 'Central API management and routing',
    category: 'infrastructure',
    status: 'degraded',
    health: 75,
    availability: 98.5,
    sla: { target: 99.9, current: 98.5, status: 'at-risk' },
    dependencies: ['Load Balancer', 'CDN'],
    consumers: ['Mobile App', 'Web App', 'Partners'],
    incidents: { total: 5, active: 1, mttr: 28 },
    performance: { responseTime: 280, throughput: 12000, errorRate: 2.5 },
    lastIncident: { time: '2 hours ago', severity: 'medium', resolved: false }
  },
  {
    id: 'srv-003',
    name: 'Database Cluster',
    description: 'Primary PostgreSQL database cluster',
    category: 'data',
    status: 'operational',
    health: 98,
    availability: 99.95,
    sla: { target: 99.9, current: 99.95, status: 'met' },
    dependencies: ['Storage Service', 'Backup Service'],
    consumers: ['All Services'],
    incidents: { total: 1, active: 0, mttr: 45 },
    performance: { responseTime: 12, throughput: 25000, errorRate: 0.05 },
    lastIncident: { time: '5 days ago', severity: 'low', resolved: true }
  },
  {
    id: 'srv-004',
    name: 'Message Queue',
    description: 'RabbitMQ message broker service',
    category: 'infrastructure',
    status: 'maintenance',
    health: 100,
    availability: 95.0,
    sla: { target: 99.5, current: 95.0, status: 'breached' },
    dependencies: ['Database Cluster'],
    consumers: ['Batch Processing', 'Email Service', 'Notifications'],
    incidents: { total: 3, active: 0, mttr: 20 },
    performance: { responseTime: 5, throughput: 50000, errorRate: 0.0 }
  },
  {
    id: 'srv-005',
    name: 'CDN Service',
    description: 'Content delivery network',
    category: 'network',
    status: 'operational',
    health: 100,
    availability: 99.99,
    sla: { target: 99.9, current: 99.99, status: 'met' },
    dependencies: ['Origin Servers'],
    consumers: ['Web Applications', 'Mobile Apps'],
    incidents: { total: 0, active: 0, mttr: 0 },
    performance: { responseTime: 15, throughput: 100000, errorRate: 0.001 }
  },
  {
    id: 'srv-006',
    name: 'Email Service',
    description: 'Transactional email delivery',
    category: 'application',
    status: 'partial-outage',
    health: 45,
    availability: 85.3,
    sla: { target: 99.0, current: 85.3, status: 'breached' },
    dependencies: ['Message Queue', 'SMTP Gateway'],
    consumers: ['All Applications'],
    incidents: { total: 8, active: 2, mttr: 35 },
    performance: { responseTime: 500, throughput: 1000, errorRate: 15.2 },
    lastIncident: { time: '30 minutes ago', severity: 'high', resolved: false }
  }
]

export default function ServicesPage() {
  const [services] = useState<Service[]>(mockServices)
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [selectedStatus, setSelectedStatus] = useState<string>('all')
  const [view, setView] = useState<'catalog' | 'health'>('catalog')

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'operational': return 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30'
      case 'degraded': return 'text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30'
      case 'partial-outage': return 'text-orange-600 dark:text-orange-400 bg-orange-100 dark:bg-orange-900/30'
      case 'major-outage': return 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30'
      case 'maintenance': return 'text-purple-600 dark:text-purple-400 bg-purple-100 dark:bg-purple-900/30'
      default: return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900/30'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'operational': return <CheckCircle className="w-5 h-5" />
      case 'degraded': return <AlertTriangle className="w-5 h-5" />
      case 'partial-outage': return <AlertCircle className="w-5 h-5" />
      case 'major-outage': return <XCircle className="w-5 h-5" />
      case 'maintenance': return <Settings className="w-5 h-5" />
      default: return <Activity className="w-5 h-5" />
    }
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'infrastructure': return <Server className="w-5 h-5" />
      case 'application': return <Globe className="w-5 h-5" />
      case 'data': return <Database className="w-5 h-5" />
      case 'security': return <Lock className="w-5 h-5" />
      case 'network': return <Network className="w-5 h-5" />
      default: return <Cloud className="w-5 h-5" />
    }
  }

  const filteredServices = services.filter(service => {
    if (selectedCategory !== 'all' && service.category !== selectedCategory) return false
    if (selectedStatus !== 'all' && service.status !== selectedStatus) return false
    return true
  })

  const stats = {
    total: services.length,
    operational: services.filter(s => s.status === 'operational').length,
    degraded: services.filter(s => s.status === 'degraded').length,
    outages: services.filter(s => s.status.includes('outage')).length,
    maintenance: services.filter(s => s.status === 'maintenance').length,
    activeIncidents: services.reduce((sum, s) => sum + s.incidents.active, 0),
    avgAvailability: (services.reduce((sum, s) => sum + s.availability, 0) / services.length).toFixed(2)
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground dark:text-white">Service Catalog</h1>
          <p className="text-muted-foreground dark:text-gray-400 mt-1">
            Monitor service health, dependencies, and SLA compliance
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setView(view === 'catalog' ? 'health' : 'catalog')}
            className="px-4 py-2 bg-muted text-foreground rounded-lg hover:bg-accent transition-colors"
          >
            {view === 'catalog' ? 'Health View' : 'Catalog View'}
          </button>
          <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
            Add Service
          </button>
        </div>
      </div>

      {/* Overall Status Banner */}
      <div className={`rounded-lg p-4 border ${
        stats.outages > 0 ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' :
        stats.degraded > 0 ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800' :
        'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {stats.outages > 0 ? (
              <AlertCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
            ) : stats.degraded > 0 ? (
              <AlertTriangle className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
            ) : (
              <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
            )}
            <div>
              <h2 className={`text-lg font-semibold ${
                stats.outages > 0 ? 'text-red-900 dark:text-red-300' :
                stats.degraded > 0 ? 'text-yellow-900 dark:text-yellow-300' :
                'text-green-900 dark:text-green-300'
              }`}>
                {stats.outages > 0 ? 'Service Disruption Detected' :
                 stats.degraded > 0 ? 'Some Services Degraded' :
                 'All Systems Operational'}
              </h2>
              <p className={`text-sm ${
                stats.outages > 0 ? 'text-red-700 dark:text-red-400' :
                stats.degraded > 0 ? 'text-yellow-700 dark:text-yellow-400' :
                'text-green-700 dark:text-green-400'
              }`}>
                {stats.operational}/{stats.total} services operational • 
                {stats.activeIncidents} active incident{stats.activeIncidents !== 1 ? 's' : ''} • 
                {stats.avgAvailability}% average availability
              </p>
            </div>
          </div>
          <button className="px-3 py-1 bg-white dark:bg-gray-800 rounded border border-gray-300 dark:border-gray-600 text-sm">
            View Status Page
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-6 gap-4">
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-foreground dark:text-white">{stats.total}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Total Services</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">{stats.operational}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Operational</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{stats.degraded}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Degraded</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-red-600 dark:text-red-400">{stats.outages}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Outages</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">{stats.maintenance}</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Maintenance</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{stats.avgAvailability}%</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Avg Availability</div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
        <div className="flex gap-4">
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="all">All Categories</option>
            <option value="infrastructure">Infrastructure</option>
            <option value="application">Application</option>
            <option value="data">Data</option>
            <option value="security">Security</option>
            <option value="network">Network</option>
          </select>

          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="px-3 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="all">All Status</option>
            <option value="operational">Operational</option>
            <option value="degraded">Degraded</option>
            <option value="partial-outage">Partial Outage</option>
            <option value="major-outage">Major Outage</option>
            <option value="maintenance">Maintenance</option>
          </select>
        </div>
      </div>

      {/* Services Grid */}
      {view === 'catalog' ? (
        <div className="grid grid-cols-2 gap-6">
          {filteredServices.map((service) => (
            <div key={service.id} className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
              {/* Service Header */}
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-start gap-3">
                  <div className="p-2 bg-muted dark:bg-gray-900 rounded">
                    {getCategoryIcon(service.category)}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-foreground dark:text-white">{service.name}</h3>
                    <p className="text-sm text-muted-foreground dark:text-gray-400">{service.description}</p>
                    <div className="flex items-center gap-2 mt-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(service.status)} inline-flex items-center gap-1`}>
                        {getStatusIcon(service.status)}
                        <span className="capitalize">{service.status.replace('-', ' ')}</span>
                      </span>
                      <span className="text-xs text-muted-foreground dark:text-gray-400 capitalize">
                        {service.category}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Health & Availability */}
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <div className="text-sm text-muted-foreground dark:text-gray-400 mb-1">Health Score</div>
                  <div className="flex items-center gap-2">
                    <div className={`text-xl font-bold ${
                      service.health >= 90 ? 'text-green-600 dark:text-green-400' :
                      service.health >= 70 ? 'text-yellow-600 dark:text-yellow-400' :
                      service.health >= 50 ? 'text-orange-600 dark:text-orange-400' :
                      'text-red-600 dark:text-red-400'
                    }`}>
                      {service.health}%
                    </div>
                    <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          service.health >= 90 ? 'bg-green-500' :
                          service.health >= 70 ? 'bg-yellow-500' :
                          service.health >= 50 ? 'bg-orange-500' :
                          'bg-red-500'
                        }`}
                        style={{ width: `${service.health}%` }}
                      />
                    </div>
                  </div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground dark:text-gray-400 mb-1">Availability</div>
                  <div className="flex items-center gap-2">
                    <div className="text-xl font-bold text-foreground dark:text-white">
                      {service.availability}%
                    </div>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      service.sla.status === 'met' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                      service.sla.status === 'at-risk' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                      'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                    }`}>
                      SLA: {service.sla.status}
                    </span>
                  </div>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="grid grid-cols-3 gap-3 mb-4 p-3 bg-muted dark:bg-gray-900 rounded">
                <div>
                  <div className="text-xs text-muted-foreground dark:text-gray-400">Response</div>
                  <div className="text-sm font-medium text-foreground dark:text-white">{service.performance.responseTime}ms</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground dark:text-gray-400">Throughput</div>
                  <div className="text-sm font-medium text-foreground dark:text-white">
                    {service.performance.throughput > 1000 ? `${(service.performance.throughput / 1000).toFixed(1)}K` : service.performance.throughput}/s
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground dark:text-gray-400">Error Rate</div>
                  <div className="text-sm font-medium text-foreground dark:text-white">{service.performance.errorRate}%</div>
                </div>
              </div>

              {/* Incidents & Dependencies */}
              <div className="flex justify-between items-center pt-4 border-t border-border dark:border-gray-700">
                <div className="flex items-center gap-4 text-sm">
                  <div className="flex items-center gap-1">
                    <AlertCircle className="w-4 h-4 text-orange-600 dark:text-orange-400" />
                    <span className="text-foreground dark:text-white">{service.incidents.active} active</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                    <span className="text-foreground dark:text-white">{service.incidents.mttr}m MTTR</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Network className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                    <span className="text-foreground dark:text-white">{service.dependencies.length} deps</span>
                  </div>
                </div>
                <button className="p-1 hover:bg-muted dark:hover:bg-gray-700 rounded">
                  <Settings className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                </button>
              </div>

              {/* Last Incident */}
              {service.lastIncident && (
                <div className={`mt-3 p-2 rounded text-xs ${
                  service.lastIncident.resolved ? 'bg-gray-100 dark:bg-gray-900' : 'bg-orange-100 dark:bg-orange-900/30'
                }`}>
                  <span className={`font-medium ${
                    service.lastIncident.resolved ? 'text-gray-600 dark:text-gray-400' : 'text-orange-700 dark:text-orange-400'
                  }`}>
                    Last incident: {service.lastIncident.time} • {service.lastIncident.severity} severity
                    {!service.lastIncident.resolved && ' • In progress'}
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        /* Health Dashboard View */
        <div className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
          <h3 className="text-lg font-semibold text-foreground dark:text-white mb-4">Service Health Matrix</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b border-border dark:border-gray-700">
                <tr>
                  <th className="text-left pb-3 text-sm font-medium text-foreground dark:text-white">Service</th>
                  <th className="text-center pb-3 text-sm font-medium text-foreground dark:text-white">Status</th>
                  <th className="text-center pb-3 text-sm font-medium text-foreground dark:text-white">Health</th>
                  <th className="text-center pb-3 text-sm font-medium text-foreground dark:text-white">Availability</th>
                  <th className="text-center pb-3 text-sm font-medium text-foreground dark:text-white">Response Time</th>
                  <th className="text-center pb-3 text-sm font-medium text-foreground dark:text-white">Error Rate</th>
                  <th className="text-center pb-3 text-sm font-medium text-foreground dark:text-white">Incidents</th>
                </tr>
              </thead>
              <tbody>
                {filteredServices.map((service) => (
                  <tr key={service.id} className="border-b border-border dark:border-gray-700">
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        {getCategoryIcon(service.category)}
                        <div>
                          <div className="font-medium text-foreground dark:text-white">{service.name}</div>
                          <div className="text-xs text-muted-foreground dark:text-gray-400">{service.category}</div>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 text-center">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(service.status)} inline-flex items-center gap-1`}>
                        {getStatusIcon(service.status)}
                      </span>
                    </td>
                    <td className="py-3 text-center">
                      <div className={`font-medium ${
                        service.health >= 90 ? 'text-green-600 dark:text-green-400' :
                        service.health >= 70 ? 'text-yellow-600 dark:text-yellow-400' :
                        service.health >= 50 ? 'text-orange-600 dark:text-orange-400' :
                        'text-red-600 dark:text-red-400'
                      }`}>
                        {service.health}%
                      </div>
                    </td>
                    <td className="py-3 text-center">
                      <div className="font-medium text-foreground dark:text-white">{service.availability}%</div>
                      <div className={`text-xs ${
                        service.sla.status === 'met' ? 'text-green-600 dark:text-green-400' :
                        service.sla.status === 'at-risk' ? 'text-yellow-600 dark:text-yellow-400' :
                        'text-red-600 dark:text-red-400'
                      }`}>
                        SLA: {service.sla.current}%/{service.sla.target}%
                      </div>
                    </td>
                    <td className="py-3 text-center">
                      <div className="font-medium text-foreground dark:text-white">{service.performance.responseTime}ms</div>
                    </td>
                    <td className="py-3 text-center">
                      <div className={`font-medium ${
                        service.performance.errorRate < 1 ? 'text-green-600 dark:text-green-400' :
                        service.performance.errorRate < 5 ? 'text-yellow-600 dark:text-yellow-400' :
                        'text-red-600 dark:text-red-400'
                      }`}>
                        {service.performance.errorRate}%
                      </div>
                    </td>
                    <td className="py-3 text-center">
                      <div className="flex items-center justify-center gap-2">
                        {service.incidents.active > 0 && (
                          <span className="px-2 py-0.5 bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 rounded text-xs font-medium">
                            {service.incidents.active} active
                          </span>
                        )}
                        <span className="text-sm text-muted-foreground dark:text-gray-400">
                          {service.incidents.total} total
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}