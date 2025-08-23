'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { 
  Server, Cloud, Activity, AlertCircle, Package, Database,
  CheckCircle, XCircle, AlertTriangle, Clock, Wrench, FileText,
  TrendingUp, TrendingDown, DollarSign, Cpu, HardDrive, Network,
  BarChart3, Users, Settings, Zap, Shield, Globe
} from 'lucide-react'

interface ResourceSummary {
  total: number
  running: number
  stopped: number
  idle: number
  orphaned: number
  degraded: number
  scheduled: number
  maintenance: number
  decommissioned: number
}

interface CloudProvider {
  name: string
  icon: string
  color: string
  resources: ResourceSummary
  health: number
  cost: number
  incidents: number
  changes: number
}

interface ServiceHealth {
  name: string
  status: 'healthy' | 'degraded' | 'critical' | 'maintenance'
  availability: number
  incidents: number
  lastIncident?: string
}

export default function ITSMDashboard() {
  const [resourceSummary, setResourceSummary] = useState<ResourceSummary>({
    total: 1247,
    running: 892,
    stopped: 156,
    idle: 87,
    orphaned: 45,
    degraded: 23,
    scheduled: 28,
    maintenance: 12,
    decommissioned: 4
  })

  const [cloudProviders] = useState<CloudProvider[]>([
    {
      name: 'Azure',
      icon: '☁️',
      color: 'blue',
      resources: {
        total: 687,
        running: 512,
        stopped: 89,
        idle: 42,
        orphaned: 21,
        degraded: 12,
        scheduled: 8,
        maintenance: 3,
        decommissioned: 0
      },
      health: 94.5,
      cost: 45230,
      incidents: 3,
      changes: 7
    },
    {
      name: 'AWS',
      icon: '🔶',
      color: 'orange',
      resources: {
        total: 423,
        running: 298,
        stopped: 54,
        idle: 31,
        orphaned: 18,
        degraded: 9,
        scheduled: 10,
        maintenance: 3,
        decommissioned: 0
      },
      health: 92.3,
      cost: 28750,
      incidents: 2,
      changes: 4
    },
    {
      name: 'GCP',
      icon: '🌐',
      color: 'green',
      resources: {
        total: 137,
        running: 82,
        stopped: 13,
        idle: 14,
        orphaned: 6,
        degraded: 2,
        scheduled: 10,
        maintenance: 6,
        decommissioned: 4
      },
      health: 96.8,
      cost: 12890,
      incidents: 0,
      changes: 2
    }
  ])

  const [criticalServices] = useState<ServiceHealth[]>([
    { name: 'Authentication Service', status: 'healthy', availability: 99.99, incidents: 0 },
    { name: 'Database Cluster', status: 'healthy', availability: 99.95, incidents: 1, lastIncident: '2 days ago' },
    { name: 'API Gateway', status: 'degraded', availability: 98.5, incidents: 3, lastIncident: '4 hours ago' },
    { name: 'Storage Service', status: 'healthy', availability: 99.98, incidents: 0 },
    { name: 'Message Queue', status: 'maintenance', availability: 95.0, incidents: 2, lastIncident: 'In maintenance' },
    { name: 'CDN', status: 'healthy', availability: 99.99, incidents: 0 }
  ])

  const [recentActivities] = useState([
    { time: '5 min ago', type: 'incident', message: 'API Gateway response time degraded', severity: 'warning' },
    { time: '15 min ago', type: 'change', message: 'Database backup policy updated', severity: 'info' },
    { time: '1 hour ago', type: 'resource', message: '12 idle VMs detected in Azure East US', severity: 'info' },
    { time: '2 hours ago', type: 'cost', message: 'Monthly spending threshold 80% reached', severity: 'warning' },
    { time: '3 hours ago', type: 'compliance', message: 'New compliance policy applied to 45 resources', severity: 'success' }
  ])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
      case 'healthy':
        return 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30'
      case 'stopped':
        return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900/30'
      case 'idle':
        return 'text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30'
      case 'orphaned':
        return 'text-orange-600 dark:text-orange-400 bg-orange-100 dark:bg-orange-900/30'
      case 'degraded':
      case 'critical':
        return 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30'
      case 'scheduled':
        return 'text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/30'
      case 'maintenance':
        return 'text-purple-600 dark:text-purple-400 bg-purple-100 dark:bg-purple-900/30'
      case 'decommissioned':
        return 'text-gray-500 dark:text-gray-500 bg-gray-100 dark:bg-gray-900/30'
      default:
        return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900/30'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
      case 'healthy':
        return <CheckCircle className="w-4 h-4" />
      case 'stopped':
        return <XCircle className="w-4 h-4" />
      case 'idle':
        return <Clock className="w-4 h-4" />
      case 'orphaned':
        return <AlertTriangle className="w-4 h-4" />
      case 'degraded':
      case 'critical':
        return <AlertCircle className="w-4 h-4" />
      case 'scheduled':
        return <Clock className="w-4 h-4" />
      case 'maintenance':
        return <Wrench className="w-4 h-4" />
      case 'decommissioned':
        return <XCircle className="w-4 h-4" />
      default:
        return <Activity className="w-4 h-4" />
    }
  }

  const calculateHealthScore = () => {
    const weights = {
      running: 1,
      stopped: 0.7,
      idle: 0.5,
      orphaned: 0.3,
      degraded: 0.2,
      scheduled: 0.9,
      maintenance: 0.6,
      decommissioned: 0.4
    }
    
    let score = 0
    let totalWeight = 0
    
    Object.entries(resourceSummary).forEach(([key, value]) => {
      if (key !== 'total' && weights[key as keyof typeof weights]) {
        score += value * weights[key as keyof typeof weights]
        totalWeight += value
      }
    })
    
    return totalWeight > 0 ? Math.round((score / totalWeight) * 100) : 0
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground dark:text-white">Cloud ITSM Dashboard</h1>
          <p className="text-muted-foreground dark:text-gray-400 mt-1">
            Comprehensive IT Service Management across all cloud providers
          </p>
        </div>
        <div className="flex gap-2">
          <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
            Discover Resources
          </button>
          <button className="px-4 py-2 bg-muted text-foreground rounded-lg hover:bg-accent transition-colors">
            Export Report
          </button>
        </div>
      </div>

      {/* Overall Health Score */}
      <div className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-foreground dark:text-white mb-2">Infrastructure Health Score</h2>
            <div className="flex items-center gap-4">
              <div className="text-4xl font-bold text-green-600 dark:text-green-400">
                {calculateHealthScore()}%
              </div>
              <div className="text-sm text-muted-foreground dark:text-gray-400">
                <div>Last scan: 2 minutes ago</div>
                <div>Next scan: in 8 minutes</div>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{resourceSummary.total}</div>
              <div className="text-sm text-muted-foreground dark:text-gray-400">Total Resources</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                ${(cloudProviders.reduce((sum, p) => sum + p.cost, 0) / 1000).toFixed(1)}K
              </div>
              <div className="text-sm text-muted-foreground dark:text-gray-400">Monthly Cost</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                {cloudProviders.reduce((sum, p) => sum + p.incidents, 0)}
              </div>
              <div className="text-sm text-muted-foreground dark:text-gray-400">Active Incidents</div>
            </div>
          </div>
        </div>
      </div>

      {/* Resource Status Grid */}
      <div className="grid grid-cols-4 gap-4">
        {Object.entries(resourceSummary).filter(([key]) => key !== 'total').map(([status, count]) => (
          <Link
            key={status}
            href="/itsm/inventory"
            className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700 hover:shadow-lg transition-all"
          >
            <div className="flex items-center justify-between mb-2">
              <div className={`p-2 rounded-lg ${getStatusColor(status)}`}>
                {getStatusIcon(status)}
              </div>
              <span className="text-2xl font-bold text-foreground dark:text-white">{count}</span>
            </div>
            <div className="text-sm font-medium text-foreground dark:text-white capitalize">{status}</div>
            <div className="text-xs text-muted-foreground dark:text-gray-400">
              {((count / resourceSummary.total) * 100).toFixed(1)}% of total
            </div>
          </Link>
        ))}
      </div>

      {/* Cloud Providers */}
      <div className="grid grid-cols-3 gap-6">
        {cloudProviders.map((provider) => (
          <div key={provider.name} className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <span className="text-2xl">{provider.icon}</span>
                <h3 className="text-lg font-semibold text-foreground dark:text-white">{provider.name}</h3>
              </div>
              <div className={`px-2 py-1 rounded text-xs font-medium ${
                provider.health >= 95 ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                provider.health >= 90 ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
              }`}>
                {provider.health}% Health
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground dark:text-gray-400">Resources</span>
                <span className="text-sm font-medium text-foreground dark:text-white">{provider.resources.total}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground dark:text-gray-400">Monthly Cost</span>
                <span className="text-sm font-medium text-foreground dark:text-white">${provider.cost.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground dark:text-gray-400">Incidents</span>
                <span className={`text-sm font-medium ${provider.incidents > 0 ? 'text-orange-600 dark:text-orange-400' : 'text-green-600 dark:text-green-400'}`}>
                  {provider.incidents}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground dark:text-gray-400">Changes</span>
                <span className="text-sm font-medium text-blue-600 dark:text-blue-400">{provider.changes}</span>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-border dark:border-gray-700">
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="text-center">
                  <div className="font-semibold text-green-600 dark:text-green-400">{provider.resources.running}</div>
                  <div className="text-muted-foreground dark:text-gray-400">Running</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-yellow-600 dark:text-yellow-400">{provider.resources.idle}</div>
                  <div className="text-muted-foreground dark:text-gray-400">Idle</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-orange-600 dark:text-orange-400">{provider.resources.orphaned}</div>
                  <div className="text-muted-foreground dark:text-gray-400">Orphaned</div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions Grid */}
      <div className="grid grid-cols-4 gap-4">
        <Link href="/itsm/inventory" className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700 hover:shadow-lg transition-all group">
          <Package className="w-8 h-8 text-blue-600 dark:text-blue-400 mb-2 group-hover:scale-110 transition-transform" />
          <h3 className="font-semibold text-foreground dark:text-white">Resource Inventory</h3>
          <p className="text-sm text-muted-foreground dark:text-gray-400">Browse all resources</p>
        </Link>
        
        <Link href="/itsm/applications" className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700 hover:shadow-lg transition-all group">
          <Cpu className="w-8 h-8 text-green-600 dark:text-green-400 mb-2 group-hover:scale-110 transition-transform" />
          <h3 className="font-semibold text-foreground dark:text-white">Applications</h3>
          <p className="text-sm text-muted-foreground dark:text-gray-400">Manage app lifecycle</p>
        </Link>
        
        <Link href="/itsm/services" className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700 hover:shadow-lg transition-all group">
          <Shield className="w-8 h-8 text-purple-600 dark:text-purple-400 mb-2 group-hover:scale-110 transition-transform" />
          <h3 className="font-semibold text-foreground dark:text-white">Service Health</h3>
          <p className="text-sm text-muted-foreground dark:text-gray-400">Monitor services</p>
        </Link>
        
        <Link href="/itsm/cmdb" className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700 hover:shadow-lg transition-all group">
          <Database className="w-8 h-8 text-orange-600 dark:text-orange-400 mb-2 group-hover:scale-110 transition-transform" />
          <h3 className="font-semibold text-foreground dark:text-white">CMDB</h3>
          <p className="text-sm text-muted-foreground dark:text-gray-400">Configuration database</p>
        </Link>
      </div>

      {/* Service Health & Recent Activities */}
      <div className="grid grid-cols-2 gap-6">
        {/* Critical Services */}
        <div className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
          <h3 className="text-lg font-semibold text-foreground dark:text-white mb-4">Critical Services</h3>
          <div className="space-y-3">
            {criticalServices.map((service) => (
              <div key={service.name} className="flex items-center justify-between p-3 bg-muted dark:bg-gray-900 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className={`p-1.5 rounded ${getStatusColor(service.status)}`}>
                    {getStatusIcon(service.status)}
                  </div>
                  <div>
                    <div className="font-medium text-foreground dark:text-white">{service.name}</div>
                    <div className="text-xs text-muted-foreground dark:text-gray-400">
                      {service.availability}% uptime
                      {service.lastIncident && ` • Last incident: ${service.lastIncident}`}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-foreground dark:text-white">{service.incidents}</div>
                  <div className="text-xs text-muted-foreground dark:text-gray-400">incidents</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Activities */}
        <div className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
          <h3 className="text-lg font-semibold text-foreground dark:text-white mb-4">Recent Activities</h3>
          <div className="space-y-3">
            {recentActivities.map((activity, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-muted dark:bg-gray-900 rounded-lg">
                <div className={`p-1.5 rounded ${
                  activity.severity === 'warning' ? 'bg-yellow-100 text-yellow-600 dark:bg-yellow-900/30 dark:text-yellow-400' :
                  activity.severity === 'success' ? 'bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400' :
                  'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400'
                }`}>
                  {activity.type === 'incident' ? <AlertCircle className="w-4 h-4" /> :
                   activity.type === 'change' ? <Wrench className="w-4 h-4" /> :
                   activity.type === 'cost' ? <DollarSign className="w-4 h-4" /> :
                   <Activity className="w-4 h-4" />}
                </div>
                <div className="flex-1">
                  <div className="text-sm font-medium text-foreground dark:text-white">{activity.message}</div>
                  <div className="text-xs text-muted-foreground dark:text-gray-400">{activity.time}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}