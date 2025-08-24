'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  Server,
  Database,
  Cloud,
  Zap,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  TrendingUp,
  TrendingDown,
  Monitor,
  Cpu,
  HardDrive,
  Network,
  Settings,
  RefreshCw,
  Play,
  Pause,
  RotateCcw,
  Eye,
  BarChart3,
  Target,
  Users,
  Globe,
  Shield,
  Layers,
  Box,
  Gauge
} from 'lucide-react'
import ViewToggle from '@/components/ViewToggle';
import ChartContainer from '@/components/ChartContainer';
import MetricCard from '@/components/MetricCard';

export default function TacticalOperationsPage() {
  const [view, setView] = useState<'cards' | 'visualizations'>('cards');
  const [operationsMetrics, setOperationsMetrics] = useState({
    systemUptime: 99.97,
    activeServices: 47,
    healthyNodes: 156,
    avgResponseTime: 42,
    errorRate: 0.03
  })

  const [systemHealth, setSystemHealth] = useState([
    {
      name: 'Production Cluster',
      status: 'healthy',
      uptime: 99.99,
      nodes: 24,
      cpu: 67,
      memory: 72,
      disk: 45,
      network: 23,
      alerts: 0,
      region: 'US East'
    },
    {
      name: 'Staging Environment',
      status: 'healthy',
      uptime: 99.95,
      nodes: 12,
      cpu: 34,
      memory: 28,
      disk: 52,
      network: 18,
      alerts: 1,
      region: 'US West'
    },
    {
      name: 'Development Cluster',
      status: 'warning',
      uptime: 98.7,
      nodes: 8,
      cpu: 89,
      memory: 91,
      disk: 78,
      network: 45,
      alerts: 3,
      region: 'EU Central'
    },
    {
      name: 'Analytics Platform',
      status: 'healthy',
      uptime: 99.8,
      nodes: 16,
      cpu: 76,
      memory: 68,
      disk: 34,
      network: 56,
      alerts: 0,
      region: 'Asia Pacific'
    }
  ])

  const [services, setServices] = useState([
    {
      name: 'API Gateway',
      status: 'running',
      health: 'healthy',
      instances: 6,
      responseTime: 23,
      errorRate: 0.01,
      requests: 2456789,
      icon: Globe
    },
    {
      name: 'User Service',
      status: 'running',
      health: 'healthy',
      instances: 4,
      responseTime: 45,
      errorRate: 0.02,
      requests: 1234567,
      icon: Users
    },
    {
      name: 'Database Cluster',
      status: 'running',
      health: 'warning',
      instances: 3,
      responseTime: 67,
      errorRate: 0.05,
      requests: 987654,
      icon: Database
    },
    {
      name: 'Message Queue',
      status: 'running',
      health: 'healthy',
      instances: 5,
      responseTime: 12,
      errorRate: 0.001,
      requests: 3456789,
      icon: Layers
    },
    {
      name: 'Cache Layer',
      status: 'running',
      health: 'healthy',
      instances: 3,
      responseTime: 8,
      errorRate: 0.003,
      requests: 5678901,
      icon: Zap
    },
    {
      name: 'File Storage',
      status: 'maintenance',
      health: 'maintenance',
      instances: 2,
      responseTime: 156,
      errorRate: 0,
      requests: 234567,
      icon: HardDrive
    }
  ])

  const [recentIncidents, setRecentIncidents] = useState([
    {
      id: 'inc-001',
      title: 'High CPU Usage - Development Cluster',
      severity: 'medium',
      status: 'investigating',
      startTime: new Date(Date.now() - 1800000),
      affectedServices: ['Development API', 'Test Database'],
      assignee: 'John Doe'
    },
    {
      id: 'inc-002',
      title: 'Scheduled Maintenance - File Storage',
      severity: 'low',
      status: 'planned',
      startTime: new Date(Date.now() - 3600000),
      affectedServices: ['File Storage', 'Backup Service'],
      assignee: 'Maintenance Team'
    },
    {
      id: 'inc-003',
      title: 'Database Connection Pool Exhaustion',
      severity: 'high',
      status: 'resolved',
      startTime: new Date(Date.now() - 7200000),
      affectedServices: ['User Service', 'API Gateway'],
      assignee: 'Sarah Johnson'
    }
  ])

  const getStatusColor = (status: string, health?: string) => {
    if (health === 'maintenance') return 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-400'
    
    switch (status) {
      case 'healthy':
      case 'running': return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-400'
      case 'warning': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-400'
      case 'error':
      case 'critical': return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-400'
      case 'maintenance': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-400'
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-400'
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high':
      case 'critical': return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-400'
      case 'medium': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-400'
      case 'low': return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-400'
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-400'
    }
  }

  const getStatusIcon = (status: string, health?: string) => {
    if (health === 'maintenance') return <Settings className="w-4 h-4 text-blue-500" />
    
    switch (status) {
      case 'healthy':
      case 'running': return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />
      case 'error':
      case 'critical': return <XCircle className="w-4 h-4 text-red-500" />
      case 'maintenance': return <Settings className="w-4 h-4 text-blue-500" />
      default: return <Clock className="w-4 h-4 text-gray-400" />
    }
  }

  const getUtilizationColor = (utilization: number) => {
    if (utilization >= 90) return 'bg-red-500'
    if (utilization >= 75) return 'bg-yellow-500'
    if (utilization >= 50) return 'bg-blue-500'
    return 'bg-green-500'
  }

  const metrics = [
    {
      id: 'system-uptime',
      title: 'System Uptime',
      value: `${operationsMetrics.systemUptime}%`,
      change: 0.02,
      trend: 'up' as const,
      sparklineData: [99.95, 99.96, 99.97, 99.96, 99.98, operationsMetrics.systemUptime],
      alert: `${systemHealth.filter(s => s.status === 'warning').length} systems need attention`
    },
    {
      id: 'active-services',
      title: 'Active Services',
      value: operationsMetrics.activeServices,
      change: 2.1,
      trend: 'up' as const,
      sparklineData: [44, 45, 46, 46, 47, operationsMetrics.activeServices]
    },
    {
      id: 'response-time',
      title: 'Avg Response Time',
      value: `${operationsMetrics.avgResponseTime}ms`,
      change: -8.3,
      trend: 'down' as const,
      sparklineData: [52, 48, 45, 44, 43, operationsMetrics.avgResponseTime]
    },
    {
      id: 'error-rate',
      title: 'Error Rate',
      value: `${operationsMetrics.errorRate}%`,
      change: -15.2,
      trend: 'down' as const,
      sparklineData: [0.05, 0.04, 0.035, 0.032, 0.031, operationsMetrics.errorRate]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-3">
              <Activity className="h-10 w-10 text-blue-600" />
              Tactical Operations Command
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Real-time system monitoring and operational management dashboard
            </p>
          </div>
          <div className="flex gap-3">
            <ViewToggle view={view} onViewChange={setView} />
            <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg">
              <RefreshCw className="h-5 w-5" />
            </button>
            <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg">
              <Settings className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* System Status Alert */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 p-4 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 border border-green-200 dark:border-green-800 rounded-xl"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-full">
              <Monitor className="w-8 h-8 text-green-600 dark:text-green-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-green-900 dark:text-green-100">
                Operations Status: {operationsMetrics.systemUptime}% Uptime
              </h3>
              <p className="text-green-700 dark:text-green-300">
                {operationsMetrics.activeServices} services running across {operationsMetrics.healthyNodes} nodes. 
                Average response time: {operationsMetrics.avgResponseTime}ms.
              </p>
            </div>
            <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
              View Status Page
            </button>
          </div>
        </motion.div>

        {/* Operations Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          {metrics.map((metric) => (
            <MetricCard
              key={metric.id}
              title={metric.title}
              value={metric.value}
              change={metric.change}
              trend={metric.trend}
              sparklineData={metric.sparklineData}
              alert={metric.alert}
            />
          ))}
        </div>

        {view === 'cards' ? (
          <>
            {/* System Health Overview */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                  <Server className="w-6 h-6 text-blue-500" />
                  System Health Overview
                </h2>
                <div className="space-y-4">
                  {systemHealth.map((system) => (
                    <motion.div
                      key={system.name}
                      whileHover={{ scale: 1.01 }}
                      className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <h3 className="font-semibold text-gray-900 dark:text-white">
                            {system.name}
                          </h3>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(system.status)}`}>
                            {system.status.toUpperCase()}
                          </span>
                          {system.alerts > 0 && (
                            <span className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 text-xs rounded-full">
                              {system.alerts} alerts
                            </span>
                          )}
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-semibold text-gray-900 dark:text-white">
                            {system.uptime}%
                          </div>
                          <div className="text-xs text-gray-500">
                            Uptime
                          </div>
                        </div>
                      </div>
                      <div className="grid grid-cols-4 gap-3 text-sm">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <Cpu className="w-3 h-3 text-gray-400" />
                            <span className="text-xs">{system.cpu}%</span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full ${getUtilizationColor(system.cpu)}`}
                              style={{ width: `${system.cpu}%` }}
                            />
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <Monitor className="w-3 h-3 text-gray-400" />
                            <span className="text-xs">{system.memory}%</span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full ${getUtilizationColor(system.memory)}`}
                              style={{ width: `${system.memory}%` }}
                            />
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <HardDrive className="w-3 h-3 text-gray-400" />
                            <span className="text-xs">{system.disk}%</span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full ${getUtilizationColor(system.disk)}`}
                              style={{ width: `${system.disk}%` }}
                            />
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <Network className="w-3 h-3 text-gray-400" />
                            <span className="text-xs">{system.network}%</span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full ${getUtilizationColor(system.network)}`}
                              style={{ width: `${system.network}%` }}
                            />
                          </div>
                        </div>
                      </div>
                      <div className="flex justify-between items-center mt-3 text-xs text-gray-500">
                        <span>{system.nodes} nodes</span>
                        <span>{system.region}</span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Services Status */}
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                  <Box className="w-6 h-6 text-green-500" />
                  Services Status
                </h2>
                <div className="space-y-4">
                  {services.map((service) => {
                    const Icon = service.icon
                    return (
                      <div key={service.name} className="p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <Icon className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                            <h3 className="font-semibold text-gray-900 dark:text-white">
                              {service.name}
                            </h3>
                            {getStatusIcon(service.status, service.health)}
                          </div>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(service.status, service.health)}`}>
                            {service.health.toUpperCase()}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-gray-600 dark:text-gray-400">Instances</span>
                            <div className="font-medium text-gray-900 dark:text-white">
                              {service.instances}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600 dark:text-gray-400">Response Time</span>
                            <div className="font-medium text-gray-900 dark:text-white">
                              {service.responseTime}ms
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600 dark:text-gray-400">Error Rate</span>
                            <div className="font-medium text-red-600 dark:text-red-400">
                              {(service.errorRate * 100).toFixed(3)}%
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600 dark:text-gray-400">Requests</span>
                            <div className="font-medium text-gray-900 dark:text-white">
                              {service.requests.toLocaleString()}
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>

            {/* Recent Incidents */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 mb-8">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                <AlertTriangle className="w-6 h-6 text-orange-500" />
                Recent Incidents
              </h2>
              <div className="space-y-4">
                {recentIncidents.map((incident) => (
                  <motion.div
                    key={incident.id}
                    whileHover={{ scale: 1.01 }}
                    className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(incident.severity)}`}>
                            {incident.severity.toUpperCase()}
                          </span>
                          <h3 className="font-semibold text-gray-900 dark:text-white">
                            {incident.title}
                          </h3>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            incident.status === 'resolved' 
                              ? 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-400'
                              : incident.status === 'investigating'
                              ? 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-400'
                              : 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-400'
                          }`}>
                            {incident.status.toUpperCase()}
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-2 mb-2">
                          {incident.affectedServices.map((service, idx) => (
                            <span key={idx} className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 text-xs rounded">
                              {service}
                            </span>
                          ))}
                        </div>
                        <div className="flex items-center gap-4 text-xs text-gray-500">
                          <span>Started: {incident.startTime.toLocaleString()}</span>
                          <span>Assignee: {incident.assignee}</span>
                        </div>
                      </div>
                      <button className="ml-4 px-3 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 transition-colors">
                        View Details
                      </button>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Quick Operations Actions */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">
                Quick Operations Actions
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { name: 'Scale Services', icon: TrendingUp, color: 'green' },
                  { name: 'Restart Service', icon: RotateCcw, color: 'blue' },
                  { name: 'View Logs', icon: Eye, color: 'purple' },
                  { name: 'Create Incident', icon: AlertTriangle, color: 'red' }
                ].map((action) => {
                  const Icon = action.icon
                  return (
                    <motion.button
                      key={action.name}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className={`p-4 rounded-lg border-2 border-dashed border-${action.color}-300 hover:border-${action.color}-500 hover:bg-${action.color}-50 dark:hover:bg-${action.color}-900/20 transition-all`}
                    >
                      <Icon className={`w-6 h-6 text-${action.color}-600 dark:text-${action.color}-400 mx-auto mb-2`} />
                      <div className="text-sm font-medium text-gray-900 dark:text-white">
                        {action.name}
                      </div>
                    </motion.button>
                  )
                })}
              </div>
            </div>
          </>
        ) : (
          <>
            {/* Visualization Mode */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              <ChartContainer
                title="System Performance Metrics"
                onDrillIn={() => console.log('Drill into performance metrics')}
              >
                <div className="p-4">
                  <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                    <p className="text-gray-500">System performance timeline visualization</p>
                  </div>
                </div>
              </ChartContainer>
              <ChartContainer
                title="Resource Utilization"
                onDrillIn={() => console.log('Drill into resource utilization')}
              >
                <div className="p-4">
                  <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                    <p className="text-gray-500">Resource utilization heatmap</p>
                  </div>
                </div>
              </ChartContainer>
            </div>
            
            {/* Service Performance Summary */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Gauge className="h-6 w-6 text-blue-600" />
                Service Performance Summary
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-2xl font-bold text-green-600">99.97%</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">System Uptime</div>
                </div>
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-2xl font-bold text-blue-600">42ms</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Response Time</div>
                </div>
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-2xl font-bold text-purple-600">47</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Active Services</div>
                </div>
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-2xl font-bold text-red-600">0.03%</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Error Rate</div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}