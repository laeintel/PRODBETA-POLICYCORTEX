'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Activity,
  Lock,
  Key,
  Bug,
  UserCheck,
  Globe,
  Database,
  Code,
  Search,
  Eye,
  Settings,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Target,
  Zap,
  Brain,
  FileSearch,
  Users,
  Server,
  Cloud
} from 'lucide-react'
import ViewToggle from '@/components/ViewToggle';
import ChartContainer from '@/components/ChartContainer';
import MetricCard from '@/components/MetricCard';

export default function TacticalSecurityPage() {
  const [view, setView] = useState<'cards' | 'visualizations'>('cards');
  const [securityMetrics, setSecurityMetrics] = useState({
    threatLevel: 'medium',
    activeThreats: 23,
    blockedAttacks: 1247,
    vulnerabilities: 89,
    complianceScore: 94.2
  })

  const [securityAlerts, setSecurityAlerts] = useState([
    {
      id: 'alert-001',
      type: 'vulnerability',
      severity: 'high',
      title: 'Critical SQL Injection Vulnerability Detected',
      description: 'Potential SQL injection in user authentication module',
      timestamp: new Date(Date.now() - 1800000),
      status: 'investigating',
      affectedSystems: ['web-app-prod', 'api-gateway']
    },
    {
      id: 'alert-002',
      type: 'access',
      severity: 'medium',
      title: 'Unusual Login Activity',
      description: 'Multiple failed login attempts from suspicious IP addresses',
      timestamp: new Date(Date.now() - 3600000),
      status: 'monitoring',
      affectedSystems: ['identity-service']
    },
    {
      id: 'alert-003',
      type: 'compliance',
      severity: 'low',
      title: 'Policy Violation Detected',
      description: 'Resource created without proper tagging compliance',
      timestamp: new Date(Date.now() - 7200000),
      status: 'resolved',
      affectedSystems: ['azure-subscription']
    }
  ])

  const threatCategories = [
    {
      name: 'Web Application',
      threats: 12,
      blocked: 156,
      severity: 'high',
      trend: 'up',
      icon: Globe,
      color: 'red'
    },
    {
      name: 'Infrastructure',
      threats: 8,
      blocked: 89,
      severity: 'medium',
      trend: 'stable',
      icon: Server,
      color: 'yellow'
    },
    {
      name: 'API Security',
      threats: 15,
      blocked: 234,
      severity: 'high',
      trend: 'down',
      icon: Code,
      color: 'red'
    },
    {
      name: 'Identity & Access',
      threats: 6,
      blocked: 67,
      severity: 'low',
      trend: 'down',
      icon: UserCheck,
      color: 'green'
    },
    {
      name: 'Data Protection',
      threats: 4,
      blocked: 23,
      severity: 'medium',
      trend: 'stable',
      icon: Database,
      color: 'yellow'
    },
    {
      name: 'Cloud Security',
      threats: 11,
      blocked: 145,
      severity: 'medium',
      trend: 'up',
      icon: Cloud,
      color: 'yellow'
    }
  ]

  const securityTools = [
    {
      name: 'SIEM Dashboard',
      status: 'operational',
      lastScan: '2 minutes ago',
      findings: 34,
      icon: Eye
    },
    {
      name: 'Vulnerability Scanner',
      status: 'operational',
      lastScan: '15 minutes ago',
      findings: 89,
      icon: Search
    },
    {
      name: 'Access Monitor',
      status: 'warning',
      lastScan: '5 minutes ago',
      findings: 12,
      icon: Lock
    },
    {
      name: 'Threat Intelligence',
      status: 'operational',
      lastScan: '1 minute ago',
      findings: 156,
      icon: Brain
    }
  ]

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-400'
      case 'high': return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-400'
      case 'medium': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-400'
      case 'low': return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-400'
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'operational': return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />
      case 'error': return <XCircle className="w-4 h-4 text-red-500" />
      default: return <Clock className="w-4 h-4 text-gray-400" />
    }
  }

  const metrics = [
    {
      id: 'threat-level',
      title: 'Current Threat Level',
      value: securityMetrics.threatLevel.toUpperCase(),
      change: 0,
      trend: 'stable' as const,
      sparklineData: [2, 2, 3, 2, 2, 2],
      alert: `${securityMetrics.activeThreats} active threats`
    },
    {
      id: 'blocked-attacks',
      title: 'Blocked Attacks (24h)',
      value: securityMetrics.blockedAttacks,
      change: -12.3,
      trend: 'down' as const,
      sparklineData: [1400, 1350, 1300, 1280, 1250, securityMetrics.blockedAttacks]
    },
    {
      id: 'vulnerabilities',
      title: 'Active Vulnerabilities',
      value: securityMetrics.vulnerabilities,
      change: -8.7,
      trend: 'down' as const,
      sparklineData: [98, 95, 92, 90, 91, securityMetrics.vulnerabilities]
    },
    {
      id: 'compliance-score',
      title: 'Compliance Score',
      value: `${securityMetrics.complianceScore}%`,
      change: 2.1,
      trend: 'up' as const,
      sparklineData: [91.5, 92.1, 92.8, 93.2, 93.8, securityMetrics.complianceScore]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-3">
              <Shield className="h-10 w-10 text-red-600" />
              Tactical Security Command
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Real-time security monitoring and threat response dashboard
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

        {/* Threat Level Alert */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 p-4 bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 border border-red-200 dark:border-red-800 rounded-xl"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded-full">
              <Shield className="w-8 h-8 text-red-600 dark:text-red-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-red-900 dark:text-red-100">
                Security Status: {securityMetrics.threatLevel.toUpperCase()} THREAT LEVEL
              </h3>
              <p className="text-red-700 dark:text-red-300">
                {securityMetrics.activeThreats} active security events require attention. 
                {securityMetrics.blockedAttacks} threats blocked in the last 24 hours.
              </p>
            </div>
            <button className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors">
              View Incidents
            </button>
          </div>
        </motion.div>

        {/* Security Metrics */}
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
            {/* Threat Categories */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
              {threatCategories.map((category) => {
                const Icon = category.icon
                return (
                  <motion.div
                    key={category.name}
                    whileHover={{ scale: 1.02 }}
                    className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div className={`p-3 rounded-lg bg-${category.color}-50 dark:bg-${category.color}-900/20`}>
                        <Icon className={`w-6 h-6 text-${category.color}-600 dark:text-${category.color}-400`} />
                      </div>
                      <div className="flex items-center gap-2">
                        {category.trend === 'up' && <TrendingUp className="w-4 h-4 text-red-500" />}
                        {category.trend === 'down' && <TrendingDown className="w-4 h-4 text-green-500" />}
                        {category.trend === 'stable' && <Activity className="w-4 h-4 text-gray-500" />}
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(category.severity)}`}>
                          {category.severity.toUpperCase()}
                        </span>
                      </div>
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                      {category.name}
                    </h3>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Active Threats</span>
                        <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                          {category.threats}
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Blocked</span>
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                          {category.blocked}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )
              })}
            </div>

            {/* Security Alerts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                  <AlertTriangle className="w-6 h-6 text-orange-500" />
                  Recent Security Alerts
                </h2>
                <div className="space-y-4">
                  {securityAlerts.map((alert) => (
                    <motion.div
                      key={alert.id}
                      whileHover={{ scale: 1.01 }}
                      className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(alert.severity)}`}>
                            {alert.severity.toUpperCase()}
                          </span>
                          <h3 className="font-semibold text-gray-900 dark:text-white">
                            {alert.title}
                          </h3>
                        </div>
                        <span className="text-xs text-gray-500">
                          {alert.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {alert.description}
                      </p>
                      <div className="flex items-center justify-between">
                        <div className="flex flex-wrap gap-1">
                          {alert.affectedSystems.map((system, idx) => (
                            <span key={idx} className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs rounded">
                              {system}
                            </span>
                          ))}
                        </div>
                        <button className="px-3 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700 transition-colors">
                          Investigate
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Security Tools Status */}
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                  <Target className="w-6 h-6 text-blue-500" />
                  Security Tools Status
                </h2>
                <div className="space-y-4">
                  {securityTools.map((tool) => {
                    const Icon = tool.icon
                    return (
                      <div key={tool.name} className="p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <Icon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                            <h3 className="font-semibold text-gray-900 dark:text-white">
                              {tool.name}
                            </h3>
                          </div>
                          {getStatusIcon(tool.status)}
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-gray-600 dark:text-gray-400">Last Scan</span>
                            <div className="font-medium text-gray-900 dark:text-white">
                              {tool.lastScan}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600 dark:text-gray-400">Findings</span>
                            <div className="font-medium text-gray-900 dark:text-white">
                              {tool.findings}
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">
                Quick Security Actions
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { name: 'Run Vulnerability Scan', icon: Search, color: 'blue' },
                  { name: 'Generate Security Report', icon: FileSearch, color: 'green' },
                  { name: 'Review Access Logs', icon: Eye, color: 'purple' },
                  { name: 'Update Security Policies', icon: Shield, color: 'red' }
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
                title="Threat Detection Over Time"
                onDrillIn={() => console.log('Drill into threat detection')}
              >
                <div className="p-4">
                  <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                    <p className="text-gray-500">Threat detection timeline visualization</p>
                  </div>
                </div>
              </ChartContainer>
              <ChartContainer
                title="Security Posture Analysis"
                onDrillIn={() => console.log('Drill into security posture')}
              >
                <div className="p-4">
                  <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                    <p className="text-gray-500">Security posture radar chart</p>
                  </div>
                </div>
              </ChartContainer>
            </div>
            
            {/* Vulnerability Breakdown */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Bug className="h-6 w-6 text-orange-600" />
                Vulnerability Analysis
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-2xl font-bold text-red-600">12</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Critical</div>
                </div>
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-2xl font-bold text-orange-600">34</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">High</div>
                </div>
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-2xl font-bold text-yellow-600">56</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Medium</div>
                </div>
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-2xl font-bold text-blue-600">23</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Low</div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}