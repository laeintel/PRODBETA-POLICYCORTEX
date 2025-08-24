'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Scale,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Clock,
  Activity,
  FileText,
  Users,
  Shield,
  DollarSign,
  BarChart3,
  TrendingUp,
  TrendingDown,
  Target,
  Settings,
  RefreshCw,
  Eye,
  Download,
  Flag,
  Award,
  Gavel,
  BookOpen,
  AlertCircle,
  Lock,
  Database,
  Globe
} from 'lucide-react'
import ViewToggle from '@/components/ViewToggle';
import ChartContainer from '@/components/ChartContainer';
import MetricCard from '@/components/MetricCard';

export default function TacticalGovernancePage() {
  const [view, setView] = useState<'cards' | 'visualizations'>('cards');
  const [governanceMetrics, setGovernanceMetrics] = useState({
    complianceScore: 94.2,
    activePolicies: 156,
    policyViolations: 23,
    riskScore: 2.3,
    auditReadiness: 89
  })

  const [complianceFrameworks, setComplianceFrameworks] = useState([
    {
      name: 'SOC 2 Type II',
      status: 'compliant',
      score: 96.8,
      lastAudit: new Date(Date.now() - 2592000000), // 30 days ago
      nextAudit: new Date(Date.now() + 7776000000), // 90 days from now
      violations: 2,
      remediation: 1
    },
    {
      name: 'ISO 27001',
      status: 'compliant',
      score: 93.4,
      lastAudit: new Date(Date.now() - 5184000000), // 60 days ago
      nextAudit: new Date(Date.now() + 10368000000), // 120 days from now
      violations: 4,
      remediation: 2
    },
    {
      name: 'GDPR',
      status: 'at-risk',
      score: 87.2,
      lastAudit: new Date(Date.now() - 1296000000), // 15 days ago
      nextAudit: new Date(Date.now() + 2592000000), // 30 days from now
      violations: 8,
      remediation: 6
    },
    {
      name: 'HIPAA',
      status: 'compliant',
      score: 91.7,
      lastAudit: new Date(Date.now() - 3888000000), // 45 days ago
      nextAudit: new Date(Date.now() + 5184000000), // 60 days from now
      violations: 3,
      remediation: 1
    }
  ])

  const [policyCategories, setPolicyCategories] = useState([
    {
      name: 'Access Control',
      policies: 24,
      violations: 5,
      compliance: 91.2,
      trend: 'up',
      icon: Lock,
      color: 'blue'
    },
    {
      name: 'Data Protection',
      policies: 18,
      violations: 2,
      compliance: 96.8,
      trend: 'stable',
      icon: Database,
      color: 'green'
    },
    {
      name: 'Network Security',
      policies: 31,
      violations: 8,
      compliance: 87.3,
      trend: 'down',
      icon: Globe,
      color: 'yellow'
    },
    {
      name: 'Incident Response',
      policies: 12,
      violations: 1,
      compliance: 94.7,
      trend: 'up',
      icon: AlertCircle,
      color: 'purple'
    },
    {
      name: 'Risk Management',
      policies: 22,
      violations: 4,
      compliance: 89.5,
      trend: 'stable',
      icon: Shield,
      color: 'orange'
    },
    {
      name: 'Asset Management',
      policies: 19,
      violations: 3,
      compliance: 92.1,
      trend: 'up',
      icon: Target,
      color: 'indigo'
    }
  ])

  const [recentActions, setRecentActions] = useState([
    {
      id: 'action-001',
      type: 'policy_update',
      title: 'Updated Data Retention Policy',
      description: 'Extended retention period for audit logs from 1 year to 3 years',
      timestamp: new Date(Date.now() - 3600000),
      user: 'Sarah Johnson',
      status: 'completed',
      impact: 'low'
    },
    {
      id: 'action-002',
      type: 'violation_resolved',
      title: 'Resolved GDPR Violation',
      description: 'Updated user consent mechanism for data processing',
      timestamp: new Date(Date.now() - 7200000),
      user: 'Mike Chen',
      status: 'completed',
      impact: 'high'
    },
    {
      id: 'action-003',
      type: 'audit_scheduled',
      title: 'SOC 2 Audit Scheduled',
      description: 'External audit scheduled for Q2 2024 compliance certification',
      timestamp: new Date(Date.now() - 14400000),
      user: 'Lisa Rodriguez',
      status: 'pending',
      impact: 'medium'
    }
  ])

  const getComplianceStatusColor = (status: string) => {
    switch (status) {
      case 'compliant': return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-400'
      case 'at-risk': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-400'
      case 'non-compliant': return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-400'
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-400'
    }
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="w-4 h-4 text-green-500" />
      case 'down': return <TrendingDown className="w-4 h-4 text-red-500" />
      case 'stable': return <Activity className="w-4 h-4 text-gray-500" />
      default: return null
    }
  }

  const getActionIcon = (type: string) => {
    switch (type) {
      case 'policy_update': return <FileText className="w-4 h-4 text-blue-500" />
      case 'violation_resolved': return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'audit_scheduled': return <Clock className="w-4 h-4 text-purple-500" />
      default: return <Activity className="w-4 h-4 text-gray-500" />
    }
  }

  const metrics = [
    {
      id: 'compliance-score',
      title: 'Overall Compliance Score',
      value: `${governanceMetrics.complianceScore}%`,
      change: 2.3,
      trend: 'up' as const,
      sparklineData: [89.1, 90.5, 91.8, 92.4, 93.1, governanceMetrics.complianceScore],
      alert: `${complianceFrameworks.filter(f => f.status === 'at-risk').length} frameworks at risk`
    },
    {
      id: 'active-policies',
      title: 'Active Policies',
      value: governanceMetrics.activePolicies,
      change: 8.7,
      trend: 'up' as const,
      sparklineData: [142, 145, 148, 152, 154, governanceMetrics.activePolicies]
    },
    {
      id: 'violations',
      title: 'Policy Violations',
      value: governanceMetrics.policyViolations,
      change: -12.8,
      trend: 'down' as const,
      sparklineData: [32, 28, 26, 25, 24, governanceMetrics.policyViolations]
    },
    {
      id: 'audit-readiness',
      title: 'Audit Readiness',
      value: `${governanceMetrics.auditReadiness}%`,
      change: 4.2,
      trend: 'up' as const,
      sparklineData: [82, 84, 86, 87, 88, governanceMetrics.auditReadiness]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-3">
              <Scale className="h-10 w-10 text-purple-600" />
              Tactical Governance Command
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Real-time governance monitoring and compliance management dashboard
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

        {/* Compliance Status Alert */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 p-4 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 border border-purple-200 dark:border-purple-800 rounded-xl"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-full">
              <Award className="w-8 h-8 text-purple-600 dark:text-purple-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-purple-900 dark:text-purple-100">
                Governance Status: {governanceMetrics.complianceScore}% Compliant
              </h3>
              <p className="text-purple-700 dark:text-purple-300">
                {governanceMetrics.activePolicies} active policies with {governanceMetrics.policyViolations} violations. 
                Next audit in {Math.ceil((Math.min(...complianceFrameworks.map(f => f.nextAudit.getTime())) - Date.now()) / (1000 * 60 * 60 * 24))} days.
              </p>
            </div>
            <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
              View Compliance Report
            </button>
          </div>
        </motion.div>

        {/* Governance Metrics */}
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
            {/* Compliance Frameworks */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                  <Gavel className="w-6 h-6 text-purple-500" />
                  Compliance Frameworks
                </h2>
                <div className="space-y-4">
                  {complianceFrameworks.map((framework) => (
                    <motion.div
                      key={framework.name}
                      whileHover={{ scale: 1.01 }}
                      className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-3">
                          <h3 className="font-semibold text-gray-900 dark:text-white">
                            {framework.name}
                          </h3>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getComplianceStatusColor(framework.status)}`}>
                            {framework.status.replace('-', ' ').toUpperCase()}
                          </span>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-bold text-gray-900 dark:text-white">
                            {framework.score}%
                          </div>
                          <div className="text-xs text-gray-500">
                            Score
                          </div>
                        </div>
                      </div>
                      <div className="grid grid-cols-3 gap-4 text-sm mt-3">
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">Violations</span>
                          <div className="font-medium text-red-600 dark:text-red-400">
                            {framework.violations}
                          </div>
                        </div>
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">Remediation</span>
                          <div className="font-medium text-yellow-600 dark:text-yellow-400">
                            {framework.remediation}
                          </div>
                        </div>
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">Next Audit</span>
                          <div className="font-medium text-gray-900 dark:text-white">
                            {Math.ceil((framework.nextAudit.getTime() - Date.now()) / (1000 * 60 * 60 * 24))}d
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Policy Categories */}
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                  <BookOpen className="w-6 h-6 text-blue-500" />
                  Policy Categories
                </h2>
                <div className="space-y-4">
                  {policyCategories.map((category) => {
                    const Icon = category.icon
                    return (
                      <div key={category.name} className="p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <div className={`p-2 rounded-lg bg-${category.color}-50 dark:bg-${category.color}-900/20`}>
                              <Icon className={`w-4 h-4 text-${category.color}-600 dark:text-${category.color}-400`} />
                            </div>
                            <h3 className="font-semibold text-gray-900 dark:text-white">
                              {category.name}
                            </h3>
                          </div>
                          <div className="flex items-center gap-2">
                            {getTrendIcon(category.trend)}
                            <span className="text-sm font-medium text-gray-900 dark:text-white">
                              {category.compliance}%
                            </span>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-gray-600 dark:text-gray-400">Policies</span>
                            <div className="font-medium text-gray-900 dark:text-white">
                              {category.policies}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600 dark:text-gray-400">Violations</span>
                            <div className="font-medium text-red-600 dark:text-red-400">
                              {category.violations}
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>

            {/* Recent Governance Actions */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 mb-8">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                <Activity className="w-6 h-6 text-green-500" />
                Recent Governance Actions
              </h2>
              <div className="space-y-4">
                {recentActions.map((action) => (
                  <motion.div
                    key={action.id}
                    whileHover={{ scale: 1.01 }}
                    className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          {getActionIcon(action.type)}
                          <h3 className="font-semibold text-gray-900 dark:text-white">
                            {action.title}
                          </h3>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            action.status === 'completed' 
                              ? 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-400'
                              : action.status === 'pending'
                              ? 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-400'
                              : 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-400'
                          }`}>
                            {action.status.toUpperCase()}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                          {action.description}
                        </p>
                        <div className="flex items-center gap-4 text-xs text-gray-500">
                          <span>By: {action.user}</span>
                          <span>{action.timestamp.toLocaleString()}</span>
                          <span className={`px-2 py-1 rounded ${
                            action.impact === 'high'
                              ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                              : action.impact === 'medium'
                              ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                              : 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                          }`}>
                            {action.impact.toUpperCase()} IMPACT
                          </span>
                        </div>
                      </div>
                      <button className="ml-4 px-3 py-1 bg-purple-600 text-white text-xs rounded hover:bg-purple-700 transition-colors">
                        View Details
                      </button>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">
                Quick Governance Actions
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { name: 'Generate Compliance Report', icon: Download, color: 'blue' },
                  { name: 'Schedule Audit', icon: Clock, color: 'purple' },
                  { name: 'Review Policy Violations', icon: Eye, color: 'red' },
                  { name: 'Update Risk Assessment', icon: Target, color: 'green' }
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
                title="Compliance Trends Over Time"
                onDrillIn={() => console.log('Drill into compliance trends')}
              >
                <div className="p-4">
                  <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                    <p className="text-gray-500">Compliance trends line chart visualization</p>
                  </div>
                </div>
              </ChartContainer>
              <ChartContainer
                title="Risk Assessment Matrix"
                onDrillIn={() => console.log('Drill into risk assessment')}
              >
                <div className="p-4">
                  <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                    <p className="text-gray-500">Risk assessment heat map visualization</p>
                  </div>
                </div>
              </ChartContainer>
            </div>
            
            {/* Policy Effectiveness */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <BarChart3 className="h-6 w-6 text-green-600" />
                Policy Effectiveness Analysis
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-3xl font-bold text-green-600">94.2%</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Overall Compliance</div>
                </div>
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-3xl font-bold text-yellow-600">23</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Active Violations</div>
                </div>
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-3xl font-bold text-blue-600">156</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Active Policies</div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}