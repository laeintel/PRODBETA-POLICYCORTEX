'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  GitBranch,
  Package,
  Rocket,
  Settings,
  Activity,
  CheckCircle,
  AlertCircle,
  Clock,
  TrendingUp,
  Server,
  Database,
  Cloud,
  Shield,
  Zap,
  BarChart3
} from 'lucide-react'

export default function TacticalDevOpsPage() {
  const [selectedPipeline, setSelectedPipeline] = useState<string | null>(null)
  const [deploymentMetrics, setDeploymentMetrics] = useState({
    success_rate: 98.5,
    avg_deployment_time: 4.2,
    rollback_rate: 1.2,
    total_deployments: 1247
  })

  const pipelines = [
    {
      id: 'main-api',
      name: 'Main API Pipeline',
      status: 'success',
      lastRun: '2 minutes ago',
      duration: '3m 45s',
      stages: ['Build', 'Test', 'Security Scan', 'Deploy'],
      successRate: 99.2
    },
    {
      id: 'frontend',
      name: 'Frontend Pipeline',
      status: 'running',
      lastRun: 'Running now',
      duration: '2m 15s',
      stages: ['Build', 'Test', 'Deploy'],
      successRate: 97.8
    },
    {
      id: 'ml-training',
      name: 'ML Training Pipeline',
      status: 'failed',
      lastRun: '1 hour ago',
      duration: '45m 30s',
      stages: ['Data Prep', 'Training', 'Validation', 'Deploy'],
      successRate: 94.5
    }
  ]

  const environments = [
    { name: 'Production', status: 'healthy', resources: 42, uptime: 99.99 },
    { name: 'Staging', status: 'healthy', resources: 28, uptime: 99.95 },
    { name: 'Development', status: 'warning', resources: 35, uptime: 98.2 }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Tactical DevOps
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Real-time CI/CD pipeline monitoring and deployment management
          </p>
        </div>

        {/* Deployment Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700"
          >
            <div className="flex items-center justify-between mb-4">
              <CheckCircle className="w-8 h-8 text-green-500" />
              <span className="text-2xl font-bold text-gray-900 dark:text-white">
                {deploymentMetrics.success_rate}%
              </span>
            </div>
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">Success Rate</h3>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700"
          >
            <div className="flex items-center justify-between mb-4">
              <Clock className="w-8 h-8 text-blue-500" />
              <span className="text-2xl font-bold text-gray-900 dark:text-white">
                {deploymentMetrics.avg_deployment_time}m
              </span>
            </div>
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Deploy Time</h3>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700"
          >
            <div className="flex items-center justify-between mb-4">
              <AlertCircle className="w-8 h-8 text-orange-500" />
              <span className="text-2xl font-bold text-gray-900 dark:text-white">
                {deploymentMetrics.rollback_rate}%
              </span>
            </div>
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">Rollback Rate</h3>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700"
          >
            <div className="flex items-center justify-between mb-4">
              <Rocket className="w-8 h-8 text-purple-500" />
              <span className="text-2xl font-bold text-gray-900 dark:text-white">
                {deploymentMetrics.total_deployments}
              </span>
            </div>
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Deploys</h3>
          </motion.div>
        </div>

        {/* Pipelines */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">
              Active Pipelines
            </h2>
            <div className="space-y-4">
              {pipelines.map((pipeline) => (
                <motion.div
                  key={pipeline.id}
                  whileHover={{ scale: 1.01 }}
                  onClick={() => setSelectedPipeline(pipeline.id)}
                  className={`p-4 rounded-lg border cursor-pointer transition-all ${
                    selectedPipeline === pipeline.id
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-purple-400'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-gray-900 dark:text-white">
                      {pipeline.name}
                    </h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      pipeline.status === 'success'
                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                        : pipeline.status === 'running'
                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                        : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                    }`}>
                      {pipeline.status}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {pipeline.lastRun}
                    </span>
                    <span className="flex items-center gap-1">
                      <Activity className="w-3 h-3" />
                      {pipeline.duration}
                    </span>
                    <span className="flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" />
                      {pipeline.successRate}%
                    </span>
                  </div>
                  <div className="mt-3 flex gap-2">
                    {pipeline.stages.map((stage, idx) => (
                      <div
                        key={idx}
                        className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden"
                      >
                        <div
                          className={`h-full transition-all ${
                            pipeline.status === 'success' || idx < 2
                              ? 'bg-green-500'
                              : pipeline.status === 'running' && idx === 2
                              ? 'bg-blue-500 animate-pulse'
                              : 'bg-gray-300 dark:bg-gray-600'
                          }`}
                          style={{ width: pipeline.status === 'failed' && idx > 2 ? '0%' : '100%' }}
                        />
                      </div>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Environments */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">
              Environment Status
            </h2>
            <div className="space-y-4">
              {environments.map((env) => (
                <div
                  key={env.name}
                  className="p-4 rounded-lg border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold text-gray-900 dark:text-white">{env.name}</h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      env.status === 'healthy'
                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                        : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                    }`}>
                      {env.status}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Resources</span>
                      <div className="font-semibold text-gray-900 dark:text-white">{env.resources}</div>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Uptime</span>
                      <div className="font-semibold text-gray-900 dark:text-white">{env.uptime}%</div>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Health</span>
                      <div className="flex gap-1 mt-1">
                        <Server className="w-4 h-4 text-green-500" />
                        <Database className="w-4 h-4 text-green-500" />
                        <Cloud className="w-4 h-4 text-green-500" />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Cloud Integration Status */}
            <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                Multi-Cloud Integration
              </h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="w-12 h-12 mx-auto mb-2 bg-blue-500 rounded-lg flex items-center justify-center">
                    <Cloud className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-sm font-medium text-gray-900 dark:text-white">Azure</div>
                  <div className="text-xs text-green-600 dark:text-green-400">Connected</div>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 mx-auto mb-2 bg-orange-500 rounded-lg flex items-center justify-center">
                    <Cloud className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-sm font-medium text-gray-900 dark:text-white">AWS</div>
                  <div className="text-xs text-green-600 dark:text-green-400">Connected</div>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 mx-auto mb-2 bg-green-500 rounded-lg flex items-center justify-center">
                    <Cloud className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-sm font-medium text-gray-900 dark:text-white">GCP</div>
                  <div className="text-xs text-yellow-600 dark:text-yellow-400">Pending</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => console.log('Deploy clicked')}
            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 transition-all"
          >
            <Rocket className="w-4 h-4 inline mr-2" />
            Deploy to Production
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => console.log('Settings clicked')}
            className="px-6 py-3 bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg font-medium hover:bg-gray-300 dark:hover:bg-gray-600 transition-all"
          >
            <Settings className="w-4 h-4 inline mr-2" />
            Pipeline Settings
          </motion.button>
        </div>
      </div>
    </div>
  )
}