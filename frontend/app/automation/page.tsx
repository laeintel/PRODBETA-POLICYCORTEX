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

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Cpu, Play, Square, Clock, Zap, GitBranch, CheckCircle, AlertCircle } from 'lucide-react'
import AppLayout from '../../components/AppLayout'

export default function AutomationPage() {
  const [automationData, setAutomationData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchAutomationData()
  }, [])

  const fetchAutomationData = async () => {
    try {
      const response = await fetch('/api/v1/automation')
      const data = await response.json()
      setAutomationData(data)
    } catch (error) {
      console.error('Failed to fetch automation data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center min-h-screen">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout>
      <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center">
          <Cpu className="w-8 h-8 mr-3 text-orange-500" />
          Automation Center
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Manage and monitor automated workflows and remediation actions
        </p>
      </div>

      {/* Automation Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <Play className="w-8 h-8 text-green-500" />
            <span className="text-3xl font-bold">24</span>
          </div>
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Active Workflows</p>
          <p className="text-xs text-gray-500 mt-1">Running now</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <Zap className="w-8 h-8 text-yellow-500" />
            <span className="text-3xl font-bold">156</span>
          </div>
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Executions Today</p>
          <p className="text-xs text-gray-500 mt-1">+23% from yesterday</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <CheckCircle className="w-8 h-8 text-blue-500" />
            <span className="text-3xl font-bold">94%</span>
          </div>
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Success Rate</p>
          <p className="text-xs text-gray-500 mt-1">Last 7 days</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <Clock className="w-8 h-8 text-purple-500" />
            <span className="text-3xl font-bold">2.3h</span>
          </div>
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Time Saved</p>
          <p className="text-xs text-gray-500 mt-1">Today</p>
        </motion.div>
      </div>

      {/* Active Workflows */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm mb-8"
      >
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Active Workflows</h2>
        <div className="space-y-4">
          {[
            {
              name: 'Auto-Scale Based on Traffic',
              status: 'running',
              trigger: 'Schedule',
              lastRun: '5 mins ago',
              nextRun: 'In 10 mins'
            },
            {
              name: 'Security Patch Deployment',
              status: 'running',
              trigger: 'Event',
              lastRun: '1 hour ago',
              nextRun: 'On trigger'
            },
            {
              name: 'Cost Optimization Cleanup',
              status: 'scheduled',
              trigger: 'Schedule',
              lastRun: 'Yesterday',
              nextRun: 'Tonight 2 AM'
            },
            {
              name: 'Compliance Remediation',
              status: 'paused',
              trigger: 'Manual',
              lastRun: '3 days ago',
              nextRun: 'Manual trigger'
            }
          ].map((workflow, idx) => (
            <div key={idx} className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <div className="flex items-center space-x-4">
                <div className={`w-3 h-3 rounded-full ${
                  workflow.status === 'running' ? 'bg-green-500 animate-pulse' :
                  workflow.status === 'scheduled' ? 'bg-blue-500' :
                  'bg-gray-400'
                }`} />
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">{workflow.name}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {workflow.trigger} • Last: {workflow.lastRun} • Next: {workflow.nextRun}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {workflow.status === 'running' ? (
                  <button className="p-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200">
                    <Square className="w-4 h-4" />
                  </button>
                ) : (
                  <button className="p-2 bg-green-100 text-green-600 rounded-lg hover:bg-green-200">
                    <Play className="w-4 h-4" />
                  </button>
                )}
                <button className="px-3 py-1 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600">
                  Edit
                </button>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Recent Executions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
      >
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Recent Executions</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left border-b border-gray-200 dark:border-gray-700">
                <th className="pb-3 text-sm font-medium text-gray-600 dark:text-gray-400">Workflow</th>
                <th className="pb-3 text-sm font-medium text-gray-600 dark:text-gray-400">Status</th>
                <th className="pb-3 text-sm font-medium text-gray-600 dark:text-gray-400">Duration</th>
                <th className="pb-3 text-sm font-medium text-gray-600 dark:text-gray-400">Started</th>
                <th className="pb-3 text-sm font-medium text-gray-600 dark:text-gray-400">Actions</th>
              </tr>
            </thead>
            <tbody>
              {[
                { workflow: 'Database Backup', status: 'success', duration: '12m 34s', started: '10:30 AM' },
                { workflow: 'SSL Certificate Renewal', status: 'success', duration: '2m 15s', started: '09:45 AM' },
                { workflow: 'Resource Tagging', status: 'failed', duration: '5m 12s', started: '09:00 AM' },
                { workflow: 'Log Archive', status: 'success', duration: '45m 20s', started: '08:30 AM' },
                { workflow: 'Security Scan', status: 'success', duration: '8m 45s', started: '08:00 AM' }
              ].map((execution, idx) => (
                <tr key={idx} className="border-b border-gray-100 dark:border-gray-700">
                  <td className="py-3 text-sm">{execution.workflow}</td>
                  <td className="py-3">
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      execution.status === 'success' ? 'bg-green-100 text-green-700' :
                      'bg-red-100 text-red-700'
                    }`}>
                      {execution.status}
                    </span>
                  </td>
                  <td className="py-3 text-sm text-gray-600 dark:text-gray-400">{execution.duration}</td>
                  <td className="py-3 text-sm text-gray-600 dark:text-gray-400">{execution.started}</td>
                  <td className="py-3">
                    <button className="text-blue-500 hover:text-blue-600 text-sm">View Logs</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
    </AppLayout>
  )
}