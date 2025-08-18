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
import { Monitor, Activity, Bell, AlertTriangle, CheckCircle, Clock, BarChart3, TrendingUp } from 'lucide-react'
import AppLayout from '../../components/AppLayout'
import toast from 'react-hot-toast'
import { api } from '../../lib/api-client'

export default function MonitoringPage() {
  const [monitoringData, setMonitoringData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchMonitoringData()
  }, [])

  const fetchMonitoringData = async () => {
    try {
      const resp = await api.getMonitoring()
      if (!resp.error) setMonitoringData(resp.data)
    } catch (error) {
      console.error('Failed to fetch monitoring data:', error)
      toast.error('Failed to load monitoring data')
    } finally {
      setLoading(false)
    }
  }

  const triggerAction = async (actionType: string, params?: any) => {
    try {
      const resp = await api.createAction('global', actionType, params)
      if (resp.error || resp.status >= 400) {
        toast.error(`Action failed: ${actionType}`)
        return
      }
      toast.success(`${actionType.replace('_',' ')} started`)
      const id = resp.data?.action_id || resp.data?.id
      if (id) {
        const stop = api.streamActionEvents(String(id), (m) => console.log('[monitoring-action]', id, m))
        setTimeout(stop, 60000)
      }
    } catch (e) {
      toast.error(`Action error: ${actionType}`)
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
          <Monitor className="w-8 h-8 mr-3 text-cyan-500" />
          Monitoring Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Real-time monitoring and alerting across your Azure infrastructure
        </p>
      </div>

      {/* Alert Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <AlertTriangle className="w-6 h-6 text-red-600" />
            <span className="text-2xl font-bold text-red-600">3</span>
          </div>
          <p className="text-sm font-medium text-red-800 dark:text-red-400">Critical Alerts</p>
          <p className="text-xs text-red-600 dark:text-red-500 mt-1">Immediate action required</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <Bell className="w-6 h-6 text-yellow-600" />
            <span className="text-2xl font-bold text-yellow-600">12</span>
          </div>
          <p className="text-sm font-medium text-yellow-800 dark:text-yellow-400">Warnings</p>
          <p className="text-xs text-yellow-600 dark:text-yellow-500 mt-1">Review recommended</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-6 h-6 text-blue-600" />
            <span className="text-2xl font-bold text-blue-600">156</span>
          </div>
          <p className="text-sm font-medium text-blue-800 dark:text-blue-400">Active Monitors</p>
          <p className="text-xs text-blue-600 dark:text-blue-500 mt-1">All systems tracked</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <CheckCircle className="w-6 h-6 text-green-600" />
            <span className="text-2xl font-bold text-green-600">98.5%</span>
          </div>
          <p className="text-sm font-medium text-green-800 dark:text-green-400">Health Score</p>
          <p className="text-xs text-green-600 dark:text-green-500 mt-1">Systems healthy</p>
        </motion.div>
      </div>

      {/* Active Incidents */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm mb-8"
      >
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Active Incidents</h2>
        <div className="space-y-4">
          {[
            {
              severity: 'critical',
              title: 'Database Connection Pool Exhausted',
              resource: 'SQL-PROD-01',
              time: '5 mins ago',
              status: 'investigating'
            },
            {
              severity: 'warning',
              title: 'High Memory Usage Detected',
              resource: 'VM-APP-03',
              time: '15 mins ago',
              status: 'monitoring'
            },
            {
              severity: 'critical',
              title: 'SSL Certificate Expiring',
              resource: 'APP-GATEWAY-01',
              time: '1 hour ago',
              status: 'assigned'
            }
          ].map((incident, idx) => (
            <div key={idx} className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <div className="flex items-center space-x-4">
                <div className={`w-2 h-12 rounded-full ${
                  incident.severity === 'critical' ? 'bg-red-500' : 'bg-yellow-500'
                }`} />
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">{incident.title}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {incident.resource} • {incident.time}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <span className={`text-xs px-2 py-1 rounded-full ${
                  incident.status === 'investigating' ? 'bg-orange-100 text-orange-700' :
                  incident.status === 'assigned' ? 'bg-blue-100 text-blue-700' :
                  'bg-gray-100 text-gray-700'
                }`}>
                  {incident.status}
                </span>
                <button className="px-3 py-1 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600">
                  View
                </button>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* System Health */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Service Health</h2>
          <div className="space-y-3">
            {[
              { service: 'Azure Active Directory', status: 'operational', uptime: '100%' },
              { service: 'Azure SQL Database', status: 'degraded', uptime: '99.2%' },
              { service: 'Azure Storage', status: 'operational', uptime: '99.9%' },
              { service: 'Azure Functions', status: 'operational', uptime: '100%' },
              { service: 'Azure CDN', status: 'operational', uptime: '99.8%' }
            ].map((service, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    service.status === 'operational' ? 'bg-green-500' :
                    service.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
                  }`} />
                  <span className="font-medium text-sm">{service.service}</span>
                </div>
                <div className="flex items-center space-x-3">
                  <span className="text-xs text-gray-500">{service.uptime}</span>
                  <span className={`text-xs capitalize ${
                    service.status === 'operational' ? 'text-green-600' :
                    service.status === 'degraded' ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {service.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Alert Rules</h2>
          <div className="space-y-3">
            {[
              { rule: 'CPU > 80%', triggers: 23, lastTriggered: '2 hours ago' },
              { rule: 'Memory > 90%', triggers: 12, lastTriggered: '5 hours ago' },
              { rule: 'Disk Space < 10%', triggers: 3, lastTriggered: '1 day ago' },
              { rule: 'Response Time > 500ms', triggers: 45, lastTriggered: '30 mins ago' },
              { rule: 'Error Rate > 1%', triggers: 8, lastTriggered: '4 hours ago' }
            ].map((rule, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div>
                  <p className="font-medium text-sm">{rule.rule}</p>
                  <p className="text-xs text-gray-500 mt-1">Last: {rule.lastTriggered}</p>
                </div>
                <div className="text-right">
                  <p className="text-lg font-semibold">{rule.triggers}</p>
                  <p className="text-xs text-gray-500">triggers</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
    </AppLayout>
  )
}