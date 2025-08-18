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
import toast from 'react-hot-toast'
import { api } from '../../lib/api-client'
import { Shield, Lock, Key, AlertTriangle, CheckCircle, XCircle, Eye, TrendingUp, Users, Activity } from 'lucide-react'
import AppLayout from '../../components/AppLayout'

export default function SecurityPage() {
  const [securityData, setSecurityData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchSecurityData()
  }, [])

  const fetchSecurityData = async () => {
    try {
      const resp = await api.request<any>('/api/v1/security')
      if (!resp.error) setSecurityData(resp.data)
    } catch (error) {
      console.error('Failed to fetch security data:', error)
      toast.error('Failed to load security data')
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
          <Shield className="w-8 h-8 mr-3 text-blue-500" />
          Security Center
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Monitor and manage your Azure security posture
        </p>
      </div>

      {/* Security Score */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl p-8 text-white mb-8"
      >
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">Security Score</h2>
            <div className="text-5xl font-bold">87%</div>
            <p className="text-blue-100 mt-2">+5% from last month</p>
          </div>
          <div className="text-right">
            <div className="text-xl mb-2">Recommendations</div>
            <div className="text-3xl font-bold">23</div>
            <p className="text-blue-100">High priority: 5</p>
          </div>
        </div>
      </motion.div>

      {/* Security Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <Lock className="w-8 h-8 text-green-500" />
            <span className="text-sm bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 px-2 py-1 rounded-full">
              Secure
            </span>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            Identity & Access
          </h3>
          <div className="text-3xl font-bold text-gray-900 dark:text-white">98%</div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
            MFA enabled for all admin accounts
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <Key className="w-8 h-8 text-yellow-500" />
            <span className="text-sm bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300 px-2 py-1 rounded-full">
              Warning
            </span>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            Key Management
          </h3>
          <div className="text-3xl font-bold text-gray-900 dark:text-white">12</div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
            Keys expiring in 30 days
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <AlertTriangle className="w-8 h-8 text-red-500" />
            <span className="text-sm bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 px-2 py-1 rounded-full">
              Critical
            </span>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            Security Alerts
          </h3>
          <div className="text-3xl font-bold text-gray-900 dark:text-white">3</div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
            Requires immediate attention
          </p>
        </motion.div>
      </div>

      {/* Threat Intelligence */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
      >
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
          Threat Intelligence Dashboard
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold mb-3">Recent Threats</h3>
            <div className="space-y-3">
              {[
                { type: 'Brute Force', count: 23, trend: 'down' },
                { type: 'SQL Injection', count: 5, trend: 'up' },
                { type: 'DDoS Attempts', count: 2, trend: 'stable' }
              ].map((threat, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <span className="font-medium">{threat.type}</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm">{threat.count} incidents</span>
                    {threat.trend === 'up' && <TrendingUp className="w-4 h-4 text-red-500" />}
                    {threat.trend === 'down' && <TrendingUp className="w-4 h-4 text-green-500 rotate-180" />}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-3">Protection Status</h3>
            <div className="space-y-3">
              {[
                { service: 'Azure Sentinel', status: 'active', health: 'healthy' },
                { service: 'Azure Firewall', status: 'active', health: 'healthy' },
                { service: 'DDoS Protection', status: 'active', health: 'warning' }
              ].map((service, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <span className="font-medium">{service.service}</span>
                  <div className="flex items-center space-x-2">
                    {service.health === 'healthy' ? (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    ) : (
                      <AlertTriangle className="w-5 h-5 text-yellow-500" />
                    )}
                    <span className="text-sm capitalize">{service.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
    </AppLayout>
  )
}