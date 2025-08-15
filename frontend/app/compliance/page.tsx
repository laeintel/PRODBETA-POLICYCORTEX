'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { FileCheck, Shield, AlertCircle, CheckCircle, XCircle, TrendingUp, BarChart3, Clock } from 'lucide-react'
import AppLayout from '../../components/AppLayout'

export default function CompliancePage() {
  const [complianceData, setComplianceData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchComplianceData()
  }, [])

  const fetchComplianceData = async () => {
    try {
      const response = await fetch('/api/v1/compliance')
      const data = await response.json()
      setComplianceData(data)
    } catch (error) {
      console.error('Failed to fetch compliance data:', error)
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
          <FileCheck className="w-8 h-8 mr-3 text-green-500" />
          Compliance Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Track and manage regulatory compliance across your Azure resources
        </p>
      </div>

      {/* Compliance Score */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-green-500 to-teal-600 rounded-2xl p-8 text-white mb-8"
      >
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">Overall Compliance</h2>
            <div className="text-5xl font-bold">92%</div>
            <p className="text-green-100 mt-2">+3% from last assessment</p>
          </div>
          <div className="text-right">
            <div className="text-xl mb-2">Policies</div>
            <div className="text-3xl font-bold">156</div>
            <p className="text-green-100">Non-compliant: 12</p>
          </div>
        </div>
      </motion.div>

      {/* Compliance by Framework */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {[
          { framework: 'ISO 27001', score: 95, status: 'compliant' },
          { framework: 'SOC 2', score: 88, status: 'partial' },
          { framework: 'GDPR', score: 92, status: 'compliant' },
          { framework: 'HIPAA', score: 78, status: 'non-compliant' }
        ].map((framework, idx) => (
          <motion.div
            key={framework.framework}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
          >
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                {framework.framework}
              </span>
              {framework.status === 'compliant' && (
                <CheckCircle className="w-5 h-5 text-green-500" />
              )}
              {framework.status === 'partial' && (
                <AlertCircle className="w-5 h-5 text-yellow-500" />
              )}
              {framework.status === 'non-compliant' && (
                <XCircle className="w-5 h-5 text-red-500" />
              )}
            </div>
            <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              {framework.score}%
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${
                  framework.score >= 90 ? 'bg-green-500' :
                  framework.score >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${framework.score}%` }}
              />
            </div>
          </motion.div>
        ))}
      </div>

      {/* Recent Assessments */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm mb-8"
      >
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
          Recent Compliance Assessments
        </h2>
        <div className="space-y-4">
          {[
            { resource: 'Storage Accounts', date: '2024-01-15', score: 98, trend: 'up' },
            { resource: 'Virtual Networks', date: '2024-01-14', score: 85, trend: 'stable' },
            { resource: 'Key Vaults', date: '2024-01-13', score: 92, trend: 'up' },
            { resource: 'SQL Databases', date: '2024-01-12', score: 76, trend: 'down' }
          ].map((assessment, idx) => (
            <div key={idx} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-4">
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                  assessment.score >= 90 ? 'bg-green-100 dark:bg-green-900' :
                  assessment.score >= 70 ? 'bg-yellow-100 dark:bg-yellow-900' : 'bg-red-100 dark:bg-red-900'
                }`}>
                  <span className="font-bold text-sm">{assessment.score}%</span>
                </div>
                <div>
                  <p className="font-semibold text-gray-900 dark:text-white">{assessment.resource}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Assessed on {assessment.date}</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {assessment.trend === 'up' && (
                  <TrendingUp className="w-4 h-4 text-green-500" />
                )}
                {assessment.trend === 'down' && (
                  <TrendingUp className="w-4 h-4 text-red-500 rotate-180" />
                )}
                {assessment.trend === 'stable' && (
                  <BarChart3 className="w-4 h-4 text-gray-500" />
                )}
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Compliance Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
      >
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
          Required Actions
        </h2>
        <div className="space-y-3">
          {[
            { action: 'Enable encryption for Storage Account "prod-data"', priority: 'high', deadline: '2024-01-20' },
            { action: 'Update network security rules for VM Scale Set', priority: 'medium', deadline: '2024-01-25' },
            { action: 'Review and update RBAC permissions', priority: 'low', deadline: '2024-02-01' }
          ].map((action, idx) => (
            <div key={idx} className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className={`w-2 h-8 rounded-full ${
                  action.priority === 'high' ? 'bg-red-500' :
                  action.priority === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
                }`} />
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">{action.action}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 flex items-center mt-1">
                    <Clock className="w-3 h-3 mr-1" />
                    Due: {action.deadline}
                  </p>
                </div>
              </div>
              <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
                Take Action
              </button>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
    </AppLayout>
  )
}