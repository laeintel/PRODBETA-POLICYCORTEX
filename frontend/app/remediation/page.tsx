'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Shield, 
  Play, 
  Pause, 
  CheckCircle, 
  AlertTriangle, 
  Clock, 
  Users, 
  Settings, 
  Eye,
  GitBranch,
  ArrowRight,
  RefreshCw,
  Zap,
  FileText,
  Bell
} from 'lucide-react'
import AppLayout from '../../components/AppLayout'

interface RemediationStatus {
  id: string
  name: string
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'awaiting_approval'
  progress: number
  startedAt: string
  estimatedCompletion?: string
  resourcesAffected: number
  riskLevel: 'low' | 'medium' | 'high'
  type: string
}

interface PendingApproval {
  id: string
  requestedBy: string
  remediationType: string
  resourcesAffected: number
  riskLevel: 'low' | 'medium' | 'high'
  expiresAt: string
  description: string
}

export default function RemediationPage() {
  const [remediations, setRemediations] = useState<RemediationStatus[]>([])
  const [pendingApprovals, setPendingApprovals] = useState<PendingApproval[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<'dashboard' | 'approvals' | 'history'>('dashboard')

  useEffect(() => {
    fetchRemediationData()
    fetchPendingApprovals()
  }, [])

  const fetchRemediationData = async () => {
    try {
      // Mock data - in production, fetch from API
      const mockRemediations: RemediationStatus[] = [
        {
          id: '1',
          name: 'Security Group Rule Remediation',
          status: 'in_progress',
          progress: 65,
          startedAt: '2025-08-16T10:30:00Z',
          estimatedCompletion: '2025-08-16T11:15:00Z',
          resourcesAffected: 8,
          riskLevel: 'medium',
          type: 'Security'
        },
        {
          id: '2',
          name: 'Storage Account Encryption',
          status: 'completed',
          progress: 100,
          startedAt: '2025-08-16T09:00:00Z',
          resourcesAffected: 3,
          riskLevel: 'high',
          type: 'Encryption'
        },
        {
          id: '3',
          name: 'Resource Tagging Compliance',
          status: 'awaiting_approval',
          progress: 0,
          startedAt: '2025-08-16T11:00:00Z',
          resourcesAffected: 15,
          riskLevel: 'low',
          type: 'Compliance'
        }
      ]
      setRemediations(mockRemediations)
    } catch (error) {
      console.error('Failed to fetch remediation data:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchPendingApprovals = async () => {
    try {
      // Mock data - in production, fetch from API
      const mockApprovals: PendingApproval[] = [
        {
          id: 'app-1',
          requestedBy: 'security.team@company.com',
          remediationType: 'Network Security Update',
          resourcesAffected: 12,
          riskLevel: 'high',
          expiresAt: '2025-08-16T18:00:00Z',
          description: 'Update NSG rules to block unauthorized traffic patterns'
        },
        {
          id: 'app-2',
          requestedBy: 'compliance.admin@company.com',
          remediationType: 'Policy Assignment',
          resourcesAffected: 25,
          riskLevel: 'medium',
          expiresAt: '2025-08-17T12:00:00Z',
          description: 'Apply mandatory resource tagging policy to production resources'
        }
      ]
      setPendingApprovals(mockApprovals)
    } catch (error) {
      console.error('Failed to fetch pending approvals:', error)
    }
  }

  const handleApproval = async (approvalId: string, decision: 'approve' | 'reject') => {
    try {
      const response = await fetch(`/api/v1/remediation/approvals/${approvalId}/approve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          decision,
          reason: decision === 'reject' ? 'Risk assessment required' : 'Approved for remediation'
        })
      })

      if (response.ok) {
        // Refresh pending approvals
        fetchPendingApprovals()
        fetchRemediationData()
      }
    } catch (error) {
      console.error('Failed to process approval:', error)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'in_progress':
        return <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />
      case 'failed':
        return <AlertTriangle className="w-5 h-5 text-red-500" />
      case 'awaiting_approval':
        return <Clock className="w-5 h-5 text-yellow-500" />
      default:
        return <Clock className="w-5 h-5 text-gray-500" />
    }
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'high':
        return 'text-red-600 bg-red-100'
      case 'medium':
        return 'text-yellow-600 bg-yellow-100'
      case 'low':
        return 'text-green-600 bg-green-100'
      default:
        return 'text-gray-600 bg-gray-100'
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
            Remediation Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Monitor and manage automated remediation workflows with approval oversight
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1 mb-6">
          {[
            { id: 'dashboard', label: 'Dashboard', icon: Shield },
            { id: 'approvals', label: 'Pending Approvals', icon: Users },
            { id: 'history', label: 'History', icon: FileText }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center px-4 py-2 rounded-md transition-colors ${
                activeTab === tab.id
                  ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
              }`}
            >
              <tab.icon className="w-4 h-4 mr-2" />
              {tab.label}
              {tab.id === 'approvals' && pendingApprovals.length > 0 && (
                <span className="ml-2 px-2 py-1 bg-red-500 text-white text-xs rounded-full">
                  {pendingApprovals.length}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <>
            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
              >
                <div className="flex items-center justify-between mb-4">
                  <Zap className="w-8 h-8 text-blue-500" />
                  <span className="text-3xl font-bold">{remediations.length}</span>
                </div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Active Remediations</p>
                <p className="text-xs text-gray-500 mt-1">Currently running</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 }}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
              >
                <div className="flex items-center justify-between mb-4">
                  <Clock className="w-8 h-8 text-yellow-500" />
                  <span className="text-3xl font-bold">{pendingApprovals.length}</span>
                </div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Pending Approvals</p>
                <p className="text-xs text-gray-500 mt-1">Awaiting decision</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
              >
                <div className="flex items-center justify-between mb-4">
                  <CheckCircle className="w-8 h-8 text-green-500" />
                  <span className="text-3xl font-bold">24</span>
                </div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Completed Today</p>
                <p className="text-xs text-gray-500 mt-1">+15% from yesterday</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 }}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
              >
                <div className="flex items-center justify-between mb-4">
                  <Shield className="w-8 h-8 text-purple-500" />
                  <span className="text-3xl font-bold">98%</span>
                </div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Success Rate</p>
                <p className="text-xs text-gray-500 mt-1">Last 30 days</p>
              </motion.div>
            </div>

            {/* Active Remediations */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm mb-8"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">Active Remediations</h2>
                <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center">
                  <Play className="w-4 h-4 mr-2" />
                  New Remediation
                </button>
              </div>

              <div className="space-y-4">
                {remediations.map((remediation) => (
                  <div
                    key={remediation.id}
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(remediation.status)}
                        <div>
                          <h3 className="font-semibold text-gray-900 dark:text-white">
                            {remediation.name}
                          </h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            {remediation.type} • {remediation.resourcesAffected} resources
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <span className={`px-2 py-1 text-xs rounded-full ${getRiskColor(remediation.riskLevel)}`}>
                          {remediation.riskLevel.toUpperCase()}
                        </span>
                        <button className="p-2 text-gray-500 hover:text-gray-700 rounded-lg hover:bg-gray-100">
                          <Eye className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    {remediation.status === 'in_progress' && (
                      <div className="mb-3">
                        <div className="flex items-center justify-between text-sm mb-1">
                          <span className="text-gray-600 dark:text-gray-400">Progress</span>
                          <span className="text-gray-900 dark:text-white font-medium">
                            {remediation.progress}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${remediation.progress}%` }}
                          />
                        </div>
                      </div>
                    )}

                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">
                        Started: {new Date(remediation.startedAt).toLocaleString()}
                      </span>
                      {remediation.estimatedCompletion && (
                        <span className="text-gray-600 dark:text-gray-400">
                          ETA: {new Date(remediation.estimatedCompletion).toLocaleTimeString()}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </>
        )}

        {/* Approvals Tab */}
        {activeTab === 'approvals' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center">
                <Bell className="w-5 h-5 mr-2 text-yellow-500" />
                Pending Approvals ({pendingApprovals.length})
              </h2>
            </div>

            {pendingApprovals.length === 0 ? (
              <div className="text-center py-12">
                <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  All caught up!
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  No pending approvals requiring your attention.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {pendingApprovals.map((approval) => (
                  <div
                    key={approval.id}
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-6"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                          {approval.remediationType}
                        </h3>
                        <p className="text-gray-600 dark:text-gray-400 mb-3">
                          {approval.description}
                        </p>
                        <div className="flex items-center space-x-4 text-sm">
                          <span className="text-gray-600 dark:text-gray-400">
                            Requested by: <span className="font-medium">{approval.requestedBy}</span>
                          </span>
                          <span className="text-gray-600 dark:text-gray-400">
                            Resources: <span className="font-medium">{approval.resourcesAffected}</span>
                          </span>
                          <span className={`px-2 py-1 text-xs rounded-full ${getRiskColor(approval.riskLevel)}`}>
                            {approval.riskLevel.toUpperCase()} RISK
                          </span>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-red-600 dark:text-red-400">
                          Expires: {new Date(approval.expiresAt).toLocaleString()}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
                      <button className="px-4 py-2 text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200">
                        View Details
                      </button>
                      <div className="flex space-x-3">
                        <button
                          onClick={() => handleApproval(approval.id, 'reject')}
                          className="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
                        >
                          Reject
                        </button>
                        <button
                          onClick={() => handleApproval(approval.id, 'approve')}
                          className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center"
                        >
                          <CheckCircle className="w-4 h-4 mr-2" />
                          Approve & Execute
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
          >
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Remediation History</h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left border-b border-gray-200 dark:border-gray-700">
                    <th className="pb-3 text-sm font-medium text-gray-600 dark:text-gray-400">Remediation</th>
                    <th className="pb-3 text-sm font-medium text-gray-600 dark:text-gray-400">Status</th>
                    <th className="pb-3 text-sm font-medium text-gray-600 dark:text-gray-400">Resources</th>
                    <th className="pb-3 text-sm font-medium text-gray-600 dark:text-gray-400">Duration</th>
                    <th className="pb-3 text-sm font-medium text-gray-600 dark:text-gray-400">Completed</th>
                    <th className="pb-3 text-sm font-medium text-gray-600 dark:text-gray-400">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { name: 'Backup Policy Enforcement', status: 'success', resources: 12, duration: '8m 23s', completed: '2 hours ago' },
                    { name: 'Access Key Rotation', status: 'success', resources: 5, duration: '3m 45s', completed: '4 hours ago' },
                    { name: 'Network Security Update', status: 'failed', resources: 8, duration: '12m 10s', completed: '6 hours ago' },
                    { name: 'Cost Tag Application', status: 'success', resources: 25, duration: '15m 32s', completed: '1 day ago' },
                    { name: 'SSL Certificate Update', status: 'success', resources: 3, duration: '5m 18s', completed: '2 days ago' }
                  ].map((item, idx) => (
                    <tr key={idx} className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-4 text-sm font-medium">{item.name}</td>
                      <td className="py-4">
                        <span className={`text-xs px-3 py-1 rounded-full ${
                          item.status === 'success' ? 'bg-green-100 text-green-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {item.status}
                        </span>
                      </td>
                      <td className="py-4 text-sm text-gray-600 dark:text-gray-400">{item.resources}</td>
                      <td className="py-4 text-sm text-gray-600 dark:text-gray-400">{item.duration}</td>
                      <td className="py-4 text-sm text-gray-600 dark:text-gray-400">{item.completed}</td>
                      <td className="py-4">
                        <button className="text-blue-500 hover:text-blue-600 text-sm mr-3">View</button>
                        <button className="text-gray-500 hover:text-gray-600 text-sm">Logs</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}
      </div>
    </AppLayout>
  )
}