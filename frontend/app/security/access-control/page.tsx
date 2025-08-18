/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform  
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2026 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Lock,
  Users,
  Shield,
  Key,
  UserCheck,
  UserX,
  Settings,
  AlertTriangle,
  CheckCircle,
  Eye,
  EyeOff,
  Plus,
  Edit,
  Trash2,
  Download,
  RefreshCw
} from 'lucide-react'

interface Permission {
  id: string
  name: string
  resource: string
  actions: string[]
  effect: 'Allow' | 'Deny'
}

interface Role {
  id: string
  name: string
  description: string
  permissions: Permission[]
  users: number
  createdAt: string
  lastModified: string
  isBuiltIn: boolean
}

interface AccessRequest {
  id: string
  user: string
  role: string
  resource: string
  reason: string
  requestedAt: string
  status: 'pending' | 'approved' | 'denied'
  approver?: string
}

export default function AccessControlPage() {
  const [roles, setRoles] = useState<Role[]>([])
  const [accessRequests, setAccessRequests] = useState<AccessRequest[]>([])
  const [selectedTab, setSelectedTab] = useState<'roles' | 'permissions' | 'requests'>('roles')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => {
      setRoles([
        {
          id: 'role-001',
          name: 'Administrator',
          description: 'Full system access with all permissions',
          permissions: [
            { id: 'p1', name: 'All Access', resource: '*', actions: ['*'], effect: 'Allow' }
          ],
          users: 3,
          createdAt: '6 months ago',
          lastModified: '1 week ago',
          isBuiltIn: true
        },
        {
          id: 'role-002',
          name: 'Developer',
          description: 'Development environment access',
          permissions: [
            { id: 'p2', name: 'Read Resources', resource: 'dev/*', actions: ['read', 'list'], effect: 'Allow' },
            { id: 'p3', name: 'Deploy Code', resource: 'dev/deployments', actions: ['create', 'update'], effect: 'Allow' }
          ],
          users: 24,
          createdAt: '5 months ago',
          lastModified: '2 days ago',
          isBuiltIn: false
        },
        {
          id: 'role-003',
          name: 'Security Analyst',
          description: 'Security monitoring and audit access',
          permissions: [
            { id: 'p4', name: 'View Logs', resource: 'logs/*', actions: ['read'], effect: 'Allow' },
            { id: 'p5', name: 'Security Scan', resource: 'security/*', actions: ['read', 'execute'], effect: 'Allow' }
          ],
          users: 8,
          createdAt: '4 months ago',
          lastModified: '1 week ago',
          isBuiltIn: false
        },
        {
          id: 'role-004',
          name: 'Read Only',
          description: 'View-only access to all resources',
          permissions: [
            { id: 'p6', name: 'Read All', resource: '*', actions: ['read', 'list'], effect: 'Allow' },
            { id: 'p7', name: 'Deny Modifications', resource: '*', actions: ['create', 'update', 'delete'], effect: 'Deny' }
          ],
          users: 45,
          createdAt: '6 months ago',
          lastModified: '1 month ago',
          isBuiltIn: true
        }
      ])

      setAccessRequests([
        {
          id: 'req-001',
          user: 'john.doe@company.com',
          role: 'Developer',
          resource: 'production/database',
          reason: 'Need to debug production issue #1234',
          requestedAt: '2 hours ago',
          status: 'pending'
        },
        {
          id: 'req-002',
          user: 'jane.smith@company.com',
          role: 'Administrator',
          resource: 'billing/accounts',
          reason: 'Quarterly audit requirement',
          requestedAt: '1 day ago',
          status: 'approved',
          approver: 'admin@company.com'
        },
        {
          id: 'req-003',
          user: 'bob.wilson@company.com',
          role: 'Security Analyst',
          resource: 'logs/sensitive',
          reason: 'Security incident investigation',
          requestedAt: '3 days ago',
          status: 'denied',
          approver: 'security@company.com'
        }
      ])

      setLoading(false)
    }, 1000)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved': return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'denied': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'pending': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-4 mb-2">
          <div className="p-3 bg-gradient-to-br from-blue-500 to-purple-500 rounded-xl">
            <Lock className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Access Control</h1>
            <p className="text-gray-400 mt-1">Role-based access control and permissions management</p>
          </div>
        </div>
      </motion.div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <Shield className="w-6 h-6 text-purple-400" />
            <span className="text-2xl font-bold text-white">{roles.length}</span>
          </div>
          <p className="text-gray-400 text-sm">Active Roles</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <Users className="w-6 h-6 text-blue-400" />
            <span className="text-2xl font-bold text-white">80</span>
          </div>
          <p className="text-gray-400 text-sm">Total Users</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <Key className="w-6 h-6 text-green-400" />
            <span className="text-2xl font-bold text-white">156</span>
          </div>
          <p className="text-gray-400 text-sm">Permissions</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <AlertTriangle className="w-6 h-6 text-yellow-400" />
            <span className="text-2xl font-bold text-white">
              {accessRequests.filter(r => r.status === 'pending').length}
            </span>
          </div>
          <p className="text-gray-400 text-sm">Pending Requests</p>
        </motion.div>
      </div>

      {/* Tabs */}
      <div className="flex gap-4 mb-6">
        {(['roles', 'permissions', 'requests'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setSelectedTab(tab)}
            className={`px-4 py-2 rounded-lg transition-colors ${
              selectedTab === tab
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Content */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <>
          {selectedTab === 'roles' && (
            <div className="space-y-4">
              {roles.map((role, index) => (
                <motion.div
                  key={role.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-6"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                        {role.name}
                        {role.isBuiltIn && (
                          <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 text-xs rounded">
                            Built-in
                          </span>
                        )}
                      </h3>
                      <p className="text-sm text-gray-400 mt-1">{role.description}</p>
                    </div>
                    <div className="flex gap-2">
                      <button className="p-2 hover:bg-white/10 rounded-lg transition-colors">
                        <Edit className="w-4 h-4 text-gray-400" />
                      </button>
                      {!role.isBuiltIn && (
                        <button className="p-2 hover:bg-white/10 rounded-lg transition-colors">
                          <Trash2 className="w-4 h-4 text-red-400" />
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="bg-black/20 rounded-lg p-3">
                      <p className="text-xs text-gray-400 mb-1">Users</p>
                      <p className="text-lg font-semibold text-white">{role.users}</p>
                    </div>
                    <div className="bg-black/20 rounded-lg p-3">
                      <p className="text-xs text-gray-400 mb-1">Permissions</p>
                      <p className="text-lg font-semibold text-white">{role.permissions.length}</p>
                    </div>
                    <div className="bg-black/20 rounded-lg p-3">
                      <p className="text-xs text-gray-400 mb-1">Last Modified</p>
                      <p className="text-sm text-white">{role.lastModified}</p>
                    </div>
                  </div>

                  <div className="border-t border-white/10 pt-4">
                    <p className="text-xs text-gray-400 mb-2">Permissions</p>
                    <div className="flex flex-wrap gap-2">
                      {role.permissions.map((perm) => (
                        <span
                          key={perm.id}
                          className={`px-2 py-1 rounded text-xs ${
                            perm.effect === 'Allow'
                              ? 'bg-green-500/20 text-green-400'
                              : 'bg-red-500/20 text-red-400'
                          }`}
                        >
                          {perm.name}
                        </span>
                      ))}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}

          {selectedTab === 'requests' && (
            <div className="space-y-4">
              {accessRequests.map((request, index) => (
                <motion.div
                  key={request.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-6"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="font-medium text-white">{request.user}</p>
                      <p className="text-sm text-gray-400 mt-1">
                        Requesting <span className="text-purple-400">{request.role}</span> access to{' '}
                        <span className="text-blue-400">{request.resource}</span>
                      </p>
                      <p className="text-sm text-gray-300 mt-2">Reason: {request.reason}</p>
                      <p className="text-xs text-gray-500 mt-2">Requested {request.requestedAt}</p>
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(request.status)}`}>
                        {request.status.toUpperCase()}
                      </span>
                      {request.approver && (
                        <p className="text-xs text-gray-400">by {request.approver}</p>
                      )}
                      {request.status === 'pending' && (
                        <div className="flex gap-2 mt-2">
                          <button className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-white text-sm">
                            Approve
                          </button>
                          <button className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-white text-sm">
                            Deny
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}