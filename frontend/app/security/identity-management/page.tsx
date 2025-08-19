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
  Users,
  UserPlus,
  UserCheck,
  UserX,
  Shield,
  Key,
  Lock,
  Smartphone,
  Mail,
  Globe,
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Settings,
  RefreshCw,
  Download,
  Search,
  Filter
} from 'lucide-react'

interface User {
  id: string
  email: string
  name: string
  department: string
  role: string
  status: 'active' | 'inactive' | 'suspended' | 'pending'
  mfaEnabled: boolean
  lastLogin: string
  createdAt: string
  riskScore: number
  groups: string[]
  permissions: number
}

interface IdentityProvider {
  id: string
  name: string
  type: string
  status: 'connected' | 'disconnected' | 'error'
  users: number
  lastSync: string
}

export default function IdentityManagementPage() {
  const [users, setUsers] = useState<User[]>([])
  const [providers, setProviders] = useState<IdentityProvider[]>([])
  const [selectedTab, setSelectedTab] = useState<'users' | 'providers' | 'settings'>('users')
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => {
      setUsers([
        {
          id: 'usr-001',
          email: 'admin@company.com',
          name: 'System Administrator',
          department: 'IT',
          role: 'Administrator',
          status: 'active',
          mfaEnabled: true,
          lastLogin: '2 hours ago',
          createdAt: '1 year ago',
          riskScore: 5,
          groups: ['Admins', 'Security', 'DevOps'],
          permissions: 156
        },
        {
          id: 'usr-002',
          email: 'john.doe@company.com',
          name: 'John Doe',
          department: 'Engineering',
          role: 'Developer',
          status: 'active',
          mfaEnabled: true,
          lastLogin: '1 day ago',
          createdAt: '6 months ago',
          riskScore: 12,
          groups: ['Developers', 'Project-Alpha'],
          permissions: 48
        },
        {
          id: 'usr-003',
          email: 'jane.smith@company.com',
          name: 'Jane Smith',
          department: 'Security',
          role: 'Security Analyst',
          status: 'active',
          mfaEnabled: true,
          lastLogin: '5 minutes ago',
          createdAt: '8 months ago',
          riskScore: 8,
          groups: ['Security', 'Compliance', 'Audit'],
          permissions: 89
        },
        {
          id: 'usr-004',
          email: 'bob.wilson@company.com',
          name: 'Bob Wilson',
          department: 'Finance',
          role: 'Viewer',
          status: 'suspended',
          mfaEnabled: false,
          lastLogin: '1 week ago',
          createdAt: '3 months ago',
          riskScore: 75,
          groups: ['Finance'],
          permissions: 12
        },
        {
          id: 'usr-005',
          email: 'alice.johnson@company.com',
          name: 'Alice Johnson',
          department: 'HR',
          role: 'HR Manager',
          status: 'active',
          mfaEnabled: false,
          lastLogin: '3 days ago',
          createdAt: '1 year ago',
          riskScore: 35,
          groups: ['HR', 'Management'],
          permissions: 34
        }
      ])

      setProviders([
        {
          id: 'idp-001',
          name: 'Azure Active Directory',
          type: 'SAML',
          status: 'connected',
          users: 1250,
          lastSync: '10 minutes ago'
        },
        {
          id: 'idp-002',
          name: 'Okta',
          type: 'OAuth2',
          status: 'connected',
          users: 450,
          lastSync: '1 hour ago'
        },
        {
          id: 'idp-003',
          name: 'Google Workspace',
          type: 'OAuth2',
          status: 'connected',
          users: 320,
          lastSync: '30 minutes ago'
        },
        {
          id: 'idp-004',
          name: 'On-Premise AD',
          type: 'LDAP',
          status: 'error',
          users: 0,
          lastSync: '3 days ago'
        }
      ])

      setLoading(false)
    }, 1000)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
      case 'connected':
        return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'inactive':
      case 'disconnected':
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
      case 'suspended':
      case 'error':
        return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'pending':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  const getRiskColor = (score: number) => {
    if (score < 20) return 'text-green-400'
    if (score < 50) return 'text-yellow-400'
    return 'text-red-400'
  }

  const filteredUsers = users.filter(user =>
    user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.department.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const activeUsers = users.filter(u => u.status === 'active').length
  const mfaEnabled = users.filter(u => u.mfaEnabled).length
  const totalProviders = providers.filter(p => p.status === 'connected').length

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-4 mb-2">
          <div className="p-3 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-xl">
            <Users className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Identity Management</h1>
            <p className="text-gray-400 mt-1">User identities and authentication management</p>
          </div>
        </div>
      </motion.div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <Users className="w-6 h-6 text-purple-400" />
            <span className="text-2xl font-bold text-white">{users.length}</span>
          </div>
          <p className="text-gray-400 text-sm">Total Users</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <UserCheck className="w-6 h-6 text-green-400" />
            <span className="text-2xl font-bold text-white">{activeUsers}</span>
          </div>
          <p className="text-gray-400 text-sm">Active</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <Smartphone className="w-6 h-6 text-blue-400" />
            <span className="text-2xl font-bold text-white">{mfaEnabled}</span>
          </div>
          <p className="text-gray-400 text-sm">MFA Enabled</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <Globe className="w-6 h-6 text-indigo-400" />
            <span className="text-2xl font-bold text-white">{totalProviders}</span>
          </div>
          <p className="text-gray-400 text-sm">ID Providers</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-4 border border-white/20"
        >
          <div className="flex items-center justify-between mb-2">
            <AlertTriangle className="w-6 h-6 text-yellow-400" />
            <span className="text-2xl font-bold text-white">
              {users.filter(u => u.riskScore > 50).length}
            </span>
          </div>
          <p className="text-gray-400 text-sm">High Risk</p>
        </motion.div>
      </div>

      {/* Tabs */}
      <div className="flex gap-4 mb-6">
        {(['users', 'providers', 'settings'] as const).map((tab) => (
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

      {/* Search Bar */}
      {selectedTab === 'users' && (
        <div className="mb-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search users..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
            />
          </div>
        </div>
      )}

      {/* Content */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <>
          {selectedTab === 'users' && (
            <div className="space-y-4">
              {filteredUsers.map((user, index) => (
                <motion.div
                  key={user.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-6"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-4">
                      <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                        <span className="text-white font-semibold">
                          {user.name.split(' ').map(n => n[0]).join('')}
                        </span>
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-white">{user.name}</h3>
                        <p className="text-sm text-gray-400">{user.email}</p>
                        <p className="text-sm text-gray-500 mt-1">
                          {user.department} • {user.role}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(user.status)}`}>
                        {user.status.toUpperCase()}
                      </span>
                      {user.mfaEnabled && (
                        <span className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs rounded">
                          MFA
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-4 gap-4 mt-4">
                    <div className="bg-black/20 rounded-lg p-2">
                      <p className="text-xs text-gray-400">Last Login</p>
                      <p className="text-sm text-white">{user.lastLogin}</p>
                    </div>
                    <div className="bg-black/20 rounded-lg p-2">
                      <p className="text-xs text-gray-400">Risk Score</p>
                      <p className={`text-sm font-semibold ${getRiskColor(user.riskScore)}`}>
                        {user.riskScore}%
                      </p>
                    </div>
                    <div className="bg-black/20 rounded-lg p-2">
                      <p className="text-xs text-gray-400">Groups</p>
                      <p className="text-sm text-white">{user.groups.length}</p>
                    </div>
                    <div className="bg-black/20 rounded-lg p-2">
                      <p className="text-xs text-gray-400">Permissions</p>
                      <p className="text-sm text-white">{user.permissions}</p>
                    </div>
                  </div>

                  <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/10">
                    <div className="flex flex-wrap gap-2">
                      {user.groups.slice(0, 3).map((group) => (
                        <span key={group} className="px-2 py-1 bg-purple-500/20 text-purple-400 text-xs rounded">
                          {group}
                        </span>
                      ))}
                      {user.groups.length > 3 && (
                        <span className="px-2 py-1 bg-gray-500/20 text-gray-400 text-xs rounded">
                          +{user.groups.length - 3} more
                        </span>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <button className="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-white text-sm">
                        Edit
                      </button>
                      {user.status === 'suspended' ? (
                        <button className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-white text-sm">
                          Activate
                        </button>
                      ) : (
                        <button className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-white text-sm">
                          Suspend
                        </button>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}

          {selectedTab === 'providers' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {providers.map((provider, index) => (
                <motion.div
                  key={provider.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-6"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white">{provider.name}</h3>
                      <p className="text-sm text-gray-400 mt-1">Type: {provider.type}</p>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(provider.status)}`}>
                      {provider.status.toUpperCase()}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="bg-black/20 rounded-lg p-3">
                      <p className="text-xs text-gray-400">Users</p>
                      <p className="text-xl font-semibold text-white">{provider.users}</p>
                    </div>
                    <div className="bg-black/20 rounded-lg p-3">
                      <p className="text-xs text-gray-400">Last Sync</p>
                      <p className="text-sm text-white">{provider.lastSync}</p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <button className="flex-1 px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded text-white text-sm">
                      Sync Now
                    </button>
                    <button className="px-3 py-2 bg-white/10 hover:bg-white/20 border border-white/20 rounded text-white text-sm">
                      Configure
                    </button>
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