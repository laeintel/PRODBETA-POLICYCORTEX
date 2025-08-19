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
  Database,
  Activity,
  HardDrive,
  Cpu,
  MemoryStick,
  Shield,
  DollarSign,
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap,
  TrendingUp,
  Users,
  Lock,
  Globe,
  BarChart3,
  Settings,
  RefreshCw
} from 'lucide-react'

interface DatabaseInstance {
  id: string
  name: string
  type: string
  engine: string
  version: string
  resourceGroup: string
  location: string
  tier: string
  status: 'Online' | 'Offline' | 'Updating' | 'Error'
  size: { current: number; max: number; unit: string }
  connections: { current: number; max: number }
  cpu: number
  memory: number
  iops: number
  storage: { used: number; total: number }
  backup: { enabled: boolean; lastBackup: string; retention: number }
  replication: { enabled: boolean; replicas: number }
  cost: { hourly: number; monthly: number }
  compliance: { score: number; issues: string[] }
}

export default function DatabasesPage() {
  const [databases, setDatabases] = useState<DatabaseInstance[]>([])
  const [selectedDb, setSelectedDb] = useState<DatabaseInstance | null>(null)
  const [filter, setFilter] = useState('all')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => {
      setDatabases([
        {
          id: 'db-001',
          name: 'prod-sql-primary',
          type: 'Azure SQL Database',
          engine: 'SQL Server',
          version: '15.0.4198.2',
          resourceGroup: 'production-rg',
          location: 'East US',
          tier: 'Premium P4',
          status: 'Online',
          size: { current: 450, max: 1024, unit: 'GB' },
          connections: { current: 125, max: 300 },
          cpu: 78,
          memory: 82,
          iops: 5000,
          storage: { used: 450, total: 1024 },
          backup: { enabled: true, lastBackup: '2 hours ago', retention: 35 },
          replication: { enabled: true, replicas: 2 },
          cost: { hourly: 2.4, monthly: 1728 },
          compliance: { score: 96, issues: ['TDE not enabled', 'Audit logs retention < 90 days'] }
        },
        {
          id: 'db-002',
          name: 'analytics-postgres',
          type: 'Azure Database for PostgreSQL',
          engine: 'PostgreSQL',
          version: '14.7',
          resourceGroup: 'analytics-rg',
          location: 'West US 2',
          tier: 'General Purpose',
          status: 'Online',
          size: { current: 256, max: 512, unit: 'GB' },
          connections: { current: 45, max: 100 },
          cpu: 45,
          memory: 58,
          iops: 3000,
          storage: { used: 256, total: 512 },
          backup: { enabled: true, lastBackup: '4 hours ago', retention: 30 },
          replication: { enabled: false, replicas: 0 },
          cost: { hourly: 0.8, monthly: 576 },
          compliance: { score: 92, issues: ['SSL enforcement disabled'] }
        },
        {
          id: 'db-003',
          name: 'cache-redis-cluster',
          type: 'Azure Cache for Redis',
          engine: 'Redis',
          version: '6.2.5',
          resourceGroup: 'cache-rg',
          location: 'East US 2',
          tier: 'Premium P1',
          status: 'Online',
          size: { current: 6, max: 6, unit: 'GB' },
          connections: { current: 850, max: 10000 },
          cpu: 35,
          memory: 72,
          iops: 10000,
          storage: { used: 4.5, total: 6 },
          backup: { enabled: true, lastBackup: '1 hour ago', retention: 7 },
          replication: { enabled: true, replicas: 1 },
          cost: { hourly: 0.45, monthly: 324 },
          compliance: { score: 98, issues: [] }
        },
        {
          id: 'db-004',
          name: 'cosmos-global',
          type: 'Azure Cosmos DB',
          engine: 'Cosmos DB',
          version: 'Latest',
          resourceGroup: 'global-rg',
          location: 'Global',
          tier: 'Autoscale',
          status: 'Online',
          size: { current: 120, max: 1000, unit: 'GB' },
          connections: { current: 2500, max: 10000 },
          cpu: 62,
          memory: 68,
          iops: 25000,
          storage: { used: 120, total: 1000 },
          backup: { enabled: true, lastBackup: 'Continuous', retention: 30 },
          replication: { enabled: true, replicas: 4 },
          cost: { hourly: 3.2, monthly: 2304 },
          compliance: { score: 100, issues: [] }
        }
      ])
      setLoading(false)
    }, 1000)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Online': return 'text-green-400 bg-green-500/20'
      case 'Offline': return 'text-red-400 bg-red-500/20'
      case 'Updating': return 'text-yellow-400 bg-yellow-500/20'
      case 'Error': return 'text-red-400 bg-red-500/20'
      default: return 'text-gray-400 bg-gray-500/20'
    }
  }

  const totalCost = databases.reduce((sum, db) => sum + db.cost.monthly, 0)
  const totalStorage = databases.reduce((sum, db) => sum + db.storage.used, 0)

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-4 mb-2">
          <div className="p-3 bg-gray-800 rounded-xl">
            <Database className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Databases</h1>
            <p className="text-gray-400 mt-1">Manage your database instances and clusters</p>
          </div>
        </div>
      </motion.div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <Database className="w-8 h-8 text-purple-400" />
            <span className="text-2xl font-bold text-white">{databases.length}</span>
          </div>
          <p className="text-gray-400 text-sm">Total Databases</p>
          <p className="text-xs text-green-400 mt-1">All online</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <HardDrive className="w-8 h-8 text-blue-400" />
            <span className="text-2xl font-bold text-white">{(totalStorage / 1024).toFixed(1)} TB</span>
          </div>
          <p className="text-gray-400 text-sm">Total Storage</p>
          <p className="text-xs text-blue-400 mt-1">Across all databases</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <DollarSign className="w-8 h-8 text-green-400" />
            <span className="text-2xl font-bold text-white">${totalCost}</span>
          </div>
          <p className="text-gray-400 text-sm">Monthly Cost</p>
          <p className="text-xs text-yellow-400 mt-1">↑ 8% from last month</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <Shield className="w-8 h-8 text-green-400" />
            <span className="text-2xl font-bold text-white">96%</span>
          </div>
          <p className="text-gray-400 text-sm">Avg Compliance</p>
          <p className="text-xs text-red-400 mt-1">3 issues found</p>
        </motion.div>
      </div>

      {/* Database Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {loading ? (
          <div className="col-span-2 flex items-center justify-center py-12">
            <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          databases.map((db, index) => (
            <motion.div
              key={db.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 overflow-hidden hover:bg-white/15 transition-colors cursor-pointer"
              onClick={() => setSelectedDb(db)}
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-purple-500/20 rounded-lg">
                      <Database className="w-6 h-6 text-purple-400" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">{db.name}</h3>
                      <p className="text-sm text-gray-400">{db.type}</p>
                      <div className="flex items-center gap-2 mt-2">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(db.status)}`}>
                          {db.status}
                        </span>
                        <span className="text-xs text-gray-400">{db.tier}</span>
                        <span className="text-xs text-gray-400">{db.engine} {db.version}</span>
                      </div>
                    </div>
                  </div>
                  {db.location === 'Global' && (
                    <Globe className="w-5 h-5 text-purple-400" />
                  )}
                </div>

                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="bg-black/20 rounded-lg p-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-400">CPU</span>
                      <span className="text-xs font-medium text-white">{db.cpu}%</span>
                    </div>
                    <div className="bg-black/30 rounded-full h-1">
                      <div
                        className="bg-blue-400 h-1 rounded-full"
                        style={{ width: `${db.cpu}%` }}
                      />
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-400">Memory</span>
                      <span className="text-xs font-medium text-white">{db.memory}%</span>
                    </div>
                    <div className="bg-black/30 rounded-full h-1">
                      <div
                        className="bg-green-400 h-1 rounded-full"
                        style={{ width: `${db.memory}%` }}
                      />
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-400">Storage</span>
                      <span className="text-xs font-medium text-white">
                        {db.storage.used}/{db.storage.total} GB
                      </span>
                    </div>
                    <div className="bg-black/30 rounded-full h-1">
                      <div
                        className="bg-purple-400 h-1 rounded-full"
                        style={{ width: `${(db.storage.used / db.storage.total) * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-400">Connections</span>
                      <span className="text-xs font-medium text-white">
                        {db.connections.current}/{db.connections.max}
                      </span>
                    </div>
                    <div className="bg-black/30 rounded-full h-1">
                      <div
                        className="bg-yellow-400 h-1 rounded-full"
                        style={{ width: `${(db.connections.current / db.connections.max) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-white/10">
                  <div className="flex items-center gap-3 text-xs">
                    <div className="flex items-center gap-1">
                      <Shield className="w-3 h-3 text-green-400" />
                      <span className="text-white">{db.compliance.score}%</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3 text-gray-400" />
                      <span className="text-gray-400">Backup: {db.backup.lastBackup}</span>
                    </div>
                    {db.replication.enabled && (
                      <div className="flex items-center gap-1">
                        <RefreshCw className="w-3 h-3 text-blue-400" />
                        <span className="text-gray-400">{db.replication.replicas} replicas</span>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    <DollarSign className="w-4 h-4 text-green-400" />
                    <span className="text-sm font-medium text-white">${db.cost.monthly}/mo</span>
                  </div>
                </div>

                {db.compliance.issues.length > 0 && (
                  <div className="mt-3 p-2 bg-red-500/10 border border-red-500/30 rounded-lg">
                    <div className="flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5" />
                      <div>
                        <p className="text-xs text-red-400 font-medium">Compliance Issues:</p>
                        {db.compliance.issues.map((issue, idx) => (
                          <p key={idx} className="text-xs text-gray-400">• {issue}</p>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          ))
        )}
      </div>
    </div>
  )
}