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
  HardDrive,
  Cloud,
  Archive,
  Shield,
  DollarSign,
  Activity,
  Lock,
  Globe,
  Download,
  Upload,
  Folder,
  File,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  BarChart3,
  Settings,
  RefreshCw
} from 'lucide-react'

interface StorageAccount {
  id: string
  name: string
  resourceGroup: string
  location: string
  kind: string
  tier: string
  replication: string
  status: 'Available' | 'Degraded' | 'Unavailable'
  capacity: { used: number; total: number; unit: string }
  containers: number
  blobs: number
  files: number
  queues: number
  tables: number
  bandwidth: { ingress: number; egress: number }
  transactions: number
  encryption: { enabled: boolean; type: string }
  publicAccess: boolean
  cost: { monthly: number; trend: number }
  compliance: { score: number; issues: string[] }
  lastAccessed: string
}

export default function StoragePage() {
  const [storageAccounts, setStorageAccounts] = useState<StorageAccount[]>([])
  const [selectedTier, setSelectedTier] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => {
      setStorageAccounts([
        {
          id: 'storage-001',
          name: 'prodstorageeast',
          resourceGroup: 'production-rg',
          location: 'East US',
          kind: 'StorageV2',
          tier: 'Hot',
          replication: 'GRS',
          status: 'Available',
          capacity: { used: 2.4, total: 5, unit: 'TB' },
          containers: 45,
          blobs: 125000,
          files: 8900,
          queues: 12,
          tables: 8,
          bandwidth: { ingress: 450, egress: 890 },
          transactions: 2450000,
          encryption: { enabled: true, type: 'AES-256' },
          publicAccess: false,
          cost: { monthly: 145, trend: 12 },
          compliance: { score: 98, issues: [] },
          lastAccessed: '2 minutes ago'
        },
        {
          id: 'storage-002',
          name: 'backupstoragewest',
          resourceGroup: 'backup-rg',
          location: 'West US',
          kind: 'BlobStorage',
          tier: 'Cool',
          replication: 'LRS',
          status: 'Available',
          capacity: { used: 8.7, total: 10, unit: 'TB' },
          containers: 12,
          blobs: 450000,
          files: 0,
          queues: 0,
          tables: 0,
          bandwidth: { ingress: 120, egress: 45 },
          transactions: 125000,
          encryption: { enabled: true, type: 'AES-256' },
          publicAccess: false,
          cost: { monthly: 89, trend: -5 },
          compliance: { score: 95, issues: ['Lifecycle policy not configured'] },
          lastAccessed: '1 hour ago'
        },
        {
          id: 'storage-003',
          name: 'archivecold',
          resourceGroup: 'archive-rg',
          location: 'Central US',
          kind: 'StorageV2',
          tier: 'Archive',
          replication: 'LRS',
          status: 'Available',
          capacity: { used: 45, total: 100, unit: 'TB' },
          containers: 8,
          blobs: 1250000,
          files: 0,
          queues: 0,
          tables: 0,
          bandwidth: { ingress: 10, egress: 5 },
          transactions: 5000,
          encryption: { enabled: true, type: 'AES-256' },
          publicAccess: false,
          cost: { monthly: 45, trend: 2 },
          compliance: { score: 100, issues: [] },
          lastAccessed: '3 days ago'
        },
        {
          id: 'storage-004',
          name: 'cdnstaticassets',
          resourceGroup: 'cdn-rg',
          location: 'Global',
          kind: 'StorageV2',
          tier: 'Hot',
          replication: 'RA-GRS',
          status: 'Available',
          capacity: { used: 0.8, total: 2, unit: 'TB' },
          containers: 25,
          blobs: 45000,
          files: 12000,
          queues: 4,
          tables: 2,
          bandwidth: { ingress: 890, egress: 2450 },
          transactions: 8900000,
          encryption: { enabled: true, type: 'AES-256' },
          publicAccess: true,
          cost: { monthly: 234, trend: 18 },
          compliance: { score: 92, issues: ['Public access enabled', 'CORS not configured'] },
          lastAccessed: 'Just now'
        }
      ])
      setLoading(false)
    }, 1000)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Available': return 'text-green-400 bg-green-500/20'
      case 'Degraded': return 'text-yellow-400 bg-yellow-500/20'
      case 'Unavailable': return 'text-red-400 bg-red-500/20'
      default: return 'text-gray-400 bg-gray-500/20'
    }
  }

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'Hot': return 'text-red-400 bg-red-500/20'
      case 'Cool': return 'text-blue-400 bg-blue-500/20'
      case 'Archive': return 'text-gray-400 bg-gray-500/20'
      default: return 'text-purple-400 bg-purple-500/20'
    }
  }

  const filteredAccounts = storageAccounts.filter(account => {
    const matchesSearch = account.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          account.resourceGroup.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesTier = selectedTier === 'all' || account.tier === selectedTier
    return matchesSearch && matchesTier
  })

  const totalStorage = storageAccounts.reduce((sum, acc) => sum + acc.capacity.used, 0)
  const totalCost = storageAccounts.reduce((sum, acc) => sum + acc.cost.monthly, 0)
  const totalBlobs = storageAccounts.reduce((sum, acc) => sum + acc.blobs, 0)

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
            <HardDrive className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Storage Accounts</h1>
            <p className="text-gray-400 mt-1">Manage your cloud storage resources</p>
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
            <Cloud className="w-8 h-8 text-purple-400" />
            <span className="text-2xl font-bold text-white">{storageAccounts.length}</span>
          </div>
          <p className="text-gray-400 text-sm">Storage Accounts</p>
          <p className="text-xs text-green-400 mt-1">All available</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <HardDrive className="w-8 h-8 text-blue-400" />
            <span className="text-2xl font-bold text-white">{totalStorage.toFixed(1)} TB</span>
          </div>
          <p className="text-gray-400 text-sm">Total Used</p>
          <p className="text-xs text-blue-400 mt-1">{(totalBlobs / 1000000).toFixed(1)}M objects</p>
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
          <p className="text-xs text-yellow-400 mt-1">↑ 15% avg trend</p>
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

      {/* Filters */}
      <div className="flex flex-wrap gap-4 mb-6">
        <input
          type="text"
          placeholder="Search storage accounts..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
        />
        
        <select
          value={selectedTier}
          onChange={(e) => setSelectedTier(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="all">All Tiers</option>
          <option value="Hot">Hot</option>
          <option value="Cool">Cool</option>
          <option value="Archive">Archive</option>
        </select>

        <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors">
          + Create Storage
        </button>
      </div>

      {/* Storage Accounts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {loading ? (
          <div className="col-span-2 flex items-center justify-center py-12">
            <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          filteredAccounts.map((account, index) => (
            <motion.div
              key={account.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 overflow-hidden hover:bg-white/15 transition-colors"
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-purple-500/20 rounded-lg">
                      <Cloud className="w-6 h-6 text-purple-400" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">{account.name}</h3>
                      <p className="text-sm text-gray-400">{account.resourceGroup} • {account.location}</p>
                      <div className="flex items-center gap-2 mt-2">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(account.status)}`}>
                          {account.status}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTierColor(account.tier)}`}>
                          {account.tier}
                        </span>
                        <span className="text-xs text-gray-400">{account.replication}</span>
                      </div>
                    </div>
                  </div>
                  {account.location === 'Global' && (
                    <Globe className="w-5 h-5 text-purple-400" />
                  )}
                </div>

                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="bg-black/20 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-400">Capacity</span>
                      <HardDrive className="w-4 h-4 text-purple-400" />
                    </div>
                    <div className="flex items-baseline gap-1">
                      <span className="text-lg font-semibold text-white">
                        {account.capacity.used} / {account.capacity.total}
                      </span>
                      <span className="text-xs text-gray-400">{account.capacity.unit}</span>
                    </div>
                    <div className="mt-2 bg-black/30 rounded-full h-1.5">
                      <div
                        className="bg-purple-400 h-1.5 rounded-full"
                        style={{ width: `${(account.capacity.used / account.capacity.total) * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-400">Objects</span>
                      <Folder className="w-4 h-4 text-blue-400" />
                    </div>
                    <div>
                      <span className="text-lg font-semibold text-white">
                        {account.blobs > 1000000 ? `${(account.blobs / 1000000).toFixed(1)}M` : 
                         account.blobs > 1000 ? `${(account.blobs / 1000).toFixed(0)}K` : account.blobs}
                      </span>
                      <div className="flex gap-2 mt-1">
                        <span className="text-xs text-gray-400">{account.containers} containers</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-400">Bandwidth</span>
                      <Activity className="w-4 h-4 text-green-400" />
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-1">
                        <Download className="w-3 h-3 text-green-400" />
                        <span className="text-xs text-white">{account.bandwidth.ingress} MB/s</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Upload className="w-3 h-3 text-blue-400" />
                        <span className="text-xs text-white">{account.bandwidth.egress} MB/s</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-400">Transactions</span>
                      <BarChart3 className="w-4 h-4 text-yellow-400" />
                    </div>
                    <span className="text-lg font-semibold text-white">
                      {account.transactions > 1000000 ? `${(account.transactions / 1000000).toFixed(1)}M` : 
                       account.transactions > 1000 ? `${(account.transactions / 1000).toFixed(0)}K` : account.transactions}
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-white/10">
                  <div className="flex items-center gap-3 text-xs">
                    {account.encryption.enabled && (
                      <div className="flex items-center gap-1">
                        <Lock className="w-3 h-3 text-green-400" />
                        <span className="text-green-400">Encrypted</span>
                      </div>
                    )}
                    {account.publicAccess && (
                      <div className="flex items-center gap-1">
                        <AlertTriangle className="w-3 h-3 text-yellow-400" />
                        <span className="text-yellow-400">Public</span>
                      </div>
                    )}
                    <div className="flex items-center gap-1">
                      <Shield className="w-3 h-3 text-green-400" />
                      <span className="text-white">{account.compliance.score}%</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">{account.lastAccessed}</span>
                    <div className="flex items-center gap-1">
                      <DollarSign className="w-4 h-4 text-green-400" />
                      <span className="text-sm font-medium text-white">${account.cost.monthly}/mo</span>
                      {account.cost.trend !== 0 && (
                        <span className={`text-xs ${account.cost.trend > 0 ? 'text-red-400' : 'text-green-400'}`}>
                          {account.cost.trend > 0 ? '+' : ''}{account.cost.trend}%
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {account.compliance.issues.length > 0 && (
                  <div className="mt-3 p-2 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                    <div className="flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5" />
                      <div>
                        <p className="text-xs text-yellow-400 font-medium">Compliance Issues:</p>
                        {account.compliance.issues.map((issue, idx) => (
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