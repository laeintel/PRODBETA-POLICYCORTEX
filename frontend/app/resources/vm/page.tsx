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
  Server,
  Activity,
  HardDrive,
  Cpu,
  MemoryStick,
  Network,
  Shield,
  DollarSign,
  Power,
  AlertTriangle,
  CheckCircle,
  Clock,
  MoreVertical,
  Play,
  Pause,
  RotateCw,
  Trash2,
  Settings,
  Monitor,
  Zap
} from 'lucide-react'

interface VirtualMachine {
  id: string
  name: string
  resourceGroup: string
  location: string
  size: string
  status: 'Running' | 'Stopped' | 'Starting' | 'Stopping' | 'Deallocated'
  os: string
  publicIP?: string
  privateIP: string
  cpu: { usage: number; cores: number }
  memory: { usage: number; total: number }
  disk: { usage: number; total: number }
  network: { in: number; out: number }
  cost: { daily: number; monthly: number }
  tags: Record<string, string>
  uptime: string
  compliance: { score: number; issues: number }
}

export default function VirtualMachinesPage() {
  const [vms, setVms] = useState<VirtualMachine[]>([])
  const [selectedVms, setSelectedVms] = useState<Set<string>>(new Set())
  const [filter, setFilter] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate loading VMs
    setTimeout(() => {
      setVms([
        {
          id: 'vm-001',
          name: 'prod-web-01',
          resourceGroup: 'production-rg',
          location: 'East US',
          size: 'Standard_D4s_v3',
          status: 'Running',
          os: 'Ubuntu 20.04',
          publicIP: '52.188.123.45',
          privateIP: '10.0.1.4',
          cpu: { usage: 65, cores: 4 },
          memory: { usage: 72, total: 16 },
          disk: { usage: 45, total: 256 },
          network: { in: 125, out: 89 },
          cost: { daily: 4.8, monthly: 144 },
          tags: { environment: 'production', team: 'platform' },
          uptime: '15 days',
          compliance: { score: 95, issues: 2 }
        },
        {
          id: 'vm-002',
          name: 'prod-api-01',
          resourceGroup: 'production-rg',
          location: 'East US',
          size: 'Standard_D8s_v3',
          status: 'Running',
          os: 'Windows Server 2019',
          publicIP: '52.188.123.46',
          privateIP: '10.0.1.5',
          cpu: { usage: 45, cores: 8 },
          memory: { usage: 58, total: 32 },
          disk: { usage: 62, total: 512 },
          network: { in: 245, out: 189 },
          cost: { daily: 9.6, monthly: 288 },
          tags: { environment: 'production', team: 'api' },
          uptime: '45 days',
          compliance: { score: 98, issues: 1 }
        },
        {
          id: 'vm-003',
          name: 'dev-test-01',
          resourceGroup: 'development-rg',
          location: 'West US',
          size: 'Standard_B2s',
          status: 'Stopped',
          os: 'Ubuntu 22.04',
          privateIP: '10.1.1.4',
          cpu: { usage: 0, cores: 2 },
          memory: { usage: 0, total: 4 },
          disk: { usage: 23, total: 64 },
          network: { in: 0, out: 0 },
          cost: { daily: 0, monthly: 0 },
          tags: { environment: 'development', team: 'qa' },
          uptime: '0 days',
          compliance: { score: 88, issues: 5 }
        },
        {
          id: 'vm-004',
          name: 'ml-training-gpu',
          resourceGroup: 'ml-rg',
          location: 'East US 2',
          size: 'Standard_NC6s_v3',
          status: 'Running',
          os: 'Ubuntu 20.04',
          privateIP: '10.2.1.10',
          cpu: { usage: 89, cores: 6 },
          memory: { usage: 85, total: 112 },
          disk: { usage: 78, total: 1024 },
          network: { in: 512, out: 128 },
          cost: { daily: 28.8, monthly: 864 },
          tags: { environment: 'production', team: 'ml', gpu: 'enabled' },
          uptime: '7 days',
          compliance: { score: 92, issues: 3 }
        }
      ])
      setLoading(false)
    }, 1000)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Running': return 'text-green-400 bg-green-500/20'
      case 'Stopped': return 'text-gray-400 bg-gray-500/20'
      case 'Deallocated': return 'text-gray-400 bg-gray-500/20'
      case 'Starting': return 'text-yellow-400 bg-yellow-500/20'
      case 'Stopping': return 'text-orange-400 bg-orange-500/20'
      default: return 'text-gray-400 bg-gray-500/20'
    }
  }

  const filteredVms = vms.filter(vm => {
    const matchesSearch = vm.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          vm.resourceGroup.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesFilter = filter === 'all' || 
                         (filter === 'running' && vm.status === 'Running') ||
                         (filter === 'stopped' && (vm.status === 'Stopped' || vm.status === 'Deallocated'))
    return matchesSearch && matchesFilter
  })

  const totalCost = vms.reduce((sum, vm) => sum + vm.cost.monthly, 0)
  const runningVms = vms.filter(vm => vm.status === 'Running').length

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
            <Server className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Virtual Machines</h1>
            <p className="text-gray-400 mt-1">Manage and monitor your compute instances</p>
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
            <Server className="w-8 h-8 text-purple-400" />
            <span className="text-2xl font-bold text-white">{vms.length}</span>
          </div>
          <p className="text-gray-400 text-sm">Total VMs</p>
          <p className="text-xs text-green-400 mt-1">{runningVms} running</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <Cpu className="w-8 h-8 text-blue-400" />
            <span className="text-2xl font-bold text-white">62%</span>
          </div>
          <p className="text-gray-400 text-sm">Avg CPU Usage</p>
          <p className="text-xs text-blue-400 mt-1">Across all VMs</p>
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
          <p className="text-xs text-yellow-400 mt-1">↑ 12% from last month</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-white/10 backdrop-blur-xl rounded-xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between mb-4">
            <Shield className="w-8 h-8 text-green-400" />
            <span className="text-2xl font-bold text-white">93%</span>
          </div>
          <p className="text-gray-400 text-sm">Avg Compliance</p>
          <p className="text-xs text-red-400 mt-1">11 total issues</p>
        </motion.div>
      </div>

      {/* Filters and Actions */}
      <div className="flex flex-wrap gap-4 mb-6">
        <input
          type="text"
          placeholder="Search VMs..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
        />
        
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value="all">All VMs</option>
          <option value="running">Running</option>
          <option value="stopped">Stopped</option>
        </select>

        <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors">
          + Create VM
        </button>

        {selectedVms.size > 0 && (
          <>
            <button className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-white transition-colors flex items-center gap-2">
              <Play className="w-4 h-4" />
              Start Selected
            </button>
            <button className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-white transition-colors flex items-center gap-2">
              <Pause className="w-4 h-4" />
              Stop Selected
            </button>
          </>
        )}
      </div>

      {/* VMs Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {loading ? (
          <div className="col-span-2 flex items-center justify-center py-12">
            <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          filteredVms.map((vm, index) => (
            <motion.div
              key={vm.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 overflow-hidden hover:bg-white/15 transition-colors"
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-4">
                    <input
                      type="checkbox"
                      checked={selectedVms.has(vm.id)}
                      onChange={(e) => {
                        const updated = new Set(selectedVms)
                        if (e.target.checked) {
                          updated.add(vm.id)
                        } else {
                          updated.delete(vm.id)
                        }
                        setSelectedVms(updated)
                      }}
                      className="mt-1"
                    />
                    <div className="p-3 bg-purple-500/20 rounded-lg">
                      <Monitor className="w-6 h-6 text-purple-400" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">{vm.name}</h3>
                      <p className="text-sm text-gray-400">{vm.resourceGroup} • {vm.location}</p>
                      <div className="flex items-center gap-2 mt-2">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(vm.status)}`}>
                          {vm.status}
                        </span>
                        <span className="text-xs text-gray-400">{vm.size}</span>
                      </div>
                    </div>
                  </div>
                  <button className="p-2 hover:bg-white/10 rounded-lg transition-colors">
                    <MoreVertical className="w-5 h-5 text-gray-400" />
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-black/20 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-400">CPU</span>
                      <Cpu className="w-4 h-4 text-blue-400" />
                    </div>
                    <div className="flex items-baseline gap-1">
                      <span className="text-lg font-semibold text-white">{vm.cpu.usage}%</span>
                      <span className="text-xs text-gray-400">/ {vm.cpu.cores} cores</span>
                    </div>
                    <div className="mt-2 bg-black/30 rounded-full h-1.5">
                      <div
                        className="bg-blue-400 h-1.5 rounded-full"
                        style={{ width: `${vm.cpu.usage}%` }}
                      />
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-400">Memory</span>
                      <MemoryStick className="w-4 h-4 text-green-400" />
                    </div>
                    <div className="flex items-baseline gap-1">
                      <span className="text-lg font-semibold text-white">{vm.memory.usage}%</span>
                      <span className="text-xs text-gray-400">/ {vm.memory.total} GB</span>
                    </div>
                    <div className="mt-2 bg-black/30 rounded-full h-1.5">
                      <div
                        className="bg-green-400 h-1.5 rounded-full"
                        style={{ width: `${vm.memory.usage}%` }}
                      />
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-400">Disk</span>
                      <HardDrive className="w-4 h-4 text-purple-400" />
                    </div>
                    <div className="flex items-baseline gap-1">
                      <span className="text-lg font-semibold text-white">{vm.disk.usage}%</span>
                      <span className="text-xs text-gray-400">/ {vm.disk.total} GB</span>
                    </div>
                    <div className="mt-2 bg-black/30 rounded-full h-1.5">
                      <div
                        className="bg-purple-400 h-1.5 rounded-full"
                        style={{ width: `${vm.disk.usage}%` }}
                      />
                    </div>
                  </div>

                  <div className="bg-black/20 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-400">Network</span>
                      <Network className="w-4 h-4 text-yellow-400" />
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-400">↓ {vm.network.in} MB/s</span>
                      <span className="text-xs text-gray-400">↑ {vm.network.out} MB/s</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-white/10">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1">
                      <DollarSign className="w-4 h-4 text-green-400" />
                      <span className="text-sm text-white">${vm.cost.daily}/day</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-400">{vm.uptime}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Shield className="w-4 h-4 text-green-400" />
                      <span className="text-sm text-white">{vm.compliance.score}%</span>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    {vm.status === 'Running' ? (
                      <button className="p-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg transition-colors">
                        <Pause className="w-4 h-4 text-red-400" />
                      </button>
                    ) : (
                      <button className="p-2 bg-green-500/20 hover:bg-green-500/30 rounded-lg transition-colors">
                        <Play className="w-4 h-4 text-green-400" />
                      </button>
                    )}
                    <button className="p-2 bg-blue-500/20 hover:bg-blue-500/30 rounded-lg transition-colors">
                      <RotateCw className="w-4 h-4 text-blue-400" />
                    </button>
                    <button className="p-2 bg-purple-500/20 hover:bg-purple-500/30 rounded-lg transition-colors">
                      <Settings className="w-4 h-4 text-purple-400" />
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          ))
        )}
      </div>
    </div>
  )
}