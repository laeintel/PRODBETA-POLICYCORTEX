/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2026 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  HardDrive,
  Database,
  Cloud,
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  Users,
  Zap,
  Shield,
  Settings,
  Monitor,
  BarChart3,
  Eye,
  RefreshCw,
  Download,
  Upload,
  Search,
  Filter,
  MoreVertical,
  Play,
  Pause,
  StopCircle,
  Power,
  PowerOff,
  RotateCcw,
  Scale,
  Maximize2,
  Minimize2,
  Network,
  DollarSign,
  AlertCircle,
  XCircle,
  ChevronRight,
  ExternalLink,
  Terminal,
  Container,
  Layers,
  Archive,
  Folder,
  FileText,
  Image,
  Video,
  Music,
  BookOpen,
  Lock,
  Unlock,
  Key,
  Globe,
  Server,
  Cpu
} from 'lucide-react'
import { Line, Bar, Doughnut } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler
)

export default function StorageSystemsPage() {
  const [loading, setLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [activeTab, setActiveTab] = useState('overview')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [storageData, setStorageData] = useState<any>(null)
  const [realTimeMetrics, setRealTimeMetrics] = useState<any[]>([])
  const [selectedAccounts, setSelectedAccounts] = useState<string[]>([])
  const [filterType, setFilterType] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    loadAllData()
    const interval = autoRefresh ? setInterval(loadRealTimeData, 5000) : null
    return () => { if (interval) clearInterval(interval) }
  }, [autoRefresh])

  const loadAllData = () => {
    setTimeout(() => {
      setStorageData({
        overview: {
          totalCapacity: '156.8 TB',
          usedCapacity: '98.4 TB',
          availableCapacity: '58.4 TB',
          utilizationPercent: 62.8,
          storageAccounts: 45,
          totalObjects: 2847536,
          avgIOPS: 12456,
          avgThroughput: 2.34,
          monthlyBudget: 8934.50,
          currentSpend: 5678.90
        },
        accounts: [
          {
            id: 'sa-001',
            name: 'prodstorageacct001',
            type: 'BlobStorage',
            tier: 'Standard',
            replication: 'GRS',
            location: 'East US',
            capacity: '23.4 TB',
            used: '18.7 TB',
            utilization: 79.9,
            objects: 456789,
            transactions: 2345678,
            costPerMonth: 892.45,
            status: 'active',
            encryption: 'enabled',
            backup: 'enabled',
            accessTier: 'Hot'
          },
          {
            id: 'sa-002',
            name: 'devstorageacct002',
            type: 'FileStorage',
            tier: 'Premium',
            replication: 'LRS',
            location: 'West US',
            capacity: '8.9 TB',
            used: '4.2 TB',
            utilization: 47.2,
            objects: 12345,
            transactions: 345678,
            costPerMonth: 234.67,
            status: 'active',
            encryption: 'enabled',
            backup: 'disabled',
            accessTier: 'Cool'
          },
          {
            id: 'sa-003',
            name: 'backupstorageacct003',
            type: 'BlockBlobStorage',
            tier: 'Standard',
            replication: 'ZRS',
            location: 'Central US',
            capacity: '45.6 TB',
            used: '32.1 TB',
            utilization: 70.4,
            objects: 789123,
            transactions: 567890,
            costPerMonth: 567.89,
            status: 'active',
            encryption: 'enabled',
            backup: 'enabled',
            accessTier: 'Archive'
          },
          {
            id: 'sa-004',
            name: 'logsstorageacct004',
            type: 'BlobStorage',
            tier: 'Standard',
            replication: 'RA-GRS',
            location: 'North Europe',
            capacity: '12.3 TB',
            used: '8.9 TB',
            utilization: 72.4,
            objects: 234567,
            transactions: 890123,
            costPerMonth: 345.12,
            status: 'warning',
            encryption: 'enabled',
            backup: 'enabled',
            accessTier: 'Hot'
          },
          {
            id: 'sa-005',
            name: 'mediastorageacct005',
            type: 'BlobStorage',
            tier: 'Premium',
            replication: 'LRS',
            location: 'West Europe',
            capacity: '67.8 TB',
            used: '45.2 TB',
            utilization: 66.7,
            objects: 1234567,
            transactions: 456789,
            costPerMonth: 1234.56,
            status: 'active',
            encryption: 'enabled',
            backup: 'enabled',
            accessTier: 'Hot'
          }
        ],
        containers: [
          {
            id: 'c-001',
            name: 'web-assets',
            account: 'prodstorageacct001',
            type: 'blob',
            objects: 45678,
            size: '2.3 TB',
            accessTier: 'Hot',
            lastModified: '2 hours ago',
            publicAccess: false
          },
          {
            id: 'c-002',
            name: 'backup-data',
            account: 'backupstorageacct003',
            type: 'blob',
            objects: 123456,
            size: '18.7 TB',
            accessTier: 'Archive',
            lastModified: '1 day ago',
            publicAccess: false
          },
          {
            id: 'c-003',
            name: 'log-files',
            account: 'logsstorageacct004',
            type: 'blob',
            objects: 789012,
            size: '8.9 TB',
            accessTier: 'Cool',
            lastModified: '10 minutes ago',
            publicAccess: false
          }
        ],
        analytics: {
          hotStoragePercent: 45.6,
          coolStoragePercent: 32.1,
          archiveStoragePercent: 22.3,
          averageObjectSize: '2.4 MB',
          compressionRatio: 2.8,
          deduplicationSavings: '12.3 TB'
        }
      })

      setRealTimeMetrics(generateRealTimeData())
      setLoading(false)
    }, 1000)
  }

  const loadRealTimeData = () => {
    setRealTimeMetrics(prev => {
      const newData = [...prev, {
        timestamp: new Date(),
        iops: 10000 + Math.random() * 5000,
        throughput: 2 + Math.random() * 1,
        utilization: 60 + Math.random() * 20,
        transactions: 50000 + Math.random() * 20000
      }]
      return newData.slice(-20)
    })
  }

  const generateRealTimeData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 60000),
      iops: 10000 + Math.random() * 5000,
      throughput: 2 + Math.random() * 1,
      utilization: 60 + Math.random() * 20,
      transactions: 50000 + Math.random() * 20000
    }))
  }

  const performanceData = {
    labels: realTimeMetrics.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [
      {
        label: 'IOPS',
        data: realTimeMetrics.map(d => d.iops),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true,
        yAxisID: 'y'
      },
      {
        label: 'Throughput (GB/s)',
        data: realTimeMetrics.map(d => d.throughput),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
        fill: true,
        yAxisID: 'y1'
      }
    ]
  }

  const utilizationData = {
    labels: storageData?.accounts.map((acc: any) => acc.name) || [],
    datasets: [{
      label: 'Utilization %',
      data: storageData?.accounts.map((acc: any) => acc.utilization) || [],
      backgroundColor: storageData?.accounts.map((acc: any) => 
        acc.utilization > 80 ? 'rgba(239, 68, 68, 0.8)' :
        acc.utilization > 60 ? 'rgba(245, 158, 11, 0.8)' :
        'rgba(16, 185, 129, 0.8)'
      ) || [],
      borderWidth: 0
    }]
  }

  const accessTierData = {
    labels: ['Hot', 'Cool', 'Archive'],
    datasets: [{
      data: [
        storageData?.analytics.hotStoragePercent || 0,
        storageData?.analytics.coolStoragePercent || 0,
        storageData?.analytics.archiveStoragePercent || 0
      ],
      backgroundColor: [
        'rgba(239, 68, 68, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(59, 130, 246, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const storageTypeData = {
    labels: ['Blob Storage', 'File Storage', 'Queue Storage', 'Table Storage'],
    datasets: [{
      data: [76.2, 15.8, 4.3, 3.7],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(139, 92, 246, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const filteredAccounts = storageData?.accounts.filter((account: any) => {
    const matchesType = filterType === 'all' || account.type === filterType
    const matchesSearch = account.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         account.location.toLowerCase().includes(searchTerm.toLowerCase())
    return matchesType && matchesSearch
  }) || []

  const handleAccountAction = (action: string, accountId: string) => {
    console.log(`${action} account: ${accountId}`)
  }

  const handleBulkAction = (action: string) => {
    console.log(`${action} accounts:`, selectedAccounts)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading Storage Systems...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="bg-gray-950 border-b border-gray-800 sticky top-0 z-50">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <HardDrive className="w-8 h-8 text-green-500" />
              <div>
                <h1 className="text-2xl font-bold">Storage Systems</h1>
                <p className="text-sm text-gray-500">Cloud storage accounts, containers, and data management</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">{storageData.overview.utilizationPercent}% UTILIZATION</span>
              </div>
              <button 
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`p-2 rounded ${autoRefresh ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400'}`}
              >
                <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              </button>
              <select 
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value)}
                className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="1h">Last Hour</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
              <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors">
                CREATE STORAGE
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'accounts', 'containers', 'analytics', 'backup'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-3 px-1 border-b-2 transition-colors capitalize ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-500'
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </header>

      <div className="p-6">
        {activeTab === 'overview' && (
          <>
            {/* Overview Metrics */}
            <div className="grid grid-cols-6 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <HardDrive className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Total Capacity</span>
                </div>
                <p className="text-2xl font-bold font-mono">{storageData.overview.totalCapacity}</p>
                <p className="text-xs text-gray-500 mt-1">{storageData.overview.storageAccounts} accounts</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Database className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Used</span>
                </div>
                <p className="text-2xl font-bold font-mono">{storageData.overview.usedCapacity}</p>
                <p className="text-xs text-gray-500 mt-1">{storageData.overview.utilizationPercent}% utilization</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Cloud className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Available</span>
                </div>
                <p className="text-2xl font-bold font-mono">{storageData.overview.availableCapacity}</p>
                <p className="text-xs text-gray-500 mt-1">free space</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Archive className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Objects</span>
                </div>
                <p className="text-2xl font-bold font-mono">{(storageData.overview.totalObjects / 1000000).toFixed(1)}M</p>
                <p className="text-xs text-gray-500 mt-1">total objects</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Activity className="w-5 h-5 text-pink-500" />
                  <span className="text-xs text-gray-500">IOPS</span>
                </div>
                <p className="text-2xl font-bold font-mono">{(storageData.overview.avgIOPS / 1000).toFixed(1)}K</p>
                <p className="text-xs text-gray-500 mt-1">average IOPS</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <DollarSign className="w-5 h-5 text-orange-500" />
                  <span className="text-xs text-gray-500">Cost</span>
                </div>
                <p className="text-2xl font-bold font-mono">${(storageData.overview.currentSpend / 1000).toFixed(1)}K</p>
                <p className="text-xs text-gray-500 mt-1">of ${(storageData.overview.monthlyBudget / 1000).toFixed(1)}K budget</p>
              </motion.div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {/* Performance Metrics */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">PERFORMANCE METRICS</h3>
                  <div className="flex items-center space-x-2">
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full" />
                      <span className="text-xs text-gray-500">IOPS</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      <span className="text-xs text-gray-500">Throughput</span>
                    </div>
                  </div>
                </div>
                <div className="h-64">
                  <Line data={performanceData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                      mode: 'index',
                      intersect: false,
                    },
                    plugins: {
                      legend: { display: false },
                      tooltip: {
                        mode: 'index',
                        intersect: false,
                      }
                    },
                    scales: {
                      x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
                      },
                      y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
                      },
                      y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: { drawOnChartArea: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
                      }
                    }
                  }} />
                </div>
              </div>

              {/* Access Tier Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">ACCESS TIER DISTRIBUTION</h3>
                <div className="h-64">
                  <Doughnut data={accessTierData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: { color: 'rgba(255, 255, 255, 0.7)', font: { size: 11 } }
                      }
                    }
                  }} />
                </div>
              </div>

              {/* Storage Type Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">STORAGE TYPE DISTRIBUTION</h3>
                <div className="h-64">
                  <Doughnut data={storageTypeData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: { color: 'rgba(255, 255, 255, 0.7)', font: { size: 11 } }
                      }
                    }
                  }} />
                </div>
              </div>
            </div>

            {/* Storage Account Utilization */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg mb-6">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">STORAGE ACCOUNT UTILIZATION</h3>
              </div>
              <div className="p-4">
                <div className="h-64">
                  <Bar data={utilizationData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false }
                    },
                    scales: {
                      x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
                      },
                      y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } },
                        max: 100
                      }
                    }
                  }} />
                </div>
              </div>
            </div>

            {/* Storage Analytics Summary */}
            <div className="grid grid-cols-4 gap-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">OPTIMIZATION METRICS</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Avg Object Size</span>
                    <span className="font-mono text-sm">{storageData.analytics.averageObjectSize}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Compression Ratio</span>
                    <span className="font-mono text-sm text-green-500">{storageData.analytics.compressionRatio}:1</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Dedup Savings</span>
                    <span className="font-mono text-sm text-blue-500">{storageData.analytics.deduplicationSavings}</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">ACCESS PATTERNS</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Hot Storage</span>
                    <span className="font-mono text-sm text-red-500">{storageData.analytics.hotStoragePercent}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Cool Storage</span>
                    <span className="font-mono text-sm text-yellow-500">{storageData.analytics.coolStoragePercent}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Archive Storage</span>
                    <span className="font-mono text-sm text-blue-500">{storageData.analytics.archiveStoragePercent}%</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">SECURITY STATUS</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Encrypted Accounts</span>
                    <span className="font-mono text-sm text-green-500">100%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Backup Enabled</span>
                    <span className="font-mono text-sm text-green-500">89%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Public Access</span>
                    <span className="font-mono text-sm text-red-500">0%</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">QUICK ACTIONS</h3>
                </div>
                <div className="p-4 space-y-2">
                  <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group">
                    <span>Optimize Tiers</span>
                    <TrendingUp className="w-4 h-4 text-gray-500 group-hover:text-white" />
                  </button>
                  <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group">
                    <span>Backup Audit</span>
                    <Shield className="w-4 h-4 text-gray-500 group-hover:text-white" />
                  </button>
                  <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group">
                    <span>Cost Analysis</span>
                    <BarChart3 className="w-4 h-4 text-gray-500 group-hover:text-white" />
                  </button>
                  <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group">
                    <span>Generate Report</span>
                    <Download className="w-4 h-4 text-gray-500 group-hover:text-white" />
                  </button>
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'accounts' && (
          <>
            {/* Account Management Controls */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <Search className="w-4 h-4 text-gray-500 absolute left-3 top-1/2 transform -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder="Search storage accounts..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <select
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value)}
                    className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                  >
                    <option value="all">All Types</option>
                    <option value="BlobStorage">Blob Storage</option>
                    <option value="FileStorage">File Storage</option>
                    <option value="BlockBlobStorage">Block Blob Storage</option>
                    <option value="QueueStorage">Queue Storage</option>
                  </select>
                </div>
                <div className="flex items-center space-x-2">
                  {selectedAccounts.length > 0 && (
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-400">{selectedAccounts.length} selected</span>
                      <button 
                        onClick={() => handleBulkAction('backup')}
                        className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded"
                      >
                        Backup
                      </button>
                      <button 
                        onClick={() => handleBulkAction('optimize')}
                        className="px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white text-sm rounded"
                      >
                        Optimize
                      </button>
                    </div>
                  )}
                  <button className="p-2 hover:bg-gray-800 rounded">
                    <Filter className="w-4 h-4 text-gray-500" />
                  </button>
                  <button className="p-2 hover:bg-gray-800 rounded">
                    <Download className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </div>
            </div>

            {/* Storage Accounts Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left">
                        <input
                          type="checkbox"
                          checked={selectedAccounts.length === filteredAccounts.length && filteredAccounts.length > 0}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedAccounts(filteredAccounts.map((acc: any) => acc.id))
                            } else {
                              setSelectedAccounts([])
                            }
                          }}
                          className="rounded border-gray-600 bg-gray-700 text-blue-600"
                        />
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Name</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Type</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Tier</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Location</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Capacity</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Utilization</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Objects</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Cost/Month</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {filteredAccounts.map((account: any) => (
                      <motion.tr
                        key={account.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <input
                            type="checkbox"
                            checked={selectedAccounts.includes(account.id)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedAccounts([...selectedAccounts, account.id])
                              } else {
                                setSelectedAccounts(selectedAccounts.filter(id => id !== account.id))
                              }
                            }}
                            className="rounded border-gray-600 bg-gray-700 text-blue-600"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <div>
                            <div className="font-medium">{account.name}</div>
                            <div className="text-sm text-gray-400">{account.replication}</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="inline-flex items-center px-2 py-1 text-xs rounded bg-blue-900/30 text-blue-500">
                            {account.type}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center px-2 py-1 text-xs rounded ${
                            account.tier === 'Premium' ? 'bg-purple-900/30 text-purple-500' :
                            'bg-gray-900/30 text-gray-500'
                          }`}>
                            {account.tier}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm">{account.location}</td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="font-mono">{account.capacity}</div>
                            <div className="text-gray-400 text-xs">{account.used} used</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="flex items-center space-x-1">
                              <span>{account.utilization.toFixed(1)}%</span>
                              <div className="w-12 bg-gray-800 rounded-full h-1.5">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    account.utilization > 80 ? 'bg-red-500' :
                                    account.utilization > 60 ? 'bg-yellow-500' :
                                    'bg-green-500'
                                  }`}
                                  style={{ width: `${Math.min(account.utilization, 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{account.objects.toLocaleString()}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">${account.costPerMonth}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            account.status === 'active' ? 'text-green-500' :
                            account.status === 'warning' ? 'text-yellow-500' :
                            'text-red-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              account.status === 'active' ? 'bg-green-500' :
                              account.status === 'warning' ? 'bg-yellow-500' :
                              'bg-red-500'
                            }`} />
                            <span className="uppercase">{account.status}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button 
                              onClick={() => handleAccountAction('backup', account.id)}
                              className="p-1 hover:bg-gray-700 rounded text-blue-500"
                              title="Backup Account"
                            >
                              <Archive className="w-4 h-4" />
                            </button>
                            <button 
                              onClick={() => handleAccountAction('optimize', account.id)}
                              className="p-1 hover:bg-gray-700 rounded text-green-500"
                              title="Optimize Storage"
                            >
                              <TrendingUp className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded">
                              <Settings className="w-4 h-4 text-gray-400" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded">
                              <MoreVertical className="w-4 h-4 text-gray-400" />
                            </button>
                          </div>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {activeTab === 'containers' && (
          <>
            {/* Container Management */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <Search className="w-4 h-4 text-gray-500 absolute left-3 top-1/2 transform -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder="Search containers..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <select className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm">
                    <option value="all">All Containers</option>
                    <option value="blob">Blob Containers</option>
                    <option value="file">File Shares</option>
                    <option value="queue">Queues</option>
                    <option value="table">Tables</option>
                  </select>
                  <select className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm">
                    <option value="all">All Access Tiers</option>
                    <option value="hot">Hot</option>
                    <option value="cool">Cool</option>
                    <option value="archive">Archive</option>
                  </select>
                </div>
                <div className="flex items-center space-x-2">
                  <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded">
                    CREATE CONTAINER
                  </button>
                  <button className="p-2 hover:bg-gray-800 rounded">
                    <Download className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </div>
            </div>

            {/* Containers Grid */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {storageData?.containers.map((container: any) => (
                <motion.div
                  key={container.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors cursor-pointer"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <Container className="w-5 h-5 text-blue-500" />
                      <span className="font-medium">{container.name}</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <button className="p-1 hover:bg-gray-800 rounded">
                        <Settings className="w-4 h-4 text-gray-500" />
                      </button>
                      <button className="p-1 hover:bg-gray-800 rounded">
                        <MoreVertical className="w-4 h-4 text-gray-500" />
                      </button>
                    </div>
                  </div>
                  
                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Account:</span>
                      <span className="font-mono text-xs">{container.account}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Size:</span>
                      <span className="font-mono">{container.size}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Objects:</span>
                      <span className="font-mono">{container.objects.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Access Tier:</span>
                      <span className={`px-2 py-1 text-xs rounded ${
                        container.accessTier === 'Hot' ? 'bg-red-900/30 text-red-500' :
                        container.accessTier === 'Cool' ? 'bg-yellow-900/30 text-yellow-500' :
                        'bg-blue-900/30 text-blue-500'
                      }`}>
                        {container.accessTier}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Modified:</span>
                      <span className="text-xs">{container.lastModified}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-3 border-t border-gray-800">
                    <div className="flex items-center space-x-2">
                      {container.publicAccess ? (
                        <div className="flex items-center space-x-1 text-red-500">
                          <Unlock className="w-4 h-4" />
                          <span className="text-xs">Public</span>
                        </div>
                      ) : (
                        <div className="flex items-center space-x-1 text-green-500">
                          <Lock className="w-4 h-4" />
                          <span className="text-xs">Private</span>
                        </div>
                      )}
                    </div>
                    <div className="flex items-center space-x-2">
                      <button className="px-2 py-1 bg-blue-600 hover:bg-blue-700 text-xs rounded">
                        BROWSE
                      </button>
                      <button className="px-2 py-1 bg-green-600 hover:bg-green-700 text-xs rounded">
                        UPLOAD
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Lifecycle Management */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">LIFECYCLE MANAGEMENT POLICIES</h3>
              </div>
              <div className="p-4">
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-gray-800/30 rounded">
                    <div className="flex items-center space-x-4">
                      <div className="w-3 h-3 bg-green-500 rounded-full" />
                      <div>
                        <div className="text-sm font-medium">Auto-tier to Cool</div>
                        <div className="text-xs text-gray-400">Move blobs to Cool tier after 30 days of inactivity</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-green-500">ACTIVE</span>
                      <button className="text-xs text-blue-500 hover:text-blue-400">EDIT</button>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between p-4 bg-gray-800/30 rounded">
                    <div className="flex items-center space-x-4">
                      <div className="w-3 h-3 bg-blue-500 rounded-full" />
                      <div>
                        <div className="text-sm font-medium">Archive Old Backups</div>
                        <div className="text-xs text-gray-400">Move backup blobs to Archive tier after 90 days</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-blue-500">ACTIVE</span>
                      <button className="text-xs text-blue-500 hover:text-blue-400">EDIT</button>
                    </div>
                  </div>

                  <div className="flex items-center justify-between p-4 bg-gray-800/30 rounded">
                    <div className="flex items-center space-x-4">
                      <div className="w-3 h-3 bg-red-500 rounded-full" />
                      <div>
                        <div className="text-sm font-medium">Delete Temporary Files</div>
                        <div className="text-xs text-gray-400">Delete temp files after 7 days</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-red-500">PAUSED</span>
                      <button className="text-xs text-blue-500 hover:text-blue-400">EDIT</button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'analytics' && (
          <>
            {/* Cost Analysis */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">COST BREAKDOWN</h3>
                </div>
                <div className="p-4">
                  <div className="h-64">
                    <Doughnut data={{
                      labels: ['Hot Storage', 'Cool Storage', 'Archive Storage', 'Transactions', 'Bandwidth'],
                      datasets: [{
                        data: [45.2, 28.7, 12.3, 8.9, 4.9],
                        backgroundColor: [
                          'rgba(239, 68, 68, 0.8)',
                          'rgba(245, 158, 11, 0.8)',
                          'rgba(59, 130, 246, 0.8)',
                          'rgba(16, 185, 129, 0.8)',
                          'rgba(139, 92, 246, 0.8)'
                        ],
                        borderWidth: 0
                      }]
                    }} options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'bottom' as const,
                          labels: { color: 'rgba(255, 255, 255, 0.7)', font: { size: 11 } }
                        }
                      }
                    }} />
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">STORAGE TRENDS (30 DAYS)</h3>
                </div>
                <div className="p-4">
                  <div className="h-64">
                    <Line data={{
                      labels: Array.from({ length: 30 }, (_, i) => `Day ${i + 1}`),
                      datasets: [
                        {
                          label: 'Total Storage (TB)',
                          data: Array.from({ length: 30 }, (_, i) => 95 + i * 0.3 + Math.random() * 2),
                          borderColor: 'rgb(59, 130, 246)',
                          backgroundColor: 'rgba(59, 130, 246, 0.1)',
                          tension: 0.4,
                          fill: true
                        },
                        {
                          label: 'Hot Storage (TB)',
                          data: Array.from({ length: 30 }, (_, i) => 45 + Math.random() * 5),
                          borderColor: 'rgb(239, 68, 68)',
                          backgroundColor: 'rgba(239, 68, 68, 0.1)',
                          tension: 0.4,
                          fill: true
                        }
                      ]
                    }} options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'bottom' as const,
                          labels: { color: 'rgba(255, 255, 255, 0.7)', font: { size: 11 } }
                        }
                      },
                      scales: {
                        x: {
                          grid: { color: 'rgba(255, 255, 255, 0.05)' },
                          ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
                        },
                        y: {
                          grid: { color: 'rgba(255, 255, 255, 0.05)' },
                          ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
                        }
                      }
                    }} />
                  </div>
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-500 mb-2">{(storageData.overview.avgIOPS / 1000).toFixed(1)}K</div>
                  <div className="text-xs text-gray-400 uppercase mb-4">Average IOPS</div>
                  <div className="h-16">
                    <Line data={{
                      labels: Array.from({ length: 12 }, (_, i) => `${i * 2}h`),
                      datasets: [{
                        data: Array.from({ length: 12 }, () => 10000 + Math.random() * 5000),
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                      }]
                    }} options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: { legend: { display: false } },
                      scales: {
                        x: { display: false },
                        y: { display: false }
                      },
                      elements: { point: { radius: 0 } }
                    }} />
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-500 mb-2">{storageData.overview.avgThroughput}GB/s</div>
                  <div className="text-xs text-gray-400 uppercase mb-4">Throughput</div>
                  <div className="h-16">
                    <Line data={{
                      labels: Array.from({ length: 12 }, (_, i) => `${i * 2}h`),
                      datasets: [{
                        data: Array.from({ length: 12 }, () => 2 + Math.random() * 1),
                        borderColor: 'rgb(16, 185, 129)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: true
                      }]
                    }} options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: { legend: { display: false } },
                      scales: {
                        x: { display: false },
                        y: { display: false }
                      },
                      elements: { point: { radius: 0 } }
                    }} />
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-yellow-500 mb-2">{storageData.overview.utilizationPercent}%</div>
                  <div className="text-xs text-gray-400 uppercase mb-4">Utilization</div>
                  <div className="h-16">
                    <Line data={{
                      labels: Array.from({ length: 12 }, (_, i) => `${i * 2}h`),
                      datasets: [{
                        data: Array.from({ length: 12 }, () => 60 + Math.random() * 20),
                        borderColor: 'rgb(245, 158, 11)',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4,
                        fill: true
                      }]
                    }} options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: { legend: { display: false } },
                      scales: {
                        x: { display: false },
                        y: { display: false }
                      },
                      elements: { point: { radius: 0 } }
                    }} />
                  </div>
                </div>
              </div>
            </div>

            {/* Access Patterns Analysis */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">ACCESS PATTERNS ANALYSIS</h3>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-4 gap-6">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-500 mb-1">2.4M</div>
                    <div className="text-xs text-gray-400">Daily Requests</div>
                    <div className="text-xs text-green-500 mt-1">↑ 12.3%</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-500 mb-1">15.7GB</div>
                    <div className="text-xs text-gray-400">Data Transferred</div>
                    <div className="text-xs text-red-500 mt-1">↓ 5.2%</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-500 mb-1">98.7%</div>
                    <div className="text-xs text-gray-400">Cache Hit Ratio</div>
                    <div className="text-xs text-green-500 mt-1">↑ 2.1%</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-500 mb-1">45ms</div>
                    <div className="text-xs text-gray-400">Avg Latency</div>
                    <div className="text-xs text-green-500 mt-1">↓ 8.4%</div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'backup' && (
          <>
            {/* Backup Status */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Archive className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Backup Enabled</span>
                </div>
                <p className="text-2xl font-bold font-mono text-green-500">89%</p>
                <p className="text-xs text-gray-500 mt-1">of storage accounts</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Clock className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Last Backup</span>
                </div>
                <p className="text-2xl font-bold font-mono">2h</p>
                <p className="text-xs text-gray-500 mt-1">ago</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <HardDrive className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Backup Size</span>
                </div>
                <p className="text-2xl font-bold font-mono">45.2TB</p>
                <p className="text-xs text-gray-500 mt-1">total backup data</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Success Rate</span>
                </div>
                <p className="text-2xl font-bold font-mono text-green-500">99.8%</p>
                <p className="text-xs text-gray-500 mt-1">last 30 days</p>
              </motion.div>
            </div>

            {/* Backup Policies */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-gray-400 uppercase">BACKUP POLICIES</h3>
                    <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-xs rounded">CREATE POLICY</button>
                  </div>
                </div>
                <div className="p-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-green-500 rounded-full" />
                        <div>
                          <div className="text-sm font-medium">Production Backup</div>
                          <div className="text-xs text-gray-400">Daily at 2:00 AM UTC</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-green-500">ACTIVE</span>
                        <button className="text-xs text-blue-500 hover:text-blue-400">EDIT</button>
                      </div>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-yellow-500 rounded-full" />
                        <div>
                          <div className="text-sm font-medium">Archive Backup</div>
                          <div className="text-xs text-gray-400">Weekly on Sunday</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-yellow-500">SCHEDULED</span>
                        <button className="text-xs text-blue-500 hover:text-blue-400">EDIT</button>
                      </div>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-red-500 rounded-full" />
                        <div>
                          <div className="text-sm font-medium">Development Backup</div>
                          <div className="text-xs text-gray-400">On-demand only</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-red-500">DISABLED</span>
                        <button className="text-xs text-blue-500 hover:text-blue-400">EDIT</button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">BACKUP HISTORY</h3>
                </div>
                <div className="p-4">
                  <div className="space-y-3">
                    {Array.from({ length: 8 }, (_, i) => {
                      const isRecent = i < 3
                      const status = isRecent ? 'completed' : Math.random() > 0.1 ? 'completed' : 'failed'
                      const date = new Date(Date.now() - (i * 24 * 60 * 60 * 1000))
                      
                      return (
                        <div key={i} className="flex items-center justify-between p-2 hover:bg-gray-800/30 rounded">
                          <div className="flex items-center space-x-3">
                            <div className={`w-2 h-2 rounded-full ${
                              status === 'completed' ? 'bg-green-500' : 'bg-red-500'
                            }`} />
                            <div>
                              <div className="text-sm font-medium">
                                {date.toLocaleDateString()}
                              </div>
                              <div className="text-xs text-gray-400">
                                {(15 + Math.random() * 20).toFixed(1)}GB backed up
                              </div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className={`text-xs font-medium ${
                              status === 'completed' ? 'text-green-500' : 'text-red-500'
                            }`}>
                              {status === 'completed' ? 'SUCCESS' : 'FAILED'}
                            </div>
                            <div className="text-xs text-gray-400">
                              {Math.floor(Math.random() * 120 + 30)}min
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            </div>

            {/* Point-in-Time Recovery */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">POINT-IN-TIME RECOVERY</h3>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-3 gap-6">
                  <div>
                    <div className="text-xs text-gray-400 mb-3 uppercase">Recovery Points Available</div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Last 24 Hours</span>
                        <span className="text-sm text-green-500">Continuous</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Last 7 Days</span>
                        <span className="text-sm text-blue-500">Hourly</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Last 30 Days</span>
                        <span className="text-sm text-yellow-500">Daily</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-400 mb-3 uppercase">Recovery Options</div>
                    <div className="space-y-2">
                      <button className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 text-sm rounded text-left">
                        Restore to Original Location
                      </button>
                      <button className="w-full px-3 py-2 bg-green-600 hover:bg-green-700 text-sm rounded text-left">
                        Restore to Alternate Location
                      </button>
                      <button className="w-full px-3 py-2 bg-purple-600 hover:bg-purple-700 text-sm rounded text-left">
                        Create Recovery Instance
                      </button>
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-400 mb-3 uppercase">Geo-Redundancy</div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Primary Region</span>
                        <span className="text-sm text-green-500">East US</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Secondary Region</span>
                        <span className="text-sm text-blue-500">West US</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Replication Status</span>
                        <span className="text-sm text-green-500">Synced</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}