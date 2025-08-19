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
  Cpu,
  Server,
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
  HardDrive,
  // Memory icon not in lucide-react; use HardDrive as substitute
  Network,
  DollarSign,
  AlertCircle,
  XCircle,
  ChevronRight,
  ExternalLink,
  Terminal,
  Database,
  Container,
  Layers
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

export default function ComputeResourcesPage() {
  const [loading, setLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [activeTab, setActiveTab] = useState('overview')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [computeData, setComputeData] = useState<any>(null)
  const [realTimeMetrics, setRealTimeMetrics] = useState<any[]>([])
  const [selectedVMs, setSelectedVMs] = useState<string[]>([])
  const [filterStatus, setFilterStatus] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    loadAllData()
    const interval = autoRefresh ? setInterval(loadRealTimeData, 5000) : null
    return () => { if (interval) clearInterval(interval) }
  }, [autoRefresh])

  const loadAllData = () => {
    setTimeout(() => {
      setComputeData({
        overview: {
          totalInstances: 234,
          runningInstances: 218,
          stoppedInstances: 16,
          totalCores: 1568,
          totalMemoryGB: 6272,
          avgCpuUtilization: 67.3,
          avgMemoryUtilization: 72.8,
          monthlyBudget: 45678.90,
          currentSpend: 23456.78
        },
        virtualMachines: [
          {
            id: 'vm-001',
            name: 'web-server-01',
            status: 'running',
            size: 'Standard_D4s_v3',
            cores: 4,
            memoryGB: 16,
            location: 'East US',
            resourceGroup: 'web-services-rg',
            os: 'Ubuntu 22.04',
            uptime: '45d 12h 30m',
            cpuUsage: 78.5,
            memoryUsage: 82.3,
            networkIn: 125.6,
            networkOut: 89.4,
            costPerHour: 0.192,
            tags: { environment: 'production', owner: 'web-team' }
          },
          {
            id: 'vm-002',
            name: 'db-server-01',
            status: 'running',
            size: 'Standard_E8s_v3',
            cores: 8,
            memoryGB: 64,
            location: 'East US',
            resourceGroup: 'database-rg',
            os: 'Windows Server 2022',
            uptime: '23d 8h 15m',
            cpuUsage: 45.2,
            memoryUsage: 78.9,
            networkIn: 256.8,
            networkOut: 189.2,
            costPerHour: 0.768,
            tags: { environment: 'production', owner: 'db-team' }
          },
          {
            id: 'vm-003',
            name: 'test-server-01',
            status: 'stopped',
            size: 'Standard_B2s',
            cores: 2,
            memoryGB: 4,
            location: 'West US',
            resourceGroup: 'testing-rg',
            os: 'Ubuntu 20.04',
            uptime: '0d 0h 0m',
            cpuUsage: 0,
            memoryUsage: 0,
            networkIn: 0,
            networkOut: 0,
            costPerHour: 0.048,
            tags: { environment: 'development', owner: 'dev-team' }
          },
          {
            id: 'vm-004',
            name: 'app-server-01',
            status: 'running',
            size: 'Standard_D2s_v3',
            cores: 2,
            memoryGB: 8,
            location: 'East US',
            resourceGroup: 'app-services-rg',
            os: 'CentOS 8',
            uptime: '12d 4h 22m',
            cpuUsage: 89.7,
            memoryUsage: 67.4,
            networkIn: 78.3,
            networkOut: 56.8,
            costPerHour: 0.096,
            tags: { environment: 'production', owner: 'app-team' }
          },
          {
            id: 'vm-005',
            name: 'cache-server-01',
            status: 'running',
            size: 'Standard_D8s_v3',
            cores: 8,
            memoryGB: 32,
            location: 'Central US',
            resourceGroup: 'cache-rg',
            os: 'Redis Enterprise',
            uptime: '67d 15h 45m',
            cpuUsage: 23.4,
            memoryUsage: 91.2,
            networkIn: 345.7,
            networkOut: 298.5,
            costPerHour: 0.384,
            tags: { environment: 'production', owner: 'cache-team' }
          }
        ],
        scaleSets: [
          {
            id: 'vmss-001',
            name: 'web-tier-vmss',
            instanceCount: 8,
            targetCapacity: 10,
            minInstances: 3,
            maxInstances: 20,
            avgCpuUsage: 67.8,
            autoScaleEnabled: true,
            lastScaleEvent: '2 hours ago',
            status: 'healthy'
          },
          {
            id: 'vmss-002',
            name: 'api-tier-vmss',
            instanceCount: 12,
            targetCapacity: 12,
            minInstances: 5,
            maxInstances: 30,
            avgCpuUsage: 78.2,
            autoScaleEnabled: true,
            lastScaleEvent: '45 minutes ago',
            status: 'scaling'
          }
        ],
        containers: {
          registries: 5,
          totalImages: 234,
          runningContainers: 1489,
          totalPods: 2847,
          clusters: 12,
          nodes: 156
        },
        functions: {
          totalApps: 89,
          executions24h: 1234567,
          errors24h: 234,
          avgDuration: 1245,
          successRate: 99.8
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
        cpu: 60 + Math.random() * 30,
        memory: 65 + Math.random() * 25,
        network: 40 + Math.random() * 20,
        instances: 218 + Math.floor(Math.random() * 6)
      }]
      return newData.slice(-20)
    })
  }

  const generateRealTimeData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 60000),
      cpu: 60 + Math.random() * 30,
      memory: 65 + Math.random() * 25,
      network: 40 + Math.random() * 20,
      instances: 218 + Math.floor(Math.random() * 6)
    }))
  }

  const performanceData = {
    labels: realTimeMetrics.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [
      {
        label: 'CPU Usage %',
        data: realTimeMetrics.map(d => d.cpu),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Memory Usage %',
        data: realTimeMetrics.map(d => d.memory),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  }

  const vmDistributionData = {
    labels: ['Running', 'Stopped', 'Deallocated', 'Starting'],
    datasets: [{
      data: [218, 16, 8, 3],
      backgroundColor: [
        'rgba(16, 185, 129, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(156, 163, 175, 0.8)',
        'rgba(245, 158, 11, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const sizeDistributionData = {
    labels: ['Small (B-series)', 'Standard (D-series)', 'Memory Optimized (E-series)', 'Compute Optimized (F-series)'],
    datasets: [{
      data: [45, 134, 67, 23],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(139, 92, 246, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const filteredVMs = computeData?.virtualMachines.filter((vm: any) => {
    const matchesStatus = filterStatus === 'all' || vm.status === filterStatus
    const matchesSearch = vm.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         vm.resourceGroup.toLowerCase().includes(searchTerm.toLowerCase())
    return matchesStatus && matchesSearch
  }) || []

  const handleVMAction = (action: string, vmId: string) => {
    // Simulate VM action
    console.log(`${action} VM: ${vmId}`)
  }

  const handleBulkAction = (action: string) => {
    // Simulate bulk action
    console.log(`${action} VMs:`, selectedVMs)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading Compute Resources...</p>
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
              <Cpu className="w-8 h-8 text-blue-500" />
              <div>
                <h1 className="text-2xl font-bold">Compute Resources</h1>
                <p className="text-sm text-gray-500">Virtual machines, containers, and serverless compute management</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">{computeData.overview.runningInstances} INSTANCES ACTIVE</span>
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
                CREATE INSTANCE
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'virtual-machines', 'scale-sets', 'containers', 'functions'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-3 px-1 border-b-2 transition-colors capitalize ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-500'
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              {tab.replace('-', ' ')}
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
                  <Server className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Total VMs</span>
                </div>
                <p className="text-2xl font-bold font-mono">{computeData.overview.totalInstances}</p>
                <div className="flex items-center mt-1">
                  <CheckCircle className="w-3 h-3 text-green-500 mr-1" />
                  <span className="text-xs text-green-500">{computeData.overview.runningInstances} running</span>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Cpu className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Total Cores</span>
                </div>
                <p className="text-2xl font-bold font-mono">{computeData.overview.totalCores}</p>
                <p className="text-xs text-gray-500 mt-1">{computeData.overview.avgCpuUtilization}% avg usage</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <HardDrive className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Memory</span>
                </div>
                <p className="text-2xl font-bold font-mono">{computeData.overview.totalMemoryGB}GB</p>
                <p className="text-xs text-gray-500 mt-1">{computeData.overview.avgMemoryUtilization}% avg usage</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Container className="w-5 h-5 text-pink-500" />
                  <span className="text-xs text-gray-500">Containers</span>
                </div>
                <p className="text-2xl font-bold font-mono">{computeData.containers.runningContainers}</p>
                <p className="text-xs text-gray-500 mt-1">{computeData.containers.totalPods} pods</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Zap className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Functions</span>
                </div>
                <p className="text-2xl font-bold font-mono">{computeData.functions.totalApps}</p>
                <p className="text-xs text-gray-500 mt-1">{(computeData.functions.executions24h / 1000000).toFixed(1)}M executions</p>
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
                <p className="text-2xl font-bold font-mono">${(computeData.overview.currentSpend / 1000).toFixed(0)}K</p>
                <p className="text-xs text-gray-500 mt-1">of ${(computeData.overview.monthlyBudget / 1000).toFixed(0)}K budget</p>
              </motion.div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {/* Real-time Performance */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">PERFORMANCE METRICS</h3>
                  <div className="flex items-center space-x-2">
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full" />
                      <span className="text-xs text-gray-500">CPU</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      <span className="text-xs text-gray-500">Memory</span>
                    </div>
                  </div>
                </div>
                <div className="h-64">
                  <Line data={performanceData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
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
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } },
                        max: 100
                      }
                    }
                  }} />
                </div>
              </div>

              {/* VM Status Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">VM STATUS DISTRIBUTION</h3>
                <div className="h-64">
                  <Doughnut data={vmDistributionData} options={{
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

              {/* VM Size Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">SIZE DISTRIBUTION</h3>
                <div className="h-64">
                  <Doughnut data={sizeDistributionData} options={{
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

            {/* Scale Sets Status */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg mb-6">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">VIRTUAL MACHINE SCALE SETS</h3>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-2 gap-4">
                  {computeData.scaleSets.map((scaleSet: any) => (
                    <div key={scaleSet.id} className="bg-gray-800 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h4 className="font-semibold">{scaleSet.name}</h4>
                          <p className="text-sm text-gray-400">
                            {scaleSet.instanceCount}/{scaleSet.targetCapacity} instances
                          </p>
                        </div>
                        <span className={`px-2 py-1 text-xs rounded ${
                          scaleSet.status === 'healthy' ? 'bg-green-900/30 text-green-500' :
                          scaleSet.status === 'scaling' ? 'bg-yellow-900/30 text-yellow-500' :
                          'bg-red-900/30 text-red-500'
                        }`}>
                          {scaleSet.status.toUpperCase()}
                        </span>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">CPU Usage</span>
                          <span>{scaleSet.avgCpuUsage}%</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Auto Scale</span>
                          <span className={scaleSet.autoScaleEnabled ? 'text-green-500' : 'text-gray-500'}>
                            {scaleSet.autoScaleEnabled ? 'Enabled' : 'Disabled'}
                          </span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Last Scale Event</span>
                          <span>{scaleSet.lastScaleEvent}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Range</span>
                          <span>{scaleSet.minInstances} - {scaleSet.maxInstances}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'virtual-machines' && (
          <>
            {/* VM Management Controls */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <Search className="w-4 h-4 text-gray-500 absolute left-3 top-1/2 transform -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder="Search VMs..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <select
                    value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value)}
                    className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                  >
                    <option value="all">All Status</option>
                    <option value="running">Running</option>
                    <option value="stopped">Stopped</option>
                    <option value="deallocated">Deallocated</option>
                  </select>
                </div>
                <div className="flex items-center space-x-2">
                  {selectedVMs.length > 0 && (
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-400">{selectedVMs.length} selected</span>
                      <button 
                        onClick={() => handleBulkAction('start')}
                        className="px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white text-sm rounded"
                      >
                        Start
                      </button>
                      <button 
                        onClick={() => handleBulkAction('stop')}
                        className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-sm rounded"
                      >
                        Stop
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

            {/* VM Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left">
                        <input
                          type="checkbox"
                          checked={selectedVMs.length === filteredVMs.length && filteredVMs.length > 0}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedVMs(filteredVMs.map((vm: any) => vm.id))
                            } else {
                              setSelectedVMs([])
                            }
                          }}
                          className="rounded border-gray-600 bg-gray-700 text-blue-600"
                        />
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Name</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Size</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Location</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">CPU</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Memory</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Network</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Cost/Hour</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {filteredVMs.map((vm: any) => (
                      <motion.tr
                        key={vm.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <input
                            type="checkbox"
                            checked={selectedVMs.includes(vm.id)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedVMs([...selectedVMs, vm.id])
                              } else {
                                setSelectedVMs(selectedVMs.filter(id => id !== vm.id))
                              }
                            }}
                            className="rounded border-gray-600 bg-gray-700 text-blue-600"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <div>
                            <div className="font-medium">{vm.name}</div>
                            <div className="text-sm text-gray-400">{vm.resourceGroup}</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            vm.status === 'running' ? 'text-green-500' :
                            vm.status === 'stopped' ? 'text-red-500' :
                            'text-gray-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              vm.status === 'running' ? 'bg-green-500' :
                              vm.status === 'stopped' ? 'bg-red-500' :
                              'bg-gray-500'
                            }`} />
                            <span className="uppercase">{vm.status}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div>{vm.size}</div>
                            <div className="text-gray-400">{vm.cores}C / {vm.memoryGB}GB</div>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm">{vm.location}</td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="flex items-center space-x-1">
                              <span>{vm.cpuUsage.toFixed(1)}%</span>
                              <div className="w-12 bg-gray-800 rounded-full h-1.5">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    vm.cpuUsage > 80 ? 'bg-red-500' :
                                    vm.cpuUsage > 60 ? 'bg-yellow-500' :
                                    'bg-green-500'
                                  }`}
                                  style={{ width: `${Math.min(vm.cpuUsage, 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="flex items-center space-x-1">
                              <span>{vm.memoryUsage.toFixed(1)}%</span>
                              <div className="w-12 bg-gray-800 rounded-full h-1.5">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    vm.memoryUsage > 80 ? 'bg-red-500' :
                                    vm.memoryUsage > 60 ? 'bg-yellow-500' :
                                    'bg-green-500'
                                  }`}
                                  style={{ width: `${Math.min(vm.memoryUsage, 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-xs">
                            <div className="text-green-500">↓ {vm.networkIn} MB/s</div>
                            <div className="text-blue-500">↑ {vm.networkOut} MB/s</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">${vm.costPerHour}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            {vm.status === 'running' ? (
                              <button 
                                onClick={() => handleVMAction('stop', vm.id)}
                                className="p-1 hover:bg-gray-700 rounded text-red-500"
                                title="Stop VM"
                              >
                                <StopCircle className="w-4 h-4" />
                              </button>
                            ) : (
                              <button 
                                onClick={() => handleVMAction('start', vm.id)}
                                className="p-1 hover:bg-gray-700 rounded text-green-500"
                                title="Start VM"
                              >
                                <Play className="w-4 h-4" />
                              </button>
                            )}
                            <button 
                              onClick={() => handleVMAction('restart', vm.id)}
                              className="p-1 hover:bg-gray-700 rounded text-yellow-500"
                              title="Restart VM"
                            >
                              <RotateCcw className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded">
                              <Terminal className="w-4 h-4 text-gray-400" />
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

        {/* Other tabs content would go here */}
        {!['overview', 'virtual-machines'].includes(activeTab) && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-bold mb-4 capitalize">{activeTab.replace('-', ' ')} Management</h2>
            <p className="text-gray-400">Detailed {activeTab.replace('-', ' ')} management interface coming soon...</p>
          </div>
        )}
      </div>
    </div>
  )
}