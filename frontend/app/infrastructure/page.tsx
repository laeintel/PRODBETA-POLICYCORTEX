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
  Server,
  Database,
  Network,
  Cloud,
  Cpu,
  HardDrive,
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  Users,
  Zap,
  Shield,
  Globe,
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
  Wifi,
  WifiOff,
  Container,
  Layers,
  GitBranch,
  Code,
  Package,
  Truck,
  Route,
  MapPin,
  Lock,
  Unlock,
  Key,
  UserCheck,
  FileCheck,
  AlertCircle,
  XCircle,
  ChevronRight,
  ExternalLink,
  Maximize2,
  Minimize2
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

export default function InfrastructureOverviewPage() {
  const [loading, setLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [activeTab, setActiveTab] = useState('overview')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [infrastructureData, setInfrastructureData] = useState<any>(null)
  const [realTimeMetrics, setRealTimeMetrics] = useState<any[]>([])
  const [healthData, setHealthData] = useState<any>(null)
  const [capacityData, setCapacityData] = useState<any>(null)

  useEffect(() => {
    loadAllData()
    const interval = autoRefresh ? setInterval(loadRealTimeData, 5000) : null
    return () => { if (interval) clearInterval(interval) }
  }, [autoRefresh])

  const loadAllData = () => {
    setTimeout(() => {
      setInfrastructureData({
        overview: {
          totalResources: 2847,
          activeResources: 2784,
          regions: 8,
          subscriptions: 3,
          resourceGroups: 156,
          healthScore: 94.8,
          costPerHour: 2456.78,
          utilizationAvg: 73.2
        },
        compute: {
          virtualMachines: { total: 234, running: 218, stopped: 16 },
          containers: { total: 1567, running: 1489, failed: 12, pending: 66 },
          kubernetes: { clusters: 12, nodes: 156, pods: 2847, services: 234 },
          functions: { total: 89, active: 84, errors: 2 }
        },
        storage: {
          totalCapacity: '47.8 TB',
          usedCapacity: '31.2 TB',
          availableCapacity: '16.6 TB',
          utilizationPercent: 65.2,
          storageAccounts: 45,
          blobs: 234567,
          files: 12345,
          queues: 67
        },
        network: {
          virtualNetworks: 34,
          subnets: 156,
          securityGroups: 89,
          loadBalancers: 23,
          publicIPs: 78,
          privateEndpoints: 134,
          expressRouteCircuits: 4,
          vpnGateways: 8
        },
        databases: {
          sqlDatabases: 45,
          cosmosCollections: 123,
          redisInstances: 12,
          postgresInstances: 8,
          mysqlInstances: 15,
          elasticClusters: 6
        }
      })

      setHealthData({
        overallHealth: 94.8,
        compute: 96.2,
        storage: 92.1,
        network: 97.5,
        databases: 93.8,
        security: 95.4
      })

      setCapacityData({
        compute: { used: 73, available: 27 },
        storage: { used: 65, available: 35 },
        network: { used: 45, available: 55 },
        memory: { used: 68, available: 32 },
        bandwidth: { used: 34, available: 66 }
      })

      setRealTimeMetrics(generateRealTimeData())
      setLoading(false)
    }, 800)
  }

  const loadRealTimeData = () => {
    setRealTimeMetrics(prev => {
      const newData = [...prev, {
        timestamp: new Date(),
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        network: Math.random() * 100,
        storage: Math.random() * 100
      }]
      return newData.slice(-20)
    })
  }

  const generateRealTimeData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 60000),
      cpu: Math.random() * 100,
      memory: Math.random() * 100,
      network: Math.random() * 100,
      storage: Math.random() * 100
    }))
  }

  const resourceDistributionData = {
    labels: ['Compute', 'Storage', 'Network', 'Databases', 'Analytics', 'Security'],
    datasets: [{
      data: [834, 456, 267, 189, 123, 67],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(139, 92, 246, 0.8)',
        'rgba(236, 72, 153, 0.8)',
        'rgba(239, 68, 68, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const performanceData = {
    labels: realTimeMetrics.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [
      {
        label: 'CPU Usage',
        data: realTimeMetrics.map(d => d.cpu),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Memory Usage',
        data: realTimeMetrics.map(d => d.memory),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Network I/O',
        data: realTimeMetrics.map(d => d.network),
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  }

  const capacityUtilizationData = {
    labels: ['Compute', 'Storage', 'Network', 'Memory', 'Bandwidth'],
    datasets: [{
      label: 'Used %',
      data: [73, 65, 45, 68, 34],
      backgroundColor: 'rgba(59, 130, 246, 0.8)',
      borderColor: 'rgb(59, 130, 246)',
      borderWidth: 1
    }]
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading Infrastructure Overview...</p>
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
              <Server className="w-8 h-8 text-blue-500" />
              <div>
                <h1 className="text-2xl font-bold">Infrastructure Overview</h1>
                <p className="text-sm text-gray-500">Comprehensive cloud infrastructure monitoring and management</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">ALL SYSTEMS OPERATIONAL</span>
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
                RESOURCE MANAGEMENT
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'compute', 'storage', 'network', 'databases', 'monitoring'].map((tab) => (
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
            {/* Infrastructure Health Score */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 mb-6">
              <div className="grid grid-cols-4 gap-6">
                <div>
                  <h2 className="text-xl font-bold mb-4">Infrastructure Health Score</h2>
                  <div className="flex items-center space-x-4">
                    <div className="relative w-32 h-32">
                      <svg className="w-full h-full transform -rotate-90">
                        <circle cx="64" cy="64" r="56" stroke="rgba(255,255,255,0.1)" strokeWidth="8" fill="none" />
                        <circle
                          cx="64" cy="64" r="56"
                          stroke="url(#healthGradient)"
                          strokeWidth="8"
                          fill="none"
                          strokeDasharray={`${2 * Math.PI * 56}`}
                          strokeDashoffset={`${2 * Math.PI * 56 * (1 - infrastructureData.overview.healthScore / 100)}`}
                          className="transition-all duration-1000"
                        />
                        <defs>
                          <linearGradient id="healthGradient">
                            <stop offset="0%" stopColor="#10b981" />
                            <stop offset="100%" stopColor="#3b82f6" />
                          </linearGradient>
                        </defs>
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="text-3xl font-bold">{infrastructureData.overview.healthScore}%</div>
                          <div className="text-xs text-gray-500">Health</div>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <TrendingUp className="w-4 h-4 text-green-500" />
                        <span className="text-sm text-green-500">+2.3% from last week</span>
                      </div>
                      <div className="text-sm text-gray-400">
                        <div>Last Check: 5 minutes ago</div>
                        <div>Next Scan: In 10 minutes</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="border-l border-gray-800 pl-6">
                  <h3 className="text-sm font-semibold text-gray-400 mb-3">RESOURCE SUMMARY</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Total Resources</span>
                      <span className="text-sm font-mono">{infrastructureData.overview.totalResources}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Active Resources</span>
                      <span className="text-sm font-mono text-green-500">{infrastructureData.overview.activeResources}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Regions</span>
                      <span className="text-sm font-mono">{infrastructureData.overview.regions}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Resource Groups</span>
                      <span className="text-sm font-mono">{infrastructureData.overview.resourceGroups}</span>
                    </div>
                  </div>
                </div>

                <div className="border-l border-gray-800 pl-6">
                  <h3 className="text-sm font-semibold text-gray-400 mb-3">COST & UTILIZATION</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Cost/Hour</span>
                      <span className="text-sm font-mono">${infrastructureData.overview.costPerHour}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Utilization</span>
                      <span className="text-sm font-mono">{infrastructureData.overview.utilizationAvg}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Efficiency Rating</span>
                      <span className="text-sm font-mono text-green-500">Optimal</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Waste Potential</span>
                      <span className="text-sm font-mono text-yellow-500">$567/day</span>
                    </div>
                  </div>
                </div>

                <div className="border-l border-gray-800 pl-6">
                  <h3 className="text-sm font-semibold text-gray-400 mb-3">QUICK ACTIONS</h3>
                  <div className="space-y-2">
                    <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group">
                      <span>Scale Resources</span>
                      <TrendingUp className="w-4 h-4 text-gray-500 group-hover:text-white" />
                    </button>
                    <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group">
                      <span>Cost Optimization</span>
                      <BarChart3 className="w-4 h-4 text-gray-500 group-hover:text-white" />
                    </button>
                    <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group">
                      <span>Health Check</span>
                      <Activity className="w-4 h-4 text-gray-500 group-hover:text-white" />
                    </button>
                    <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group">
                      <span>Generate Report</span>
                      <Download className="w-4 h-4 text-gray-500 group-hover:text-white" />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Key Metrics Grid */}
            <div className="grid grid-cols-6 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Cpu className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Compute</span>
                </div>
                <p className="text-2xl font-bold font-mono">{infrastructureData.compute.virtualMachines.total}</p>
                <p className="text-xs text-gray-500 mt-1">virtual machines</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <HardDrive className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Storage</span>
                </div>
                <p className="text-2xl font-bold font-mono">{infrastructureData.storage.totalCapacity}</p>
                <p className="text-xs text-gray-500 mt-1">total capacity</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Network className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Network</span>
                </div>
                <p className="text-2xl font-bold font-mono">{infrastructureData.network.virtualNetworks}</p>
                <p className="text-xs text-gray-500 mt-1">virtual networks</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Database className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Databases</span>
                </div>
                <p className="text-2xl font-bold font-mono">{infrastructureData.databases.sqlDatabases}</p>
                <p className="text-xs text-gray-500 mt-1">SQL databases</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Container className="w-5 h-5 text-pink-500" />
                  <span className="text-xs text-gray-500">Containers</span>
                </div>
                <p className="text-2xl font-bold font-mono">{infrastructureData.compute.containers.total}</p>
                <p className="text-xs text-gray-500 mt-1">containers</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Layers className="w-5 h-5 text-orange-500" />
                  <span className="text-xs text-gray-500">K8s Pods</span>
                </div>
                <p className="text-2xl font-bold font-mono">{infrastructureData.compute.kubernetes.pods}</p>
                <p className="text-xs text-gray-500 mt-1">active pods</p>
              </motion.div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {/* Resource Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">RESOURCE DISTRIBUTION</h3>
                <div className="h-64">
                  <Doughnut data={resourceDistributionData} options={{
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
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-yellow-500 rounded-full" />
                      <span className="text-xs text-gray-500">Network</span>
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

              {/* Capacity Utilization */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">CAPACITY UTILIZATION</h3>
                <div className="h-64">
                  <Bar data={capacityUtilizationData} options={{
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

            {/* Infrastructure Services Status */}
            <div className="grid grid-cols-4 gap-6">
              {/* Compute Services */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">COMPUTE SERVICES</h3>
                </div>
                <div className="p-4 space-y-3">
                  {[
                    { name: 'Virtual Machines', count: infrastructureData.compute.virtualMachines.running, total: infrastructureData.compute.virtualMachines.total, status: 'healthy', icon: Server },
                    { name: 'App Services', count: 45, total: 48, status: 'healthy', icon: Cloud },
                    { name: 'Function Apps', count: infrastructureData.compute.functions.active, total: infrastructureData.compute.functions.total, status: 'warning', icon: Zap },
                    { name: 'Container Instances', count: infrastructureData.compute.containers.running, total: infrastructureData.compute.containers.total, status: 'healthy', icon: Container }
                  ].map((service, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 bg-gray-800 rounded hover:bg-gray-700 transition-colors">
                      <div className="flex items-center space-x-2">
                        <service.icon className="w-4 h-4 text-gray-500" />
                        <span className="text-sm">{service.name}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-500">{service.count}/{service.total}</span>
                        <div className={`w-2 h-2 rounded-full ${
                          service.status === 'healthy' ? 'bg-green-500' : 
                          service.status === 'warning' ? 'bg-yellow-500' : 
                          'bg-red-500'
                        }`} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Storage Services */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">STORAGE SERVICES</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-400">Total Capacity</span>
                      <span className="font-mono text-sm">{infrastructureData.storage.totalCapacity}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-400">Used</span>
                      <span className="font-mono text-sm text-blue-500">{infrastructureData.storage.usedCapacity}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-400">Available</span>
                      <span className="font-mono text-sm text-green-500">{infrastructureData.storage.availableCapacity}</span>
                    </div>
                  </div>
                  <div className="pt-2 border-t border-gray-800">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-400">Utilization</span>
                      <span className="text-sm">{infrastructureData.storage.utilizationPercent}%</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full">
                      <div className="h-2 bg-blue-500 rounded-full" style={{ width: `${infrastructureData.storage.utilizationPercent}%` }} />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 pt-2 border-t border-gray-800">
                    <div className="text-center p-2 bg-gray-800 rounded">
                      <p className="text-xs text-gray-400">Storage Accounts</p>
                      <p className="font-mono">{infrastructureData.storage.storageAccounts}</p>
                    </div>
                    <div className="text-center p-2 bg-gray-800 rounded">
                      <p className="text-xs text-gray-400">Blob Objects</p>
                      <p className="font-mono">{infrastructureData.storage.blobs.toLocaleString()}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Network Services */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">NETWORK SERVICES</h3>
                </div>
                <div className="p-4 space-y-3">
                  {[
                    { name: 'Virtual Networks', count: infrastructureData.network.virtualNetworks, icon: Network },
                    { name: 'Load Balancers', count: infrastructureData.network.loadBalancers, icon: Route },
                    { name: 'Security Groups', count: infrastructureData.network.securityGroups, icon: Shield },
                    { name: 'Public IPs', count: infrastructureData.network.publicIPs, icon: Globe },
                    { name: 'VPN Gateways', count: infrastructureData.network.vpnGateways, icon: Lock },
                    { name: 'ExpressRoute', count: infrastructureData.network.expressRouteCircuits, icon: Zap }
                  ].map((service, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 bg-gray-800 rounded hover:bg-gray-700 transition-colors">
                      <div className="flex items-center space-x-2">
                        <service.icon className="w-4 h-4 text-gray-500" />
                        <span className="text-sm">{service.name}</span>
                      </div>
                      <span className="text-sm font-mono">{service.count}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Database Services */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">DATABASE SERVICES</h3>
                </div>
                <div className="p-4 space-y-3">
                  {[
                    { name: 'SQL Databases', count: infrastructureData.databases.sqlDatabases, health: 95, icon: Database },
                    { name: 'Cosmos DB', count: infrastructureData.databases.cosmosCollections, health: 98, icon: Globe },
                    { name: 'Redis Cache', count: infrastructureData.databases.redisInstances, health: 92, icon: Zap },
                    { name: 'PostgreSQL', count: infrastructureData.databases.postgresInstances, health: 94, icon: Database },
                    { name: 'MySQL', count: infrastructureData.databases.mysqlInstances, health: 96, icon: Database },
                    { name: 'Elasticsearch', count: infrastructureData.databases.elasticClusters, health: 89, icon: Search }
                  ].map((service, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 bg-gray-800 rounded hover:bg-gray-700 transition-colors">
                      <div className="flex items-center space-x-2">
                        <service.icon className="w-4 h-4 text-gray-500" />
                        <span className="text-sm">{service.name}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-500">{service.count}</span>
                        <div className={`w-2 h-2 rounded-full ${
                          service.health > 95 ? 'bg-green-500' : 
                          service.health > 90 ? 'bg-yellow-500' : 
                          'bg-red-500'
                        }`} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}

        {/* Other tabs content would go here */}
        {activeTab !== 'overview' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-bold mb-4 capitalize">{activeTab} Management</h2>
            <p className="text-gray-400">Detailed {activeTab} management interface coming soon...</p>
          </div>
        )}
      </div>
    </div>
  )
}