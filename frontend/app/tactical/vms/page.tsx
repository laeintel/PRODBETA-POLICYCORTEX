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
  Monitor,
  Server,
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
  Settings,
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
  DollarSign,
  AlertCircle,
  XCircle,
  ChevronRight,
  ExternalLink,
  Terminal,
  Database,
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
  Cloud,
  Globe,
  MapPin,
  Link,
  Target,
  Crosshair,
  Navigation,
  Radio,
  Signal,
  Wifi,
  Shield,
  Network,
  Package,
  Copy
} from 'lucide-react'
import { Line, Bar, Doughnut, Scatter } from 'react-chartjs-2'
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
  Filler,
  ScatterController
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
  Filler,
  ScatterController
)

export default function VirtualMachinesPage() {
  const [loading, setLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [activeTab, setActiveTab] = useState('overview')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [vmData, setVmData] = useState<any>(null)
  const [realTimeMetrics, setRealTimeMetrics] = useState<any[]>([])
  const [selectedVMs, setSelectedVMs] = useState<string[]>([])
  const [filterStatus, setFilterStatus] = useState('all')
  const [filterSize, setFilterSize] = useState('all')
  const [filterEnvironment, setFilterEnvironment] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    loadAllData()
    const interval = autoRefresh ? setInterval(loadRealTimeData, 5000) : null
    return () => { if (interval) clearInterval(interval) }
  }, [autoRefresh])

  const loadAllData = () => {
    setTimeout(() => {
      setVmData({
        overview: {
          totalVMs: 147,
          runningVMs: 134,
          stoppedVMs: 8,
          deallocatedVMs: 5,
          totalCores: 1248,
          totalMemoryGB: 9856,
          totalDiskGB: 156780,
          avgCpuUtilization: 72.4,
          avgMemoryUtilization: 68.9,
          avgDiskUtilization: 45.6,
          monthlyBudget: 78945.50,
          currentSpend: 52678.90,
          totalSnapshots: 456,
          totalBackups: 789
        },
        vms: [
          {
            id: 'VM-001',
            name: 'prod-web-01',
            resourceGroup: 'rg-prod-web',
            subscription: 'Production',
            size: 'Standard_D4s_v3',
            cores: 4,
            memoryGB: 16,
            diskGB: 128,
            os: 'Ubuntu 20.04 LTS',
            status: 'running',
            powerState: 'VM running',
            environment: 'production',
            region: 'East US',
            availabilityZone: '1',
            privateIP: '10.0.1.15',
            publicIP: '52.188.45.123',
            fqdn: 'prod-web-01.eastus.cloudapp.azure.com',
            cpuUtilization: 78.5,
            memoryUtilization: 82.3,
            diskUtilization: 65.4,
            networkIn: 125.6,
            networkOut: 89.4,
            costPerHour: 0.192,
            created: '2024-01-15',
            lastStarted: '2024-12-18T08:30:00Z',
            uptime: '2d 14h 32m',
            osType: 'Linux',
            adminUsername: 'azureuser',
            extensions: ['CustomScript', 'OmsAgentForLinux'],
            tags: { Environment: 'Production', Application: 'WebServer', Owner: 'DevOps' },
            monitoringAgent: true,
            backupEnabled: true,
            lastBackup: '2024-12-18T02:00:00Z',
            snapshotCount: 3
          },
          {
            id: 'VM-002',
            name: 'prod-api-01',
            resourceGroup: 'rg-prod-api',
            subscription: 'Production',
            size: 'Standard_D8s_v3',
            cores: 8,
            memoryGB: 32,
            diskGB: 256,
            os: 'Windows Server 2022',
            status: 'running',
            powerState: 'VM running',
            environment: 'production',
            region: 'West US',
            availabilityZone: '2',
            privateIP: '10.0.2.25',
            publicIP: '40.112.78.45',
            fqdn: 'prod-api-01.westus.cloudapp.azure.com',
            cpuUtilization: 65.2,
            memoryUtilization: 75.8,
            diskUtilization: 58.7,
            networkIn: 98.3,
            networkOut: 156.2,
            costPerHour: 0.384,
            created: '2024-02-01',
            lastStarted: '2024-12-17T09:15:00Z',
            uptime: '1d 15h 45m',
            osType: 'Windows',
            adminUsername: 'vmadmin',
            extensions: ['IaaSDiagnostics', 'MicrosoftMonitoringAgent'],
            tags: { Environment: 'Production', Application: 'API', Owner: 'Backend' },
            monitoringAgent: true,
            backupEnabled: true,
            lastBackup: '2024-12-18T01:00:00Z',
            snapshotCount: 5
          },
          {
            id: 'VM-003',
            name: 'staging-app-01',
            resourceGroup: 'rg-staging',
            subscription: 'Non-Production',
            size: 'Standard_B2s',
            cores: 2,
            memoryGB: 4,
            diskGB: 64,
            os: 'Ubuntu 22.04 LTS',
            status: 'running',
            powerState: 'VM running',
            environment: 'staging',
            region: 'Central US',
            availabilityZone: '1',
            privateIP: '10.1.1.10',
            publicIP: null,
            fqdn: null,
            cpuUtilization: 45.6,
            memoryUtilization: 52.3,
            diskUtilization: 38.9,
            networkIn: 34.7,
            networkOut: 28.5,
            costPerHour: 0.0416,
            created: '2024-03-15',
            lastStarted: '2024-12-18T07:00:00Z',
            uptime: '3h 30m',
            osType: 'Linux',
            adminUsername: 'staginguser',
            extensions: ['CustomScript'],
            tags: { Environment: 'Staging', Application: 'TestApp', Owner: 'QA' },
            monitoringAgent: false,
            backupEnabled: false,
            lastBackup: null,
            snapshotCount: 1
          },
          {
            id: 'VM-004',
            name: 'dev-build-01',
            resourceGroup: 'rg-development',
            subscription: 'Development',
            size: 'Standard_D2s_v3',
            cores: 2,
            memoryGB: 8,
            diskGB: 128,
            os: 'Ubuntu 22.04 LTS',
            status: 'stopped',
            powerState: 'VM stopped (deallocated)',
            environment: 'development',
            region: 'East US 2',
            availabilityZone: null,
            privateIP: '10.2.1.5',
            publicIP: null,
            fqdn: null,
            cpuUtilization: 0,
            memoryUtilization: 0,
            diskUtilization: 22.1,
            networkIn: 0,
            networkOut: 0,
            costPerHour: 0,
            created: '2024-04-10',
            lastStarted: '2024-12-17T16:30:00Z',
            uptime: '0m',
            osType: 'Linux',
            adminUsername: 'devuser',
            extensions: ['Docker', 'CustomScript'],
            tags: { Environment: 'Development', Application: 'BuildAgent', Owner: 'DevOps' },
            monitoringAgent: false,
            backupEnabled: false,
            lastBackup: null,
            snapshotCount: 0
          },
          {
            id: 'VM-005',
            name: 'ml-gpu-01',
            resourceGroup: 'rg-ml-workloads',
            subscription: 'Production',
            size: 'Standard_NC6s_v3',
            cores: 6,
            memoryGB: 112,
            diskGB: 736,
            os: 'Ubuntu 20.04 LTS',
            status: 'running',
            powerState: 'VM running',
            environment: 'production',
            region: 'North Central US',
            availabilityZone: '3',
            privateIP: '10.3.1.100',
            publicIP: '13.89.123.45',
            fqdn: 'ml-gpu-01.northcentralus.cloudapp.azure.com',
            cpuUtilization: 89.7,
            memoryUtilization: 67.4,
            diskUtilization: 78.9,
            networkIn: 234.7,
            networkOut: 189.3,
            costPerHour: 3.168,
            created: '2024-05-20',
            lastStarted: '2024-12-18T06:00:00Z',
            uptime: '4h 30m',
            osType: 'Linux',
            adminUsername: 'mluser',
            extensions: ['NvidiaGpuDriverLinux', 'CustomScript'],
            tags: { Environment: 'Production', Application: 'MachineLearning', Owner: 'DataScience' },
            monitoringAgent: true,
            backupEnabled: true,
            lastBackup: '2024-12-18T03:00:00Z',
            snapshotCount: 2,
            gpuCount: 1,
            gpuType: 'Tesla V100'
          },
          {
            id: 'VM-006',
            name: 'test-db-01',
            resourceGroup: 'rg-test',
            subscription: 'Non-Production',
            size: 'Standard_E4s_v3',
            cores: 4,
            memoryGB: 32,
            diskGB: 512,
            os: 'Windows Server 2019',
            status: 'stopped',
            powerState: 'VM stopped (deallocated)',
            environment: 'test',
            region: 'South Central US',
            availabilityZone: null,
            privateIP: '10.4.1.50',
            publicIP: null,
            fqdn: null,
            cpuUtilization: 0,
            memoryUtilization: 0,
            diskUtilization: 45.6,
            networkIn: 0,
            networkOut: 0,
            costPerHour: 0,
            created: '2024-06-01',
            lastStarted: '2024-12-16T14:00:00Z',
            uptime: '0m',
            osType: 'Windows',
            adminUsername: 'testadmin',
            extensions: ['SqlIaaSExtension', 'IaaSDiagnostics'],
            tags: { Environment: 'Test', Application: 'Database', Owner: 'QA' },
            monitoringAgent: false,
            backupEnabled: false,
            lastBackup: null,
            snapshotCount: 1
          }
        ],
        resourceGroups: [
          { name: 'rg-prod-web', vmCount: 12, region: 'East US', totalCores: 48, totalMemoryGB: 192 },
          { name: 'rg-prod-api', vmCount: 8, region: 'West US', totalCores: 64, totalMemoryGB: 256 },
          { name: 'rg-staging', vmCount: 6, region: 'Central US', totalCores: 12, totalMemoryGB: 24 },
          { name: 'rg-development', vmCount: 15, region: 'East US 2', totalCores: 30, totalMemoryGB: 120 },
          { name: 'rg-ml-workloads', vmCount: 4, region: 'North Central US', totalCores: 24, totalMemoryGB: 448 },
          { name: 'rg-test', vmCount: 8, region: 'South Central US', totalCores: 32, totalMemoryGB: 128 }
        ],
        vmSizes: [
          { name: 'Standard_B2s', count: 23, cores: 2, memoryGB: 4, costPerHour: 0.0416 },
          { name: 'Standard_D2s_v3', count: 18, cores: 2, memoryGB: 8, costPerHour: 0.096 },
          { name: 'Standard_D4s_v3', count: 25, cores: 4, memoryGB: 16, costPerHour: 0.192 },
          { name: 'Standard_D8s_v3', count: 15, cores: 8, memoryGB: 32, costPerHour: 0.384 },
          { name: 'Standard_E4s_v3', count: 12, cores: 4, memoryGB: 32, costPerHour: 0.252 },
          { name: 'Standard_NC6s_v3', count: 3, cores: 6, memoryGB: 112, costPerHour: 3.168, hasGPU: true }
        ],
        snapshots: [
          {
            id: 'SNAP-001',
            name: 'prod-web-01_snapshot_20241218',
            vmId: 'VM-001',
            vmName: 'prod-web-01',
            sizeGB: 128,
            created: '2024-12-18T02:00:00Z',
            status: 'succeeded',
            type: 'automatic'
          },
          {
            id: 'SNAP-002',
            name: 'prod-api-01_manual_snapshot',
            vmId: 'VM-002',
            vmName: 'prod-api-01',
            sizeGB: 256,
            created: '2024-12-17T18:30:00Z',
            status: 'succeeded',
            type: 'manual'
          }
        ],
        events: [
          {
            id: 'EVT-001',
            type: 'Information',
            vmId: 'VM-001',
            vmName: 'prod-web-01',
            message: 'VM started successfully',
            timestamp: '2024-12-18T08:30:00Z',
            category: 'PowerState'
          },
          {
            id: 'EVT-002',
            type: 'Warning',
            vmId: 'VM-005',
            vmName: 'ml-gpu-01',
            message: 'High CPU utilization detected (89.7%)',
            timestamp: '2024-12-18T10:15:00Z',
            category: 'Performance'
          },
          {
            id: 'EVT-003',
            type: 'Information',
            vmId: 'VM-004',
            vmName: 'dev-build-01',
            message: 'VM stopped (deallocated)',
            timestamp: '2024-12-17T18:00:00Z',
            category: 'PowerState'
          }
        ]
      })

      setRealTimeMetrics(generateRealTimeData())
      setLoading(false)
    }, 1000)
  }

  const loadRealTimeData = () => {
    setRealTimeMetrics(prev => {
      const newData = [...prev, {
        timestamp: new Date(),
        cpuUsage: 70 + Math.random() * 20,
        memoryUsage: 65 + Math.random() * 25,
        diskUsage: 40 + Math.random() * 20,
        networkIn: 50 + Math.random() * 100,
        networkOut: 45 + Math.random() * 90
      }]
      return newData.slice(-20)
    })
  }

  const generateRealTimeData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 60000),
      cpuUsage: 70 + Math.random() * 20,
      memoryUsage: 65 + Math.random() * 25,
      diskUsage: 40 + Math.random() * 20,
      networkIn: 50 + Math.random() * 100,
      networkOut: 45 + Math.random() * 90
    }))
  }

  const performanceData = {
    labels: realTimeMetrics.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [
      {
        label: 'CPU %',
        data: realTimeMetrics.map(d => d.cpuUsage),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Memory %',
        data: realTimeMetrics.map(d => d.memoryUsage),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Disk %',
        data: realTimeMetrics.map(d => d.diskUsage),
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  }

  const vmStatusData = {
    labels: ['Running', 'Stopped', 'Deallocated', 'Starting'],
    datasets: [{
      data: [vmData?.overview.runningVMs || 0, vmData?.overview.stoppedVMs || 0, vmData?.overview.deallocatedVMs || 0, 0],
      backgroundColor: [
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(156, 163, 175, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const vmSizeDistributionData = {
    labels: vmData?.vmSizes.map((size: any) => size.name) || [],
    datasets: [{
      label: 'VM Count',
      data: vmData?.vmSizes.map((size: any) => size.count) || [],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(139, 92, 246, 0.8)',
        'rgba(236, 72, 153, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const costByResourceGroupData = {
    labels: vmData?.resourceGroups.map((rg: any) => rg.name) || [],
    datasets: [{
      label: 'Estimated Monthly Cost ($)',
      data: vmData?.resourceGroups.map((rg: any) => rg.vmCount * 150) || [], // Mock calculation
      backgroundColor: vmData?.resourceGroups.map((_: any, index: number) => 
        index % 2 === 0 ? 'rgba(59, 130, 246, 0.8)' : 'rgba(16, 185, 129, 0.8)'
      ) || [],
      borderWidth: 0
    }]
  }

  const filteredVMs = vmData?.vms.filter((vm: any) => {
    const matchesStatus = filterStatus === 'all' || vm.status === filterStatus
    const matchesSize = filterSize === 'all' || vm.size === filterSize
    const matchesEnvironment = filterEnvironment === 'all' || vm.environment === filterEnvironment
    const matchesSearch = vm.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         vm.resourceGroup.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         vm.privateIP.includes(searchTerm)
    return matchesStatus && matchesSize && matchesEnvironment && matchesSearch
  }) || []

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'running': return 'text-green-500 bg-green-900/20'
      case 'stopped': return 'text-yellow-500 bg-yellow-900/20'
      case 'deallocated': return 'text-red-500 bg-red-900/20'
      case 'starting': return 'text-blue-500 bg-blue-900/20'
      default: return 'text-gray-500 bg-gray-900/20'
    }
  }

  const getEnvironmentColor = (env: string) => {
    switch(env) {
      case 'production': return 'text-red-500 bg-red-900/20'
      case 'staging': return 'text-yellow-500 bg-yellow-900/20'
      case 'development': return 'text-green-500 bg-green-900/20'
      case 'test': return 'text-blue-500 bg-blue-900/20'
      default: return 'text-gray-500 bg-gray-900/20'
    }
  }

  const getOSIcon = (os: string) => {
    if (os.includes('Windows')) {
      return <Monitor className="w-4 h-4 text-blue-500" />
    } else {
      return <Terminal className="w-4 h-4 text-orange-500" />
    }
  }

  const handleVMAction = (action: string, vmId: string) => {
    console.log(`${action} VM: ${vmId}`)
  }

  const handleBulkAction = (action: string) => {
    console.log(`${action} VMs:`, selectedVMs)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading Virtual Machines...</p>
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
              <Monitor className="w-8 h-8 text-blue-500" />
              <div>
                <h1 className="text-2xl font-bold">Virtual Machines</h1>
                <p className="text-sm text-gray-500">Azure virtual machine management and monitoring</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">{vmData.overview.runningVMs}/{vmData.overview.totalVMs} RUNNING</span>
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
                CREATE VM
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'virtual-machines', 'resource-groups', 'snapshots', 'backups', 'maintenance', 'events'].map((tab) => (
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
                  <Monitor className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Total VMs</span>
                </div>
                <p className="text-2xl font-bold font-mono">{vmData.overview.totalVMs}</p>
                <div className="flex items-center mt-1">
                  <CheckCircle className="w-3 h-3 text-green-500 mr-1" />
                  <span className="text-xs text-green-500">{vmData.overview.runningVMs} running</span>
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
                <p className="text-2xl font-bold font-mono">{vmData.overview.totalCores}</p>
                <p className="text-xs text-gray-500 mt-1">{vmData.overview.avgCpuUtilization}% avg usage</p>
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
                <p className="text-2xl font-bold font-mono">{(vmData.overview.totalMemoryGB / 1024).toFixed(1)}TB</p>
                <p className="text-xs text-gray-500 mt-1">{vmData.overview.avgMemoryUtilization}% avg usage</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Database className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Storage</span>
                </div>
                <p className="text-2xl font-bold font-mono">{(vmData.overview.totalDiskGB / 1024).toFixed(1)}TB</p>
                <p className="text-xs text-gray-500 mt-1">{vmData.overview.avgDiskUtilization}% avg usage</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Archive className="w-5 h-5 text-indigo-500" />
                  <span className="text-xs text-gray-500">Backups</span>
                </div>
                <p className="text-2xl font-bold font-mono">{vmData.overview.totalBackups}</p>
                <p className="text-xs text-gray-500 mt-1">{vmData.overview.totalSnapshots} snapshots</p>
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
                <p className="text-2xl font-bold font-mono">${(vmData.overview.currentSpend / 1000).toFixed(1)}K</p>
                <p className="text-xs text-gray-500 mt-1">of ${(vmData.overview.monthlyBudget / 1000).toFixed(1)}K budget</p>
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
                      <span className="text-xs text-gray-500">CPU</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      <span className="text-xs text-gray-500">Memory</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-yellow-500 rounded-full" />
                      <span className="text-xs text-gray-500">Disk</span>
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
                  <Doughnut data={vmStatusData} options={{
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
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">VM SIZE DISTRIBUTION</h3>
                <div className="h-64">
                  <Bar data={vmSizeDistributionData} options={{
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
                        ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
                      }
                    }
                  }} />
                </div>
              </div>
            </div>

            {/* Cost by Resource Group */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg mb-6">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">COST BY RESOURCE GROUP</h3>
              </div>
              <div className="p-4">
                <div className="h-64">
                  <Bar data={costByResourceGroupData} options={{
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
                        ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
                      }
                    }
                  }} />
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
                  <select
                    value={filterSize}
                    onChange={(e) => setFilterSize(e.target.value)}
                    className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                  >
                    <option value="all">All Sizes</option>
                    <option value="Standard_B2s">B2s</option>
                    <option value="Standard_D2s_v3">D2s_v3</option>
                    <option value="Standard_D4s_v3">D4s_v3</option>
                    <option value="Standard_D8s_v3">D8s_v3</option>
                  </select>
                  <select
                    value={filterEnvironment}
                    onChange={(e) => setFilterEnvironment(e.target.value)}
                    className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                  >
                    <option value="all">All Environments</option>
                    <option value="production">Production</option>
                    <option value="staging">Staging</option>
                    <option value="development">Development</option>
                    <option value="test">Test</option>
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
                      <button 
                        onClick={() => handleBulkAction('restart')}
                        className="px-3 py-1.5 bg-orange-600 hover:bg-orange-700 text-white text-sm rounded"
                      >
                        Restart
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

            {/* VMs Table */}
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
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Virtual Machine</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Environment</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Size</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">CPU</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Memory</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Disk</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Uptime</th>
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
                          <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded ${getStatusColor(vm.status)}`}>
                              {getOSIcon(vm.os)}
                            </div>
                            <div>
                              <div className="font-medium">{vm.name}</div>
                              <div className="text-sm text-gray-400">{vm.resourceGroup}</div>
                              <div className="text-xs text-gray-500">{vm.privateIP}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            vm.status === 'running' ? 'text-green-500' :
                            vm.status === 'stopped' ? 'text-yellow-500' :
                            vm.status === 'deallocated' ? 'text-red-500' :
                            'text-gray-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              vm.status === 'running' ? 'bg-green-500' :
                              vm.status === 'stopped' ? 'bg-yellow-500' :
                              vm.status === 'deallocated' ? 'bg-red-500' :
                              'bg-gray-500'
                            }`} />
                            <span className="uppercase">{vm.status}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center px-2 py-1 text-xs rounded ${getEnvironmentColor(vm.environment)}`}>
                            {vm.environment.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="font-medium">{vm.size.replace('Standard_', '')}</div>
                            <div className="text-gray-400 text-xs">{vm.cores}C/{vm.memoryGB}GB</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="flex items-center space-x-1">
                              <span>{vm.cpuUtilization?.toFixed(1) || 0}%</span>
                              <div className="w-12 bg-gray-800 rounded-full h-1.5">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    vm.cpuUtilization > 80 ? 'bg-red-500' :
                                    vm.cpuUtilization > 60 ? 'bg-yellow-500' :
                                    'bg-green-500'
                                  }`}
                                  style={{ width: `${Math.min(vm.cpuUtilization || 0, 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="flex items-center space-x-1">
                              <span>{vm.memoryUtilization?.toFixed(1) || 0}%</span>
                              <div className="w-12 bg-gray-800 rounded-full h-1.5">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    vm.memoryUtilization > 80 ? 'bg-red-500' :
                                    vm.memoryUtilization > 60 ? 'bg-yellow-500' :
                                    'bg-green-500'
                                  }`}
                                  style={{ width: `${Math.min(vm.memoryUtilization || 0, 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="flex items-center space-x-1">
                              <span>{vm.diskUtilization?.toFixed(1) || 0}%</span>
                              <div className="w-12 bg-gray-800 rounded-full h-1.5">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    vm.diskUtilization > 80 ? 'bg-red-500' :
                                    vm.diskUtilization > 60 ? 'bg-yellow-500' :
                                    'bg-green-500'
                                  }`}
                                  style={{ width: `${Math.min(vm.diskUtilization || 0, 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{vm.uptime}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">${vm.costPerHour}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            {vm.status === 'running' && (
                              <>
                                <button 
                                  onClick={() => handleVMAction('stop', vm.id)}
                                  className="p-1 hover:bg-gray-700 rounded text-red-500"
                                  title="Stop VM"
                                >
                                  <StopCircle className="w-4 h-4" />
                                </button>
                                <button 
                                  onClick={() => handleVMAction('restart', vm.id)}
                                  className="p-1 hover:bg-gray-700 rounded text-orange-500"
                                  title="Restart VM"
                                >
                                  <RotateCcw className="w-4 h-4" />
                                </button>
                              </>
                            )}
                            {(vm.status === 'stopped' || vm.status === 'deallocated') && (
                              <button 
                                onClick={() => handleVMAction('start', vm.id)}
                                className="p-1 hover:bg-gray-700 rounded text-green-500"
                                title="Start VM"
                              >
                                <Play className="w-4 h-4" />
                              </button>
                            )}
                            <button 
                              onClick={() => handleVMAction('configure', vm.id)}
                              className="p-1 hover:bg-gray-700 rounded text-blue-500"
                              title="Configure VM"
                            >
                              <Settings className="w-4 h-4" />
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

        {activeTab === 'resource-groups' && (
          <>
            {/* Resource Groups Management */}
            <div className="grid grid-cols-3 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Folder className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Total Groups</span>
                </div>
                <p className="text-2xl font-bold font-mono">{vmData.resourceGroups.length}</p>
                <p className="text-xs text-gray-500 mt-1">resource groups</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Monitor className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Total VMs</span>
                </div>
                <p className="text-2xl font-bold font-mono">{vmData.resourceGroups.reduce((sum: any, rg: any) => sum + rg.vmCount, 0)}</p>
                <p className="text-xs text-gray-500 mt-1">across all groups</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Cpu className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Total Cores</span>
                </div>
                <p className="text-2xl font-bold font-mono">{vmData.resourceGroups.reduce((sum: any, rg: any) => sum + rg.totalCores, 0)}</p>
                <p className="text-xs text-gray-500 mt-1">allocated cores</p>
              </motion.div>
            </div>

            {/* Resource Groups Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">Resource Groups</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Group Name</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Region</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">VM Count</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Total Cores</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Total Memory</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Est. Monthly Cost</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {vmData.resourceGroups.map((rg: any, index: number) => (
                      <motion.tr
                        key={rg.name}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: index * 0.1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-3">
                            <div className="bg-blue-900/20 text-blue-500 p-2 rounded">
                              <Folder className="w-4 h-4" />
                            </div>
                            <div>
                              <div className="font-medium">{rg.name}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm">{rg.region}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{rg.vmCount}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{rg.totalCores}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{rg.totalMemoryGB}GB</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">${(rg.vmCount * 150).toLocaleString()}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button className="p-1 hover:bg-gray-700 rounded text-blue-500" title="View VMs">
                              <Eye className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded text-yellow-500" title="Manage">
                              <Settings className="w-4 h-4" />
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

        {activeTab === 'snapshots' && (
          <>
            {/* Snapshots Management */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Image className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Total Snapshots</span>
                </div>
                <p className="text-2xl font-bold font-mono">{vmData.snapshots.length}</p>
                <p className="text-xs text-gray-500 mt-1">disk snapshots</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Successful</span>
                </div>
                <p className="text-2xl font-bold font-mono">{vmData.snapshots.filter((s: any) => s.status === 'succeeded').length}</p>
                <p className="text-xs text-gray-500 mt-1">completed snapshots</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <HardDrive className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Total Size</span>
                </div>
                <p className="text-2xl font-bold font-mono">{(vmData.snapshots.reduce((sum: any, s: any) => sum + s.sizeGB, 0) / 1024).toFixed(1)}TB</p>
                <p className="text-xs text-gray-500 mt-1">storage used</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Clock className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Automatic</span>
                </div>
                <p className="text-2xl font-bold font-mono">{vmData.snapshots.filter((s: any) => s.type === 'automatic').length}</p>
                <p className="text-xs text-gray-500 mt-1">automated snapshots</p>
              </motion.div>
            </div>

            {/* Snapshots Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">VM Snapshots</h3>
                  <button className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded">
                    Create Snapshot
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Snapshot Name</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Source VM</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Size</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Type</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Created</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {vmData.snapshots.map((snapshot: any) => (
                      <motion.tr
                        key={snapshot.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-3">
                            <div className="bg-purple-900/20 text-purple-500 p-2 rounded">
                              <Image className="w-4 h-4" />
                            </div>
                            <div>
                              <div className="font-medium">{snapshot.name}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm">{snapshot.vmName}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{snapshot.sizeGB}GB</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center px-2 py-1 text-xs rounded ${
                            snapshot.type === 'automatic' ? 'bg-blue-900/20 text-blue-500' : 'bg-green-900/20 text-green-500'
                          }`}>
                            {snapshot.type}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm">{new Date(snapshot.created).toLocaleDateString()}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            snapshot.status === 'succeeded' ? 'text-green-500' : 'text-gray-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              snapshot.status === 'succeeded' ? 'bg-green-500' : 'bg-gray-500'
                            }`} />
                            <span className="uppercase">{snapshot.status}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button className="p-1 hover:bg-gray-700 rounded text-green-500" title="Restore">
                              <RotateCcw className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded text-blue-500" title="Clone">
                              <Copy className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded text-red-500" title="Delete">
                              <XCircle className="w-4 h-4" />
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

        {activeTab === 'backups' && (
          <>
            {/* Backups Management */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Archive className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Protected VMs</span>
                </div>
                <p className="text-2xl font-bold font-mono">{vmData.vms.filter((vm: any) => vm.backupEnabled).length}</p>
                <p className="text-xs text-gray-500 mt-1">of {vmData.overview.totalVMs} total</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Recent Backups</span>
                </div>
                <p className="text-2xl font-bold font-mono">{vmData.vms.filter((vm: any) => vm.lastBackup && new Date(vm.lastBackup) > new Date(Date.now() - 24*60*60*1000)).length}</p>
                <p className="text-xs text-gray-500 mt-1">in last 24h</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Unprotected</span>
                </div>
                <p className="text-2xl font-bold font-mono">{vmData.vms.filter((vm: any) => !vm.backupEnabled).length}</p>
                <p className="text-xs text-gray-500 mt-1">VMs without backup</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <DollarSign className="w-5 h-5 text-orange-500" />
                  <span className="text-xs text-gray-500">Backup Cost</span>
                </div>
                <p className="text-2xl font-bold font-mono">$1.2K</p>
                <p className="text-xs text-gray-500 mt-1">estimated monthly</p>
              </motion.div>
            </div>

            {/* Backup Status Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">VM Backup Status</h3>
                  <div className="flex items-center space-x-2">
                    <button className="px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white text-sm rounded">
                      Run Backup Now
                    </button>
                    <button className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded">
                      Configure Policy
                    </button>
                  </div>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">VM Name</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Backup Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Last Backup</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Next Backup</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Retention</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Backup Size</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {vmData.vms.map((vm: any) => (
                      <motion.tr
                        key={vm.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded ${vm.backupEnabled ? 'bg-green-900/20 text-green-500' : 'bg-red-900/20 text-red-500'}`}>
                              <Archive className="w-4 h-4" />
                            </div>
                            <div>
                              <div className="font-medium">{vm.name}</div>
                              <div className="text-sm text-gray-400">{vm.resourceGroup}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            vm.backupEnabled ? 'text-green-500' : 'text-red-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              vm.backupEnabled ? 'bg-green-500' : 'bg-red-500'
                            }`} />
                            <span>{vm.backupEnabled ? 'ENABLED' : 'DISABLED'}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          {vm.lastBackup ? (
                            <span className="text-sm">{new Date(vm.lastBackup).toLocaleString()}</span>
                          ) : (
                            <span className="text-sm text-gray-500">Never</span>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          {vm.backupEnabled ? (
                            <span className="text-sm">Today 2:00 AM</span>
                          ) : (
                            <span className="text-sm text-gray-500">Not scheduled</span>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm">30 days</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{Math.round(vm.diskGB * 0.6)}GB</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            {vm.backupEnabled && (
                              <button className="p-1 hover:bg-gray-700 rounded text-green-500" title="Backup Now">
                                <Play className="w-4 h-4" />
                              </button>
                            )}
                            <button className="p-1 hover:bg-gray-700 rounded text-blue-500" title="Configure">
                              <Settings className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded text-yellow-500" title="Restore">
                              <RotateCcw className="w-4 h-4" />
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

        {activeTab === 'maintenance' && (
          <>
            {/* Maintenance Management */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Settings className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Scheduled</span>
                </div>
                <p className="text-2xl font-bold font-mono">12</p>
                <p className="text-xs text-gray-500 mt-1">maintenance windows</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Shield className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Updates Available</span>
                </div>
                <p className="text-2xl font-bold font-mono">23</p>
                <p className="text-xs text-gray-500 mt-1">security patches</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Clock className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Next Window</span>
                </div>
                <p className="text-2xl font-bold font-mono">2h</p>
                <p className="text-xs text-gray-500 mt-1">until maintenance</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Up to Date</span>
                </div>
                <p className="text-2xl font-bold font-mono">{vmData.vms.filter((vm: any) => vm.status === 'running').length - 5}</p>
                <p className="text-xs text-gray-500 mt-1">VMs current</p>
              </motion.div>
            </div>

            {/* Maintenance Schedule */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg mb-6">
              <div className="p-4 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">Maintenance Schedule</h3>
                  <button className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded">
                    Schedule Maintenance
                  </button>
                </div>
              </div>
              <div className="p-4">
                <div className="space-y-4">
                  {[
                    { time: '2024-12-19 02:00', vms: ['prod-web-01', 'prod-api-01'], type: 'Security Updates', duration: '30 mins' },
                    { time: '2024-12-20 03:00', vms: ['staging-app-01'], type: 'OS Patching', duration: '45 mins' },
                    { time: '2024-12-21 01:00', vms: ['ml-gpu-01'], type: 'Driver Updates', duration: '20 mins' }
                  ].map((maintenance, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="flex items-center justify-between p-4 bg-gray-800/30 rounded-lg"
                    >
                      <div className="flex items-center space-x-4">
                        <div className="bg-blue-900/20 text-blue-500 p-2 rounded">
                          <Clock className="w-5 h-5" />
                        </div>
                        <div>
                          <div className="font-medium">{maintenance.type}</div>
                          <div className="text-sm text-gray-400">{maintenance.time} ({maintenance.duration})</div>
                          <div className="text-xs text-gray-500">VMs: {maintenance.vms.join(', ')}</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <button className="p-1 hover:bg-gray-700 rounded text-blue-500" title="Edit">
                          <Settings className="w-4 h-4" />
                        </button>
                        <button className="p-1 hover:bg-gray-700 rounded text-red-500" title="Cancel">
                          <XCircle className="w-4 h-4" />
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'events' && (
          <>
            {/* Events Management */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">VM Events</h3>
                  <div className="flex items-center space-x-2">
                    <button className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-white text-sm rounded">
                      <RefreshCw className="w-4 h-4 mr-2 inline" />
                      Refresh
                    </button>
                    <select className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm">
                      <option value="all">All Types</option>
                      <option value="Information">Information</option>
                      <option value="Warning">Warning</option>
                      <option value="Error">Error</option>
                    </select>
                  </div>
                </div>
              </div>
              <div className="max-h-96 overflow-y-auto">
                <div className="space-y-2 p-4">
                  {vmData.events.map((event: any) => (
                    <motion.div
                      key={event.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="flex items-start space-x-3 p-3 bg-gray-800/30 rounded-lg hover:bg-gray-800/50 transition-colors"
                    >
                      <div className={`flex-shrink-0 w-2 h-2 rounded-full mt-2 ${
                        event.type === 'Information' ? 'bg-green-500' :
                        event.type === 'Warning' ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium text-white">{event.vmName}</p>
                          <p className="text-xs text-gray-500">{new Date(event.timestamp).toLocaleString()}</p>
                        </div>
                        <p className="text-sm text-gray-400 mt-1">{event.message}</p>
                        <div className="flex items-center mt-2 space-x-2">
                          <span className="text-xs text-gray-500">Category: {event.category}</span>
                          <span className={`text-xs px-2 py-1 rounded ${
                            event.type === 'Information' ? 'bg-green-900/20 text-green-500' :
                            event.type === 'Warning' ? 'bg-yellow-900/20 text-yellow-500' :
                            'bg-red-900/20 text-red-500'
                          }`}>
                            {event.type}
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}

        {/* Other tabs content would go here */}
        {!['overview', 'virtual-machines', 'resource-groups', 'snapshots', 'backups', 'maintenance', 'events'].includes(activeTab) && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-bold mb-4 capitalize">{activeTab.replace('-', ' ')} Management</h2>
            <p className="text-gray-400">Detailed {activeTab.replace('-', ' ')} management interface coming soon...</p>
          </div>
        )}
      </div>
    </div>
  )
}