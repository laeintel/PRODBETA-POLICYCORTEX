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
  Network,
  Globe,
  Router,
  Wifi,
  Shield,
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  Users,
  Zap,
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
  Server,
  Cpu,
  HardDrive,
  Cloud,
  MapPin,
  Link,
  Target,
  Crosshair,
  Navigation,
  Radio,
  Signal
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

export default function NetworkTopologyPage() {
  const [loading, setLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [activeTab, setActiveTab] = useState('overview')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [networkData, setNetworkData] = useState<any>(null)
  const [realTimeMetrics, setRealTimeMetrics] = useState<any[]>([])
  const [selectedDevices, setSelectedDevices] = useState<string[]>([])
  const [filterType, setFilterType] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')
  const [topologyView, setTopologyView] = useState<'logical' | 'physical' | 'geographic'>('logical')

  useEffect(() => {
    loadAllData()
    const interval = autoRefresh ? setInterval(loadRealTimeData, 5000) : null
    return () => { if (interval) clearInterval(interval) }
  }, [autoRefresh])

  const loadAllData = () => {
    setTimeout(() => {
      setNetworkData({
        overview: {
          totalDevices: 156,
          activeDevices: 142,
          inactiveDevices: 14,
          totalBandwidth: '10.5 Gbps',
          usedBandwidth: '6.8 Gbps',
          utilizationPercent: 64.7,
          totalSubnets: 24,
          vlanCount: 45,
          avgLatency: 12.4,
          packetLoss: 0.02,
          monthlyBudget: 12456.78,
          currentSpend: 8765.43
        },
        devices: [
          {
            id: 'DEV-001',
            name: 'Core-Switch-01',
            type: 'switch',
            model: 'Cisco Nexus 9000',
            status: 'active',
            location: 'Data Center A',
            ipAddress: '10.0.1.1',
            subnet: '10.0.0.0/16',
            vlan: 'VLAN-100',
            ports: { total: 48, used: 42, available: 6 },
            utilization: {
              cpu: 23.5,
              memory: 45.2,
              bandwidth: 78.3
            },
            uptime: '245d 18h 32m',
            lastSeen: '2 minutes ago',
            firmware: '9.3.8',
            manufacturer: 'Cisco',
            serialNumber: 'FCH2345A1B2',
            powerConsumption: 145.6,
            temperature: 42.3,
            alerts: 0,
            connections: 42
          },
          {
            id: 'DEV-002',
            name: 'Firewall-01',
            type: 'firewall',
            model: 'Palo Alto PA-3220',
            status: 'active',
            location: 'DMZ',
            ipAddress: '192.168.1.1',
            subnet: '192.168.0.0/16',
            vlan: 'VLAN-200',
            ports: { total: 16, used: 12, available: 4 },
            utilization: {
              cpu: 67.8,
              memory: 78.9,
              bandwidth: 45.2
            },
            uptime: '189d 12h 45m',
            lastSeen: '1 minute ago',
            firmware: '10.1.6',
            manufacturer: 'Palo Alto Networks',
            serialNumber: 'PA3220A1B2C3',
            powerConsumption: 89.3,
            temperature: 38.7,
            alerts: 2,
            connections: 12,
            throughput: '2.3 Gbps',
            sessionsActive: 45678,
            threatsBlocked: 1234
          },
          {
            id: 'DEV-003',
            name: 'Router-Edge-01',
            type: 'router',
            model: 'Juniper MX204',
            status: 'active',
            location: 'Edge Network',
            ipAddress: '203.0.113.1',
            subnet: '203.0.113.0/24',
            vlan: 'VLAN-300',
            ports: { total: 24, used: 18, available: 6 },
            utilization: {
              cpu: 34.6,
              memory: 56.7,
              bandwidth: 89.1
            },
            uptime: '156d 8h 23m',
            lastSeen: '30 seconds ago',
            firmware: '20.4R3.8',
            manufacturer: 'Juniper',
            serialNumber: 'JN1234567890',
            powerConsumption: 234.7,
            temperature: 45.1,
            alerts: 1,
            connections: 18,
            bgpSessions: 12,
            routingTable: 156789
          },
          {
            id: 'DEV-004',
            name: 'WiFi-AP-Floor3-01',
            type: 'access_point',
            model: 'Aruba AP-635',
            status: 'active',
            location: 'Floor 3 - East Wing',
            ipAddress: '172.16.3.101',
            subnet: '172.16.0.0/12',
            vlan: 'VLAN-400',
            ports: { total: 2, used: 1, available: 1 },
            utilization: {
              cpu: 12.3,
              memory: 34.5,
              bandwidth: 67.8
            },
            uptime: '78d 14h 56m',
            lastSeen: '5 seconds ago',
            firmware: '8.10.0.3',
            manufacturer: 'Aruba',
            serialNumber: 'AR635ABC123',
            powerConsumption: 12.4,
            temperature: 32.1,
            alerts: 0,
            connections: 45,
            clientsConnected: 23,
            signalStrength: -45,
            channel: '6 (2.4GHz), 36 (5GHz)'
          },
          {
            id: 'DEV-005',
            name: 'Load-Balancer-01',
            type: 'load_balancer',
            model: 'F5 BigIP 2000s',
            status: 'warning',
            location: 'Data Center B',
            ipAddress: '10.10.10.100',
            subnet: '10.10.0.0/16',
            vlan: 'VLAN-500',
            ports: { total: 8, used: 8, available: 0 },
            utilization: {
              cpu: 89.5,
              memory: 92.3,
              bandwidth: 95.7
            },
            uptime: '123d 6h 12m',
            lastSeen: '10 seconds ago',
            firmware: '15.1.8',
            manufacturer: 'F5',
            serialNumber: 'F5LB123456789',
            powerConsumption: 156.8,
            temperature: 48.9,
            alerts: 3,
            connections: 8,
            virtualServers: 12,
            poolMembers: 24,
            requestsPerSecond: 15678
          },
          {
            id: 'DEV-006',
            name: 'Switch-Floor1-01',
            type: 'switch',
            model: 'HP Aruba 2930F',
            status: 'inactive',
            location: 'Floor 1 - Server Room',
            ipAddress: '172.20.1.10',
            subnet: '172.20.0.0/16',
            vlan: 'VLAN-600',
            ports: { total: 24, used: 0, available: 24 },
            utilization: {
              cpu: 0,
              memory: 0,
              bandwidth: 0
            },
            uptime: '0d 0h 0m',
            lastSeen: '2 hours ago',
            firmware: '16.10.0014',
            manufacturer: 'HPE',
            serialNumber: 'HP2930F789012',
            powerConsumption: 0,
            temperature: 0,
            alerts: 1,
            connections: 0
          }
        ],
        subnets: [
          {
            id: 'SUB-001',
            name: 'Management Network',
            cidr: '10.0.0.0/16',
            vlan: 'VLAN-100',
            gateway: '10.0.0.1',
            dhcpRange: '10.0.10.1 - 10.0.10.254',
            devices: 42,
            utilization: 65.4,
            status: 'active'
          },
          {
            id: 'SUB-002',
            name: 'DMZ Network',
            cidr: '192.168.0.0/16',
            vlan: 'VLAN-200',
            gateway: '192.168.0.1',
            dhcpRange: 'Static Only',
            devices: 12,
            utilization: 23.8,
            status: 'active'
          },
          {
            id: 'SUB-003',
            name: 'Guest Network',
            cidr: '172.16.0.0/12',
            vlan: 'VLAN-400',
            gateway: '172.16.0.1',
            dhcpRange: '172.16.100.1 - 172.16.200.254',
            devices: 78,
            utilization: 45.2,
            status: 'active'
          }
        ],
        flows: [
          {
            id: 'FLOW-001',
            source: '10.0.1.15',
            destination: '192.168.1.100',
            protocol: 'HTTPS',
            port: 443,
            bytesPerSecond: 1048576,
            packetsPerSecond: 1500,
            sessions: 45,
            latency: 12.3,
            jitter: 2.1,
            classification: 'Web Traffic'
          },
          {
            id: 'FLOW-002',
            source: '172.16.50.23',
            destination: '10.10.10.100',
            protocol: 'HTTP',
            port: 80,
            bytesPerSecond: 524288,
            packetsPerSecond: 750,
            sessions: 23,
            latency: 8.7,
            jitter: 1.5,
            classification: 'API Traffic'
          },
          {
            id: 'FLOW-003',
            source: '10.0.5.67',
            destination: '203.0.113.50',
            protocol: 'SSH',
            port: 22,
            bytesPerSecond: 8192,
            packetsPerSecond: 50,
            sessions: 2,
            latency: 156.7,
            jitter: 12.3,
            classification: 'Management Traffic'
          }
        ],
        security: {
          activePolicies: 156,
          blockedConnections: 2345,
          malwareDetected: 12,
          anomaliesDetected: 5,
          vpnSessions: 45,
          encryptedTraffic: 87.5
        },
        performance: {
          avgLatency: 12.4,
          maxLatency: 234.5,
          minLatency: 0.8,
          packetLoss: 0.02,
          jitter: 2.1,
          availability: 99.97,
          mtu: 1500,
          bandwidthEfficiency: 78.9
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
        bandwidth: 6.8 + Math.random() * 2,
        latency: 10 + Math.random() * 5,
        packetLoss: Math.random() * 0.1,
        activeConnections: 1500 + Math.floor(Math.random() * 500)
      }]
      return newData.slice(-20)
    })
  }

  const generateRealTimeData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 60000),
      bandwidth: 6.8 + Math.random() * 2,
      latency: 10 + Math.random() * 5,
      packetLoss: Math.random() * 0.1,
      activeConnections: 1500 + Math.floor(Math.random() * 500)
    }))
  }

  const networkTrafficData = {
    labels: realTimeMetrics.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [
      {
        label: 'Bandwidth (Gbps)',
        data: realTimeMetrics.map(d => d.bandwidth),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true,
        yAxisID: 'y'
      },
      {
        label: 'Latency (ms)',
        data: realTimeMetrics.map(d => d.latency),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
        fill: true,
        yAxisID: 'y1'
      }
    ]
  }

  const deviceTypeData = {
    labels: ['Switches', 'Routers', 'Firewalls', 'Access Points', 'Load Balancers'],
    datasets: [{
      data: [45, 24, 18, 38, 12],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(139, 92, 246, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const bandwidthUtilizationData = {
    labels: networkData?.devices.map((device: any) => device.name) || [],
    datasets: [{
      label: 'Bandwidth Utilization %',
      data: networkData?.devices.map((device: any) => device.utilization.bandwidth) || [],
      backgroundColor: networkData?.devices.map((device: any) => 
        device.utilization.bandwidth > 90 ? 'rgba(239, 68, 68, 0.8)' :
        device.utilization.bandwidth > 70 ? 'rgba(245, 158, 11, 0.8)' :
        'rgba(16, 185, 129, 0.8)'
      ) || [],
      borderWidth: 0
    }]
  }

  const subnetUtilizationData = {
    labels: networkData?.subnets.map((subnet: any) => subnet.name) || [],
    datasets: [{
      label: 'Subnet Utilization %',
      data: networkData?.subnets.map((subnet: any) => subnet.utilization) || [],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const filteredDevices = networkData?.devices.filter((device: any) => {
    const matchesType = filterType === 'all' || device.type === filterType
    const matchesSearch = device.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         device.location.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         device.ipAddress.includes(searchTerm)
    return matchesType && matchesSearch
  }) || []

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': return 'text-green-500 bg-green-900/20'
      case 'inactive': return 'text-red-500 bg-red-900/20'
      case 'warning': return 'text-yellow-500 bg-yellow-900/20'
      case 'maintenance': return 'text-blue-500 bg-blue-900/20'
      default: return 'text-gray-500 bg-gray-900/20'
    }
  }

  const getDeviceIcon = (type: string) => {
    switch(type) {
      case 'switch': return <Network className="w-4 h-4" />
      case 'router': return <Router className="w-4 h-4" />
      case 'firewall': return <Shield className="w-4 h-4" />
      case 'access_point': return <Wifi className="w-4 h-4" />
      case 'load_balancer': return <Layers className="w-4 h-4" />
      default: return <Globe className="w-4 h-4" />
    }
  }

  const handleDeviceAction = (action: string, deviceId: string) => {
    console.log(`${action} device: ${deviceId}`)
  }

  const handleBulkAction = (action: string) => {
    console.log(`${action} devices:`, selectedDevices)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading Network Topology...</p>
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
              <Network className="w-8 h-8 text-blue-500" />
              <div>
                <h1 className="text-2xl font-bold">Network Topology</h1>
                <p className="text-sm text-gray-500">Network infrastructure monitoring and management</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">{networkData.overview.utilizationPercent}% UTILIZATION</span>
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
                CONFIGURE NETWORK
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'devices', 'topology', 'flows', 'security', 'analytics'].map((tab) => (
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
                  <Network className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Total Devices</span>
                </div>
                <p className="text-2xl font-bold font-mono">{networkData.overview.totalDevices}</p>
                <div className="flex items-center mt-1">
                  <CheckCircle className="w-3 h-3 text-green-500 mr-1" />
                  <span className="text-xs text-green-500">{networkData.overview.activeDevices} active</span>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Activity className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Bandwidth</span>
                </div>
                <p className="text-2xl font-bold font-mono">{networkData.overview.usedBandwidth}</p>
                <p className="text-xs text-gray-500 mt-1">of {networkData.overview.totalBandwidth}</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Clock className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Latency</span>
                </div>
                <p className="text-2xl font-bold font-mono">{networkData.overview.avgLatency}ms</p>
                <p className="text-xs text-gray-500 mt-1">average</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Target className="w-5 h-5 text-red-500" />
                  <span className="text-xs text-gray-500">Packet Loss</span>
                </div>
                <p className="text-2xl font-bold font-mono">{networkData.overview.packetLoss}%</p>
                <p className="text-xs text-gray-500 mt-1">last 24h</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Globe className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Subnets</span>
                </div>
                <p className="text-2xl font-bold font-mono">{networkData.overview.totalSubnets}</p>
                <p className="text-xs text-gray-500 mt-1">{networkData.overview.vlanCount} VLANs</p>
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
                <p className="text-2xl font-bold font-mono">${(networkData.overview.currentSpend / 1000).toFixed(1)}K</p>
                <p className="text-xs text-gray-500 mt-1">of ${(networkData.overview.monthlyBudget / 1000).toFixed(1)}K budget</p>
              </motion.div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {/* Network Traffic */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">NETWORK TRAFFIC</h3>
                  <div className="flex items-center space-x-2">
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full" />
                      <span className="text-xs text-gray-500">Bandwidth</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      <span className="text-xs text-gray-500">Latency</span>
                    </div>
                  </div>
                </div>
                <div className="h-64">
                  <Line data={networkTrafficData} options={{
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

              {/* Device Types */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">DEVICE TYPE DISTRIBUTION</h3>
                <div className="h-64">
                  <Doughnut data={deviceTypeData} options={{
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

              {/* Bandwidth Utilization */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">BANDWIDTH UTILIZATION</h3>
                <div className="h-64">
                  <Bar data={bandwidthUtilizationData} options={{
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

            {/* Network Performance Summary */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg mb-6">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">NETWORK PERFORMANCE SUMMARY</h3>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-4 gap-6">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-500 mb-2">{networkData.performance.availability}%</div>
                    <div className="text-xs text-gray-400 uppercase">Availability</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-500 mb-2">{networkData.performance.avgLatency}ms</div>
                    <div className="text-xs text-gray-400 uppercase">Avg Latency</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-500 mb-2">{networkData.performance.packetLoss}%</div>
                    <div className="text-xs text-gray-400 uppercase">Packet Loss</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-500 mb-2">{networkData.performance.bandwidthEfficiency}%</div>
                    <div className="text-xs text-gray-400 uppercase">Efficiency</div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'devices' && (
          <>
            {/* Device Management Controls */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <Search className="w-4 h-4 text-gray-500 absolute left-3 top-1/2 transform -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder="Search devices..."
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
                    <option value="switch">Switches</option>
                    <option value="router">Routers</option>
                    <option value="firewall">Firewalls</option>
                    <option value="access_point">Access Points</option>
                    <option value="load_balancer">Load Balancers</option>
                  </select>
                </div>
                <div className="flex items-center space-x-2">
                  {selectedDevices.length > 0 && (
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-400">{selectedDevices.length} selected</span>
                      <button 
                        onClick={() => handleBulkAction('reboot')}
                        className="px-3 py-1.5 bg-orange-600 hover:bg-orange-700 text-white text-sm rounded"
                      >
                        Reboot
                      </button>
                      <button 
                        onClick={() => handleBulkAction('backup')}
                        className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded"
                      >
                        Backup Config
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

            {/* Devices Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left">
                        <input
                          type="checkbox"
                          checked={selectedDevices.length === filteredDevices.length && filteredDevices.length > 0}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedDevices(filteredDevices.map((device: any) => device.id))
                            } else {
                              setSelectedDevices([])
                            }
                          }}
                          className="rounded border-gray-600 bg-gray-700 text-blue-600"
                        />
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Device</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Type</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Location</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">IP Address</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">CPU</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Memory</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Bandwidth</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Uptime</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {filteredDevices.map((device: any) => (
                      <motion.tr
                        key={device.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <input
                            type="checkbox"
                            checked={selectedDevices.includes(device.id)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedDevices([...selectedDevices, device.id])
                              } else {
                                setSelectedDevices(selectedDevices.filter(id => id !== device.id))
                              }
                            }}
                            className="rounded border-gray-600 bg-gray-700 text-blue-600"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded ${getStatusColor(device.status)}`}>
                              {getDeviceIcon(device.type)}
                            </div>
                            <div>
                              <div className="font-medium">{device.name}</div>
                              <div className="text-sm text-gray-400">{device.model}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="inline-flex items-center px-2 py-1 text-xs rounded bg-blue-900/30 text-blue-500">
                            {device.type.replace('_', ' ').toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            device.status === 'active' ? 'text-green-500' :
                            device.status === 'inactive' ? 'text-red-500' :
                            device.status === 'warning' ? 'text-yellow-500' :
                            'text-gray-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              device.status === 'active' ? 'bg-green-500' :
                              device.status === 'inactive' ? 'bg-red-500' :
                              device.status === 'warning' ? 'bg-yellow-500' :
                              'bg-gray-500'
                            }`} />
                            <span className="uppercase">{device.status}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm">{device.location}</td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="font-mono">{device.ipAddress}</div>
                            <div className="text-gray-400 text-xs">{device.vlan}</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="flex items-center space-x-1">
                              <span>{device.utilization.cpu?.toFixed(1) || 0}%</span>
                              <div className="w-12 bg-gray-800 rounded-full h-1.5">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    device.utilization.cpu > 80 ? 'bg-red-500' :
                                    device.utilization.cpu > 60 ? 'bg-yellow-500' :
                                    'bg-green-500'
                                  }`}
                                  style={{ width: `${Math.min(device.utilization.cpu || 0, 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="flex items-center space-x-1">
                              <span>{device.utilization.memory?.toFixed(1) || 0}%</span>
                              <div className="w-12 bg-gray-800 rounded-full h-1.5">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    device.utilization.memory > 80 ? 'bg-red-500' :
                                    device.utilization.memory > 60 ? 'bg-yellow-500' :
                                    'bg-green-500'
                                  }`}
                                  style={{ width: `${Math.min(device.utilization.memory || 0, 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="flex items-center space-x-1">
                              <span>{device.utilization.bandwidth?.toFixed(1) || 0}%</span>
                              <div className="w-12 bg-gray-800 rounded-full h-1.5">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    device.utilization.bandwidth > 80 ? 'bg-red-500' :
                                    device.utilization.bandwidth > 60 ? 'bg-yellow-500' :
                                    'bg-green-500'
                                  }`}
                                  style={{ width: `${Math.min(device.utilization.bandwidth || 0, 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{device.uptime}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button 
                              onClick={() => handleDeviceAction('configure', device.id)}
                              className="p-1 hover:bg-gray-700 rounded text-blue-500"
                              title="Configure Device"
                            >
                              <Settings className="w-4 h-4" />
                            </button>
                            <button 
                              onClick={() => handleDeviceAction('monitor', device.id)}
                              className="p-1 hover:bg-gray-700 rounded text-green-500"
                              title="Monitor Device"
                            >
                              <Monitor className="w-4 h-4" />
                            </button>
                            <button 
                              onClick={() => handleDeviceAction('reboot', device.id)}
                              className="p-1 hover:bg-gray-700 rounded text-orange-500"
                              title="Reboot Device"
                            >
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

        {activeTab === 'topology' && (
          <>
            {/* Topology View Controls */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">NETWORK TOPOLOGY VIEW</h3>
                  <div className="flex items-center space-x-2 border border-gray-700 rounded">
                    <button 
                      onClick={() => setTopologyView('logical')}
                      className={`px-3 py-1 text-xs rounded-l transition-colors ${
                        topologyView === 'logical' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
                      }`}
                    >
                      LOGICAL
                    </button>
                    <button 
                      onClick={() => setTopologyView('physical')}
                      className={`px-3 py-1 text-xs transition-colors ${
                        topologyView === 'physical' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
                      }`}
                    >
                      PHYSICAL
                    </button>
                    <button 
                      onClick={() => setTopologyView('geographic')}
                      className={`px-3 py-1 text-xs rounded-r transition-colors ${
                        topologyView === 'geographic' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
                      }`}
                    >
                      GEOGRAPHIC
                    </button>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <button className="px-3 py-1 bg-green-600 hover:bg-green-700 text-xs rounded">AUTO LAYOUT</button>
                  <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-xs rounded">SAVE VIEW</button>
                  <button className="p-1 hover:bg-gray-800 rounded">
                    <Download className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </div>
            </div>

            {/* Network Topology Visualization */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg mb-6" style={{ height: '600px' }}>
              <div className="p-4 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">NETWORK {topologyView.toUpperCase()} VIEW</h3>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-green-500 rounded-full" />
                      <span className="text-xs text-gray-400">Active ({networkData.overview.activeDevices})</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full" />
                      <span className="text-xs text-gray-400">Inactive ({networkData.overview.inactiveDevices})</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-yellow-500 rounded-full" />
                      <span className="text-xs text-gray-400">Warning</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="p-4 h-full">
                {/* SVG Network Topology Visualization */}
                <div className="w-full h-full bg-gray-800/30 rounded relative overflow-hidden">
                  <svg className="w-full h-full" viewBox="0 0 1200 500">
                    {/* Network Links */}
                    <defs>
                      <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="rgba(75, 85, 99, 0.6)" />
                      </marker>
                    </defs>
                    
                    {/* Core Switch Connections */}
                    <line x1="600" y1="250" x2="300" y2="150" stroke="rgba(59, 130, 246, 0.8)" strokeWidth="2" markerEnd="url(#arrowhead)" />
                    <line x1="600" y1="250" x2="900" y2="150" stroke="rgba(59, 130, 246, 0.8)" strokeWidth="2" markerEnd="url(#arrowhead)" />
                    <line x1="600" y1="250" x2="150" y2="350" stroke="rgba(16, 185, 129, 0.8)" strokeWidth="2" markerEnd="url(#arrowhead)" />
                    <line x1="600" y1="250" x2="1050" y2="350" stroke="rgba(239, 68, 68, 0.8)" strokeWidth="2" markerEnd="url(#arrowhead)" />
                    <line x1="600" y1="250" x2="450" y2="400" stroke="rgba(245, 158, 11, 0.8)" strokeWidth="2" markerEnd="url(#arrowhead)" />
                    <line x1="600" y1="250" x2="750" y2="400" stroke="rgba(139, 92, 246, 0.8)" strokeWidth="2" markerEnd="url(#arrowhead)" />

                    {/* Core Switch */}
                    <g transform="translate(600, 250)">
                      <circle cx="0" cy="0" r="25" fill="rgba(59, 130, 246, 0.2)" stroke="rgb(59, 130, 246)" strokeWidth="2" />
                      <text x="0" y="5" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">CORE</text>
                      <text x="0" y="45" textAnchor="middle" fill="rgba(255, 255, 255, 0.7)" fontSize="8">Core-Switch-01</text>
                    </g>

                    {/* Firewall */}
                    <g transform="translate(300, 150)">
                      <rect x="-20" y="-15" width="40" height="30" fill="rgba(239, 68, 68, 0.2)" stroke="rgb(239, 68, 68)" strokeWidth="2" rx="3" />
                      <text x="0" y="5" textAnchor="middle" fill="white" fontSize="8" fontWeight="bold">FW</text>
                      <text x="0" y="-25" textAnchor="middle" fill="rgba(255, 255, 255, 0.7)" fontSize="8">Firewall-01</text>
                    </g>

                    {/* Router */}
                    <g transform="translate(900, 150)">
                      <polygon points="-18,0 0,-15 18,0 0,15" fill="rgba(16, 185, 129, 0.2)" stroke="rgb(16, 185, 129)" strokeWidth="2" />
                      <text x="0" y="5" textAnchor="middle" fill="white" fontSize="8" fontWeight="bold">RTR</text>
                      <text x="0" y="-25" textAnchor="middle" fill="rgba(255, 255, 255, 0.7)" fontSize="8">Router-Edge-01</text>
                    </g>

                    {/* Access Point */}
                    <g transform="translate(150, 350)">
                      <circle cx="0" cy="0" r="15" fill="rgba(245, 158, 11, 0.2)" stroke="rgb(245, 158, 11)" strokeWidth="2" />
                      <path d="M-8,-8 Q0,-15 8,-8 Q0,-20 -8,-8" fill="none" stroke="rgb(245, 158, 11)" strokeWidth="1.5" />
                      <text x="0" y="-35" textAnchor="middle" fill="rgba(255, 255, 255, 0.7)" fontSize="8">WiFi-AP-01</text>
                    </g>

                    {/* Load Balancer */}
                    <g transform="translate(1050, 350)">
                      <rect x="-20" y="-12" width="40" height="24" fill="rgba(139, 92, 246, 0.2)" stroke="rgb(139, 92, 246)" strokeWidth="2" rx="5" />
                      <text x="0" y="5" textAnchor="middle" fill="white" fontSize="8" fontWeight="bold">LB</text>
                      <text x="0" y="-25" textAnchor="middle" fill="rgba(255, 255, 255, 0.7)" fontSize="8">Load-Balancer-01</text>
                      <circle cx="25" cy="-8" r="3" fill="rgb(245, 158, 11)" className="animate-pulse" />
                    </g>

                    {/* Edge Switch 1 */}
                    <g transform="translate(450, 400)">
                      <circle cx="0" cy="0" r="18" fill="rgba(16, 185, 129, 0.2)" stroke="rgb(16, 185, 129)" strokeWidth="2" />
                      <text x="0" y="5" textAnchor="middle" fill="white" fontSize="8" fontWeight="bold">SW1</text>
                      <text x="0" y="35" textAnchor="middle" fill="rgba(255, 255, 255, 0.7)" fontSize="8">Switch-Floor1-01</text>
                    </g>

                    {/* Edge Switch 2 (Inactive) */}
                    <g transform="translate(750, 400)">
                      <circle cx="0" cy="0" r="18" fill="rgba(107, 114, 128, 0.2)" stroke="rgb(107, 114, 128)" strokeWidth="2" strokeDasharray="5,5" />
                      <text x="0" y="5" textAnchor="middle" fill="rgba(255, 255, 255, 0.5)" fontSize="8" fontWeight="bold">SW2</text>
                      <text x="0" y="35" textAnchor="middle" fill="rgba(255, 255, 255, 0.5)" fontSize="8">Switch-Floor2-01</text>
                      <circle cx="20" cy="-15" r="3" fill="rgb(239, 68, 68)" className="animate-pulse" />
                    </g>

                    {/* Network Traffic Flow Animations */}
                    <g>
                      <circle r="2" fill="rgb(59, 130, 246)" opacity="0.8">
                        <animateMotion dur="3s" repeatCount="indefinite">
                          <mpath href="#path1" />
                        </animateMotion>
                      </circle>
                      <path id="path1" d="M600,250 L300,150" fill="none" stroke="none" />
                    </g>

                    {/* VLAN Labels */}
                    <g transform="translate(100, 50)">
                      <rect x="0" y="0" width="200" height="80" fill="rgba(0, 0, 0, 0.8)" stroke="rgba(75, 85, 99, 0.6)" rx="5" />
                      <text x="10" y="20" fill="rgba(255, 255, 255, 0.9)" fontSize="12" fontWeight="bold">VLAN Information</text>
                      <text x="10" y="35" fill="rgb(59, 130, 246)" fontSize="9">VLAN-100: Management</text>
                      <text x="10" y="50" fill="rgb(239, 68, 68)" fontSize="9">VLAN-200: DMZ</text>
                      <text x="10" y="65" fill="rgb(245, 158, 11)" fontSize="9">VLAN-400: Guest</text>
                    </g>

                    {/* Performance Indicators */}
                    <g transform="translate(1000, 50)">
                      <rect x="0" y="0" width="180" height="120" fill="rgba(0, 0, 0, 0.8)" stroke="rgba(75, 85, 99, 0.6)" rx="5" />
                      <text x="10" y="20" fill="rgba(255, 255, 255, 0.9)" fontSize="12" fontWeight="bold">Live Metrics</text>
                      <text x="10" y="40" fill="rgba(255, 255, 255, 0.7)" fontSize="9">Bandwidth: {networkData.overview.usedBandwidth}</text>
                      <text x="10" y="55" fill="rgba(255, 255, 255, 0.7)" fontSize="9">Latency: {networkData.overview.avgLatency}ms</text>
                      <text x="10" y="70" fill="rgba(255, 255, 255, 0.7)" fontSize="9">Packet Loss: {networkData.overview.packetLoss}%</text>
                      <text x="10" y="85" fill="rgba(255, 255, 255, 0.7)" fontSize="9">Active Devices: {networkData.overview.activeDevices}</text>
                      <text x="10" y="100" fill="rgba(255, 255, 255, 0.7)" fontSize="9">Utilization: {networkData.overview.utilizationPercent}%</text>
                    </g>
                  </svg>
                </div>
              </div>
            </div>

            {/* Subnet Overview */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">SUBNET CONFIGURATION</h3>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-3 gap-6">
                  {networkData?.subnets.map((subnet: any) => (
                    <motion.div
                      key={subnet.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-medium">{subnet.name}</h4>
                        <span className={`w-3 h-3 rounded-full ${
                          subnet.status === 'active' ? 'bg-green-500' : 'bg-red-500'
                        }`} />
                      </div>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">CIDR:</span>
                          <span className="font-mono">{subnet.cidr}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">VLAN:</span>
                          <span className="font-mono text-blue-500">{subnet.vlan}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Gateway:</span>
                          <span className="font-mono">{subnet.gateway}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Devices:</span>
                          <span className="font-bold">{subnet.devices}</span>
                        </div>
                        <div className="mt-3">
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-gray-400">Utilization</span>
                            <span>{subnet.utilization.toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-800 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full transition-all duration-300 ${
                                subnet.utilization > 80 ? 'bg-red-500' :
                                subnet.utilization > 60 ? 'bg-yellow-500' :
                                'bg-green-500'
                              }`}
                              style={{ width: `${subnet.utilization}%` }}
                            />
                          </div>
                        </div>
                      </div>
                      <div className="mt-4 pt-3 border-t border-gray-800">
                        <div className="text-xs text-gray-400 mb-1">DHCP Range</div>
                        <div className="font-mono text-xs">{subnet.dhcpRange}</div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'flows' && (
          <>
            {/* Traffic Flow Analysis */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">TOP TRAFFIC FLOWS</h3>
                </div>
                <div className="p-4">
                  <div className="space-y-3">
                    {networkData?.flows.map((flow: any) => (
                      <motion.div
                        key={flow.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex items-center justify-between p-3 bg-gray-800/30 rounded"
                      >
                        <div className="flex items-center space-x-3">
                          <div className={`w-3 h-3 rounded-full ${
                            flow.classification === 'Web Traffic' ? 'bg-blue-500' :
                            flow.classification === 'API Traffic' ? 'bg-green-500' :
                            'bg-yellow-500'
                          }`} />
                          <div>
                            <div className="text-sm font-medium">
                              {flow.source} → {flow.destination}
                            </div>
                            <div className="text-xs text-gray-400">
                              {flow.protocol}:{flow.port} - {flow.classification}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-bold">
                            {(flow.bytesPerSecond / 1024 / 1024).toFixed(1)} MB/s
                          </div>
                          <div className="text-xs text-gray-400">
                            {flow.packetsPerSecond.toLocaleString()} pps
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">TRAFFIC CLASSIFICATION</h3>
                </div>
                <div className="p-4">
                  <div className="h-64">
                    <Doughnut data={{
                      labels: ['Web Traffic', 'API Traffic', 'Management Traffic', 'File Transfer', 'Other'],
                      datasets: [{
                        data: [45.2, 28.7, 12.3, 8.9, 4.9],
                        backgroundColor: [
                          'rgba(59, 130, 246, 0.8)',
                          'rgba(16, 185, 129, 0.8)',
                          'rgba(245, 158, 11, 0.8)',
                          'rgba(139, 92, 246, 0.8)',
                          'rgba(107, 114, 128, 0.8)'
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
            </div>

            {/* Flow Details Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">DETAILED FLOW ANALYSIS</h3>
                  <div className="flex items-center space-x-2">
                    <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-xs rounded">EXPORT DATA</button>
                    <button className="px-3 py-1 bg-green-600 hover:bg-green-700 text-xs rounded">CREATE POLICY</button>
                  </div>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Source</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Destination</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Protocol</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Bandwidth</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Sessions</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Latency</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Classification</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {networkData?.flows.map((flow: any) => (
                      <motion.tr
                        key={flow.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{flow.source}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{flow.destination}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="font-medium">{flow.protocol}</div>
                            <div className="text-xs text-gray-400">Port {flow.port}</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="font-mono">{(flow.bytesPerSecond / 1024 / 1024).toFixed(2)} MB/s</div>
                            <div className="text-xs text-gray-400">{flow.packetsPerSecond.toLocaleString()} pps</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{flow.sessions}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className={`font-mono ${
                              flow.latency > 100 ? 'text-red-500' :
                              flow.latency > 50 ? 'text-yellow-500' :
                              'text-green-500'
                            }`}>
                              {flow.latency}ms
                            </div>
                            <div className="text-xs text-gray-400">±{flow.jitter}ms</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center px-2 py-1 text-xs rounded ${
                            flow.classification === 'Web Traffic' ? 'bg-blue-900/30 text-blue-500' :
                            flow.classification === 'API Traffic' ? 'bg-green-900/30 text-green-500' :
                            'bg-yellow-900/30 text-yellow-500'
                          }`}>
                            {flow.classification}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button className="p-1 hover:bg-gray-700 rounded text-blue-500" title="Block Flow">
                              <Shield className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded text-yellow-500" title="Prioritize">
                              <TrendingUp className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded text-green-500" title="Allow">
                              <CheckCircle className="w-4 h-4" />
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

        {activeTab === 'security' && (
          <>
            {/* Security Overview */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Shield className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Active Policies</span>
                </div>
                <p className="text-2xl font-bold font-mono">{networkData.security.activePolicies}</p>
                <p className="text-xs text-green-500 mt-1">All enforced</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <XCircle className="w-5 h-5 text-red-500" />
                  <span className="text-xs text-gray-500">Blocked</span>
                </div>
                <p className="text-2xl font-bold font-mono text-red-500">{networkData.security.blockedConnections.toLocaleString()}</p>
                <p className="text-xs text-gray-500 mt-1">connections</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Anomalies</span>
                </div>
                <p className="text-2xl font-bold font-mono text-yellow-500">{networkData.security.anomaliesDetected}</p>
                <p className="text-xs text-gray-500 mt-1">detected</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Lock className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">VPN Sessions</span>
                </div>
                <p className="text-2xl font-bold font-mono">{networkData.security.vpnSessions}</p>
                <p className="text-xs text-blue-500 mt-1">active</p>
              </motion.div>
            </div>

            {/* Security Policies and Threat Detection */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">SECURITY POLICIES</h3>
                </div>
                <div className="p-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-green-500 rounded-full" />
                        <div>
                          <div className="text-sm font-medium">Intrusion Prevention</div>
                          <div className="text-xs text-gray-400">Block malicious traffic patterns</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-green-500">ACTIVE</span>
                        <button className="text-xs text-blue-500 hover:text-blue-400">EDIT</button>
                      </div>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-blue-500 rounded-full" />
                        <div>
                          <div className="text-sm font-medium">DDoS Protection</div>
                          <div className="text-xs text-gray-400">Rate limiting and traffic shaping</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-blue-500">ACTIVE</span>
                        <button className="text-xs text-blue-500 hover:text-blue-400">EDIT</button>
                      </div>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-yellow-500 rounded-full" />
                        <div>
                          <div className="text-sm font-medium">Geo-blocking</div>
                          <div className="text-xs text-gray-400">Block traffic from specific regions</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-yellow-500">WARNING</span>
                        <button className="text-xs text-blue-500 hover:text-blue-400">EDIT</button>
                      </div>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-red-500 rounded-full" />
                        <div>
                          <div className="text-sm font-medium">Application Control</div>
                          <div className="text-xs text-gray-400">Control application access</div>
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
                  <h3 className="text-sm font-bold text-gray-400 uppercase">THREAT DETECTION</h3>
                </div>
                <div className="p-4">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 bg-red-900/20 border border-red-800/50 rounded">
                      <div className="flex items-center space-x-3">
                        <AlertTriangle className="w-4 h-4 text-red-500" />
                        <div>
                          <div className="text-sm font-medium text-red-400">Port Scan Detected</div>
                          <div className="text-xs text-gray-400">Source: 203.0.113.45</div>
                          <div className="text-xs text-gray-500">5 minutes ago</div>
                        </div>
                      </div>
                      <div className="text-xs text-red-500 font-medium">BLOCKED</div>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-yellow-900/20 border border-yellow-800/50 rounded">
                      <div className="flex items-center space-x-3">
                        <AlertCircle className="w-4 h-4 text-yellow-500" />
                        <div>
                          <div className="text-sm font-medium text-yellow-400">Unusual Traffic Pattern</div>
                          <div className="text-xs text-gray-400">Subnet: 10.0.5.0/24</div>
                          <div className="text-xs text-gray-500">12 minutes ago</div>
                        </div>
                      </div>
                      <div className="text-xs text-yellow-500 font-medium">MONITORING</div>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-orange-900/20 border border-orange-800/50 rounded">
                      <div className="flex items-center space-x-3">
                        <AlertTriangle className="w-4 h-4 text-orange-500" />
                        <div>
                          <div className="text-sm font-medium text-orange-400">Malware Communication</div>
                          <div className="text-xs text-gray-400">Destination: C&C Server</div>
                          <div className="text-xs text-gray-500">18 minutes ago</div>
                        </div>
                      </div>
                      <div className="text-xs text-orange-500 font-medium">QUARANTINED</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Network Access Control */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">NETWORK ACCESS CONTROL</h3>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-3 gap-6">
                  <div>
                    <div className="text-xs text-gray-400 mb-3 uppercase">Access Control Lists</div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Allow Rules</span>
                        <span className="text-sm text-green-500">1,245</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Deny Rules</span>
                        <span className="text-sm text-red-500">89</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Custom Rules</span>
                        <span className="text-sm text-blue-500">156</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-400 mb-3 uppercase">Traffic Encryption</div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Encrypted Traffic</span>
                        <span className="text-sm text-green-500">{networkData.security.encryptedTraffic}%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">TLS/SSL Inspection</span>
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Certificate Validation</span>
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-400 mb-3 uppercase">Monitoring Status</div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Network Sensors</span>
                        <span className="text-sm text-green-500">12/12</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Log Collection</span>
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Real-time Analysis</span>
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'analytics' && (
          <>
            {/* Network Analytics Dashboard */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-500 mb-2">{networkData.performance.availability}%</div>
                  <div className="text-xs text-gray-400 uppercase mb-4">Network Availability</div>
                  <div className="h-16">
                    <Line data={{
                      labels: Array.from({ length: 24 }, (_, i) => `${i}h`),
                      datasets: [{
                        data: Array.from({ length: 24 }, () => 99.5 + Math.random() * 0.5),
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
                        y: { display: false, min: 99, max: 100 }
                      },
                      elements: { point: { radius: 0 } }
                    }} />
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-500 mb-2">{networkData.performance.avgLatency}ms</div>
                  <div className="text-xs text-gray-400 uppercase mb-4">Average Latency</div>
                  <div className="h-16">
                    <Line data={{
                      labels: Array.from({ length: 24 }, (_, i) => `${i}h`),
                      datasets: [{
                        data: Array.from({ length: 24 }, () => 10 + Math.random() * 10),
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
                  <div className="text-3xl font-bold text-purple-500 mb-2">{networkData.performance.bandwidthEfficiency}%</div>
                  <div className="text-xs text-gray-400 uppercase mb-4">Bandwidth Efficiency</div>
                  <div className="h-16">
                    <Line data={{
                      labels: Array.from({ length: 24 }, (_, i) => `${i}h`),
                      datasets: [{
                        data: Array.from({ length: 24 }, () => 75 + Math.random() * 15),
                        borderColor: 'rgb(139, 92, 246)',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
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

            {/* Advanced Analytics */}
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">TRAFFIC PATTERN ANALYSIS</h3>
                </div>
                <div className="p-4">
                  <div className="h-64">
                    <Scatter data={{
                      datasets: [{
                        label: 'Normal Traffic',
                        data: Array.from({ length: 50 }, () => ({
                          x: Math.random() * 100,
                          y: Math.random() * 100
                        })),
                        backgroundColor: 'rgba(59, 130, 246, 0.6)'
                      }, {
                        label: 'Anomalous Traffic',
                        data: Array.from({ length: 5 }, () => ({
                          x: 80 + Math.random() * 20,
                          y: 80 + Math.random() * 20
                        })),
                        backgroundColor: 'rgba(239, 68, 68, 0.8)'
                      }]
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
                          ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } },
                          title: { display: true, text: 'Bandwidth Usage', color: 'rgba(255, 255, 255, 0.7)' }
                        },
                        y: {
                          grid: { color: 'rgba(255, 255, 255, 0.05)' },
                          ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } },
                          title: { display: true, text: 'Connection Count', color: 'rgba(255, 255, 255, 0.7)' }
                        }
                      }
                    }} />
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">PERFORMANCE INSIGHTS</h3>
                </div>
                <div className="p-4">
                  <div className="space-y-4">
                    <div className="p-3 bg-blue-900/20 border border-blue-800/50 rounded">
                      <div className="flex items-center space-x-2 mb-2">
                        <TrendingUp className="w-4 h-4 text-blue-500" />
                        <span className="text-sm font-medium text-blue-400">Bandwidth Optimization</span>
                      </div>
                      <div className="text-xs text-gray-300">
                        Traffic compression can reduce bandwidth usage by up to 23% during peak hours.
                      </div>
                      <button className="mt-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 text-xs rounded">
                        IMPLEMENT
                      </button>
                    </div>

                    <div className="p-3 bg-green-900/20 border border-green-800/50 rounded">
                      <div className="flex items-center space-x-2 mb-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span className="text-sm font-medium text-green-400">Load Balancing</span>
                      </div>
                      <div className="text-xs text-gray-300">
                        Current load balancing configuration is optimal for traffic distribution.
                      </div>
                    </div>

                    <div className="p-3 bg-yellow-900/20 border border-yellow-800/50 rounded">
                      <div className="flex items-center space-x-2 mb-2">
                        <AlertTriangle className="w-4 h-4 text-yellow-500" />
                        <span className="text-sm font-medium text-yellow-400">QoS Configuration</span>
                      </div>
                      <div className="text-xs text-gray-300">
                        Consider updating QoS policies for better performance during peak usage.
                      </div>
                      <button className="mt-2 px-3 py-1 bg-yellow-600 hover:bg-yellow-700 text-xs rounded">
                        REVIEW
                      </button>
                    </div>

                    <div className="p-3 bg-purple-900/20 border border-purple-800/50 rounded">
                      <div className="flex items-center space-x-2 mb-2">
                        <Network className="w-4 h-4 text-purple-500" />
                        <span className="text-sm font-medium text-purple-400">Redundancy Check</span>
                      </div>
                      <div className="text-xs text-gray-300">
                        All critical network paths have redundant connections configured.
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