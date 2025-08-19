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
  Database,
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
  Monitor
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

export default function DatabaseServersPage() {
  const [loading, setLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [activeTab, setActiveTab] = useState('overview')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [databaseData, setDatabaseData] = useState<any>(null)
  const [realTimeMetrics, setRealTimeMetrics] = useState<any[]>([])
  const [selectedDatabases, setSelectedDatabases] = useState<string[]>([])
  const [filterType, setFilterType] = useState('all')
  const [filterStatus, setFilterStatus] = useState('all')
  const [filterEnvironment, setFilterEnvironment] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    loadAllData()
    const interval = autoRefresh ? setInterval(loadRealTimeData, 5000) : null
    return () => { if (interval) clearInterval(interval) }
  }, [autoRefresh])

  const loadAllData = () => {
    setTimeout(() => {
      setDatabaseData({
        overview: {
          totalDatabases: 34,
          onlineDatabases: 32,
          offlineDatabases: 2,
          totalStorage: 45.6, // TB
          usedStorage: 32.8, // TB
          totalConnections: 2847,
          activeConnections: 1923,
          avgCpuUtilization: 67.3,
          avgMemoryUtilization: 74.8,
          avgStorageUtilization: 72.1,
          monthlyBudget: 89456.50,
          currentSpend: 67234.78,
          totalBackups: 156,
          lastBackupHours: 2
        },
        databases: [
          {
            id: 'DB-001',
            name: 'prod-customer-db',
            type: 'Azure SQL Database',
            engine: 'SQL Server',
            version: '15.0.2000',
            status: 'online',
            environment: 'production',
            tier: 'Premium',
            computeSize: 'P4',
            maxSizeGB: 1024,
            usedSizeGB: 756,
            storageUsedPercent: 73.8,
            region: 'East US',
            server: 'prod-sql-server-01',
            resourceGroup: 'rg-prod-databases',
            elasticPool: null,
            cpuUtilization: 78.5,
            memoryUtilization: 82.3,
            diskUtilization: 65.4,
            ioUtilization: 45.2,
            activeConnections: 245,
            maxConnections: 800,
            dtuUsage: 78.2,
            dtuLimit: 125,
            costPerHour: 4.68,
            created: '2024-01-15',
            lastBackup: '2024-12-18T02:00:00Z',
            backupRetention: '35 days',
            encryption: 'TDE enabled',
            firewallRules: 5,
            privateEndpoint: true,
            tags: { Environment: 'Production', Application: 'CustomerDB', Owner: 'DataTeam' },
            collation: 'SQL_Latin1_General_CP1_CI_AS',
            readReplicas: 2,
            geoReplication: true,
            threats: 0,
            vulnerabilities: 0
          },
          {
            id: 'DB-002',
            name: 'prod-analytics-db',
            type: 'Azure SQL Database',
            engine: 'SQL Server',
            version: '15.0.2000',
            status: 'online',
            environment: 'production',
            tier: 'Business Critical',
            computeSize: 'BC_Gen5_8',
            maxSizeGB: 2048,
            usedSizeGB: 1456,
            storageUsedPercent: 71.1,
            region: 'West US',
            server: 'prod-sql-server-02',
            resourceGroup: 'rg-prod-analytics',
            elasticPool: null,
            cpuUtilization: 65.2,
            memoryUtilization: 75.8,
            diskUtilization: 58.7,
            ioUtilization: 67.3,
            activeConnections: 156,
            maxConnections: 1600,
            dtuUsage: null,
            dtuLimit: null,
            vCores: 8,
            costPerHour: 8.96,
            created: '2024-02-01',
            lastBackup: '2024-12-18T01:30:00Z',
            backupRetention: '35 days',
            encryption: 'TDE enabled',
            firewallRules: 8,
            privateEndpoint: true,
            tags: { Environment: 'Production', Application: 'Analytics', Owner: 'DataTeam' },
            collation: 'SQL_Latin1_General_CP1_CI_AS',
            readReplicas: 3,
            geoReplication: true,
            threats: 0,
            vulnerabilities: 1
          },
          {
            id: 'DB-003',
            name: 'prod-redis-cache',
            type: 'Azure Cache for Redis',
            engine: 'Redis',
            version: '6.2.7',
            status: 'online',
            environment: 'production',
            tier: 'Premium',
            computeSize: 'P3',
            maxSizeGB: 26,
            usedSizeGB: 18.4,
            storageUsedPercent: 70.8,
            region: 'Central US',
            server: null,
            resourceGroup: 'rg-prod-cache',
            elasticPool: null,
            cpuUtilization: 45.6,
            memoryUtilization: 70.8,
            diskUtilization: null,
            ioUtilization: 34.7,
            activeConnections: 89,
            maxConnections: 1000,
            dtuUsage: null,
            dtuLimit: null,
            vCores: null,
            costPerHour: 1.44,
            created: '2024-03-15',
            lastBackup: '2024-12-18T03:00:00Z',
            backupRetention: '7 days',
            encryption: 'In-transit and at-rest',
            firewallRules: 3,
            privateEndpoint: false,
            tags: { Environment: 'Production', Application: 'Cache', Owner: 'DevOps' },
            collation: null,
            readReplicas: 0,
            geoReplication: false,
            threats: 0,
            vulnerabilities: 0
          },
          {
            id: 'DB-004',
            name: 'staging-app-db',
            type: 'Azure SQL Database',
            engine: 'SQL Server',
            version: '15.0.2000',
            status: 'online',
            environment: 'staging',
            tier: 'Standard',
            computeSize: 'S2',
            maxSizeGB: 250,
            usedSizeGB: 89.6,
            storageUsedPercent: 35.8,
            region: 'East US 2',
            server: 'staging-sql-server-01',
            resourceGroup: 'rg-staging',
            elasticPool: 'staging-elastic-pool',
            cpuUtilization: 23.4,
            memoryUtilization: 31.7,
            diskUtilization: 22.1,
            ioUtilization: 18.9,
            activeConnections: 12,
            maxConnections: 120,
            dtuUsage: 23.8,
            dtuLimit: 50,
            costPerHour: 0.75,
            created: '2024-04-10',
            lastBackup: '2024-12-18T04:00:00Z',
            backupRetention: '7 days',
            encryption: 'TDE enabled',
            firewallRules: 2,
            privateEndpoint: false,
            tags: { Environment: 'Staging', Application: 'TestApp', Owner: 'QA' },
            collation: 'SQL_Latin1_General_CP1_CI_AS',
            readReplicas: 0,
            geoReplication: false,
            threats: 0,
            vulnerabilities: 0
          },
          {
            id: 'DB-005',
            name: 'prod-nosql-db',
            type: 'Azure Cosmos DB',
            engine: 'Cosmos DB (SQL API)',
            version: '4.0',
            status: 'online',
            environment: 'production',
            tier: 'Provisioned Throughput',
            computeSize: '4000 RU/s',
            maxSizeGB: null,
            usedSizeGB: 234.7,
            storageUsedPercent: null,
            region: 'Multi-region',
            server: null,
            resourceGroup: 'rg-prod-nosql',
            elasticPool: null,
            cpuUtilization: null,
            memoryUtilization: null,
            diskUtilization: null,
            ioUtilization: null,
            activeConnections: 67,
            maxConnections: null,
            dtuUsage: null,
            dtuLimit: null,
            vCores: null,
            requestUnits: 3456,
            maxRequestUnits: 4000,
            costPerHour: 5.76,
            created: '2024-05-20',
            lastBackup: 'Continuous',
            backupRetention: 'Continuous',
            encryption: 'Always encrypted',
            firewallRules: 4,
            privateEndpoint: true,
            tags: { Environment: 'Production', Application: 'NoSQL', Owner: 'DataTeam' },
            collation: null,
            readReplicas: 0,
            geoReplication: true,
            threats: 0,
            vulnerabilities: 0,
            regions: ['East US', 'West US', 'Europe West']
          },
          {
            id: 'DB-006',
            name: 'dev-mysql-db',
            type: 'Azure Database for MySQL',
            engine: 'MySQL',
            version: '8.0.28',
            status: 'offline',
            environment: 'development',
            tier: 'Basic',
            computeSize: 'B_Gen5_1',
            maxSizeGB: 32,
            usedSizeGB: 5.2,
            storageUsedPercent: 16.3,
            region: 'North Central US',
            server: 'dev-mysql-server-01',
            resourceGroup: 'rg-development',
            elasticPool: null,
            cpuUtilization: 0,
            memoryUtilization: 0,
            diskUtilization: 0,
            ioUtilization: 0,
            activeConnections: 0,
            maxConnections: 150,
            dtuUsage: null,
            dtuLimit: null,
            vCores: 1,
            costPerHour: 0,
            created: '2024-06-01',
            lastBackup: '2024-12-17T22:00:00Z',
            backupRetention: '7 days',
            encryption: 'SSL enforced',
            firewallRules: 1,
            privateEndpoint: false,
            tags: { Environment: 'Development', Application: 'DevApp', Owner: 'DevTeam' },
            collation: 'utf8_general_ci',
            readReplicas: 0,
            geoReplication: false,
            threats: 0,
            vulnerabilities: 0
          }
        ],
        elasticPools: [
          {
            id: 'EP-001',
            name: 'prod-elastic-pool-01',
            server: 'prod-sql-server-01',
            tier: 'Premium',
            dtuLimit: 1000,
            dtuUsed: 678,
            storageLimit: 2048,
            storageUsed: 1456,
            databaseCount: 8,
            maxDatabases: 100
          },
          {
            id: 'EP-002',
            name: 'staging-elastic-pool',
            server: 'staging-sql-server-01',
            tier: 'Standard',
            dtuLimit: 200,
            dtuUsed: 89,
            storageLimit: 500,
            storageUsed: 234,
            databaseCount: 4,
            maxDatabases: 100
          }
        ],
        backups: [
          {
            id: 'BK-001',
            databaseId: 'DB-001',
            databaseName: 'prod-customer-db',
            type: 'Automated',
            status: 'Completed',
            sizeGB: 756,
            started: '2024-12-18T02:00:00Z',
            completed: '2024-12-18T02:45:00Z',
            duration: '45m',
            retentionDays: 35
          },
          {
            id: 'BK-002',
            databaseId: 'DB-002',
            databaseName: 'prod-analytics-db',
            type: 'Automated',
            status: 'Completed',
            sizeGB: 1456,
            started: '2024-12-18T01:30:00Z',
            completed: '2024-12-18T02:58:00Z',
            duration: '1h 28m',
            retentionDays: 35
          },
          {
            id: 'BK-003',
            databaseId: 'DB-003',
            databaseName: 'prod-redis-cache',
            type: 'Manual',
            status: 'In Progress',
            sizeGB: 18.4,
            started: '2024-12-18T10:15:00Z',
            completed: null,
            duration: null,
            retentionDays: 7
          }
        ],
        performance: {
          queries: [
            {
              id: 'Q-001',
              database: 'prod-customer-db',
              query: 'SELECT * FROM customers WHERE created_date > ?',
              executionCount: 15674,
              avgDuration: 245,
              maxDuration: 2345,
              cpuTime: 12.4,
              logicalReads: 456789,
              physicalReads: 23456
            },
            {
              id: 'Q-002',
              database: 'prod-analytics-db',
              query: 'SELECT SUM(revenue) FROM sales_data WHERE date BETWEEN ? AND ?',
              executionCount: 8923,
              avgDuration: 1234,
              maxDuration: 8765,
              cpuTime: 45.7,
              logicalReads: 2345678,
              physicalReads: 456789
            }
          ],
          slowQueries: [
            {
              database: 'prod-analytics-db',
              query: 'Complex analytical query with multiple joins',
              duration: 8765,
              timestamp: '2024-12-18T10:25:00Z'
            }
          ]
        },
        security: {
          threats: [
            {
              id: 'T-001',
              database: 'prod-analytics-db',
              type: 'SQL Injection Attempt',
              severity: 'Medium',
              timestamp: '2024-12-18T09:15:00Z',
              blocked: true
            }
          ],
          vulnerabilities: [
            {
              id: 'V-001',
              database: 'prod-analytics-db',
              type: 'Weak Authentication',
              severity: 'Low',
              description: 'Consider implementing stronger authentication methods',
              recommendation: 'Enable Azure AD authentication'
            }
          ]
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
        cpuUsage: 65 + Math.random() * 20,
        memoryUsage: 70 + Math.random() * 20,
        connections: 1800 + Math.floor(Math.random() * 400),
        dtuUsage: 60 + Math.random() * 30,
        storageUsage: 70 + Math.random() * 10
      }]
      return newData.slice(-20)
    })
  }

  const generateRealTimeData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 60000),
      cpuUsage: 65 + Math.random() * 20,
      memoryUsage: 70 + Math.random() * 20,
      connections: 1800 + Math.floor(Math.random() * 400),
      dtuUsage: 60 + Math.random() * 30,
      storageUsage: 70 + Math.random() * 10
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
        label: 'DTU %',
        data: realTimeMetrics.map(d => d.dtuUsage),
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  }

  const databaseTypeData = {
    labels: ['SQL Database', 'Redis Cache', 'Cosmos DB', 'MySQL', 'PostgreSQL'],
    datasets: [{
      data: [18, 6, 4, 3, 3],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(139, 92, 246, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const storageUsageData = {
    labels: databaseData?.databases.map((db: any) => db.name) || [],
    datasets: [{
      label: 'Storage Used (GB)',
      data: databaseData?.databases.map((db: any) => db.usedSizeGB || 0) || [],
      backgroundColor: databaseData?.databases.map((db: any) => 
        db.storageUsedPercent > 80 ? 'rgba(239, 68, 68, 0.8)' :
        db.storageUsedPercent > 60 ? 'rgba(245, 158, 11, 0.8)' :
        'rgba(16, 185, 129, 0.8)'
      ) || [],
      borderWidth: 0
    }]
  }

  const connectionsData = {
    labels: realTimeMetrics.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [{
      label: 'Active Connections',
      data: realTimeMetrics.map(d => d.connections),
      borderColor: 'rgb(139, 92, 246)',
      backgroundColor: 'rgba(139, 92, 246, 0.1)',
      tension: 0.4,
      fill: true
    }]
  }

  const filteredDatabases = databaseData?.databases.filter((db: any) => {
    const matchesType = filterType === 'all' || db.type.toLowerCase().includes(filterType.toLowerCase())
    const matchesStatus = filterStatus === 'all' || db.status === filterStatus
    const matchesEnvironment = filterEnvironment === 'all' || db.environment === filterEnvironment
    const matchesSearch = db.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         db.server?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         db.engine.toLowerCase().includes(searchTerm.toLowerCase())
    return matchesType && matchesStatus && matchesEnvironment && matchesSearch
  }) || []

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'online': return 'text-green-500 bg-green-900/20'
      case 'offline': return 'text-red-500 bg-red-900/20'
      case 'paused': return 'text-yellow-500 bg-yellow-900/20'
      case 'scaling': return 'text-blue-500 bg-blue-900/20'
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

  const getEngineIcon = (engine: string) => {
    if (engine.includes('SQL Server')) {
      return <Database className="w-4 h-4 text-blue-500" />
    } else if (engine.includes('MySQL')) {
      return <Database className="w-4 h-4 text-orange-500" />
    } else if (engine.includes('Redis')) {
      return <Zap className="w-4 h-4 text-red-500" />
    } else if (engine.includes('Cosmos')) {
      return <Globe className="w-4 h-4 text-purple-500" />
    } else if (engine.includes('PostgreSQL')) {
      return <Database className="w-4 h-4 text-indigo-500" />
    }
    return <Database className="w-4 h-4 text-gray-500" />
  }

  const handleDatabaseAction = (action: string, databaseId: string) => {
    console.log(`${action} database: ${databaseId}`)
  }

  const handleBulkAction = (action: string) => {
    console.log(`${action} databases:`, selectedDatabases)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading Database Servers...</p>
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
              <Database className="w-8 h-8 text-blue-500" />
              <div>
                <h1 className="text-2xl font-bold">Database Servers</h1>
                <p className="text-sm text-gray-500">Azure database services monitoring and management</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">{databaseData.overview.onlineDatabases}/{databaseData.overview.totalDatabases} ONLINE</span>
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
                CREATE DATABASE
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'databases', 'performance', 'backups', 'security'].map((tab) => (
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
                  <Database className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Total Databases</span>
                </div>
                <p className="text-2xl font-bold font-mono">{databaseData.overview.totalDatabases}</p>
                <div className="flex items-center mt-1">
                  <CheckCircle className="w-3 h-3 text-green-500 mr-1" />
                  <span className="text-xs text-green-500">{databaseData.overview.onlineDatabases} online</span>
                </div>
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
                <p className="text-2xl font-bold font-mono">{databaseData.overview.usedStorage}TB</p>
                <p className="text-xs text-gray-500 mt-1">of {databaseData.overview.totalStorage}TB</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Users className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Connections</span>
                </div>
                <p className="text-2xl font-bold font-mono">{databaseData.overview.activeConnections}</p>
                <p className="text-xs text-gray-500 mt-1">of {databaseData.overview.totalConnections}</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Cpu className="w-5 h-5 text-red-500" />
                  <span className="text-xs text-gray-500">CPU Usage</span>
                </div>
                <p className="text-2xl font-bold font-mono">{databaseData.overview.avgCpuUtilization}%</p>
                <p className="text-xs text-gray-500 mt-1">average</p>
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
                <p className="text-2xl font-bold font-mono">{databaseData.overview.totalBackups}</p>
                <p className="text-xs text-gray-500 mt-1">{databaseData.overview.lastBackupHours}h ago</p>
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
                <p className="text-2xl font-bold font-mono">${(databaseData.overview.currentSpend / 1000).toFixed(1)}K</p>
                <p className="text-xs text-gray-500 mt-1">of ${(databaseData.overview.monthlyBudget / 1000).toFixed(1)}K budget</p>
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
                      <span className="text-xs text-gray-500">DTU</span>
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

              {/* Database Type Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">DATABASE TYPE DISTRIBUTION</h3>
                <div className="h-64">
                  <Doughnut data={databaseTypeData} options={{
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

              {/* Active Connections */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">ACTIVE CONNECTIONS</h3>
                <div className="h-64">
                  <Line data={connectionsData} options={{
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

            {/* Storage Usage by Database */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg mb-6">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">STORAGE USAGE BY DATABASE</h3>
              </div>
              <div className="p-4">
                <div className="h-64">
                  <Bar data={storageUsageData} options={{
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

        {activeTab === 'databases' && (
          <>
            {/* Database Management Controls */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <Search className="w-4 h-4 text-gray-500 absolute left-3 top-1/2 transform -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder="Search databases..."
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
                    <option value="sql">SQL Database</option>
                    <option value="redis">Redis Cache</option>
                    <option value="cosmos">Cosmos DB</option>
                    <option value="mysql">MySQL</option>
                    <option value="postgresql">PostgreSQL</option>
                  </select>
                  <select
                    value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value)}
                    className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                  >
                    <option value="all">All Status</option>
                    <option value="online">Online</option>
                    <option value="offline">Offline</option>
                    <option value="paused">Paused</option>
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
                  {selectedDatabases.length > 0 && (
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-400">{selectedDatabases.length} selected</span>
                      <button 
                        onClick={() => handleBulkAction('backup')}
                        className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded"
                      >
                        Backup
                      </button>
                      <button 
                        onClick={() => handleBulkAction('scale')}
                        className="px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white text-sm rounded"
                      >
                        Scale
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

            {/* Databases Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left">
                        <input
                          type="checkbox"
                          checked={selectedDatabases.length === filteredDatabases.length && filteredDatabases.length > 0}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedDatabases(filteredDatabases.map((db: any) => db.id))
                            } else {
                              setSelectedDatabases([])
                            }
                          }}
                          className="rounded border-gray-600 bg-gray-700 text-blue-600"
                        />
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Database</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Type</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Environment</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Tier</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Storage</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Connections</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Performance</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Cost/Hour</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {filteredDatabases.map((db: any) => (
                      <motion.tr
                        key={db.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <input
                            type="checkbox"
                            checked={selectedDatabases.includes(db.id)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedDatabases([...selectedDatabases, db.id])
                              } else {
                                setSelectedDatabases(selectedDatabases.filter(id => id !== db.id))
                              }
                            }}
                            className="rounded border-gray-600 bg-gray-700 text-blue-600"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded ${getStatusColor(db.status)}`}>
                              {getEngineIcon(db.engine)}
                            </div>
                            <div>
                              <div className="font-medium">{db.name}</div>
                              <div className="text-sm text-gray-400">{db.server || 'Serverless'}</div>
                              <div className="text-xs text-gray-500">{db.version}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="font-medium">{db.engine}</div>
                            <div className="text-gray-400 text-xs">{db.type}</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            db.status === 'online' ? 'text-green-500' :
                            db.status === 'offline' ? 'text-red-500' :
                            db.status === 'paused' ? 'text-yellow-500' :
                            'text-gray-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              db.status === 'online' ? 'bg-green-500' :
                              db.status === 'offline' ? 'bg-red-500' :
                              db.status === 'paused' ? 'bg-yellow-500' :
                              'bg-gray-500'
                            }`} />
                            <span className="uppercase">{db.status}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center px-2 py-1 text-xs rounded ${getEnvironmentColor(db.environment)}`}>
                            {db.environment.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="font-medium">{db.tier}</div>
                            <div className="text-gray-400 text-xs">{db.computeSize}</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            {db.maxSizeGB ? (
                              <>
                                <div className="font-medium">{db.usedSizeGB}GB</div>
                                <div className="text-gray-400 text-xs">of {db.maxSizeGB}GB</div>
                                <div className="w-12 bg-gray-800 rounded-full h-1.5 mt-1">
                                  <div 
                                    className={`h-1.5 rounded-full ${
                                      db.storageUsedPercent > 80 ? 'bg-red-500' :
                                      db.storageUsedPercent > 60 ? 'bg-yellow-500' :
                                      'bg-green-500'
                                    }`}
                                    style={{ width: `${Math.min(db.storageUsedPercent || 0, 100)}%` }}
                                  />
                                </div>
                              </>
                            ) : (
                              <div className="font-medium">{db.usedSizeGB}GB</div>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="font-medium">{db.activeConnections}</div>
                            {db.maxConnections && (
                              <div className="text-gray-400 text-xs">of {db.maxConnections}</div>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm space-y-1">
                            {db.cpuUtilization !== null && (
                              <div className="flex items-center space-x-1">
                                <span className="text-xs text-gray-500">CPU:</span>
                                <span>{db.cpuUtilization.toFixed(1)}%</span>
                              </div>
                            )}
                            {db.dtuUsage !== null && (
                              <div className="flex items-center space-x-1">
                                <span className="text-xs text-gray-500">DTU:</span>
                                <span>{db.dtuUsage.toFixed(1)}%</span>
                              </div>
                            )}
                            {db.requestUnits && (
                              <div className="flex items-center space-x-1">
                                <span className="text-xs text-gray-500">RU:</span>
                                <span>{db.requestUnits}</span>
                              </div>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">${db.costPerHour}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button 
                              onClick={() => handleDatabaseAction('configure', db.id)}
                              className="p-1 hover:bg-gray-700 rounded text-blue-500"
                              title="Configure Database"
                            >
                              <Settings className="w-4 h-4" />
                            </button>
                            <button 
                              onClick={() => handleDatabaseAction('scale', db.id)}
                              className="p-1 hover:bg-gray-700 rounded text-green-500"
                              title="Scale Database"
                            >
                              <Scale className="w-4 h-4" />
                            </button>
                            <button 
                              onClick={() => handleDatabaseAction('backup', db.id)}
                              className="p-1 hover:bg-gray-700 rounded text-orange-500"
                              title="Backup Database"
                            >
                              <Archive className="w-4 h-4" />
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

        {activeTab === 'performance' && (
          <>
            {/* Performance Analytics */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              {/* Top Queries */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-gray-400 uppercase">TOP RESOURCE CONSUMING QUERIES</h3>
                    <button className="text-xs text-blue-500 hover:text-blue-400">VIEW ALL</button>
                  </div>
                </div>
                <div className="p-4">
                  <div className="space-y-4">
                    {databaseData?.performance.queries.map((query: any) => (
                      <div key={query.id} className="border border-gray-800 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-medium text-blue-500">{query.database}</span>
                          <div className="flex items-center space-x-4 text-xs text-gray-500">
                            <span>Exec: {query.executionCount}</span>
                            <span>Avg: {query.avgDuration}ms</span>
                            <span>Max: {query.maxDuration}ms</span>
                          </div>
                        </div>
                        <div className="text-xs text-gray-300 mb-2 font-mono bg-gray-800/50 p-2 rounded">
                          {query.query.substring(0, 120)}...
                        </div>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4 text-xs">
                            <span className="text-yellow-500">CPU: {query.cpuTime}s</span>
                            <span className="text-green-500">Logical: {query.logicalReads.toLocaleString()}</span>
                            <span className="text-purple-500">Physical: {query.physicalReads.toLocaleString()}</span>
                          </div>
                          <button className="text-xs text-blue-500 hover:text-blue-400">OPTIMIZE</button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Query Performance Trends */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">QUERY PERFORMANCE TRENDS</h3>
                </div>
                <div className="p-4">
                  <div className="h-64">
                    <Line data={{
                      labels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
                      datasets: [{
                        label: 'Avg Query Duration (ms)',
                        data: Array.from({ length: 24 }, () => 200 + Math.random() * 300),
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                      }, {
                        label: 'Queries per Second',
                        data: Array.from({ length: 24 }, () => 50 + Math.random() * 100),
                        borderColor: 'rgb(16, 185, 129)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: true,
                        yAxisID: 'y1'
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
                          ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
                        },
                        y: {
                          grid: { color: 'rgba(255, 255, 255, 0.05)' },
                          ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
                        },
                        y1: {
                          type: 'linear' as const,
                          display: true,
                          position: 'right' as const,
                          grid: { drawOnChartArea: false },
                          ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
                        }
                      }
                    }} />
                  </div>
                </div>
              </div>
            </div>

            {/* Slow Queries Alert */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg mb-6">
              <div className="p-4 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">SLOW QUERIES ALERT</h3>
                  <span className="text-xs text-red-500">Last 24 Hours</span>
                </div>
              </div>
              <div className="p-4">
                <div className="space-y-3">
                  {databaseData?.performance.slowQueries.map((query: any, index: number) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="flex items-center justify-between p-4 bg-red-900/20 border border-red-800/50 rounded-lg"
                    >
                      <div className="flex items-center space-x-4">
                        <AlertTriangle className="w-5 h-5 text-red-500" />
                        <div>
                          <div className="text-sm font-medium text-red-400">{query.database}</div>
                          <div className="text-xs text-gray-400">{query.timestamp}</div>
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-xl font-bold text-red-500">{query.duration}ms</div>
                        <div className="text-xs text-gray-400">Duration</div>
                      </div>
                      <div className="flex-1 mx-4">
                        <div className="text-sm text-gray-300 font-mono bg-gray-800/50 p-2 rounded">
                          {query.query.substring(0, 80)}...
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-xs rounded">ANALYZE</button>
                        <button className="px-3 py-1 bg-green-600 hover:bg-green-700 text-xs rounded">OPTIMIZE</button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>

            {/* Database Performance Grid */}
            <div className="grid grid-cols-3 gap-6">
              {/* Connection Statistics */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">CONNECTION STATISTICS</h3>
                </div>
                <div className="p-4 space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Active Connections</span>
                    <span className="text-lg font-bold text-green-500">{databaseData.overview.activeConnections}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Total Connections</span>
                    <span className="text-lg font-bold text-blue-500">{databaseData.overview.totalConnections}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Connection Pool Usage</span>
                    <span className="text-lg font-bold text-yellow-500">78%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Failed Connections</span>
                    <span className="text-lg font-bold text-red-500">12</span>
                  </div>
                  <div className="mt-4">
                    <div className="h-32">
                      <Line data={connectionsData} options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                          x: {
                            display: false
                          },
                          y: {
                            display: false
                          }
                        },
                        elements: {
                          point: {
                            radius: 0
                          }
                        }
                      }} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Resource Utilization */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">RESOURCE UTILIZATION</h3>
                </div>
                <div className="p-4 space-y-4">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-400">CPU Usage</span>
                      <span className="text-sm font-bold">{databaseData.overview.avgCpuUtilization}%</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${databaseData.overview.avgCpuUtilization}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-400">Memory Usage</span>
                      <span className="text-sm font-bold">{databaseData.overview.avgMemoryUtilization}%</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${databaseData.overview.avgMemoryUtilization}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-400">Storage Usage</span>
                      <span className="text-sm font-bold">{databaseData.overview.avgStorageUtilization}%</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div 
                        className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${databaseData.overview.avgStorageUtilization}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-400">I/O Utilization</span>
                      <span className="text-sm font-bold">45.2%</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div className="bg-purple-500 h-2 rounded-full w-[45.2%] transition-all duration-300" />
                    </div>
                  </div>
                </div>
              </div>

              {/* Index Performance */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">INDEX PERFORMANCE</h3>
                </div>
                <div className="p-4 space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">Missing Indexes</span>
                      <span className="text-sm font-bold text-red-500">23</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">Unused Indexes</span>
                      <span className="text-sm font-bold text-yellow-500">45</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">Fragmented Indexes</span>
                      <span className="text-sm font-bold text-orange-500">12</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">Duplicate Indexes</span>
                      <span className="text-sm font-bold text-purple-500">8</span>
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-gray-800">
                    <div className="text-xs text-gray-400 mb-2">Index Usage Distribution</div>
                    <div className="h-20">
                      <Doughnut data={{
                        labels: ['Used', 'Unused', 'Fragmented'],
                        datasets: [{
                          data: [78, 15, 7],
                          backgroundColor: [
                            'rgba(16, 185, 129, 0.8)',
                            'rgba(245, 158, 11, 0.8)',
                            'rgba(239, 68, 68, 0.8)'
                          ],
                          borderWidth: 0
                        }]
                      }} options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: { display: false }
                        }
                      }} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'backups' && (
          <>
            {/* Backup Status Overview */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Archive className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Total Backups</span>
                </div>
                <p className="text-2xl font-bold font-mono">{databaseData.overview.totalBackups}</p>
                <p className="text-xs text-gray-500 mt-1">All databases</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Successful</span>
                </div>
                <p className="text-2xl font-bold font-mono text-green-500">153</p>
                <p className="text-xs text-gray-500 mt-1">98.1% success rate</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Clock className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Last Backup</span>
                </div>
                <p className="text-2xl font-bold font-mono">{databaseData.overview.lastBackupHours}h</p>
                <p className="text-xs text-gray-500 mt-1">ago</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <HardDrive className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Backup Size</span>
                </div>
                <p className="text-2xl font-bold font-mono">2.3TB</p>
                <p className="text-xs text-gray-500 mt-1">total size</p>
              </motion.div>
            </div>

            {/* Backup Schedule and Management */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              {/* Backup Schedule */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-gray-400 uppercase">BACKUP SCHEDULE</h3>
                    <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-xs rounded">ADD SCHEDULE</button>
                  </div>
                </div>
                <div className="p-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-green-500 rounded-full" />
                        <div>
                          <div className="text-sm font-medium">Daily Full Backup</div>
                          <div className="text-xs text-gray-400">Every day at 02:00 UTC</div>
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
                          <div className="text-sm font-medium">Incremental Backup</div>
                          <div className="text-xs text-gray-400">Every 6 hours</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-yellow-500">ACTIVE</span>
                        <button className="text-xs text-blue-500 hover:text-blue-400">EDIT</button>
                      </div>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-red-500 rounded-full" />
                        <div>
                          <div className="text-sm font-medium">Archive Backup</div>
                          <div className="text-xs text-gray-400">Weekly on Sunday</div>
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

              {/* Backup History */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">RECENT BACKUP HISTORY</h3>
                </div>
                <div className="p-4">
                  <div className="space-y-3">
                    {databaseData?.backups.map((backup: any) => (
                      <div key={backup.id} className="flex items-center justify-between p-3 border border-gray-800 rounded">
                        <div className="flex items-center space-x-3">
                          <div className={`w-3 h-3 rounded-full ${
                            backup.status === 'Completed' ? 'bg-green-500' :
                            backup.status === 'In Progress' ? 'bg-yellow-500 animate-pulse' :
                            'bg-red-500'
                          }`} />
                          <div>
                            <div className="text-sm font-medium">{backup.databaseName}</div>
                            <div className="text-xs text-gray-400">{backup.type} - {backup.sizeGB}GB</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-mono">
                            {backup.status === 'Completed' ? backup.duration : 'In Progress...'}
                          </div>
                          <div className="text-xs text-gray-400">
                            {new Date(backup.started).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Backup Details Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">ALL BACKUPS</h3>
                  <div className="flex items-center space-x-2">
                    <button className="px-3 py-1 bg-green-600 hover:bg-green-700 text-xs rounded">START BACKUP</button>
                    <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-xs rounded">RESTORE</button>
                  </div>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Database</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Type</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Size</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Started</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Duration</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Retention</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {databaseData?.backups.map((backup: any) => (
                      <motion.tr
                        key={backup.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div className="text-sm font-medium">{backup.databaseName}</div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="inline-flex items-center px-2 py-1 text-xs rounded bg-blue-900/30 text-blue-500">
                            {backup.type}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            backup.status === 'Completed' ? 'text-green-500' :
                            backup.status === 'In Progress' ? 'text-yellow-500' :
                            'text-red-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              backup.status === 'Completed' ? 'bg-green-500' :
                              backup.status === 'In Progress' ? 'bg-yellow-500 animate-pulse' :
                              'bg-red-500'
                            }`} />
                            <span>{backup.status.toUpperCase()}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{backup.sizeGB}GB</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm">{new Date(backup.started).toLocaleString()}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{backup.duration || 'N/A'}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm">{backup.retentionDays} days</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button className="p-1 hover:bg-gray-700 rounded text-blue-500" title="Download Backup">
                              <Download className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded text-green-500" title="Restore Backup">
                              <RotateCcw className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded text-red-500" title="Delete Backup">
                              <XCircle className="w-4 h-4" />
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
                  <span className="text-xs text-gray-500">Threats Detected</span>
                </div>
                <p className="text-2xl font-bold font-mono">{databaseData.security.threats.length}</p>
                <p className="text-xs text-green-500 mt-1">All blocked</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Vulnerabilities</span>
                </div>
                <p className="text-2xl font-bold font-mono">{databaseData.security.vulnerabilities.length}</p>
                <p className="text-xs text-yellow-500 mt-1">Need attention</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Lock className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Encrypted DBs</span>
                </div>
                <p className="text-2xl font-bold font-mono">100%</p>
                <p className="text-xs text-blue-500 mt-1">TDE enabled</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Users className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Active Sessions</span>
                </div>
                <p className="text-2xl font-bold font-mono">245</p>
                <p className="text-xs text-gray-500 mt-1">authenticated users</p>
              </motion.div>
            </div>

            {/* Threat Detection */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">RECENT THREATS</h3>
                </div>
                <div className="p-4">
                  <div className="space-y-3">
                    {databaseData?.security.threats.map((threat: any) => (
                      <motion.div
                        key={threat.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex items-center justify-between p-3 bg-red-900/20 border border-red-800/50 rounded"
                      >
                        <div className="flex items-center space-x-3">
                          <AlertTriangle className="w-4 h-4 text-red-500" />
                          <div>
                            <div className="text-sm font-medium text-red-400">{threat.type}</div>
                            <div className="text-xs text-gray-400">{threat.database}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`text-xs font-medium ${
                            threat.blocked ? 'text-green-500' : 'text-red-500'
                          }`}>
                            {threat.blocked ? 'BLOCKED' : 'ACTIVE'}
                          </div>
                          <div className="text-xs text-gray-400">{threat.timestamp}</div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">VULNERABILITY ASSESSMENT</h3>
                </div>
                <div className="p-4">
                  <div className="space-y-3">
                    {databaseData?.security.vulnerabilities.map((vuln: any) => (
                      <motion.div
                        key={vuln.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="p-3 border border-gray-800 rounded"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <span className={`w-2 h-2 rounded-full ${
                              vuln.severity === 'High' ? 'bg-red-500' :
                              vuln.severity === 'Medium' ? 'bg-yellow-500' :
                              'bg-green-500'
                            }`} />
                            <span className="text-sm font-medium">{vuln.type}</span>
                            <span className={`text-xs px-2 py-1 rounded ${
                              vuln.severity === 'High' ? 'bg-red-900/30 text-red-500' :
                              vuln.severity === 'Medium' ? 'bg-yellow-900/30 text-yellow-500' :
                              'bg-green-900/30 text-green-500'
                            }`}>
                              {vuln.severity}
                            </span>
                          </div>
                          <button className="text-xs text-blue-500 hover:text-blue-400">FIX</button>
                        </div>
                        <div className="text-xs text-gray-400 mb-2">{vuln.description}</div>
                        <div className="text-xs text-green-500">{vuln.recommendation}</div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Access Control and Audit */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">ACCESS CONTROL & AUDIT TRAIL</h3>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-3 gap-6">
                  <div>
                    <div className="text-xs text-gray-400 mb-3 uppercase">Authentication Methods</div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Azure AD</span>
                        <span className="text-sm text-green-500">89%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">SQL Auth</span>
                        <span className="text-sm text-yellow-500">11%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Certificate</span>
                        <span className="text-sm text-blue-500">0%</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-400 mb-3 uppercase">User Activity (24h)</div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Successful Logins</span>
                        <span className="text-sm text-green-500">2,345</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Failed Logins</span>
                        <span className="text-sm text-red-500">23</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Privilege Changes</span>
                        <span className="text-sm text-yellow-500">5</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-400 mb-3 uppercase">Security Features</div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">TDE Encryption</span>
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Always Encrypted</span>
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Row Level Security</span>
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Dynamic Data Masking</span>
                        <XCircle className="w-4 h-4 text-red-500" />
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