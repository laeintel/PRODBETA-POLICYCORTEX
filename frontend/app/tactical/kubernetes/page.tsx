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
  Hexagon,
  Container,
  Layers,
  Cloud,
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
  Network,
  Globe,
  MapPin,
  Link,
  Target,
  Crosshair,
  Navigation,
  Radio,
  Signal,
  GitBranch,
  Package
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

export default function KubernetesClustersPage() {
  const [loading, setLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [activeTab, setActiveTab] = useState('overview')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [kubernetesData, setKubernetesData] = useState<any>(null)
  const [realTimeMetrics, setRealTimeMetrics] = useState<any[]>([])
  const [selectedClusters, setSelectedClusters] = useState<string[]>([])
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
      setKubernetesData({
        overview: {
          totalClusters: 8,
          healthyClusters: 7,
          unhealthyClusters: 1,
          totalNodes: 156,
          readyNodes: 152,
          totalPods: 2847,
          runningPods: 2734,
          pendingPods: 78,
          failedPods: 35,
          totalNamespaces: 45,
          avgCpuUtilization: 67.3,
          avgMemoryUtilization: 72.8,
          monthlyBudget: 45678.90,
          currentSpend: 32456.78
        },
        clusters: [
          {
            id: 'CLUSTER-001',
            name: 'prod-k8s-east-01',
            provider: 'azure',
            version: 'v1.28.3',
            status: 'healthy',
            environment: 'production',
            region: 'East US',
            nodeCount: 24,
            readyNodes: 24,
            notReadyNodes: 0,
            podCount: 847,
            runningPods: 823,
            pendingPods: 15,
            failedPods: 9,
            namespaceCount: 12,
            cpuUtilization: 78.5,
            memoryUtilization: 82.3,
            storageUtilization: 65.4,
            networkIn: 125.6,
            networkOut: 89.4,
            costPerHour: 12.45,
            created: '2024-01-15',
            lastUpdated: '2024-12-18T10:30:00Z',
            apiServerEndpoint: 'https://prod-k8s-east-01-api.eastus.cloudapp.azure.com',
            kubernetesVersion: 'v1.28.3',
            nodePoolCount: 3,
            autoscalingEnabled: true,
            rbacEnabled: true,
            networkPolicy: 'calico',
            ingressController: 'nginx',
            monitoring: 'prometheus',
            logging: 'fluentd'
          },
          {
            id: 'CLUSTER-002',
            name: 'prod-k8s-west-01',
            provider: 'azure',
            version: 'v1.28.3',
            status: 'healthy',
            environment: 'production',
            region: 'West US',
            nodeCount: 18,
            readyNodes: 18,
            notReadyNodes: 0,
            podCount: 634,
            runningPods: 612,
            pendingPods: 12,
            failedPods: 10,
            namespaceCount: 10,
            cpuUtilization: 65.2,
            memoryUtilization: 75.8,
            storageUtilization: 58.7,
            networkIn: 98.3,
            networkOut: 76.2,
            costPerHour: 9.87,
            created: '2024-02-01',
            lastUpdated: '2024-12-18T10:28:00Z',
            apiServerEndpoint: 'https://prod-k8s-west-01-api.westus.cloudapp.azure.com',
            kubernetesVersion: 'v1.28.3',
            nodePoolCount: 2,
            autoscalingEnabled: true,
            rbacEnabled: true,
            networkPolicy: 'calico',
            ingressController: 'nginx',
            monitoring: 'prometheus',
            logging: 'fluentd'
          },
          {
            id: 'CLUSTER-003',
            name: 'staging-k8s-01',
            provider: 'azure',
            version: 'v1.29.0',
            status: 'healthy',
            environment: 'staging',
            region: 'Central US',
            nodeCount: 8,
            readyNodes: 8,
            notReadyNodes: 0,
            podCount: 234,
            runningPods: 218,
            pendingPods: 8,
            failedPods: 8,
            namespaceCount: 6,
            cpuUtilization: 45.6,
            memoryUtilization: 52.3,
            storageUtilization: 38.9,
            networkIn: 34.7,
            networkOut: 28.5,
            costPerHour: 3.45,
            created: '2024-03-15',
            lastUpdated: '2024-12-18T10:25:00Z',
            apiServerEndpoint: 'https://staging-k8s-01-api.centralus.cloudapp.azure.com',
            kubernetesVersion: 'v1.29.0',
            nodePoolCount: 2,
            autoscalingEnabled: true,
            rbacEnabled: true,
            networkPolicy: 'azure',
            ingressController: 'traefik',
            monitoring: 'prometheus',
            logging: 'azure-monitor'
          },
          {
            id: 'CLUSTER-004',
            name: 'dev-k8s-01',
            provider: 'azure',
            version: 'v1.29.0',
            status: 'healthy',
            environment: 'development',
            region: 'East US 2',
            nodeCount: 4,
            readyNodes: 4,
            notReadyNodes: 0,
            podCount: 156,
            runningPods: 145,
            pendingPods: 6,
            failedPods: 5,
            namespaceCount: 8,
            cpuUtilization: 23.4,
            memoryUtilization: 31.7,
            storageUtilization: 22.1,
            networkIn: 12.8,
            networkOut: 9.3,
            costPerHour: 1.89,
            created: '2024-04-10',
            lastUpdated: '2024-12-18T10:22:00Z',
            apiServerEndpoint: 'https://dev-k8s-01-api.eastus2.cloudapp.azure.com',
            kubernetesVersion: 'v1.29.0',
            nodePoolCount: 1,
            autoscalingEnabled: false,
            rbacEnabled: true,
            networkPolicy: 'kubenet',
            ingressController: 'nginx',
            monitoring: 'basic',
            logging: 'basic'
          },
          {
            id: 'CLUSTER-005',
            name: 'ml-k8s-gpu-01',
            provider: 'azure',
            version: 'v1.28.3',
            status: 'healthy',
            environment: 'production',
            region: 'North Central US',
            nodeCount: 12,
            readyNodes: 12,
            notReadyNodes: 0,
            podCount: 89,
            runningPods: 84,
            pendingPods: 3,
            failedPods: 2,
            namespaceCount: 4,
            cpuUtilization: 89.7,
            memoryUtilization: 67.4,
            storageUtilization: 78.9,
            networkIn: 234.7,
            networkOut: 189.3,
            costPerHour: 18.92,
            created: '2024-05-20',
            lastUpdated: '2024-12-18T10:20:00Z',
            apiServerEndpoint: 'https://ml-k8s-gpu-01-api.northcentralus.cloudapp.azure.com',
            kubernetesVersion: 'v1.28.3',
            nodePoolCount: 2,
            autoscalingEnabled: true,
            rbacEnabled: true,
            networkPolicy: 'calico',
            ingressController: 'istio',
            monitoring: 'prometheus',
            logging: 'elasticsearch',
            gpuNodeCount: 8
          },
          {
            id: 'CLUSTER-006',
            name: 'test-k8s-01',
            provider: 'azure',
            version: 'v1.29.1',
            status: 'warning',
            environment: 'test',
            region: 'South Central US',
            nodeCount: 6,
            readyNodes: 5,
            notReadyNodes: 1,
            podCount: 187,
            runningPods: 165,
            pendingPods: 12,
            failedPods: 10,
            namespaceCount: 5,
            cpuUtilization: 56.8,
            memoryUtilization: 78.9,
            storageUtilization: 45.6,
            networkIn: 45.7,
            networkOut: 38.2,
            costPerHour: 2.67,
            created: '2024-06-01',
            lastUpdated: '2024-12-18T10:15:00Z',
            apiServerEndpoint: 'https://test-k8s-01-api.southcentralus.cloudapp.azure.com',
            kubernetesVersion: 'v1.29.1',
            nodePoolCount: 2,
            autoscalingEnabled: true,
            rbacEnabled: true,
            networkPolicy: 'azure',
            ingressController: 'nginx',
            monitoring: 'prometheus',
            logging: 'azure-monitor'
          }
        ],
        nodePools: [
          {
            id: 'POOL-001',
            name: 'system-pool',
            cluster: 'prod-k8s-east-01',
            nodeCount: 3,
            vmSize: 'Standard_D4s_v3',
            osType: 'Linux',
            mode: 'System',
            autoScaling: false,
            minCount: 3,
            maxCount: 3,
            currentCount: 3,
            readyNodes: 3,
            status: 'ready'
          },
          {
            id: 'POOL-002',
            name: 'user-pool',
            cluster: 'prod-k8s-east-01',
            nodeCount: 18,
            vmSize: 'Standard_D8s_v3',
            osType: 'Linux',
            mode: 'User',
            autoScaling: true,
            minCount: 6,
            maxCount: 30,
            currentCount: 18,
            readyNodes: 18,
            status: 'ready'
          },
          {
            id: 'POOL-003',
            name: 'gpu-pool',
            cluster: 'ml-k8s-gpu-01',
            nodeCount: 8,
            vmSize: 'Standard_NC24rs_v3',
            osType: 'Linux',
            mode: 'User',
            autoScaling: true,
            minCount: 2,
            maxCount: 12,
            currentCount: 8,
            readyNodes: 8,
            status: 'ready',
            gpuType: 'V100'
          }
        ],
        workloads: [
          {
            id: 'WL-001',
            name: 'web-frontend',
            namespace: 'production',
            cluster: 'prod-k8s-east-01',
            type: 'Deployment',
            replicas: 6,
            availableReplicas: 6,
            readyReplicas: 6,
            image: 'nginx:1.21',
            cpuRequests: '200m',
            cpuLimits: '500m',
            memoryRequests: '256Mi',
            memoryLimits: '512Mi',
            status: 'healthy',
            created: '2024-11-15',
            restartCount: 0
          },
          {
            id: 'WL-002',
            name: 'api-backend',
            namespace: 'production',
            cluster: 'prod-k8s-east-01',
            type: 'Deployment',
            replicas: 12,
            availableReplicas: 12,
            readyReplicas: 12,
            image: 'node:18-alpine',
            cpuRequests: '500m',
            cpuLimits: '1000m',
            memoryRequests: '512Mi',
            memoryLimits: '1Gi',
            status: 'healthy',
            created: '2024-11-10',
            restartCount: 2
          },
          {
            id: 'WL-003',
            name: 'database',
            namespace: 'production',
            cluster: 'prod-k8s-east-01',
            type: 'StatefulSet',
            replicas: 3,
            availableReplicas: 3,
            readyReplicas: 3,
            image: 'postgres:15',
            cpuRequests: '1000m',
            cpuLimits: '2000m',
            memoryRequests: '2Gi',
            memoryLimits: '4Gi',
            status: 'healthy',
            created: '2024-10-20',
            restartCount: 0
          },
          {
            id: 'WL-004',
            name: 'ml-training',
            namespace: 'ml-workloads',
            cluster: 'ml-k8s-gpu-01',
            type: 'Job',
            replicas: 1,
            availableReplicas: 1,
            readyReplicas: 1,
            image: 'tensorflow/tensorflow:2.13.0-gpu',
            cpuRequests: '4000m',
            cpuLimits: '8000m',
            memoryRequests: '16Gi',
            memoryLimits: '32Gi',
            gpuRequests: 1,
            status: 'running',
            created: '2024-12-18',
            restartCount: 0
          }
        ],
        services: [
          {
            id: 'SVC-001',
            name: 'web-frontend-svc',
            namespace: 'production',
            cluster: 'prod-k8s-east-01',
            type: 'LoadBalancer',
            clusterIP: '10.0.245.123',
            externalIP: '52.188.45.123',
            ports: [{ port: 80, targetPort: 8080, protocol: 'TCP' }],
            selector: { app: 'web-frontend' },
            status: 'active'
          },
          {
            id: 'SVC-002',
            name: 'api-backend-svc',
            namespace: 'production',
            cluster: 'prod-k8s-east-01',
            type: 'ClusterIP',
            clusterIP: '10.0.156.78',
            ports: [{ port: 3000, targetPort: 3000, protocol: 'TCP' }],
            selector: { app: 'api-backend' },
            status: 'active'
          }
        ],
        events: [
          {
            id: 'EVT-001',
            type: 'Normal',
            reason: 'Scheduled',
            object: 'Pod/web-frontend-7b8c9d5f68-abc12',
            message: 'Successfully assigned production/web-frontend-7b8c9d5f68-abc12 to aks-nodepool1-12345-vmss000002',
            timestamp: '2024-12-18T10:30:15Z',
            cluster: 'prod-k8s-east-01'
          },
          {
            id: 'EVT-002',
            type: 'Warning',
            reason: 'FailedMount',
            object: 'Pod/database-0',
            message: 'Unable to attach or mount volumes: unmounted volumes=[data], unattached volumes=[data default-token-xyz]: timed out waiting for the condition',
            timestamp: '2024-12-18T10:25:30Z',
            cluster: 'test-k8s-01'
          },
          {
            id: 'EVT-003',
            type: 'Normal',
            reason: 'Killing',
            object: 'Pod/api-backend-6f7g8h9i0j-def34',
            message: 'Stopping container api-backend',
            timestamp: '2024-12-18T10:20:45Z',
            cluster: 'prod-k8s-east-01'
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
        cpuUsage: 65 + Math.random() * 20,
        memoryUsage: 70 + Math.random() * 20,
        podCount: 2800 + Math.floor(Math.random() * 100),
        activeNodes: 150 + Math.floor(Math.random() * 10)
      }]
      return newData.slice(-20)
    })
  }

  const generateRealTimeData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 60000),
      cpuUsage: 65 + Math.random() * 20,
      memoryUsage: 70 + Math.random() * 20,
      podCount: 2800 + Math.floor(Math.random() * 100),
      activeNodes: 150 + Math.floor(Math.random() * 10)
    }))
  }

  const performanceData = {
    labels: realTimeMetrics.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [
      {
        label: 'CPU Usage %',
        data: realTimeMetrics.map(d => d.cpuUsage),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Memory Usage %',
        data: realTimeMetrics.map(d => d.memoryUsage),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  }

  const clusterStatusData = {
    labels: ['Healthy', 'Warning', 'Critical', 'Maintenance'],
    datasets: [{
      data: [7, 1, 0, 0],
      backgroundColor: [
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(156, 163, 175, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const podDistributionData = {
    labels: kubernetesData?.clusters.map((cluster: any) => cluster.name) || [],
    datasets: [{
      label: 'Running Pods',
      data: kubernetesData?.clusters.map((cluster: any) => cluster.runningPods) || [],
      backgroundColor: 'rgba(59, 130, 246, 0.8)',
      borderWidth: 0
    }, {
      label: 'Pending Pods',
      data: kubernetesData?.clusters.map((cluster: any) => cluster.pendingPods) || [],
      backgroundColor: 'rgba(245, 158, 11, 0.8)',
      borderWidth: 0
    }, {
      label: 'Failed Pods',
      data: kubernetesData?.clusters.map((cluster: any) => cluster.failedPods) || [],
      backgroundColor: 'rgba(239, 68, 68, 0.8)',
      borderWidth: 0
    }]
  }

  const resourceUtilizationData = {
    labels: kubernetesData?.clusters.map((cluster: any) => cluster.name) || [],
    datasets: [{
      label: 'CPU Utilization %',
      data: kubernetesData?.clusters.map((cluster: any) => cluster.cpuUtilization) || [],
      backgroundColor: kubernetesData?.clusters.map((cluster: any) => 
        cluster.cpuUtilization > 80 ? 'rgba(239, 68, 68, 0.8)' :
        cluster.cpuUtilization > 60 ? 'rgba(245, 158, 11, 0.8)' :
        'rgba(16, 185, 129, 0.8)'
      ) || [],
      borderWidth: 0
    }]
  }

  const filteredClusters = kubernetesData?.clusters.filter((cluster: any) => {
    const matchesStatus = filterStatus === 'all' || cluster.status === filterStatus
    const matchesEnvironment = filterEnvironment === 'all' || cluster.environment === filterEnvironment
    const matchesSearch = cluster.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         cluster.region.toLowerCase().includes(searchTerm.toLowerCase())
    return matchesStatus && matchesEnvironment && matchesSearch
  }) || []

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'healthy': return 'text-green-500 bg-green-900/20'
      case 'warning': return 'text-yellow-500 bg-yellow-900/20'
      case 'critical': return 'text-red-500 bg-red-900/20'
      case 'maintenance': return 'text-blue-500 bg-blue-900/20'
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

  const handleClusterAction = (action: string, clusterId: string) => {
    console.log(`${action} cluster: ${clusterId}`)
  }

  const handleBulkAction = (action: string) => {
    console.log(`${action} clusters:`, selectedClusters)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading Kubernetes Clusters...</p>
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
              <Hexagon className="w-8 h-8 text-blue-500" />
              <div>
                <h1 className="text-2xl font-bold">Kubernetes Clusters</h1>
                <p className="text-sm text-gray-500">Container orchestration and cluster management</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">{kubernetesData.overview.healthyClusters}/{kubernetesData.overview.totalClusters} HEALTHY</span>
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
                CREATE CLUSTER
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'clusters', 'workloads', 'nodes', 'services', 'events'].map((tab) => (
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
                  <Hexagon className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Clusters</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.overview.totalClusters}</p>
                <div className="flex items-center mt-1">
                  <CheckCircle className="w-3 h-3 text-green-500 mr-1" />
                  <span className="text-xs text-green-500">{kubernetesData.overview.healthyClusters} healthy</span>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Server className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Nodes</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.overview.totalNodes}</p>
                <p className="text-xs text-gray-500 mt-1">{kubernetesData.overview.readyNodes} ready</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Container className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Pods</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.overview.totalPods}</p>
                <p className="text-xs text-gray-500 mt-1">{kubernetesData.overview.runningPods} running</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Folder className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Namespaces</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.overview.totalNamespaces}</p>
                <p className="text-xs text-gray-500 mt-1">across all clusters</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Cpu className="w-5 h-5 text-red-500" />
                  <span className="text-xs text-gray-500">CPU Usage</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.overview.avgCpuUtilization}%</p>
                <p className="text-xs text-gray-500 mt-1">average</p>
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
                <p className="text-2xl font-bold font-mono">${(kubernetesData.overview.currentSpend / 1000).toFixed(1)}K</p>
                <p className="text-xs text-gray-500 mt-1">of ${(kubernetesData.overview.monthlyBudget / 1000).toFixed(1)}K budget</p>
              </motion.div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {/* Resource Performance */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">RESOURCE USAGE</h3>
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

              {/* Cluster Status */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">CLUSTER STATUS</h3>
                <div className="h-64">
                  <Doughnut data={clusterStatusData} options={{
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

              {/* Pod Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">POD DISTRIBUTION</h3>
                <div className="h-64">
                  <Bar data={podDistributionData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'top',
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

            {/* Resource Utilization Summary */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg mb-6">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">CLUSTER RESOURCE UTILIZATION</h3>
              </div>
              <div className="p-4">
                <div className="h-64">
                  <Bar data={resourceUtilizationData} options={{
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
          </>
        )}

        {activeTab === 'clusters' && (
          <>
            {/* Cluster Management Controls */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <Search className="w-4 h-4 text-gray-500 absolute left-3 top-1/2 transform -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder="Search clusters..."
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
                    <option value="healthy">Healthy</option>
                    <option value="warning">Warning</option>
                    <option value="critical">Critical</option>
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
                  {selectedClusters.length > 0 && (
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-400">{selectedClusters.length} selected</span>
                      <button 
                        onClick={() => handleBulkAction('upgrade')}
                        className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded"
                      >
                        Upgrade
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

            {/* Clusters Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left">
                        <input
                          type="checkbox"
                          checked={selectedClusters.length === filteredClusters.length && filteredClusters.length > 0}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedClusters(filteredClusters.map((cluster: any) => cluster.id))
                            } else {
                              setSelectedClusters([])
                            }
                          }}
                          className="rounded border-gray-600 bg-gray-700 text-blue-600"
                        />
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Cluster</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Environment</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Version</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Nodes</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Pods</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">CPU</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Memory</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Cost/Hour</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {filteredClusters.map((cluster: any) => (
                      <motion.tr
                        key={cluster.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <input
                            type="checkbox"
                            checked={selectedClusters.includes(cluster.id)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedClusters([...selectedClusters, cluster.id])
                              } else {
                                setSelectedClusters(selectedClusters.filter(id => id !== cluster.id))
                              }
                            }}
                            className="rounded border-gray-600 bg-gray-700 text-blue-600"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded ${getStatusColor(cluster.status)}`}>
                              <Hexagon className="w-4 h-4" />
                            </div>
                            <div>
                              <div className="font-medium">{cluster.name}</div>
                              <div className="text-sm text-gray-400">{cluster.region}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            cluster.status === 'healthy' ? 'text-green-500' :
                            cluster.status === 'warning' ? 'text-yellow-500' :
                            cluster.status === 'critical' ? 'text-red-500' :
                            'text-gray-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              cluster.status === 'healthy' ? 'bg-green-500' :
                              cluster.status === 'warning' ? 'bg-yellow-500' :
                              cluster.status === 'critical' ? 'bg-red-500' :
                              'bg-gray-500'
                            }`} />
                            <span className="uppercase">{cluster.status}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center px-2 py-1 text-xs rounded ${getEnvironmentColor(cluster.environment)}`}>
                            {cluster.environment.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{cluster.version}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="font-medium">{cluster.nodeCount}</div>
                            <div className="text-gray-400 text-xs">{cluster.readyNodes} ready</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="font-medium">{cluster.podCount}</div>
                            <div className="text-gray-400 text-xs">{cluster.runningPods} running</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="flex items-center space-x-1">
                              <span>{cluster.cpuUtilization.toFixed(1)}%</span>
                              <div className="w-12 bg-gray-800 rounded-full h-1.5">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    cluster.cpuUtilization > 80 ? 'bg-red-500' :
                                    cluster.cpuUtilization > 60 ? 'bg-yellow-500' :
                                    'bg-green-500'
                                  }`}
                                  style={{ width: `${Math.min(cluster.cpuUtilization, 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="flex items-center space-x-1">
                              <span>{cluster.memoryUtilization.toFixed(1)}%</span>
                              <div className="w-12 bg-gray-800 rounded-full h-1.5">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    cluster.memoryUtilization > 80 ? 'bg-red-500' :
                                    cluster.memoryUtilization > 60 ? 'bg-yellow-500' :
                                    'bg-green-500'
                                  }`}
                                  style={{ width: `${Math.min(cluster.memoryUtilization, 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">${cluster.costPerHour}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button 
                              onClick={() => handleClusterAction('configure', cluster.id)}
                              className="p-1 hover:bg-gray-700 rounded text-blue-500"
                              title="Configure Cluster"
                            >
                              <Settings className="w-4 h-4" />
                            </button>
                            <button 
                              onClick={() => handleClusterAction('scale', cluster.id)}
                              className="p-1 hover:bg-gray-700 rounded text-green-500"
                              title="Scale Cluster"
                            >
                              <Scale className="w-4 h-4" />
                            </button>
                            <button 
                              onClick={() => handleClusterAction('upgrade', cluster.id)}
                              className="p-1 hover:bg-gray-700 rounded text-purple-500"
                              title="Upgrade Cluster"
                            >
                              <TrendingUp className="w-4 h-4" />
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

        {activeTab === 'workloads' && (
          <>
            {/* Workloads Management */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Package className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Deployments</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.workloads.filter((w: any) => w.type === 'Deployment').length}</p>
                <p className="text-xs text-gray-500 mt-1">active workloads</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Database className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">StatefulSets</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.workloads.filter((w: any) => w.type === 'StatefulSet').length}</p>
                <p className="text-xs text-gray-500 mt-1">persistent workloads</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Zap className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Jobs</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.workloads.filter((w: any) => w.type === 'Job').length}</p>
                <p className="text-xs text-gray-500 mt-1">batch workloads</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Activity className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Total Replicas</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.workloads.reduce((sum: any, w: any) => sum + w.replicas, 0)}</p>
                <p className="text-xs text-gray-500 mt-1">across all workloads</p>
              </motion.div>
            </div>

            {/* Workloads Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">Workloads</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Name</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Type</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Namespace</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Cluster</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Replicas</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Ready</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Resources</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {kubernetesData.workloads.map((workload: any) => (
                      <motion.tr
                        key={workload.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded ${workload.type === 'Deployment' ? 'bg-blue-900/20 text-blue-500' : workload.type === 'StatefulSet' ? 'bg-green-900/20 text-green-500' : 'bg-yellow-900/20 text-yellow-500'}`}>
                              {workload.type === 'Deployment' && <Package className="w-4 h-4" />}
                              {workload.type === 'StatefulSet' && <Database className="w-4 h-4" />}
                              {workload.type === 'Job' && <Zap className="w-4 h-4" />}
                            </div>
                            <div>
                              <div className="font-medium">{workload.name}</div>
                              <div className="text-sm text-gray-400">{workload.image}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-medium">{workload.type}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm">{workload.namespace}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm">{workload.cluster}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{workload.replicas}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-2">
                            <span className="text-sm font-mono">{workload.readyReplicas}/{workload.replicas}</span>
                            <div className="w-12 bg-gray-800 rounded-full h-1.5">
                              <div 
                                className="h-1.5 rounded-full bg-green-500"
                                style={{ width: `${(workload.readyReplicas / workload.replicas) * 100}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-xs text-gray-400">
                            <div>CPU: {workload.cpuRequests} / {workload.cpuLimits}</div>
                            <div>Mem: {workload.memoryRequests} / {workload.memoryLimits}</div>
                            {workload.gpuRequests && <div>GPU: {workload.gpuRequests}</div>}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            workload.status === 'healthy' ? 'text-green-500' :
                            workload.status === 'running' ? 'text-blue-500' :
                            'text-gray-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              workload.status === 'healthy' ? 'bg-green-500' :
                              workload.status === 'running' ? 'bg-blue-500' :
                              'bg-gray-500'
                            }`} />
                            <span className="uppercase">{workload.status}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button className="p-1 hover:bg-gray-700 rounded text-blue-500" title="View Pods">
                              <Eye className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded text-green-500" title="Scale">
                              <Scale className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded text-yellow-500" title="Edit">
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

        {activeTab === 'nodes' && (
          <>
            {/* Node Management */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Server className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Total Nodes</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.overview.totalNodes}</p>
                <p className="text-xs text-gray-500 mt-1">{kubernetesData.overview.readyNodes} ready</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Layers className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Node Pools</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.nodePools.length}</p>
                <p className="text-xs text-gray-500 mt-1">across clusters</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Cpu className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Avg CPU</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.overview.avgCpuUtilization}%</p>
                <p className="text-xs text-gray-500 mt-1">utilization</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <HardDrive className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Avg Memory</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.overview.avgMemoryUtilization}%</p>
                <p className="text-xs text-gray-500 mt-1">utilization</p>
              </motion.div>
            </div>

            {/* Node Pools Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">Node Pools</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Pool Name</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Cluster</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">VM Size</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">OS Type</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Mode</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Nodes</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Auto Scaling</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {kubernetesData.nodePools.map((pool: any) => (
                      <motion.tr
                        key={pool.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-3">
                            <div className="bg-blue-900/20 text-blue-500 p-2 rounded">
                              <Layers className="w-4 h-4" />
                            </div>
                            <div>
                              <div className="font-medium">{pool.name}</div>
                              {pool.gpuType && <div className="text-xs text-purple-400">GPU: {pool.gpuType}</div>}
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm">{pool.cluster}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{pool.vmSize}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm">{pool.osType}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center px-2 py-1 text-xs rounded ${
                            pool.mode === 'System' ? 'bg-red-900/20 text-red-500' : 'bg-blue-900/20 text-blue-500'
                          }`}>
                            {pool.mode}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="font-medium">{pool.currentCount}</div>
                            <div className="text-gray-400 text-xs">{pool.readyNodes} ready</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            {pool.autoScaling ? (
                              <div className="text-green-500">
                                <div className="flex items-center space-x-1">
                                  <CheckCircle className="w-3 h-3" />
                                  <span>Enabled</span>
                                </div>
                                <div className="text-xs text-gray-400">{pool.minCount}-{pool.maxCount}</div>
                              </div>
                            ) : (
                              <div className="text-gray-500">
                                <div className="flex items-center space-x-1">
                                  <XCircle className="w-3 h-3" />
                                  <span>Disabled</span>
                                </div>
                              </div>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            pool.status === 'ready' ? 'text-green-500' : 'text-gray-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              pool.status === 'ready' ? 'bg-green-500' : 'bg-gray-500'
                            }`} />
                            <span className="uppercase">{pool.status}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button className="p-1 hover:bg-gray-700 rounded text-green-500" title="Scale Pool">
                              <Scale className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded text-yellow-500" title="Configure">
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

        {activeTab === 'services' && (
          <>
            {/* Services Management */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Network className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Total Services</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.services.length}</p>
                <p className="text-xs text-gray-500 mt-1">across all clusters</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Globe className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Load Balancers</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.services.filter((s: any) => s.type === 'LoadBalancer').length}</p>
                <p className="text-xs text-gray-500 mt-1">external services</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Link className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Cluster IPs</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.services.filter((s: any) => s.type === 'ClusterIP').length}</p>
                <p className="text-xs text-gray-500 mt-1">internal services</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Activity className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Active</span>
                </div>
                <p className="text-2xl font-bold font-mono">{kubernetesData.services.filter((s: any) => s.status === 'active').length}</p>
                <p className="text-xs text-gray-500 mt-1">healthy services</p>
              </motion.div>
            </div>

            {/* Services Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase">Kubernetes Services</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Service Name</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Type</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Namespace</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Cluster</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Cluster IP</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">External IP</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Ports</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {kubernetesData.services.map((service: any) => (
                      <motion.tr
                        key={service.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded ${
                              service.type === 'LoadBalancer' ? 'bg-green-900/20 text-green-500' : 'bg-blue-900/20 text-blue-500'
                            }`}>
                              {service.type === 'LoadBalancer' ? <Globe className="w-4 h-4" /> : <Link className="w-4 h-4" />}
                            </div>
                            <div>
                              <div className="font-medium">{service.name}</div>
                              <div className="text-sm text-gray-400">{Object.keys(service.selector).map(key => `${key}=${service.selector[key]}`).join(', ')}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-medium">{service.type}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm">{service.namespace}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm">{service.cluster}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{service.clusterIP}</span>
                        </td>
                        <td className="px-4 py-3">
                          {service.externalIP ? (
                            <span className="text-sm font-mono">{service.externalIP}</span>
                          ) : (
                            <span className="text-sm text-gray-500">None</span>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            {service.ports.map((port: any, index: number) => (
                              <div key={index} className="text-xs">
                                {port.port}:{port.targetPort}/{port.protocol}
                              </div>
                            ))}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            service.status === 'active' ? 'text-green-500' : 'text-gray-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              service.status === 'active' ? 'bg-green-500' : 'bg-gray-500'
                            }`} />
                            <span className="uppercase">{service.status}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button className="p-1 hover:bg-gray-700 rounded text-blue-500" title="View Details">
                              <Eye className="w-4 h-4" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded text-yellow-500" title="Edit Service">
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

        {activeTab === 'events' && (
          <>
            {/* Events Management */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">Cluster Events</h3>
                  <div className="flex items-center space-x-2">
                    <button className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-white text-sm rounded">
                      <RefreshCw className="w-4 h-4 mr-2 inline" />
                      Refresh
                    </button>
                    <select className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm">
                      <option value="all">All Types</option>
                      <option value="Normal">Normal</option>
                      <option value="Warning">Warning</option>
                    </select>
                  </div>
                </div>
              </div>
              <div className="max-h-96 overflow-y-auto">
                <div className="space-y-2 p-4">
                  {kubernetesData.events.map((event: any) => (
                    <motion.div
                      key={event.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="flex items-start space-x-3 p-3 bg-gray-800/30 rounded-lg hover:bg-gray-800/50 transition-colors"
                    >
                      <div className={`flex-shrink-0 w-2 h-2 rounded-full mt-2 ${
                        event.type === 'Normal' ? 'bg-green-500' :
                        event.type === 'Warning' ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium text-white">{event.reason}</p>
                          <p className="text-xs text-gray-500">{new Date(event.timestamp).toLocaleString()}</p>
                        </div>
                        <p className="text-sm text-gray-400 mt-1">{event.object}</p>
                        <p className="text-xs text-gray-500 mt-1">{event.message}</p>
                        <div className="flex items-center mt-2 space-x-2">
                          <span className="text-xs text-gray-500">Cluster: {event.cluster}</span>
                          <span className={`text-xs px-2 py-1 rounded ${
                            event.type === 'Normal' ? 'bg-green-900/20 text-green-500' :
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
        {!['overview', 'clusters', 'workloads', 'nodes', 'services', 'events'].includes(activeTab) && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-bold mb-4 capitalize">{activeTab} Management</h2>
            <p className="text-gray-400">Detailed {activeTab} management interface coming soon...</p>
          </div>
        )}
      </div>
    </div>
  )
}