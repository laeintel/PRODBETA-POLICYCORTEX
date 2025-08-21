/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 18/234,567 - Conversational Governance Intelligence System  
 * - US Patent Application 19/345,678 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2026 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect } from 'react'
import { toast } from '@/hooks/useToast'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Brain,
  Cpu,
  Zap,
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  Users,
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
  Database,
  GitBranch,
  Code,
  Target,
  Lightbulb,
  Microscope,
  Beaker,
  LineChart,
  PieChart,
  CloudLightning,
  Workflow,
  Layers3
} from 'lucide-react'
import { Line, Bar, Doughnut, Radar, Scatter } from 'react-chartjs-2'
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
  RadialLinearScale
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
  RadialLinearScale
)

export default function AIDashboardPage() {
  const [loading, setLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [activeTab, setActiveTab] = useState('overview')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [aiData, setAiData] = useState<any>(null)
  const [realTimeMetrics, setRealTimeMetrics] = useState<any[]>([])
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [filterStatus, setFilterStatus] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    loadAllData()
    const interval = autoRefresh ? setInterval(loadRealTimeData, 3000) : null
    return () => { if (interval) clearInterval(interval) }
  }, [autoRefresh])

  const loadAllData = () => {
    setTimeout(() => {
      setAiData({
        overview: {
          totalModels: 89,
          activeModels: 76,
          trainingJobs: 12,
          inferenceEndpoints: 45,
          totalPredictions: 2847356,
          accuracyScore: 94.7,
          averageLatency: 125,
          monthlyComputeHours: 12456,
          currentSpend: 8934.50,
          predictionTrend: 15.6
        },
        models: [
          {
            id: 'model-001',
            name: 'governance-classifier-v3',
            type: 'Classification',
            framework: 'PyTorch',
            version: '3.2.1',
            status: 'serving',
            accuracy: 96.8,
            latency: 89,
            throughput: 1250,
            deploymentDate: '2024-03-15',
            lastTrained: '2024-03-10',
            predictions24h: 45678,
            errorRate: 0.2,
            computeHours: 156.7,
            cost: 234.56,
            environment: 'production',
            endpoint: 'https://api.policycortex.ai/v1/classify'
          },
          {
            id: 'model-002',
            name: 'anomaly-detector-v2',
            type: 'Anomaly Detection',
            framework: 'TensorFlow',
            version: '2.1.5',
            status: 'serving',
            accuracy: 92.4,
            latency: 156,
            throughput: 890,
            deploymentDate: '2024-03-12',
            lastTrained: '2024-03-08',
            predictions24h: 23456,
            errorRate: 0.8,
            computeHours: 89.3,
            cost: 145.67,
            environment: 'production',
            endpoint: 'https://api.policycortex.ai/v1/anomaly'
          },
          {
            id: 'model-003',
            name: 'policy-recommender-v1',
            type: 'Recommendation',
            framework: 'Scikit-learn',
            version: '1.4.2',
            status: 'training',
            accuracy: 88.9,
            latency: 234,
            throughput: 567,
            deploymentDate: null,
            lastTrained: '2024-03-18',
            predictions24h: 0,
            errorRate: 0.0,
            computeHours: 234.8,
            cost: 89.12,
            environment: 'development',
            endpoint: null
          },
          {
            id: 'model-004',
            name: 'compliance-predictor-v4',
            type: 'Regression',
            framework: 'XGBoost',
            version: '4.1.0',
            status: 'serving',
            accuracy: 94.2,
            latency: 67,
            throughput: 1890,
            deploymentDate: '2024-03-20',
            lastTrained: '2024-03-18',
            predictions24h: 78912,
            errorRate: 0.1,
            computeHours: 67.4,
            cost: 156.78,
            environment: 'production',
            endpoint: 'https://api.policycortex.ai/v1/predict'
          },
          {
            id: 'model-005',
            name: 'risk-assessment-v2',
            type: 'Risk Analysis',
            framework: 'PyTorch',
            version: '2.3.7',
            status: 'error',
            accuracy: 91.6,
            latency: 345,
            throughput: 234,
            deploymentDate: '2024-03-05',
            lastTrained: '2024-02-28',
            predictions24h: 0,
            errorRate: 15.6,
            computeHours: 45.2,
            cost: 67.89,
            environment: 'production',
            endpoint: 'https://api.policycortex.ai/v1/risk'
          }
        ],
        pipelines: [
          {
            id: 'pipeline-001',
            name: 'Daily Governance Analysis',
            status: 'running',
            models: ['governance-classifier-v3', 'compliance-predictor-v4'],
            schedule: '0 2 * * *',
            lastRun: '2024-03-21 02:00:00',
            nextRun: '2024-03-22 02:00:00',
            duration: '45m 32s',
            success: true,
            recordsProcessed: 234567
          },
          {
            id: 'pipeline-002',
            name: 'Real-time Anomaly Detection',
            status: 'running',
            models: ['anomaly-detector-v2'],
            schedule: 'Real-time',
            lastRun: '2024-03-21 15:30:00',
            nextRun: 'Continuous',
            duration: '156ms avg',
            success: true,
            recordsProcessed: 1234567
          },
          {
            id: 'pipeline-003',
            name: 'Weekly Policy Recommendations',
            status: 'scheduled',
            models: ['policy-recommender-v1'],
            schedule: '0 0 * * 0',
            lastRun: '2024-03-17 00:00:00',
            nextRun: '2024-03-24 00:00:00',
            duration: '2h 15m',
            success: true,
            recordsProcessed: 56789
          }
        ],
        experiments: [
          {
            id: 'exp-001',
            name: 'Governance Classifier Optimization',
            status: 'completed',
            type: 'Hyperparameter Tuning',
            startTime: '2024-03-15 09:00:00',
            endTime: '2024-03-15 18:30:00',
            duration: '9h 30m',
            bestAccuracy: 96.8,
            baseline: 94.2,
            improvement: 2.6,
            trials: 156
          },
          {
            id: 'exp-002',
            name: 'Feature Selection Analysis',
            status: 'running',
            type: 'Feature Engineering',
            startTime: '2024-03-20 14:00:00',
            endTime: null,
            duration: '1h 45m',
            bestAccuracy: 93.4,
            baseline: 91.8,
            improvement: 1.6,
            trials: 45
          }
        ],
        dataLakes: {
          totalDatasets: 234,
          totalSize: '45.7 TB',
          trainingData: '32.1 TB',
          validationData: '8.9 TB',
          testData: '4.7 TB',
          lastUpdated: '2024-03-21 08:30:00',
          quality: 94.2
        },
        infrastructure: {
          gpuClusters: 8,
          totalGPUs: 156,
          activeGPUs: 134,
          cpuCores: 2456,
          memory: '12.3 TB',
          storage: '234 TB',
          utilization: 78.4
        }
      })

      setRealTimeMetrics(generateRealTimeData())
      setLoading(false)
    }, 1200)
  }

  const loadRealTimeData = () => {
    setRealTimeMetrics(prev => {
      const newData = [...prev, {
        timestamp: new Date(),
        predictions: 1000 + Math.random() * 500,
        accuracy: 94 + Math.random() * 4,
        latency: 100 + Math.random() * 50,
        throughput: 800 + Math.random() * 400,
        errors: Math.random() * 10
      }]
      return newData.slice(-20)
    })
  }

  const generateRealTimeData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 180000),
      predictions: 1000 + Math.random() * 500,
      accuracy: 94 + Math.random() * 4,
      latency: 100 + Math.random() * 50,
      throughput: 800 + Math.random() * 400,
      errors: Math.random() * 10
    }))
  }

  const performanceData = {
    labels: realTimeMetrics.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [
      {
        label: 'Predictions/min',
        data: realTimeMetrics.map(d => d.predictions),
        borderColor: 'rgb(139, 92, 246)',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        tension: 0.4,
        fill: true,
        yAxisID: 'y'
      },
      {
        label: 'Avg Latency (ms)',
        data: realTimeMetrics.map(d => d.latency),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4,
        fill: true,
        yAxisID: 'y1'
      }
    ]
  }

  const accuracyData = {
    labels: realTimeMetrics.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [
      {
        label: 'Model Accuracy %',
        data: realTimeMetrics.map(d => d.accuracy),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  }

  const modelTypeData = {
    labels: ['Classification', 'Regression', 'Anomaly Detection', 'Recommendation', 'Risk Analysis'],
    datasets: [{
      data: [35, 28, 18, 12, 7],
      backgroundColor: [
        'rgba(139, 92, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(59, 130, 246, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(239, 68, 68, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const frameworkData = {
    labels: ['PyTorch', 'TensorFlow', 'Scikit-learn', 'XGBoost', 'Keras'],
    datasets: [{
      data: [42, 28, 15, 10, 5],
      backgroundColor: [
        'rgba(239, 68, 68, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(59, 130, 246, 0.8)',
        'rgba(139, 92, 246, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const modelCapabilitiesData = {
    labels: ['Accuracy', 'Speed', 'Scalability', 'Robustness', 'Interpretability', 'Cost Efficiency'],
    datasets: [{
      label: 'Current Performance',
      data: [94.7, 87.3, 91.8, 89.2, 76.4, 82.9],
      backgroundColor: 'rgba(139, 92, 246, 0.2)',
      borderColor: 'rgb(139, 92, 246)',
      pointBackgroundColor: 'rgb(139, 92, 246)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgb(139, 92, 246)'
    }, {
      label: 'Target Performance',
      data: [98, 92, 95, 93, 85, 88],
      backgroundColor: 'rgba(16, 185, 129, 0.2)',
      borderColor: 'rgb(16, 185, 129)',
      pointBackgroundColor: 'rgb(16, 185, 129)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgb(16, 185, 129)'
    }]
  }

  const filteredModels = aiData?.models.filter((model: any) => {
    const matchesStatus = filterStatus === 'all' || model.status === filterStatus
    const matchesSearch = model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         model.type.toLowerCase().includes(searchTerm.toLowerCase())
    return matchesStatus && matchesSearch
  }) || []

  const handleModelAction = (action: string, modelId: string) => {
    console.log(`${action} model: ${modelId}`)
  }

  const handleBulkAction = (action: string) => {
    console.log(`${action} models:`, selectedModels)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Initializing AI Intelligence Center...</p>
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
              <Brain className="w-8 h-8 text-purple-500" />
              <div>
                <h1 className="text-2xl font-bold">AI Intelligence Center</h1>
                <p className="text-sm text-gray-500">Machine learning models, pipelines, and AI-driven insights</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">{aiData.overview.activeModels} MODELS ACTIVE</span>
              </div>
              <div className="flex items-center space-x-2">
                <Target className="w-4 h-4 text-green-500" />
                <span className="text-sm text-gray-400">{aiData.overview.accuracyScore}% AVG ACCURACY</span>
              </div>
              <button type="button" 
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`p-2 rounded ${autoRefresh ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400'}`}
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
              <button
                type="button"
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded transition-colors"
                onClick={() => toast({ title: 'Deploy', description: 'Deploy model flow coming soon' })}
              >
                DEPLOY MODEL
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'models', 'training', 'pipelines', 'experiments', 'data'].map((tab) => (
            <button type="button"
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-3 px-1 border-b-2 transition-colors capitalize ${
                activeTab === tab
                  ? 'border-purple-500 text-purple-500'
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
            {/* AI Overview Metrics */}
            <div className="grid grid-cols-6 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Brain className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Total Models</span>
                </div>
                <p className="text-2xl font-bold font-mono">{aiData.overview.totalModels}</p>
                <div className="flex items-center mt-1">
                  <CheckCircle className="w-3 h-3 text-green-500 mr-1" />
                  <span className="text-xs text-green-500">{aiData.overview.activeModels} active</span>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Target className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Accuracy</span>
                </div>
                <p className="text-2xl font-bold font-mono">{aiData.overview.accuracyScore}%</p>
                <p className="text-xs text-gray-500 mt-1">average score</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Zap className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Latency</span>
                </div>
                <p className="text-2xl font-bold font-mono">{aiData.overview.averageLatency}ms</p>
                <p className="text-xs text-gray-500 mt-1">average response</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Activity className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Predictions</span>
                </div>
                <p className="text-2xl font-bold font-mono">{(aiData.overview.totalPredictions / 1000000).toFixed(1)}M</p>
                <div className="flex items-center mt-1">
                  <TrendingUp className="w-3 h-3 text-green-500 mr-1" />
                  <span className="text-xs text-green-500">+{aiData.overview.predictionTrend}% today</span>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Beaker className="w-5 h-5 text-pink-500" />
                  <span className="text-xs text-gray-500">Training</span>
                </div>
                <p className="text-2xl font-bold font-mono">{aiData.overview.trainingJobs}</p>
                <p className="text-xs text-gray-500 mt-1">active jobs</p>
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
                <p className="text-2xl font-bold font-mono">${(aiData.overview.currentSpend / 1000).toFixed(1)}K</p>
                <p className="text-xs text-gray-500 mt-1">monthly spend</p>
              </motion.div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {/* Real-time Performance */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">REAL-TIME PERFORMANCE</h3>
                  <div className="flex items-center space-x-2">
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-purple-500 rounded-full" />
                      <span className="text-xs text-gray-500">Predictions</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-red-500 rounded-full" />
                      <span className="text-xs text-gray-500">Latency</span>
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

              {/* Model Type Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">MODEL TYPE DISTRIBUTION</h3>
                <div className="h-64">
                  <Doughnut data={modelTypeData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: { color: 'rgba(255, 255, 255, 0.7)', font: { size: 10 } }
                      }
                    }
                  }} />
                </div>
              </div>

              {/* Framework Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">FRAMEWORK DISTRIBUTION</h3>
                <div className="h-64">
                  <Doughnut data={frameworkData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: { color: 'rgba(255, 255, 255, 0.7)', font: { size: 10 } }
                      }
                    }
                  }} />
                </div>
              </div>
            </div>

            {/* Model Accuracy Trend */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase">MODEL ACCURACY TREND</h3>
                <div className="flex items-center space-x-2">
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full" />
                    <span className="text-xs text-gray-500">Average Accuracy</span>
                  </div>
                </div>
              </div>
              <div className="h-64">
                <Line data={accuracyData} options={{
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
                      min: 90,
                      max: 100
                    }
                  }
                }} />
              </div>
            </div>

            {/* AI Capabilities Radar */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
              <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">AI CAPABILITIES ASSESSMENT</h3>
              <div className="h-80">
                <Radar data={modelCapabilitiesData} options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'bottom',
                      labels: { color: 'rgba(255, 255, 255, 0.7)' }
                    }
                  },
                  scales: {
                    r: {
                      beginAtZero: true,
                      max: 100,
                      ticks: { 
                        color: 'rgba(255, 255, 255, 0.5)',
                        backdropColor: 'transparent'
                      },
                      grid: { color: 'rgba(255, 255, 255, 0.1)' },
                      pointLabels: { color: 'rgba(255, 255, 255, 0.7)' }
                    }
                  }
                }} />
              </div>
            </div>

            {/* Status Cards Grid */}
            <div className="grid grid-cols-4 gap-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">INFRASTRUCTURE</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">GPU Clusters</span>
                    <span className="font-mono text-sm">{aiData.infrastructure.gpuClusters}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Active GPUs</span>
                    <span className="font-mono text-sm text-green-500">{aiData.infrastructure.activeGPUs}/{aiData.infrastructure.totalGPUs}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">CPU Cores</span>
                    <span className="font-mono text-sm">{aiData.infrastructure.cpuCores}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Memory</span>
                    <span className="font-mono text-sm">{aiData.infrastructure.memory}</span>
                  </div>
                  <div className="pt-2 border-t border-gray-800">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-400">Utilization</span>
                      <span className="text-sm">{aiData.infrastructure.utilization}%</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full">
                      <div className="h-2 bg-purple-500 rounded-full" style={{ width: `${aiData.infrastructure.utilization}%` }} />
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">DATA LAKES</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Total Datasets</span>
                    <span className="font-mono text-sm">{aiData.dataLakes.totalDatasets}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Total Size</span>
                    <span className="font-mono text-sm">{aiData.dataLakes.totalSize}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Training Data</span>
                    <span className="font-mono text-sm text-blue-500">{aiData.dataLakes.trainingData}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Data Quality</span>
                    <span className="font-mono text-sm text-green-500">{aiData.dataLakes.quality}%</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">ACTIVE PIPELINES</h3>
                </div>
                <div className="p-4 space-y-3">
                  {aiData.pipelines.filter((p: any) => p.status === 'running').map((pipeline: any, idx: number) => (
                    <div key={idx} className="flex items-center justify-between p-2 bg-gray-800 rounded hover:bg-gray-700 transition-colors">
                      <div className="flex items-center space-x-2">
                        <Workflow className="w-4 h-4 text-gray-500" />
                        <span className="text-sm">{pipeline.name}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                        <span className="text-xs text-gray-500">{pipeline.duration}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">QUICK ACTIONS</h3>
                </div>
                <div className="p-4 space-y-2">
                  <button
                    type="button"
                    className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group"
                    onClick={() => toast({ title: 'Train model', description: 'Training job queued' })}
                  >
                    <span>Train New Model</span>
                    <Beaker className="w-4 h-4 text-gray-500 group-hover:text-white" />
                  </button>
                  <button
                    type="button"
                    className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group"
                    onClick={() => toast({ title: 'Experiment', description: 'Experiment started' })}
                  >
                    <span>Run Experiment</span>
                    <Beaker className="w-4 h-4 text-gray-500 group-hover:text-white" />
                  </button>
                  <button
                    type="button"
                    className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group"
                    onClick={() => toast({ title: 'Pipeline', description: 'Pipeline deployment queued' })}
                  >
                    <span>Deploy Pipeline</span>
                    <Workflow className="w-4 h-4 text-gray-500 group-hover:text-white" />
                  </button>
                  <button
                    type="button"
                    className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center justify-between group"
                    onClick={() => toast({ title: 'Report', description: 'Generating performance report...' })}
                  >
                    <span>Performance Report</span>
                    <Download className="w-4 h-4 text-gray-500 group-hover:text-white" />
                  </button>
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'models' && (
          <>
            {/* Model Management Controls */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <Search className="w-4 h-4 text-gray-500 absolute left-3 top-1/2 transform -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder="Search models..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm text-white placeholder-gray-500 focus:border-purple-500 focus:outline-none"
                    />
                  </div>
                  <select
                    value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value)}
                    className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                  >
                    <option value="all">All Status</option>
                    <option value="serving">Serving</option>
                    <option value="training">Training</option>
                    <option value="error">Error</option>
                    <option value="stopped">Stopped</option>
                  </select>
                </div>
                <div className="flex items-center space-x-2">
                  {selectedModels.length > 0 && (
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-400">{selectedModels.length} selected</span>
                      <button type="button" 
                        onClick={() => handleBulkAction('deploy')}
                        className="px-3 py-1.5 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded"
                      >
                        Deploy
                      </button>
                      <button type="button" 
                        onClick={() => handleBulkAction('stop')}
                        className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-sm rounded"
                      >
                        Stop
                      </button>
                    </div>
                  )}
                  <button type="button" className="p-2 hover:bg-gray-800 rounded" onClick={() => toast({ title: 'Filters', description: 'Filter dialog coming soon' })}>
                    <Filter className="w-4 h-4 text-gray-500" />
                  </button>
                  <button type="button" className="p-2 hover:bg-gray-800 rounded" onClick={() => toast({ title: 'Export', description: 'Exporting dashboard data...' })}>
                    <Download className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </div>
            </div>

            {/* Models Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left">
                        <input
                          type="checkbox"
                          checked={selectedModels.length === filteredModels.length && filteredModels.length > 0}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedModels(filteredModels.map((model: any) => model.id))
                            } else {
                              setSelectedModels([])
                            }
                          }}
                          className="rounded border-gray-600 bg-gray-700 text-purple-600"
                        />
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Name</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Type</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Framework</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Accuracy</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Latency</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Predictions/24h</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Cost</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {filteredModels.map((model: any) => (
                      <motion.tr
                        key={model.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <input
                            type="checkbox"
                            checked={selectedModels.includes(model.id)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedModels([...selectedModels, model.id])
                              } else {
                                setSelectedModels(selectedModels.filter(id => id !== model.id))
                              }
                            }}
                            className="rounded border-gray-600 bg-gray-700 text-purple-600"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <div>
                            <div className="font-medium">{model.name}</div>
                            <div className="text-sm text-gray-400">v{model.version}</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="inline-flex items-center px-2 py-1 text-xs rounded bg-purple-900/30 text-purple-500">
                            {model.type}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="inline-flex items-center px-2 py-1 text-xs rounded bg-blue-900/30 text-blue-500">
                            {model.framework}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center space-x-1 text-xs font-medium ${
                            model.status === 'serving' ? 'text-green-500' :
                            model.status === 'training' ? 'text-yellow-500' :
                            model.status === 'error' ? 'text-red-500' :
                            'text-gray-500'
                          }`}>
                            <span className={`w-2 h-2 rounded-full ${
                              model.status === 'serving' ? 'bg-green-500' :
                              model.status === 'training' ? 'bg-yellow-500 animate-pulse' :
                              model.status === 'error' ? 'bg-red-500' :
                              'bg-gray-500'
                            }`} />
                            <span className="uppercase">{model.status}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm">
                            <div className="flex items-center space-x-1">
                              <span>{model.accuracy.toFixed(1)}%</span>
                              <div className="w-12 bg-gray-800 rounded-full h-1.5">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    model.accuracy > 95 ? 'bg-green-500' :
                                    model.accuracy > 90 ? 'bg-yellow-500' :
                                    'bg-red-500'
                                  }`}
                                  style={{ width: `${Math.min(model.accuracy, 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`text-sm font-mono ${
                            model.latency < 100 ? 'text-green-500' :
                            model.latency < 200 ? 'text-yellow-500' :
                            'text-red-500'
                          }`}>
                            {model.latency}ms
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">{model.predictions24h.toLocaleString()}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono">${model.cost}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            {model.status === 'serving' ? (
                              <button type="button" 
                                onClick={() => handleModelAction('stop', model.id)}
                                className="p-1 hover:bg-gray-700 rounded text-red-500"
                                title="Stop Model"
                              >
                                <StopCircle className="w-4 h-4" />
                              </button>
                            ) : (
                              <button type="button" 
                                onClick={() => handleModelAction('deploy', model.id)}
                                className="p-1 hover:bg-gray-700 rounded text-green-500"
                                title="Deploy Model"
                              >
                                <Play className="w-4 h-4" />
                              </button>
                            )}
                            <button type="button" 
                              onClick={() => handleModelAction('retrain', model.id)}
                              className="p-1 hover:bg-gray-700 rounded text-purple-500"
                              title="Retrain Model"
                            >
                              <RotateCcw className="w-4 h-4" />
                            </button>
                            <button type="button" className="p-1 hover:bg-gray-700 rounded" onClick={() => toast({ title: 'Monitor', description: 'Opening monitoring view (coming soon)' })}>
                              <Monitor className="w-4 h-4 text-gray-400" />
                            </button>
                            <button type="button" className="p-1 hover:bg-gray-700 rounded" onClick={() => toast({ title: 'More', description: 'More actions menu (coming soon)' })}>
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
        {!['overview', 'models'].includes(activeTab) && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-bold mb-4 capitalize">{activeTab} Management</h2>
            <p className="text-gray-400">Detailed {activeTab} management interface coming soon...</p>
          </div>
        )}
      </div>
    </div>
  )
}