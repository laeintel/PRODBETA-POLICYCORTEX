'use client'

import React, { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Activity,
  Bell,
  Filter,
  Download,
  Search,
  Calendar,
  Eye,
  EyeOff,
  RefreshCw,
  Settings,
  ChevronRight,
  Clock,
  DollarSign,
  Server,
  Network,
  Database,
  Shield,
  Zap,
  BarChart3,
  ArrowUp,
  ArrowDown,
  ArrowRight,
  AlertCircle,
  CheckCircle,
  XCircle,
  Info,
  Target,
  Gauge,
  Brain,
  Sparkles,
  Hash,
  Percent,
  Users,
  Package,
  Cpu,
  HardDrive,
  Wifi,
  Lock,
  Unlock,
  GitBranch,
  X,
  Check,
  Plus,
  Minus,
  Edit2,
  Trash2,
  Save,
  Copy,
  Share2,
  ExternalLink,
  FileText,
  PieChart,
  LineChart,
  Layers,
  Grid,
  List
} from 'lucide-react'
import { Line, Bar, Doughnut, Scatter, Radar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'
import { Progress } from '@/components/ui/progress'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Alert, AlertDescription } from '@/components/ui/alert'

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
)

// Enhanced mock data for comprehensive anomaly detection
const mockAnomalies = [
  {
    id: 'ANM-001',
    type: 'cost',
    category: 'Billing',
    severity: 'critical',
    confidence: 95,
    status: 'active',
    resource: 'Azure SQL Database',
    resourceId: 'sql-prod-001',
    metric: 'Daily Cost',
    baseline: 450,
    current: 1250,
    deviation: '+177.8%',
    impact: 800,
    detected: '2024-01-19T10:30:00Z',
    duration: '4h 30m',
    pattern: 'sudden_spike',
    correlations: ['CPU Usage', 'Storage Growth', 'Query Count'],
    affectedServices: ['API Gateway', 'Web App', 'Analytics'],
    predictedCost: 37500,
    recommendations: [
      'Review database SKU sizing',
      'Enable auto-pause for dev/test',
      'Implement query optimization'
    ],
    rootCause: 'Unoptimized queries causing high DTU consumption',
    autoResolve: true,
    mlScore: 0.92,
    tags: ['production', 'database', 'cost-optimization']
  },
  {
    id: 'ANM-002',
    type: 'performance',
    category: 'Compute',
    severity: 'high',
    confidence: 88,
    status: 'investigating',
    resource: 'Virtual Machine Scale Set',
    resourceId: 'vmss-web-prod',
    metric: 'Response Time',
    baseline: 200,
    current: 850,
    deviation: '+325%',
    impact: 650,
    detected: '2024-01-19T11:45:00Z',
    duration: '3h 15m',
    pattern: 'gradual_degradation',
    correlations: ['Memory Usage', 'Network Latency', 'Disk I/O'],
    affectedServices: ['User Portal', 'Admin Dashboard'],
    predictedCost: 0,
    recommendations: [
      'Scale out instances',
      'Clear application cache',
      'Review memory allocation'
    ],
    rootCause: 'Memory leak in application code',
    autoResolve: false,
    mlScore: 0.85,
    tags: ['performance', 'vmss', 'critical-path']
  },
  {
    id: 'ANM-003',
    type: 'security',
    category: 'Access',
    severity: 'medium',
    confidence: 76,
    status: 'resolved',
    resource: 'Storage Account',
    resourceId: 'storage-backup-001',
    metric: 'Failed Auth Attempts',
    baseline: 5,
    current: 187,
    deviation: '+3640%',
    impact: 182,
    detected: '2024-01-19T08:00:00Z',
    duration: '7h 0m',
    pattern: 'brute_force',
    correlations: ['IP Reputation', 'Geo Location', 'Time Pattern'],
    affectedServices: ['Backup Service'],
    predictedCost: 0,
    recommendations: [
      'Enable IP filtering',
      'Implement MFA',
      'Review access policies'
    ],
    rootCause: 'Attempted brute force attack from suspicious IPs',
    autoResolve: true,
    mlScore: 0.73,
    tags: ['security', 'storage', 'resolved']
  },
  {
    id: 'ANM-004',
    type: 'availability',
    category: 'Network',
    severity: 'low',
    confidence: 92,
    status: 'monitoring',
    resource: 'Application Gateway',
    resourceId: 'appgw-main',
    metric: 'Health Probe Failures',
    baseline: 0,
    current: 12,
    deviation: '+∞',
    impact: 12,
    detected: '2024-01-19T13:20:00Z',
    duration: '1h 40m',
    pattern: 'intermittent',
    correlations: ['Backend Health', 'SSL Certificate'],
    affectedServices: ['Public API'],
    predictedCost: 0,
    recommendations: [
      'Check backend health',
      'Review probe configuration',
      'Monitor SSL expiry'
    ],
    rootCause: 'Intermittent backend connectivity issues',
    autoResolve: false,
    mlScore: 0.88,
    tags: ['network', 'availability', 'monitoring']
  },
  {
    id: 'ANM-005',
    type: 'capacity',
    category: 'Storage',
    severity: 'high',
    confidence: 97,
    status: 'active',
    resource: 'Cosmos DB',
    resourceId: 'cosmos-prod-001',
    metric: 'Storage Usage',
    baseline: 75,
    current: 94,
    deviation: '+25.3%',
    impact: 19,
    detected: '2024-01-19T09:15:00Z',
    duration: '5h 45m',
    pattern: 'linear_growth',
    correlations: ['Document Count', 'Partition Size'],
    affectedServices: ['Data Pipeline', 'Analytics Engine'],
    predictedCost: 2500,
    recommendations: [
      'Implement data archival',
      'Review retention policies',
      'Scale storage capacity'
    ],
    rootCause: 'Rapid data growth exceeding projections',
    autoResolve: false,
    mlScore: 0.94,
    tags: ['capacity', 'database', 'scaling']
  }
]

// Time series data for charts
const generateTimeSeriesData = (hours: number = 24) => {
  const data = []
  const now = new Date()
  for (let i = hours; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 60 * 60 * 1000)
    data.push({
      time: time.toISOString(),
      normal: Math.floor(Math.random() * 20) + 80,
      anomalies: Math.floor(Math.random() * 15) + 5,
      critical: Math.floor(Math.random() * 5),
      resolved: Math.floor(Math.random() * 10) + 10
    })
  }
  return data
}

// ML model confidence scores
const mlModels = [
  { name: 'Time Series Forecasting', accuracy: 94, status: 'active' },
  { name: 'Pattern Recognition', accuracy: 89, status: 'active' },
  { name: 'Correlation Analysis', accuracy: 91, status: 'training' },
  { name: 'Root Cause Analysis', accuracy: 86, status: 'active' },
  { name: 'Impact Prediction', accuracy: 88, status: 'active' }
]

export default function AnomaliesPage() {
  const [selectedAnomaly, setSelectedAnomaly] = useState<any>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedSeverity, setSelectedSeverity] = useState('all')
  const [selectedType, setSelectedType] = useState('all')
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [showResolved, setShowResolved] = useState(true)
  const [autoResolve, setAutoResolve] = useState(true)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [showDetailsDialog, setShowDetailsDialog] = useState(false)
  const [refreshInterval, setRefreshInterval] = useState(30)
  const [selectedAnomalies, setSelectedAnomalies] = useState<string[]>([])
  const [sensitivityThreshold, setSensitivityThreshold] = useState([75])
  const [showPredictions, setShowPredictions] = useState(true)
  const [groupByCategory, setGroupByCategory] = useState(false)
  const [notificationSettings, setNotificationSettings] = useState({
    email: true,
    sms: false,
    slack: true,
    teams: false,
    webhook: false
  })

  // Filter anomalies based on search and filters
  const filteredAnomalies = useMemo(() => {
    return mockAnomalies.filter(anomaly => {
      const matchesSearch = anomaly.resource.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           anomaly.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           anomaly.rootCause.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesSeverity = selectedSeverity === 'all' || anomaly.severity === selectedSeverity
      const matchesType = selectedType === 'all' || anomaly.type === selectedType
      const matchesStatus = showResolved || anomaly.status !== 'resolved'
      const matchesSensitivity = anomaly.confidence >= sensitivityThreshold[0]
      
      return matchesSearch && matchesSeverity && matchesType && matchesStatus && matchesSensitivity
    })
  }, [searchQuery, selectedSeverity, selectedType, showResolved, sensitivityThreshold])

  // Group anomalies by category if enabled
  const groupedAnomalies = useMemo(() => {
    if (!groupByCategory) return { 'All': filteredAnomalies }
    
    return filteredAnomalies.reduce((acc, anomaly) => {
      const category = anomaly.category
      if (!acc[category]) acc[category] = []
      acc[category].push(anomaly)
      return acc
    }, {} as Record<string, typeof mockAnomalies>)
  }, [filteredAnomalies, groupByCategory])

  // Calculate statistics
  const stats = useMemo(() => {
    const active = filteredAnomalies.filter(a => a.status === 'active').length
    const critical = filteredAnomalies.filter(a => a.severity === 'critical').length
    const totalImpact = filteredAnomalies.reduce((sum, a) => sum + (a.predictedCost || 0), 0)
    const avgConfidence = filteredAnomalies.reduce((sum, a) => sum + a.confidence, 0) / (filteredAnomalies.length || 1)
    
    return { active, critical, totalImpact, avgConfidence }
  }, [filteredAnomalies])

  // Time series data
  const timeSeriesData = useMemo(() => generateTimeSeriesData(), [])

  // Chart configurations
  const anomalyTrendChart = {
    labels: timeSeriesData.map(d => new Date(d.time).toLocaleTimeString()),
    datasets: [
      {
        label: 'Anomalies Detected',
        data: timeSeriesData.map(d => d.anomalies),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Critical',
        data: timeSeriesData.map(d => d.critical),
        borderColor: 'rgb(220, 38, 38)',
        backgroundColor: 'rgba(220, 38, 38, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Resolved',
        data: timeSeriesData.map(d => d.resolved),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  }

  const severityDistribution = {
    labels: ['Critical', 'High', 'Medium', 'Low'],
    datasets: [{
      data: [
        mockAnomalies.filter(a => a.severity === 'critical').length,
        mockAnomalies.filter(a => a.severity === 'high').length,
        mockAnomalies.filter(a => a.severity === 'medium').length,
        mockAnomalies.filter(a => a.severity === 'low').length
      ],
      backgroundColor: [
        'rgba(220, 38, 38, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(251, 146, 60, 0.8)',
        'rgba(250, 204, 21, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const typeDistribution = {
    labels: ['Cost', 'Performance', 'Security', 'Availability', 'Capacity'],
    datasets: [{
      label: 'Anomaly Count',
      data: [
        mockAnomalies.filter(a => a.type === 'cost').length,
        mockAnomalies.filter(a => a.type === 'performance').length,
        mockAnomalies.filter(a => a.type === 'security').length,
        mockAnomalies.filter(a => a.type === 'availability').length,
        mockAnomalies.filter(a => a.type === 'capacity').length
      ],
      backgroundColor: 'rgba(59, 130, 246, 0.8)',
      borderColor: 'rgb(59, 130, 246)',
      borderWidth: 1
    }]
  }

  const mlPerformanceChart = {
    labels: mlModels.map(m => m.name),
    datasets: [{
      label: 'Model Accuracy',
      data: mlModels.map(m => m.accuracy),
      backgroundColor: [
        'rgba(34, 197, 94, 0.8)',
        'rgba(34, 197, 94, 0.8)',
        'rgba(251, 146, 60, 0.8)',
        'rgba(34, 197, 94, 0.8)',
        'rgba(34, 197, 94, 0.8)'
      ],
      borderColor: [
        'rgb(34, 197, 94)',
        'rgb(34, 197, 94)',
        'rgb(251, 146, 60)',
        'rgb(34, 197, 94)',
        'rgb(34, 197, 94)'
      ],
      borderWidth: 1
    }]
  }

  // Pattern analysis radar chart
  const patternAnalysis = {
    labels: ['Sudden Spike', 'Gradual Degradation', 'Intermittent', 'Linear Growth', 'Cyclic', 'Random'],
    datasets: [{
      label: 'Pattern Frequency',
      data: [65, 45, 30, 55, 40, 20],
      backgroundColor: 'rgba(147, 51, 234, 0.2)',
      borderColor: 'rgb(147, 51, 234)',
      pointBackgroundColor: 'rgb(147, 51, 234)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgb(147, 51, 234)'
    }]
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-500 bg-red-500/10 border-red-500/20'
      case 'high': return 'text-orange-500 bg-orange-500/10 border-orange-500/20'
      case 'medium': return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/20'
      case 'low': return 'text-blue-500 bg-blue-500/10 border-blue-500/20'
      default: return 'text-gray-500 bg-gray-500/10 border-gray-500/20'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <AlertCircle className="w-4 h-4 text-red-500" />
      case 'investigating': return <Eye className="w-4 h-4 text-yellow-500" />
      case 'monitoring': return <Activity className="w-4 h-4 text-blue-500" />
      case 'resolved': return <CheckCircle className="w-4 h-4 text-green-500" />
      default: return <Info className="w-4 h-4 text-gray-500" />
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'cost': return <DollarSign className="w-5 h-5" />
      case 'performance': return <Gauge className="w-5 h-5" />
      case 'security': return <Shield className="w-5 h-5" />
      case 'availability': return <Activity className="w-5 h-5" />
      case 'capacity': return <HardDrive className="w-5 h-5" />
      default: return <AlertTriangle className="w-5 h-5" />
    }
  }

  const handleBulkAction = (action: string) => {
    console.log(`Performing ${action} on`, selectedAnomalies)
    setSelectedAnomalies([])
  }

  const exportData = () => {
    const data = filteredAnomalies.map(a => ({
      id: a.id,
      type: a.type,
      severity: a.severity,
      resource: a.resource,
      metric: a.metric,
      deviation: a.deviation,
      status: a.status,
      detected: a.detected
    }))
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `anomalies-${new Date().toISOString()}.json`
    a.click()
  }

  return (
    <div className="flex-1 space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Anomaly Detection</h1>
          <p className="text-gray-400">AI-powered anomaly detection and pattern recognition</p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowDetailsDialog(true)}
            className="border-gray-700 hover:bg-gray-800"
          >
            <Brain className="w-4 h-4 mr-2" />
            ML Models
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={exportData}
            className="border-gray-700 hover:bg-gray-800"
          >
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700">
            <Plus className="w-4 h-4 mr-2" />
            Configure Alert
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <AlertTriangle className="w-8 h-8 text-red-500" />
              <Badge className="bg-red-500/20 text-red-400 border-red-500/30">Live</Badge>
            </div>
            <p className="text-2xl font-bold text-white">{stats.active}</p>
            <p className="text-sm text-gray-400">Active Anomalies</p>
            <div className="mt-2 flex items-center text-xs">
              <ArrowUp className="w-3 h-3 text-red-400 mr-1" />
              <span className="text-red-400">+23%</span>
              <span className="text-gray-500 ml-1">vs last hour</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <AlertCircle className="w-8 h-8 text-orange-500" />
              <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/30">Critical</Badge>
            </div>
            <p className="text-2xl font-bold text-white">{stats.critical}</p>
            <p className="text-sm text-gray-400">Critical Issues</p>
            <div className="mt-2 flex items-center text-xs">
              <ArrowUp className="w-3 h-3 text-orange-400 mr-1" />
              <span className="text-orange-400">2 new</span>
              <span className="text-gray-500 ml-1">in last hour</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <DollarSign className="w-8 h-8 text-yellow-500" />
              <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">Impact</Badge>
            </div>
            <p className="text-2xl font-bold text-white">${stats.totalImpact.toLocaleString()}</p>
            <p className="text-sm text-gray-400">Predicted Cost Impact</p>
            <div className="mt-2 flex items-center text-xs">
              <TrendingUp className="w-3 h-3 text-yellow-400 mr-1" />
              <span className="text-yellow-400">Monthly projection</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Brain className="w-8 h-8 text-purple-500" />
              <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30">
                {Math.round(stats.avgConfidence)}%
              </Badge>
            </div>
            <p className="text-2xl font-bold text-white">{Math.round(stats.avgConfidence)}%</p>
            <p className="text-sm text-gray-400">ML Confidence</p>
            <Progress value={stats.avgConfidence} className="mt-2 h-1" />
          </CardContent>
        </Card>
      </div>

      {/* Filters and Search */}
      <Card className="bg-gray-900/50 border-gray-800">
        <CardContent className="p-4">
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                placeholder="Search anomalies..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-gray-800 border-gray-700 text-white"
              />
            </div>
            
            <div className="flex flex-wrap gap-2">
              <Select value={selectedSeverity} onValueChange={setSelectedSeverity}>
                <SelectTrigger className="w-[140px] bg-gray-800 border-gray-700">
                  <SelectValue placeholder="Severity" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Severities</SelectItem>
                  <SelectItem value="critical">Critical</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="low">Low</SelectItem>
                </SelectContent>
              </Select>

              <Select value={selectedType} onValueChange={setSelectedType}>
                <SelectTrigger className="w-[140px] bg-gray-800 border-gray-700">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="cost">Cost</SelectItem>
                  <SelectItem value="performance">Performance</SelectItem>
                  <SelectItem value="security">Security</SelectItem>
                  <SelectItem value="availability">Availability</SelectItem>
                  <SelectItem value="capacity">Capacity</SelectItem>
                </SelectContent>
              </Select>

              <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
                <SelectTrigger className="w-[120px] bg-gray-800 border-gray-700">
                  <SelectValue placeholder="Time" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1h">Last Hour</SelectItem>
                  <SelectItem value="24h">Last 24h</SelectItem>
                  <SelectItem value="7d">Last 7 Days</SelectItem>
                  <SelectItem value="30d">Last 30 Days</SelectItem>
                </SelectContent>
              </Select>

              <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded-lg border border-gray-700">
                <Switch
                  checked={showResolved}
                  onCheckedChange={setShowResolved}
                  className="scale-75"
                />
                <Label className="text-sm text-gray-300 cursor-pointer">
                  Show Resolved
                </Label>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
                  className="border-gray-700 hover:bg-gray-800"
                >
                  {viewMode === 'grid' ? <List className="w-4 h-4" /> : <Grid className="w-4 h-4" />}
                </Button>
              </div>
            </div>
          </div>

          {/* Advanced Filters */}
          <div className="mt-4 flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Label className="text-sm text-gray-400">Sensitivity:</Label>
              <div className="w-32">
                <Slider
                  value={sensitivityThreshold}
                  onValueChange={setSensitivityThreshold}
                  min={0}
                  max={100}
                  step={5}
                  className="w-full"
                />
              </div>
              <span className="text-sm text-gray-300">{sensitivityThreshold[0]}%</span>
            </div>

            <div className="flex items-center gap-2">
              <Switch
                checked={groupByCategory}
                onCheckedChange={setGroupByCategory}
                className="scale-75"
              />
              <Label className="text-sm text-gray-300 cursor-pointer">
                Group by Category
              </Label>
            </div>

            <div className="flex items-center gap-2">
              <Switch
                checked={showPredictions}
                onCheckedChange={setShowPredictions}
                className="scale-75"
              />
              <Label className="text-sm text-gray-300 cursor-pointer">
                Show Predictions
              </Label>
            </div>

            <div className="flex items-center gap-2">
              <Switch
                checked={autoResolve}
                onCheckedChange={setAutoResolve}
                className="scale-75"
              />
              <Label className="text-sm text-gray-300 cursor-pointer">
                Auto-Resolve
              </Label>
            </div>
          </div>

          {/* Bulk Actions */}
          {selectedAnomalies.length > 0 && (
            <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg flex items-center justify-between">
              <span className="text-sm text-blue-400">
                {selectedAnomalies.length} anomalies selected
              </span>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleBulkAction('acknowledge')}
                  className="border-blue-500/50 text-blue-400 hover:bg-blue-500/20"
                >
                  Acknowledge
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleBulkAction('resolve')}
                  className="border-green-500/50 text-green-400 hover:bg-green-500/20"
                >
                  Resolve
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleBulkAction('ignore')}
                  className="border-gray-500/50 text-gray-400 hover:bg-gray-500/20"
                >
                  Ignore
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setSelectedAnomalies([])}
                  className="text-gray-400"
                >
                  Clear
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Anomalies List/Grid */}
        <div className="lg:col-span-2 space-y-4">
          <Tabs defaultValue="anomalies" className="w-full">
            <TabsList className="bg-gray-800 border-gray-700">
              <TabsTrigger value="anomalies">Anomalies</TabsTrigger>
              <TabsTrigger value="trends">Trends</TabsTrigger>
              <TabsTrigger value="patterns">Patterns</TabsTrigger>
              <TabsTrigger value="predictions">Predictions</TabsTrigger>
            </TabsList>

            <TabsContent value="anomalies" className="space-y-4">
              {Object.entries(groupedAnomalies).map(([category, anomalies]) => (
                <div key={category}>
                  {groupByCategory && (
                    <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                      {category}
                      <Badge variant="outline" className="ml-2">
                        {anomalies.length}
                      </Badge>
                    </h3>
                  )}
                  
                  <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 gap-4' : 'space-y-3'}>
                    {anomalies.map((anomaly) => (
                      <motion.div
                        key={anomaly.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="relative"
                      >
                        <Card
                          className={`bg-gray-900/50 border-gray-800 hover:border-gray-700 transition-all cursor-pointer ${
                            selectedAnomaly?.id === anomaly.id ? 'ring-2 ring-blue-500' : ''
                          } ${selectedAnomalies.includes(anomaly.id) ? 'ring-1 ring-blue-500/50' : ''}`}
                          onClick={() => setSelectedAnomaly(anomaly)}
                        >
                          <CardContent className="p-4">
                            <div className="flex items-start justify-between mb-3">
                              <div className="flex items-center gap-3">
                                <div className={`p-2 rounded-lg ${getSeverityColor(anomaly.severity)}`}>
                                  {getTypeIcon(anomaly.type)}
                                </div>
                                <div>
                                  <div className="flex items-center gap-2">
                                    <p className="font-semibold text-white">{anomaly.resource}</p>
                                    {getStatusIcon(anomaly.status)}
                                  </div>
                                  <p className="text-xs text-gray-400">{anomaly.id}</p>
                                </div>
                              </div>
                              <Checkbox
                                checked={selectedAnomalies.includes(anomaly.id)}
                                onCheckedChange={(checked) => {
                                  if (checked) {
                                    setSelectedAnomalies([...selectedAnomalies, anomaly.id])
                                  } else {
                                    setSelectedAnomalies(selectedAnomalies.filter(id => id !== anomaly.id))
                                  }
                                }}
                                onClick={(e) => e.stopPropagation()}
                              />
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <span className="text-sm text-gray-400">{anomaly.metric}</span>
                                <span className={`text-sm font-bold ${
                                  anomaly.deviation.startsWith('+') ? 'text-red-400' : 'text-green-400'
                                }`}>
                                  {anomaly.deviation}
                                </span>
                              </div>

                              <div className="flex items-center justify-between text-xs">
                                <span className="text-gray-500">Baseline: {anomaly.baseline}</span>
                                <span className="text-gray-300">Current: {anomaly.current}</span>
                              </div>

                              <Progress
                                value={(anomaly.current / (anomaly.baseline || 1)) * 100}
                                className="h-1"
                              />

                              <div className="flex items-center justify-between pt-2">
                                <div className="flex items-center gap-2">
                                  <Badge className={getSeverityColor(anomaly.severity)}>
                                    {anomaly.severity}
                                  </Badge>
                                  <Badge variant="outline" className="text-xs">
                                    {anomaly.confidence}% confidence
                                  </Badge>
                                </div>
                                <div className="flex items-center gap-1 text-xs text-gray-400">
                                  <Clock className="w-3 h-3" />
                                  {anomaly.duration}
                                </div>
                              </div>

                              {anomaly.predictedCost > 0 && (
                                <Alert className="mt-2 bg-yellow-500/10 border-yellow-500/20">
                                  <DollarSign className="w-4 h-4 text-yellow-400" />
                                  <AlertDescription className="text-xs text-yellow-400">
                                    Predicted cost impact: ${anomaly.predictedCost.toLocaleString()}/month
                                  </AlertDescription>
                                </Alert>
                              )}

                              {anomaly.autoResolve && autoResolve && (
                                <div className="flex items-center gap-1 text-xs text-green-400 mt-2">
                                  <Sparkles className="w-3 h-3" />
                                  Auto-resolve enabled
                                </div>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      </motion.div>
                    ))}
                  </div>
                </div>
              ))}
            </TabsContent>

            <TabsContent value="trends" className="space-y-4">
              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white">Anomaly Trends</CardTitle>
                </CardHeader>
                <CardContent>
                  <Line
                    data={anomalyTrendChart}
                    options={{
                      responsive: true,
                      plugins: {
                        legend: { display: true, labels: { color: 'white' } },
                        tooltip: {
                          backgroundColor: 'rgba(0, 0, 0, 0.8)',
                          titleColor: 'white',
                          bodyColor: 'white',
                          borderColor: 'rgba(255, 255, 255, 0.2)',
                          borderWidth: 1
                        }
                      },
                      scales: {
                        x: {
                          grid: { color: 'rgba(255, 255, 255, 0.1)' },
                          ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                        },
                        y: {
                          grid: { color: 'rgba(255, 255, 255, 0.1)' },
                          ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                        }
                      }
                    }}
                  />
                </CardContent>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="bg-gray-900/50 border-gray-800">
                  <CardHeader>
                    <CardTitle className="text-white text-base">By Severity</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Doughnut
                      data={severityDistribution}
                      options={{
                        responsive: true,
                        plugins: {
                          legend: { display: true, position: 'bottom', labels: { color: 'white' } }
                        }
                      }}
                    />
                  </CardContent>
                </Card>

                <Card className="bg-gray-900/50 border-gray-800">
                  <CardHeader>
                    <CardTitle className="text-white text-base">By Type</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Bar
                      data={typeDistribution}
                      options={{
                        responsive: true,
                        plugins: {
                          legend: { display: false }
                        },
                        scales: {
                          x: {
                            grid: { display: false },
                            ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                          },
                          y: {
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                          }
                        }
                      }}
                    />
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="patterns" className="space-y-4">
              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white">Pattern Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <Radar
                    data={patternAnalysis}
                    options={{
                      responsive: true,
                      plugins: {
                        legend: { display: false }
                      },
                      scales: {
                        r: {
                          grid: { color: 'rgba(255, 255, 255, 0.1)' },
                          angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                          pointLabels: { color: 'rgba(255, 255, 255, 0.7)' },
                          ticks: { display: false }
                        }
                      }
                    }}
                  />
                </CardContent>
              </Card>

              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white">Common Patterns</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {['Sudden Spike', 'Gradual Degradation', 'Intermittent', 'Linear Growth'].map((pattern) => (
                    <div key={pattern} className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                      <div className="flex items-center gap-3">
                        <GitBranch className="w-4 h-4 text-purple-400" />
                        <span className="text-sm text-white">{pattern}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          {Math.floor(Math.random() * 50) + 10} occurrences
                        </Badge>
                        <span className="text-xs text-gray-400">
                          {Math.floor(Math.random() * 30) + 70}% accuracy
                        </span>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="predictions" className="space-y-4">
              {showPredictions && (
                <>
                  <Alert className="bg-blue-500/10 border-blue-500/20">
                    <Brain className="w-4 h-4 text-blue-400" />
                    <AlertDescription className="text-blue-400">
                      ML models predict 3 potential anomalies in the next 4 hours based on current patterns
                    </AlertDescription>
                  </Alert>

                  <Card className="bg-gray-900/50 border-gray-800">
                    <CardHeader>
                      <CardTitle className="text-white">Predicted Anomalies</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="p-4 bg-gray-800 rounded-lg border border-yellow-500/20">
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <TrendingUp className="w-4 h-4 text-yellow-400" />
                            <span className="font-medium text-white">Storage Account - Capacity Warning</span>
                          </div>
                          <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">
                            In 2 hours
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-400 mb-2">
                          Storage capacity expected to reach 90% based on current growth rate
                        </p>
                        <div className="flex items-center gap-4 text-xs">
                          <span className="text-gray-400">Confidence: 87%</span>
                          <span className="text-gray-400">Impact: Medium</span>
                        </div>
                      </div>

                      <div className="p-4 bg-gray-800 rounded-lg border border-orange-500/20">
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <DollarSign className="w-4 h-4 text-orange-400" />
                            <span className="font-medium text-white">VM Scale Set - Cost Spike</span>
                          </div>
                          <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/30">
                            In 3 hours
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-400 mb-2">
                          Auto-scaling likely to trigger based on traffic patterns, estimated $450 increase
                        </p>
                        <div className="flex items-center gap-4 text-xs">
                          <span className="text-gray-400">Confidence: 73%</span>
                          <span className="text-gray-400">Impact: High</span>
                        </div>
                      </div>

                      <div className="p-4 bg-gray-800 rounded-lg border border-blue-500/20">
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-blue-400" />
                            <span className="font-medium text-white">API Gateway - Latency Increase</span>
                          </div>
                          <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30">
                            In 4 hours
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-400 mb-2">
                          Response time expected to increase by 150ms during peak hours
                        </p>
                        <div className="flex items-center gap-4 text-xs">
                          <span className="text-gray-400">Confidence: 68%</span>
                          <span className="text-gray-400">Impact: Low</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </>
              )}
            </TabsContent>
          </Tabs>
        </div>

        {/* Details Panel */}
        <div className="space-y-4">
          {selectedAnomaly ? (
            <Card className="bg-gray-900/50 border-gray-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center justify-between">
                  Anomaly Details
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setSelectedAnomaly(null)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-sm text-gray-400 mb-1">Resource</p>
                  <p className="text-white font-medium">{selectedAnomaly.resource}</p>
                  <p className="text-xs text-gray-500">{selectedAnomaly.resourceId}</p>
                </div>

                <div>
                  <p className="text-sm text-gray-400 mb-1">Root Cause Analysis</p>
                  <p className="text-sm text-white">{selectedAnomaly.rootCause}</p>
                </div>

                <div>
                  <p className="text-sm text-gray-400 mb-2">Correlated Metrics</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedAnomaly.correlations.map((correlation: string) => (
                      <Badge key={correlation} variant="outline" className="text-xs">
                        {correlation}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-sm text-gray-400 mb-2">Affected Services</p>
                  <div className="space-y-1">
                    {selectedAnomaly.affectedServices.map((service: string) => (
                      <div key={service} className="flex items-center gap-2 text-sm">
                        <ChevronRight className="w-3 h-3 text-gray-500" />
                        <span className="text-gray-300">{service}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-sm text-gray-400 mb-2">Recommendations</p>
                  <div className="space-y-2">
                    {selectedAnomaly.recommendations.map((rec: string, idx: number) => (
                      <div key={idx} className="flex items-start gap-2">
                        <span className="text-blue-400 text-xs mt-0.5">{idx + 1}.</span>
                        <p className="text-sm text-gray-300">{rec}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex gap-2 pt-4">
                  <Button
                    size="sm"
                    className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600"
                  >
                    <Check className="w-4 h-4 mr-1" />
                    Resolve
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="flex-1 border-gray-700"
                  >
                    <Eye className="w-4 h-4 mr-1" />
                    Investigate
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="bg-gray-900/50 border-gray-800">
              <CardContent className="p-6 text-center">
                <AlertTriangle className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                <p className="text-gray-400">Select an anomaly to view details</p>
              </CardContent>
            </Card>
          )}

          {/* ML Models Performance */}
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white text-base">ML Model Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <Bar
                data={mlPerformanceChart}
                options={{
                  indexAxis: 'y',
                  responsive: true,
                  plugins: {
                    legend: { display: false }
                  },
                  scales: {
                    x: {
                      grid: { color: 'rgba(255, 255, 255, 0.1)' },
                      ticks: { color: 'rgba(255, 255, 255, 0.5)' },
                      max: 100
                    },
                    y: {
                      grid: { display: false },
                      ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
                    }
                  }
                }}
              />
            </CardContent>
          </Card>

          {/* Notification Settings */}
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white text-base">Alert Channels</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {Object.entries(notificationSettings).map(([channel, enabled]) => (
                <div key={channel} className="flex items-center justify-between">
                  <Label className="text-sm text-gray-300 capitalize cursor-pointer">
                    {channel}
                  </Label>
                  <Switch
                    checked={enabled}
                    onCheckedChange={(checked) => 
                      setNotificationSettings({...notificationSettings, [channel]: checked})
                    }
                    className="scale-75"
                  />
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Auto Refresh */}
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white text-base">Auto Refresh</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="text-sm text-gray-300">Interval (seconds)</Label>
                  <span className="text-sm text-white">{refreshInterval}s</span>
                </div>
                <Slider
                  value={[refreshInterval]}
                  onValueChange={([value]) => setRefreshInterval(value)}
                  min={10}
                  max={300}
                  step={10}
                />
                <Button
                  size="sm"
                  variant="outline"
                  className="w-full border-gray-700"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh Now
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* ML Models Dialog */}
      <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
        <DialogContent className="bg-gray-900 border-gray-800 max-w-2xl">
          <DialogHeader>
            <DialogTitle className="text-white">Machine Learning Models</DialogTitle>
            <DialogDescription className="text-gray-400">
              Overview of AI models powering anomaly detection
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            {mlModels.map((model) => (
              <div key={model.name} className="p-4 bg-gray-800 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <Brain className="w-5 h-5 text-purple-400" />
                    <span className="font-medium text-white">{model.name}</span>
                  </div>
                  <Badge className={model.status === 'active' ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}>
                    {model.status}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Accuracy</span>
                  <div className="flex items-center gap-2">
                    <Progress value={model.accuracy} className="w-32 h-2" />
                    <span className="text-sm text-white">{model.accuracy}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}