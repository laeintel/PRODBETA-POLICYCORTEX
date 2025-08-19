'use client'

import React, { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Package,
  Archive,
  Download,
  Upload,
  Search,
  Filter,
  MoreVertical,
  ChevronRight,
  Clock,
  Tag,
  GitBranch,
  Shield,
  AlertCircle,
  CheckCircle,
  XCircle,
  Info,
  Copy,
  Trash2,
  Edit2,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  Star,
  Share2,
  ExternalLink,
  FileCode,
  FileText,
  FileImage,
  FileArchive,
  File,
  Folder,
  FolderOpen,
  RefreshCw,
  Settings,
  Plus,
  Minus,
  ArrowUp,
  ArrowDown,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  BarChart3,
  PieChart,
  Activity,
  Zap,
  Database,
  Server,
  HardDrive,
  Cpu,
  Hash,
  Key,
  Globe,
  Link,
  Layers,
  Box,
  Terminal,
  Code,
  GitCommit,
  GitPullRequest,
  GitMerge,
  Calendar,
  User,
  Users,
  Bell,
  CheckSquare,
  Square,
  Circle,
  CircleDot,
  X
} from 'lucide-react'
import { Line, Bar, Doughnut } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
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
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'
import { Progress } from '@/components/ui/progress'
import { Textarea } from '@/components/ui/textarea'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Slider } from '@/components/ui/slider'

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

// Mock artifact data
const mockArtifacts = [
  {
    id: 'ART-001',
    name: 'policycortex-api',
    type: 'docker',
    version: '2.14.3',
    registry: 'crpcxdev.azurecr.io',
    repository: 'policycortex/api',
    tags: ['latest', 'v2.14.3', 'production'],
    size: '245.8 MB',
    layers: 12,
    os: 'linux',
    architecture: 'amd64',
    created: '2024-01-19T10:30:00Z',
    lastModified: '2024-01-19T14:45:00Z',
    pushedBy: 'CI Pipeline',
    digest: 'sha256:3b4c5d6e7f8a9b',
    signed: true,
    scanned: true,
    vulnerabilities: { critical: 0, high: 2, medium: 5, low: 12 },
    downloads: 1847,
    retention: '30 days',
    immutable: false,
    platform: 'linux/amd64',
    dependencies: ['node:18-alpine', 'nginx:1.24'],
    metadata: {
      buildId: 'build-4521',
      commitSha: 'a3f4d5e',
      branch: 'main',
      environment: 'production'
    }
  },
  {
    id: 'ART-002',
    name: 'frontend-bundle',
    type: 'npm',
    version: '1.8.2',
    registry: 'npm.pkg.github.com',
    repository: '@policycortex/frontend',
    tags: ['latest', 'stable', 'v1.8.2'],
    size: '18.4 MB',
    files: 2847,
    license: 'MIT',
    created: '2024-01-19T09:15:00Z',
    lastModified: '2024-01-19T09:15:00Z',
    publishedBy: 'github-actions',
    integrity: 'sha512-xVuGz5JrKLR...',
    signed: true,
    scanned: true,
    vulnerabilities: { critical: 0, high: 0, medium: 3, low: 8 },
    downloads: 523,
    retention: '90 days',
    deprecated: false,
    dependencies: 48,
    devDependencies: 32,
    metadata: {
      node: '>=18.0.0',
      npm: '>=9.0.0',
      homepage: 'https://policycortex.com'
    }
  },
  {
    id: 'ART-003',
    name: 'ml-models',
    type: 'python',
    version: '3.2.1',
    registry: 'pypi.org',
    repository: 'policycortex-ml',
    tags: ['latest', 'v3.2.1', 'stable'],
    size: '456.3 MB',
    format: 'wheel',
    pythonVersion: '>=3.9',
    created: '2024-01-18T16:20:00Z',
    lastModified: '2024-01-18T16:20:00Z',
    uploadedBy: 'ml-team',
    hash: 'md5:7f3a2b5c8d9e0f1a',
    signed: true,
    scanned: true,
    vulnerabilities: { critical: 0, high: 1, medium: 4, low: 15 },
    downloads: 892,
    retention: '180 days',
    requires: ['tensorflow>=2.12', 'scikit-learn>=1.3', 'pandas>=2.0'],
    metadata: {
      framework: 'tensorflow',
      modelType: 'classification',
      accuracy: 0.94
    }
  },
  {
    id: 'ART-004',
    name: 'helm-charts',
    type: 'helm',
    version: '0.5.8',
    registry: 'charts.policycortex.io',
    repository: 'policycortex/charts',
    tags: ['v0.5.8', 'stable'],
    size: '124 KB',
    apiVersion: 'v2',
    appVersion: '2.14.3',
    created: '2024-01-19T11:00:00Z',
    lastModified: '2024-01-19T11:00:00Z',
    maintainer: 'DevOps Team',
    digest: 'sha256:9e8f7d6c5b4a3',
    signed: true,
    validated: true,
    dependencies: ['postgresql-11.2.3', 'redis-7.0.5'],
    downloads: 234,
    retention: '60 days',
    values: {
      replicas: 3,
      resources: { cpu: '500m', memory: '512Mi' }
    }
  },
  {
    id: 'ART-005',
    name: 'terraform-modules',
    type: 'terraform',
    version: '1.3.0',
    registry: 'registry.terraform.io',
    repository: 'policycortex/azure-infra',
    tags: ['v1.3.0', 'latest'],
    size: '89 KB',
    provider: 'azurerm',
    terraformVersion: '>=1.5.0',
    created: '2024-01-17T14:30:00Z',
    lastModified: '2024-01-17T14:30:00Z',
    publishedBy: 'infra-team',
    checksum: 'h1:abc123def456...',
    signed: true,
    verified: true,
    downloads: 567,
    retention: '365 days',
    modules: 12,
    resources: 48,
    metadata: {
      cloud: 'azure',
      region: 'eastus',
      environment: 'production'
    }
  }
]

// Mock repository data
const mockRepositories = [
  { name: 'Docker Registry', type: 'docker', count: 48, size: '12.4 GB', status: 'healthy' },
  { name: 'NPM Registry', type: 'npm', count: 156, size: '3.2 GB', status: 'healthy' },
  { name: 'PyPI Registry', type: 'python', count: 23, size: '8.7 GB', status: 'healthy' },
  { name: 'Helm Repository', type: 'helm', count: 12, size: '245 MB', status: 'healthy' },
  { name: 'Maven Repository', type: 'maven', count: 67, size: '5.1 GB', status: 'degraded' }
]

// Generate time series data
const generateTimeSeriesData = (days: number = 30) => {
  const data = []
  const now = new Date()
  for (let i = days; i >= 0; i--) {
    const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000)
    data.push({
      date: date.toISOString().split('T')[0],
      uploads: Math.floor(Math.random() * 50) + 10,
      downloads: Math.floor(Math.random() * 200) + 100,
      storage: Math.floor(Math.random() * 100) + 400
    })
  }
  return data
}

export default function ArtifactsPage() {
  const [artifacts, setArtifacts] = useState(mockArtifacts)
  const [selectedArtifact, setSelectedArtifact] = useState<any>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedType, setSelectedType] = useState('all')
  const [selectedRegistry, setSelectedRegistry] = useState('all')
  const [showVulnerable, setShowVulnerable] = useState(false)
  const [showSigned, setShowSigned] = useState(false)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('list')
  const [showUploadDialog, setShowUploadDialog] = useState(false)
  const [showScanDialog, setShowScanDialog] = useState(false)
  const [selectedArtifacts, setSelectedArtifacts] = useState<string[]>([])
  const [retentionPolicy, setRetentionPolicy] = useState('30')
  const [autoScan, setAutoScan] = useState(true)
  const [requireSigning, setRequireSigning] = useState(true)
  const [showPromoteDialog, setShowPromoteDialog] = useState(false)
  const [quotaLimit, setQuotaLimit] = useState([80])

  // Filter artifacts
  const filteredArtifacts = useMemo(() => {
    return artifacts.filter(artifact => {
      const matchesSearch = artifact.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           artifact.version.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           artifact.repository.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesType = selectedType === 'all' || artifact.type === selectedType
      const matchesRegistry = selectedRegistry === 'all' || artifact.registry.includes(selectedRegistry)
      const matchesVulnerable = !showVulnerable || 
                                ((artifact.vulnerabilities?.critical ?? 0) > 0 || (artifact.vulnerabilities?.high ?? 0) > 0)
      const matchesSigned = !showSigned || artifact.signed
      
      return matchesSearch && matchesType && matchesRegistry && matchesVulnerable && matchesSigned
    })
  }, [artifacts, searchQuery, selectedType, selectedRegistry, showVulnerable, showSigned])

  // Calculate statistics
  const stats = useMemo(() => {
    const totalSize = artifacts.reduce((sum, a) => {
      const size = parseFloat(a.size.replace(/[^\d.]/g, ''))
      const unit = a.size.match(/[A-Z]+/)?.[0] || 'MB'
      const multiplier = unit === 'GB' ? 1000 : unit === 'KB' ? 0.001 : 1
      return sum + (size * multiplier)
    }, 0)

    const totalDownloads = artifacts.reduce((sum, a) => sum + a.downloads, 0)
    const vulnerableCount = artifacts.filter(a => 
      (a.vulnerabilities?.critical ?? 0) > 0 || (a.vulnerabilities?.high ?? 0) > 0
    ).length
    const signedCount = artifacts.filter(a => a.signed).length

    return {
      total: artifacts.length,
      totalSize: totalSize.toFixed(1),
      totalDownloads,
      vulnerableCount,
      signedCount,
      signedPercentage: Math.round((signedCount / artifacts.length) * 100)
    }
  }, [artifacts])

  // Time series data
  const timeSeriesData = useMemo(() => generateTimeSeriesData(), [])

  // Chart configurations
  const uploadTrendChart = {
    labels: timeSeriesData.slice(-7).map(d => d.date.split('-').slice(1).join('/')),
    datasets: [
      {
        label: 'Uploads',
        data: timeSeriesData.slice(-7).map(d => d.uploads),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Downloads',
        data: timeSeriesData.slice(-7).map(d => d.downloads),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  }

  const typeDistribution = {
    labels: ['Docker', 'NPM', 'Python', 'Helm', 'Terraform'],
    datasets: [{
      data: [
        artifacts.filter(a => a.type === 'docker').length,
        artifacts.filter(a => a.type === 'npm').length,
        artifacts.filter(a => a.type === 'python').length,
        artifacts.filter(a => a.type === 'helm').length,
        artifacts.filter(a => a.type === 'terraform').length
      ],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(34, 197, 94, 0.8)',
        'rgba(168, 85, 247, 0.8)',
        'rgba(251, 146, 60, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const storageUsage = {
    labels: mockRepositories.map(r => r.name),
    datasets: [{
      label: 'Storage (GB)',
      data: mockRepositories.map(r => parseFloat(r.size.replace(/[^\d.]/g, ''))),
      backgroundColor: 'rgba(147, 51, 234, 0.8)',
      borderColor: 'rgb(147, 51, 234)',
      borderWidth: 1
    }]
  }

  const vulnerabilityChart = {
    labels: ['Critical', 'High', 'Medium', 'Low'],
    datasets: [{
      label: 'Vulnerabilities',
      data: [
        artifacts.reduce((sum, a) => sum + (a.vulnerabilities?.critical ?? 0), 0),
        artifacts.reduce((sum, a) => sum + (a.vulnerabilities?.high ?? 0), 0),
        artifacts.reduce((sum, a) => sum + (a.vulnerabilities?.medium ?? 0), 0),
        artifacts.reduce((sum, a) => sum + (a.vulnerabilities?.low ?? 0), 0)
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

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'docker': return <Box className="w-5 h-5 text-blue-400" />
      case 'npm': return <Package className="w-5 h-5 text-red-400" />
      case 'python': return <Code className="w-5 h-5 text-green-400" />
      case 'helm': return <Layers className="w-5 h-5 text-purple-400" />
      case 'terraform': return <Terminal className="w-5 h-5 text-orange-400" />
      default: return <Archive className="w-5 h-5 text-gray-400" />
    }
  }

  const getVulnerabilityBadge = (vulnerabilities: any) => {
    if (vulnerabilities?.critical > 0) {
      return <Badge className="bg-red-500/20 text-red-400 border-red-500/30">Critical</Badge>
    }
    if (vulnerabilities?.high > 0) {
      return <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/30">High</Badge>
    }
    if (vulnerabilities?.medium > 0) {
      return <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">Medium</Badge>
    }
    return <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Clean</Badge>
  }

  const handleBulkAction = (action: string) => {
    console.log(`Performing ${action} on`, selectedArtifacts)
    setSelectedArtifacts([])
  }

  const handlePromote = (artifact: any, environment: string) => {
    console.log(`Promoting ${artifact.name} to ${environment}`)
    setShowPromoteDialog(false)
  }

  const handleScan = (artifact: any) => {
    console.log(`Scanning ${artifact.name}`)
    setShowScanDialog(false)
  }

  return (
    <div className="flex-1 space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Artifact Registry</h1>
          <p className="text-gray-400">Manage container images, packages, and build artifacts</p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            className="border-gray-700 hover:bg-gray-800"
          >
            <Settings className="w-4 h-4 mr-2" />
            Registry Settings
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="border-gray-700 hover:bg-gray-800"
          >
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button 
            className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700"
            onClick={() => setShowUploadDialog(true)}
          >
            <Upload className="w-4 h-4 mr-2" />
            Upload Artifact
          </Button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Archive className="w-8 h-8 text-blue-500" />
              <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30">Total</Badge>
            </div>
            <p className="text-2xl font-bold text-white">{stats.total}</p>
            <p className="text-sm text-gray-400">Artifacts</p>
            <div className="mt-2 flex items-center text-xs">
              <TrendingUp className="w-3 h-3 text-green-400 mr-1" />
              <span className="text-green-400">+12%</span>
              <span className="text-gray-500 ml-1">vs last month</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <HardDrive className="w-8 h-8 text-purple-500" />
              <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30">Storage</Badge>
            </div>
            <p className="text-2xl font-bold text-white">{stats.totalSize} GB</p>
            <p className="text-sm text-gray-400">Total Size</p>
            <Progress value={quotaLimit[0]} className="mt-2 h-1" />
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Download className="w-8 h-8 text-green-500" />
              <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Usage</Badge>
            </div>
            <p className="text-2xl font-bold text-white">{stats.totalDownloads.toLocaleString()}</p>
            <p className="text-sm text-gray-400">Total Downloads</p>
            <div className="mt-2 flex items-center text-xs">
              <Activity className="w-3 h-3 text-gray-400 mr-1" />
              <span className="text-gray-300">523 today</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Shield className="w-8 h-8 text-yellow-500" />
              {stats.vulnerableCount > 0 ? (
                <Badge className="bg-red-500/20 text-red-400 border-red-500/30">{stats.vulnerableCount}</Badge>
              ) : (
                <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Secure</Badge>
              )}
            </div>
            <p className="text-2xl font-bold text-white">
              {stats.vulnerableCount > 0 ? stats.vulnerableCount : 'All Clear'}
            </p>
            <p className="text-sm text-gray-400">
              {stats.vulnerableCount > 0 ? 'Vulnerable' : 'No Issues'}
            </p>
            <div className="mt-2 flex items-center text-xs">
              <AlertCircle className="w-3 h-3 text-yellow-400 mr-1" />
              <span className="text-yellow-400">2 critical</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Key className="w-8 h-8 text-cyan-500" />
              <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/30">
                {stats.signedPercentage}%
              </Badge>
            </div>
            <p className="text-2xl font-bold text-white">{stats.signedCount}/{stats.total}</p>
            <p className="text-sm text-gray-400">Signed Artifacts</p>
            <Progress value={stats.signedPercentage} className="mt-2 h-1" />
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card className="bg-gray-900/50 border-gray-800">
        <CardContent className="p-4">
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                placeholder="Search artifacts..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-gray-800 border-gray-700 text-white"
              />
            </div>
            
            <div className="flex flex-wrap gap-2">
              <Select value={selectedType} onValueChange={setSelectedType}>
                <SelectTrigger className="w-[140px] bg-gray-800 border-gray-700">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="docker">Docker</SelectItem>
                  <SelectItem value="npm">NPM</SelectItem>
                  <SelectItem value="python">Python</SelectItem>
                  <SelectItem value="helm">Helm</SelectItem>
                  <SelectItem value="terraform">Terraform</SelectItem>
                </SelectContent>
              </Select>

              <Select value={selectedRegistry} onValueChange={setSelectedRegistry}>
                <SelectTrigger className="w-[160px] bg-gray-800 border-gray-700">
                  <SelectValue placeholder="Registry" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Registries</SelectItem>
                  <SelectItem value="azurecr">Azure CR</SelectItem>
                  <SelectItem value="github">GitHub</SelectItem>
                  <SelectItem value="pypi">PyPI</SelectItem>
                  <SelectItem value="terraform">Terraform</SelectItem>
                </SelectContent>
              </Select>

              <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded-lg border border-gray-700">
                <Switch
                  checked={showVulnerable}
                  onCheckedChange={setShowVulnerable}
                  className="scale-75"
                />
                <Label className="text-sm text-gray-300 cursor-pointer">
                  Vulnerable Only
                </Label>
              </div>

              <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded-lg border border-gray-700">
                <Switch
                  checked={showSigned}
                  onCheckedChange={setShowSigned}
                  className="scale-75"
                />
                <Label className="text-sm text-gray-300 cursor-pointer">
                  Signed Only
                </Label>
              </div>
            </div>
          </div>

          {/* Bulk Actions */}
          {selectedArtifacts.length > 0 && (
            <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg flex items-center justify-between">
              <span className="text-sm text-blue-400">
                {selectedArtifacts.length} artifacts selected
              </span>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleBulkAction('scan')}
                  className="border-blue-500/50 text-blue-400 hover:bg-blue-500/20"
                >
                  <Shield className="w-4 h-4 mr-1" />
                  Scan
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleBulkAction('sign')}
                  className="border-green-500/50 text-green-400 hover:bg-green-500/20"
                >
                  <Key className="w-4 h-4 mr-1" />
                  Sign
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleBulkAction('delete')}
                  className="border-red-500/50 text-red-400 hover:bg-red-500/20"
                >
                  <Trash2 className="w-4 h-4 mr-1" />
                  Delete
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setSelectedArtifacts([])}
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
        {/* Artifacts List */}
        <div className="lg:col-span-2">
          <Tabs defaultValue="artifacts" className="w-full">
            <TabsList className="bg-gray-800 border-gray-700">
              <TabsTrigger value="artifacts">Artifacts</TabsTrigger>
              <TabsTrigger value="repositories">Repositories</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
              <TabsTrigger value="security">Security</TabsTrigger>
            </TabsList>

            <TabsContent value="artifacts" className="space-y-4">
              {filteredArtifacts.map((artifact) => (
                <motion.div
                  key={artifact.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <Card
                    className={`bg-gray-900/50 border-gray-800 hover:border-gray-700 transition-all cursor-pointer ${
                      selectedArtifact?.id === artifact.id ? 'ring-2 ring-blue-500' : ''
                    } ${selectedArtifacts.includes(artifact.id) ? 'ring-1 ring-blue-500/50' : ''}`}
                    onClick={() => setSelectedArtifact(artifact)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-lg bg-gray-800">
                            {getTypeIcon(artifact.type)}
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <p className="font-semibold text-white">{artifact.name}</p>
                              <Badge variant="outline" className="text-xs">
                                v{artifact.version}
                              </Badge>
                              {artifact.signed && (
                                <Shield className="w-4 h-4 text-green-400" />
                              )}
                            </div>
                            <p className="text-xs text-gray-400">{artifact.repository}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Checkbox
                            checked={selectedArtifacts.includes(artifact.id)}
                            onCheckedChange={(checked) => {
                              if (checked) {
                                setSelectedArtifacts([...selectedArtifacts, artifact.id])
                              } else {
                                setSelectedArtifacts(selectedArtifacts.filter(id => id !== artifact.id))
                              }
                            }}
                            onClick={(e) => e.stopPropagation()}
                          />
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={(e) => {
                              e.stopPropagation()
                              setShowPromoteDialog(true)
                            }}
                          >
                            <MoreVertical className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                        <div>
                          <p className="text-xs text-gray-400">Size</p>
                          <p className="text-sm text-white">{artifact.size}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Downloads</p>
                          <p className="text-sm text-white">{artifact.downloads.toLocaleString()}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Created</p>
                          <p className="text-sm text-white">
                            {new Date(artifact.created).toLocaleDateString()}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Retention</p>
                          <p className="text-sm text-white">{artifact.retention}</p>
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          {artifact.tags.slice(0, 3).map((tag) => (
                            <Badge key={tag} variant="outline" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                          {artifact.tags.length > 3 && (
                            <span className="text-xs text-gray-500">+{artifact.tags.length - 3}</span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          {getVulnerabilityBadge(artifact.vulnerabilities)}
                          {artifact.scanned && (
                            <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30 text-xs">
                              Scanned
                            </Badge>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </TabsContent>

            <TabsContent value="repositories" className="space-y-4">
              {mockRepositories.map((repo) => (
                <Card key={repo.name} className="bg-gray-900/50 border-gray-800">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-gray-800">
                          {getTypeIcon(repo.type)}
                        </div>
                        <div>
                          <p className="font-semibold text-white">{repo.name}</p>
                          <p className="text-xs text-gray-400">Type: {repo.type}</p>
                        </div>
                      </div>
                      <Badge className={repo.status === 'healthy' ? 
                        'bg-green-500/20 text-green-400 border-green-500/30' :
                        'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
                      }>
                        {repo.status}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      <div>
                        <p className="text-xs text-gray-400">Artifacts</p>
                        <p className="text-sm text-white">{repo.count}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Total Size</p>
                        <p className="text-sm text-white">{repo.size}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Usage</p>
                        <Progress value={Math.random() * 100} className="mt-1 h-1" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </TabsContent>

            <TabsContent value="analytics" className="space-y-4">
              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white">Upload & Download Trends</CardTitle>
                </CardHeader>
                <CardContent>
                  <Line
                    data={uploadTrendChart}
                    options={{
                      responsive: true,
                      plugins: {
                        legend: { display: true, labels: { color: 'white' } },
                        tooltip: {
                          backgroundColor: 'rgba(0, 0, 0, 0.8)',
                          titleColor: 'white',
                          bodyColor: 'white'
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
                    <CardTitle className="text-white text-base">By Type</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Doughnut
                      data={typeDistribution}
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
                    <CardTitle className="text-white text-base">Storage Usage</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Bar
                      data={storageUsage}
                      options={{
                        responsive: true,
                        plugins: {
                          legend: { display: false }
                        },
                        scales: {
                          x: {
                            grid: { display: false },
                            ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { size: 10 } }
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

            <TabsContent value="security" className="space-y-4">
              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white">Vulnerability Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <Doughnut
                    data={vulnerabilityChart}
                    options={{
                      responsive: true,
                      plugins: {
                        legend: { display: true, position: 'right', labels: { color: 'white' } }
                      }
                    }}
                  />
                </CardContent>
              </Card>

              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white text-base">Security Policies</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                    <div className="flex items-center gap-3">
                      <Shield className="w-4 h-4 text-blue-400" />
                      <span className="text-sm text-white">Vulnerability Scanning</span>
                    </div>
                    <Switch
                      checked={autoScan}
                      onCheckedChange={setAutoScan}
                      className="scale-75"
                    />
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                    <div className="flex items-center gap-3">
                      <Key className="w-4 h-4 text-green-400" />
                      <span className="text-sm text-white">Require Signing</span>
                    </div>
                    <Switch
                      checked={requireSigning}
                      onCheckedChange={setRequireSigning}
                      className="scale-75"
                    />
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                    <div className="flex items-center gap-3">
                      <Lock className="w-4 h-4 text-yellow-400" />
                      <span className="text-sm text-white">Immutable Tags</span>
                    </div>
                    <Switch className="scale-75" />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {selectedArtifact ? (
            <Card className="bg-gray-900/50 border-gray-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center justify-between">
                  Artifact Details
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setSelectedArtifact(null)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-sm text-gray-400 mb-1">Registry</p>
                  <p className="text-white font-medium">{selectedArtifact.registry}</p>
                </div>

                <div>
                  <p className="text-sm text-gray-400 mb-1">Digest</p>
                  <div className="flex items-center gap-2">
                    <code className="text-xs text-gray-300 bg-gray-800 px-2 py-1 rounded">
                      {selectedArtifact.digest}
                    </code>
                    <Button size="sm" variant="ghost">
                      <Copy className="w-3 h-3" />
                    </Button>
                  </div>
                </div>

                <div>
                  <p className="text-sm text-gray-400 mb-2">Tags</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedArtifact.tags.map((tag: string) => (
                      <Badge key={tag} variant="outline" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-sm text-gray-400 mb-2">Vulnerabilities</p>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-red-400">Critical</span>
                      <span className="text-xs text-white">{selectedArtifact.vulnerabilities?.critical ?? 0}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-orange-400">High</span>
                      <span className="text-xs text-white">{selectedArtifact.vulnerabilities?.high ?? 0}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-yellow-400">Medium</span>
                      <span className="text-xs text-white">{selectedArtifact.vulnerabilities?.medium ?? 0}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-blue-400">Low</span>
                      <span className="text-xs text-white">{selectedArtifact.vulnerabilities?.low ?? 0}</span>
                    </div>
                  </div>
                </div>

                <div>
                  <p className="text-sm text-gray-400 mb-2">Metadata</p>
                  <div className="space-y-1 text-xs">
                    {Object.entries(selectedArtifact.metadata).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span className="text-gray-400">{key}:</span>
                        <span className="text-gray-300">{value as string}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex gap-2 pt-4">
                  <Button
                    size="sm"
                    className="flex-1"
                    onClick={() => setShowScanDialog(true)}
                  >
                    <Shield className="w-4 h-4 mr-1" />
                    Scan
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="flex-1 border-gray-700"
                  >
                    <Download className="w-4 h-4 mr-1" />
                    Pull
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="bg-gray-900/50 border-gray-800">
              <CardContent className="p-6 text-center">
                <Archive className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                <p className="text-gray-400">Select an artifact to view details</p>
              </CardContent>
            </Card>
          )}

          {/* Retention Policy */}
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white text-base">Retention Policy</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <Label className="text-sm text-gray-400">Default Retention (days)</Label>
                <Select value={retentionPolicy} onValueChange={setRetentionPolicy}>
                  <SelectTrigger className="mt-1 bg-gray-800 border-gray-700">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="7">7 days</SelectItem>
                    <SelectItem value="30">30 days</SelectItem>
                    <SelectItem value="90">90 days</SelectItem>
                    <SelectItem value="180">180 days</SelectItem>
                    <SelectItem value="365">365 days</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <Alert className="bg-blue-500/10 border-blue-500/20">
                <Info className="w-4 h-4 text-blue-400" />
                <AlertDescription className="text-xs text-blue-400">
                  Artifacts older than retention period will be automatically deleted
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>

          {/* Storage Quota */}
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white text-base">Storage Quota</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="text-sm text-gray-400">Usage Limit</Label>
                  <span className="text-sm text-white">{quotaLimit[0]}%</span>
                </div>
                <Slider
                  value={quotaLimit}
                  onValueChange={setQuotaLimit}
                  min={50}
                  max={100}
                  step={10}
                />
                <div className="text-xs text-gray-400">
                  Alert when storage exceeds {quotaLimit[0]}% capacity
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Upload Dialog */}
      <Dialog open={showUploadDialog} onOpenChange={setShowUploadDialog}>
        <DialogContent className="bg-gray-900 border-gray-800 max-w-2xl">
          <DialogHeader>
            <DialogTitle className="text-white">Upload Artifact</DialogTitle>
            <DialogDescription className="text-gray-400">
              Upload a new artifact to the registry
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label className="text-gray-300">Registry</Label>
              <Select>
                <SelectTrigger className="mt-1 bg-gray-800 border-gray-700">
                  <SelectValue placeholder="Select registry" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="docker">Docker Registry</SelectItem>
                  <SelectItem value="npm">NPM Registry</SelectItem>
                  <SelectItem value="pypi">PyPI Registry</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-gray-300">Repository</Label>
              <Input className="mt-1 bg-gray-800 border-gray-700" placeholder="e.g., myapp/api" />
            </div>
            <div>
              <Label className="text-gray-300">Version</Label>
              <Input className="mt-1 bg-gray-800 border-gray-700" placeholder="e.g., 1.0.0" />
            </div>
            <div>
              <Label className="text-gray-300">Tags</Label>
              <Input className="mt-1 bg-gray-800 border-gray-700" placeholder="latest, stable" />
            </div>
            <div className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center">
              <Upload className="w-12 h-12 text-gray-500 mx-auto mb-3" />
              <p className="text-gray-400">Drop files here or click to browse</p>
              <p className="text-xs text-gray-500 mt-1">Supports Docker images, NPM packages, Python wheels</p>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowUploadDialog(false)} className="border-gray-700">
              Cancel
            </Button>
            <Button className="bg-gradient-to-r from-blue-600 to-cyan-600">
              Upload
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Promote Dialog */}
      <Dialog open={showPromoteDialog} onOpenChange={setShowPromoteDialog}>
        <DialogContent className="bg-gray-900 border-gray-800">
          <DialogHeader>
            <DialogTitle className="text-white">Promote Artifact</DialogTitle>
            <DialogDescription className="text-gray-400">
              Promote artifact to another environment
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label className="text-gray-300">Target Environment</Label>
              <Select>
                <SelectTrigger className="mt-1 bg-gray-800 border-gray-700">
                  <SelectValue placeholder="Select environment" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="staging">Staging</SelectItem>
                  <SelectItem value="production">Production</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-gray-300">Additional Tags</Label>
              <Input className="mt-1 bg-gray-800 border-gray-700" placeholder="e.g., stable, release" />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowPromoteDialog(false)} className="border-gray-700">
              Cancel
            </Button>
            <Button className="bg-gradient-to-r from-green-600 to-emerald-600">
              Promote
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}