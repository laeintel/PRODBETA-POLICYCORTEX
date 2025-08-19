'use client'

import React, { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  GitBranch,
  GitCommit,
  GitMerge,
  GitPullRequest,
  Shield,
  Lock,
  Unlock,
  Users,
  User,
  Clock,
  Calendar,
  Activity,
  AlertCircle,
  CheckCircle,
  XCircle,
  Info,
  Search,
  Filter,
  Settings,
  MoreVertical,
  ChevronRight,
  RefreshCw,
  Plus,
  Trash2,
  Edit2,
  Copy,
  ExternalLink,
  FileCode,
  FileText,
  Eye,
  EyeOff,
  Code,
  Terminal,
  Package,
  Tag,
  Hash,
  ArrowUp,
  ArrowDown,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  BarChart3,
  PieChart,
  Zap,
  Bell,
  CheckSquare,
  Square,
  Circle,
  CircleDot,
  Star,
  Award,
  Flag,
  Bookmark,
  Archive,
  Download,
  Upload,
  Share2,
  Link,
  Database,
  Server,
  Globe,
  Layers,
  Box,
  Folder,
  FolderOpen,
  Play,
  Pause,
  StopCircle,
  RotateCw,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
  ShieldOff,
  Key,
  UserCheck,
  UserX,
  UserPlus,
  X
} from 'lucide-react'
import { Line, Bar, Doughnut, Scatter } from 'react-chartjs-2'
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

// Mock branch data
const mockBranches = [
  {
    id: 'BR-001',
    name: 'main',
    repository: 'policycortex/api',
    status: 'active',
    protection: 'enabled',
    lastCommit: {
      sha: 'a3f4d5e',
      message: 'feat: add new API endpoints',
      author: 'John Doe',
      date: '2024-01-19T14:30:00Z'
    },
    ahead: 0,
    behind: 0,
    pullRequests: 3,
    openIssues: 12,
    protectionRules: {
      requirePR: true,
      requireApprovals: 2,
      dismissStaleReviews: true,
      requireCodeOwners: true,
      requireStatusChecks: true,
      requireUpToDate: true,
      includeAdmins: false,
      restrictPushes: true,
      allowForcePushes: false,
      allowDeletions: false,
      linearHistory: true,
      signedCommits: true
    },
    policies: ['no-direct-push', 'require-ci-pass', 'auto-delete-head'],
    reviewers: ['alice', 'bob', 'charlie'],
    statusChecks: ['build', 'test', 'security-scan', 'code-quality'],
    mergeStrategies: ['squash', 'merge', 'rebase'],
    autoMerge: false,
    deleteBranchOnMerge: true,
    compliance: 98,
    riskScore: 'low'
  },
  {
    id: 'BR-002',
    name: 'develop',
    repository: 'policycortex/api',
    status: 'active',
    protection: 'enabled',
    lastCommit: {
      sha: 'b5e6f7g',
      message: 'fix: resolve merge conflicts',
      author: 'Jane Smith',
      date: '2024-01-19T13:15:00Z'
    },
    ahead: 12,
    behind: 3,
    pullRequests: 8,
    openIssues: 24,
    protectionRules: {
      requirePR: true,
      requireApprovals: 1,
      dismissStaleReviews: true,
      requireCodeOwners: false,
      requireStatusChecks: true,
      requireUpToDate: false,
      includeAdmins: false,
      restrictPushes: false,
      allowForcePushes: false,
      allowDeletions: false,
      linearHistory: false,
      signedCommits: false
    },
    policies: ['require-ci-pass', 'auto-update'],
    reviewers: ['alice', 'david'],
    statusChecks: ['build', 'test'],
    mergeStrategies: ['squash', 'merge'],
    autoMerge: true,
    deleteBranchOnMerge: false,
    compliance: 85,
    riskScore: 'medium'
  },
  {
    id: 'BR-003',
    name: 'feature/auth-refactor',
    repository: 'policycortex/frontend',
    status: 'active',
    protection: 'disabled',
    lastCommit: {
      sha: 'c8h9i0j',
      message: 'wip: refactor authentication flow',
      author: 'Alice Johnson',
      date: '2024-01-19T11:45:00Z'
    },
    ahead: 45,
    behind: 8,
    pullRequests: 1,
    openIssues: 3,
    protectionRules: {
      requirePR: false,
      requireApprovals: 0,
      dismissStaleReviews: false,
      requireCodeOwners: false,
      requireStatusChecks: false,
      requireUpToDate: false,
      includeAdmins: false,
      restrictPushes: false,
      allowForcePushes: true,
      allowDeletions: true,
      linearHistory: false,
      signedCommits: false
    },
    policies: [],
    reviewers: [],
    statusChecks: [],
    mergeStrategies: ['merge'],
    autoMerge: false,
    deleteBranchOnMerge: true,
    compliance: 45,
    riskScore: 'high'
  },
  {
    id: 'BR-004',
    name: 'release/v2.15.0',
    repository: 'policycortex/api',
    status: 'protected',
    protection: 'enabled',
    lastCommit: {
      sha: 'd1k2l3m',
      message: 'chore: bump version to v2.15.0',
      author: 'Release Bot',
      date: '2024-01-19T10:00:00Z'
    },
    ahead: 0,
    behind: 0,
    pullRequests: 0,
    openIssues: 0,
    protectionRules: {
      requirePR: true,
      requireApprovals: 3,
      dismissStaleReviews: true,
      requireCodeOwners: true,
      requireStatusChecks: true,
      requireUpToDate: true,
      includeAdmins: true,
      restrictPushes: true,
      allowForcePushes: false,
      allowDeletions: false,
      linearHistory: true,
      signedCommits: true
    },
    policies: ['no-direct-push', 'require-ci-pass', 'require-security-scan', 'require-release-notes'],
    reviewers: ['alice', 'bob', 'charlie', 'david', 'eve'],
    statusChecks: ['build', 'test', 'security-scan', 'code-quality', 'performance', 'integration'],
    mergeStrategies: ['merge'],
    autoMerge: false,
    deleteBranchOnMerge: false,
    compliance: 100,
    riskScore: 'low'
  },
  {
    id: 'BR-005',
    name: 'hotfix/security-patch',
    repository: 'policycortex/api',
    status: 'active',
    protection: 'enabled',
    lastCommit: {
      sha: 'e4n5o6p',
      message: 'security: patch critical vulnerability',
      author: 'Security Team',
      date: '2024-01-19T09:30:00Z'
    },
    ahead: 2,
    behind: 0,
    pullRequests: 1,
    openIssues: 1,
    protectionRules: {
      requirePR: true,
      requireApprovals: 2,
      dismissStaleReviews: true,
      requireCodeOwners: true,
      requireStatusChecks: true,
      requireUpToDate: true,
      includeAdmins: false,
      restrictPushes: true,
      allowForcePushes: false,
      allowDeletions: false,
      linearHistory: true,
      signedCommits: true
    },
    policies: ['emergency-merge', 'bypass-ci-on-approval', 'notify-security-team'],
    reviewers: ['security-team', 'alice', 'bob'],
    statusChecks: ['security-scan', 'test'],
    mergeStrategies: ['merge'],
    autoMerge: false,
    deleteBranchOnMerge: true,
    compliance: 95,
    riskScore: 'critical'
  }
]

// Mock policy templates
const policyTemplates = [
  { name: 'Standard Development', rules: 5, repos: 12 },
  { name: 'High Security', rules: 8, repos: 3 },
  { name: 'Open Source', rules: 3, repos: 7 },
  { name: 'Release Branch', rules: 10, repos: 4 },
  { name: 'Feature Branch', rules: 2, repos: 25 }
]

// Generate time series data
const generateTimeSeriesData = (days: number = 30) => {
  const data = []
  const now = new Date()
  for (let i = days; i >= 0; i--) {
    const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000)
    data.push({
      date: date.toISOString().split('T')[0],
      commits: Math.floor(Math.random() * 100) + 20,
      merges: Math.floor(Math.random() * 20) + 5,
      violations: Math.floor(Math.random() * 10)
    })
  }
  return data
}

export default function BranchesPage() {
  const [branches, setBranches] = useState(mockBranches)
  const [selectedBranch, setSelectedBranch] = useState<any>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedRepo, setSelectedRepo] = useState('all')
  const [selectedStatus, setSelectedStatus] = useState('all')
  const [showProtectedOnly, setShowProtectedOnly] = useState(false)
  const [showViolations, setShowViolations] = useState(false)
  const [showPolicyDialog, setShowPolicyDialog] = useState(false)
  const [showProtectionDialog, setShowProtectionDialog] = useState(false)
  const [selectedBranches, setSelectedBranches] = useState<string[]>([])
  const [autoEnforcement, setAutoEnforcement] = useState(true)
  const [requireSignedCommits, setRequireSignedCommits] = useState(true)
  const [allowBypass, setAllowBypass] = useState(false)
  const [minApprovals, setMinApprovals] = useState([2])

  // Filter branches
  const filteredBranches = useMemo(() => {
    return branches.filter(branch => {
      const matchesSearch = branch.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           branch.repository.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesRepo = selectedRepo === 'all' || branch.repository.includes(selectedRepo)
      const matchesStatus = selectedStatus === 'all' || branch.status === selectedStatus
      const matchesProtection = !showProtectedOnly || branch.protection === 'enabled'
      const matchesViolations = !showViolations || branch.compliance < 90
      
      return matchesSearch && matchesRepo && matchesStatus && matchesProtection && matchesViolations
    })
  }, [branches, searchQuery, selectedRepo, selectedStatus, showProtectedOnly, showViolations])

  // Calculate statistics
  const stats = useMemo(() => {
    const protectedCount = branches.filter(b => b.protection === 'enabled').length
    const activeCount = branches.filter(b => b.status === 'active').length
    const totalPRs = branches.reduce((sum, b) => sum + b.pullRequests, 0)
    const avgCompliance = branches.reduce((sum, b) => sum + b.compliance, 0) / branches.length
    const highRiskCount = branches.filter(b => b.riskScore === 'high' || b.riskScore === 'critical').length

    return {
      total: branches.length,
      protected: protectedCount,
      active: activeCount,
      pullRequests: totalPRs,
      compliance: Math.round(avgCompliance),
      highRisk: highRiskCount
    }
  }, [branches])

  // Time series data
  const timeSeriesData = useMemo(() => generateTimeSeriesData(), [])

  // Chart configurations
  const activityChart = {
    labels: timeSeriesData.slice(-7).map(d => d.date.split('-').slice(1).join('/')),
    datasets: [
      {
        label: 'Commits',
        data: timeSeriesData.slice(-7).map(d => d.commits),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Merges',
        data: timeSeriesData.slice(-7).map(d => d.merges),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Violations',
        data: timeSeriesData.slice(-7).map(d => d.violations),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  }

  const protectionDistribution = {
    labels: ['Protected', 'Unprotected'],
    datasets: [{
      data: [
        branches.filter(b => b.protection === 'enabled').length,
        branches.filter(b => b.protection === 'disabled').length
      ],
      backgroundColor: [
        'rgba(34, 197, 94, 0.8)',
        'rgba(239, 68, 68, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const complianceChart = {
    labels: branches.map(b => b.name.substring(0, 15)),
    datasets: [{
      label: 'Compliance Score',
      data: branches.map(b => b.compliance),
      backgroundColor: branches.map(b => 
        b.compliance >= 90 ? 'rgba(34, 197, 94, 0.8)' :
        b.compliance >= 70 ? 'rgba(251, 146, 60, 0.8)' :
        'rgba(239, 68, 68, 0.8)'
      ),
      borderColor: branches.map(b => 
        b.compliance >= 90 ? 'rgb(34, 197, 94)' :
        b.compliance >= 70 ? 'rgb(251, 146, 60)' :
        'rgb(239, 68, 68)'
      ),
      borderWidth: 1
    }]
  }

  const riskDistribution = {
    labels: ['Low', 'Medium', 'High', 'Critical'],
    datasets: [{
      label: 'Risk Level',
      data: [
        branches.filter(b => b.riskScore === 'low').length,
        branches.filter(b => b.riskScore === 'medium').length,
        branches.filter(b => b.riskScore === 'high').length,
        branches.filter(b => b.riskScore === 'critical').length
      ],
      backgroundColor: [
        'rgba(34, 197, 94, 0.8)',
        'rgba(251, 146, 60, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(139, 0, 0, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CircleDot className="w-4 h-4 text-green-500" />
      case 'protected': return <Shield className="w-4 h-4 text-blue-500" />
      case 'inactive': return <Circle className="w-4 h-4 text-gray-500" />
      default: return <Circle className="w-4 h-4 text-gray-500" />
    }
  }

  const getRiskBadge = (risk: string) => {
    switch (risk) {
      case 'low': return <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Low Risk</Badge>
      case 'medium': return <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">Medium Risk</Badge>
      case 'high': return <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/30">High Risk</Badge>
      case 'critical': return <Badge className="bg-red-500/20 text-red-400 border-red-500/30">Critical</Badge>
      default: return <Badge>Unknown</Badge>
    }
  }

  const handleBulkAction = (action: string) => {
    console.log(`Performing ${action} on`, selectedBranches)
    setSelectedBranches([])
  }

  const applyPolicyTemplate = (template: string) => {
    console.log(`Applying ${template} policy template`)
    setShowPolicyDialog(false)
  }

  return (
    <div className="flex-1 space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Branch Management</h1>
          <p className="text-gray-400">Configure branch protection rules and policies</p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowPolicyDialog(true)}
            className="border-gray-700 hover:bg-gray-800"
          >
            <FileText className="w-4 h-4 mr-2" />
            Policy Templates
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="border-gray-700 hover:bg-gray-800"
          >
            <Download className="w-4 h-4 mr-2" />
            Export Rules
          </Button>
          <Button
            className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700"
            onClick={() => setShowProtectionDialog(true)}
          >
            <Shield className="w-4 h-4 mr-2" />
            New Protection Rule
          </Button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <GitBranch className="w-8 h-8 text-blue-500" />
              <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30">Total</Badge>
            </div>
            <p className="text-2xl font-bold text-white">{stats.total}</p>
            <p className="text-sm text-gray-400">Branches</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Shield className="w-8 h-8 text-green-500" />
              <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Protected</Badge>
            </div>
            <p className="text-2xl font-bold text-white">{stats.protected}</p>
            <p className="text-sm text-gray-400">Protected</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Activity className="w-8 h-8 text-purple-500" />
              <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30">Active</Badge>
            </div>
            <p className="text-2xl font-bold text-white">{stats.active}</p>
            <p className="text-sm text-gray-400">Active</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <GitPullRequest className="w-8 h-8 text-cyan-500" />
              <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/30">PRs</Badge>
            </div>
            <p className="text-2xl font-bold text-white">{stats.pullRequests}</p>
            <p className="text-sm text-gray-400">Open PRs</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <CheckCircle className="w-8 h-8 text-yellow-500" />
              <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">
                {stats.compliance}%
              </Badge>
            </div>
            <p className="text-2xl font-bold text-white">{stats.compliance}%</p>
            <p className="text-sm text-gray-400">Compliance</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <AlertTriangle className="w-8 h-8 text-red-500" />
              {stats.highRisk > 0 && (
                <Badge className="bg-red-500/20 text-red-400 border-red-500/30">{stats.highRisk}</Badge>
              )}
            </div>
            <p className="text-2xl font-bold text-white">{stats.highRisk}</p>
            <p className="text-sm text-gray-400">High Risk</p>
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
                placeholder="Search branches..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-gray-800 border-gray-700 text-white"
              />
            </div>
            
            <div className="flex flex-wrap gap-2">
              <Select value={selectedRepo} onValueChange={setSelectedRepo}>
                <SelectTrigger className="w-[160px] bg-gray-800 border-gray-700">
                  <SelectValue placeholder="Repository" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Repositories</SelectItem>
                  <SelectItem value="api">policycortex/api</SelectItem>
                  <SelectItem value="frontend">policycortex/frontend</SelectItem>
                  <SelectItem value="ml">policycortex/ml</SelectItem>
                </SelectContent>
              </Select>

              <Select value={selectedStatus} onValueChange={setSelectedStatus}>
                <SelectTrigger className="w-[140px] bg-gray-800 border-gray-700">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="active">Active</SelectItem>
                  <SelectItem value="protected">Protected</SelectItem>
                  <SelectItem value="inactive">Inactive</SelectItem>
                </SelectContent>
              </Select>

              <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded-lg border border-gray-700">
                <Switch
                  checked={showProtectedOnly}
                  onCheckedChange={setShowProtectedOnly}
                  className="scale-75"
                />
                <Label className="text-sm text-gray-300 cursor-pointer">
                  Protected Only
                </Label>
              </div>

              <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded-lg border border-gray-700">
                <Switch
                  checked={showViolations}
                  onCheckedChange={setShowViolations}
                  className="scale-75"
                />
                <Label className="text-sm text-gray-300 cursor-pointer">
                  Violations
                </Label>
              </div>
            </div>
          </div>

          {/* Bulk Actions */}
          {selectedBranches.length > 0 && (
            <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg flex items-center justify-between">
              <span className="text-sm text-blue-400">
                {selectedBranches.length} branches selected
              </span>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleBulkAction('protect')}
                  className="border-green-500/50 text-green-400 hover:bg-green-500/20"
                >
                  <Shield className="w-4 h-4 mr-1" />
                  Protect
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleBulkAction('unprotect')}
                  className="border-yellow-500/50 text-yellow-400 hover:bg-yellow-500/20"
                >
                  <Unlock className="w-4 h-4 mr-1" />
                  Unprotect
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
                  onClick={() => setSelectedBranches([])}
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
        {/* Branches List */}
        <div className="lg:col-span-2">
          <Tabs defaultValue="branches" className="w-full">
            <TabsList className="bg-gray-800 border-gray-700">
              <TabsTrigger value="branches">Branches</TabsTrigger>
              <TabsTrigger value="policies">Policies</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
              <TabsTrigger value="compliance">Compliance</TabsTrigger>
            </TabsList>

            <TabsContent value="branches" className="space-y-4">
              {filteredBranches.map((branch) => (
                <motion.div
                  key={branch.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <Card
                    className={`bg-gray-900/50 border-gray-800 hover:border-gray-700 transition-all cursor-pointer ${
                      selectedBranch?.id === branch.id ? 'ring-2 ring-blue-500' : ''
                    } ${selectedBranches.includes(branch.id) ? 'ring-1 ring-blue-500/50' : ''}`}
                    onClick={() => setSelectedBranch(branch)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-lg bg-gray-800">
                            <GitBranch className="w-5 h-5 text-blue-400" />
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <p className="font-semibold text-white">{branch.name}</p>
                              {getStatusIcon(branch.status)}
                              {branch.protection === 'enabled' && (
                                <Lock className="w-4 h-4 text-yellow-400" />
                              )}
                            </div>
                            <p className="text-xs text-gray-400">{branch.repository}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Checkbox
                            checked={selectedBranches.includes(branch.id)}
                            onCheckedChange={(checked) => {
                              if (checked) {
                                setSelectedBranches([...selectedBranches, branch.id])
                              } else {
                                setSelectedBranches(selectedBranches.filter(id => id !== branch.id))
                              }
                            }}
                            onClick={(e) => e.stopPropagation()}
                          />
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <MoreVertical className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                        <div>
                          <p className="text-xs text-gray-400">Last Commit</p>
                          <p className="text-sm text-white font-mono">{branch.lastCommit.sha}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Ahead/Behind</p>
                          <div className="flex items-center gap-2 text-sm">
                            <span className="text-green-400">+{branch.ahead}</span>
                            <span className="text-red-400">-{branch.behind}</span>
                          </div>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Pull Requests</p>
                          <p className="text-sm text-white">{branch.pullRequests}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Compliance</p>
                          <p className="text-sm text-white">{branch.compliance}%</p>
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          {branch.policies.slice(0, 2).map((policy) => (
                            <Badge key={policy} variant="outline" className="text-xs">
                              {policy}
                            </Badge>
                          ))}
                          {branch.policies.length > 2 && (
                            <span className="text-xs text-gray-500">+{branch.policies.length - 2}</span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          {getRiskBadge(branch.riskScore)}
                          {branch.autoMerge && (
                            <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30 text-xs">
                              Auto-merge
                            </Badge>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </TabsContent>

            <TabsContent value="policies" className="space-y-4">
              {policyTemplates.map((template) => (
                <Card key={template.name} className="bg-gray-900/50 border-gray-800">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-gray-800">
                          <FileText className="w-5 h-5 text-purple-400" />
                        </div>
                        <div>
                          <p className="font-semibold text-white">{template.name}</p>
                          <p className="text-xs text-gray-400">{template.rules} rules</p>
                        </div>
                      </div>
                      <Badge variant="outline">
                        {template.repos} repos
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button size="sm" variant="outline" className="flex-1 border-gray-700">
                        <Eye className="w-4 h-4 mr-1" />
                        View
                      </Button>
                      <Button size="sm" className="flex-1 bg-gradient-to-r from-blue-600 to-cyan-600">
                        Apply
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </TabsContent>

            <TabsContent value="analytics" className="space-y-4">
              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white">Branch Activity</CardTitle>
                </CardHeader>
                <CardContent>
                  <Line
                    data={activityChart}
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
                    <CardTitle className="text-white text-base">Protection Status</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Doughnut
                      data={protectionDistribution}
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
                    <CardTitle className="text-white text-base">Risk Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Doughnut
                      data={riskDistribution}
                      options={{
                        responsive: true,
                        plugins: {
                          legend: { display: true, position: 'bottom', labels: { color: 'white' } }
                        }
                      }}
                    />
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="compliance" className="space-y-4">
              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white">Compliance Scores</CardTitle>
                </CardHeader>
                <CardContent>
                  <Bar
                    data={complianceChart}
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
                          ticks: { color: 'rgba(255, 255, 255, 0.5)' },
                          max: 100
                        }
                      }
                    }}
                  />
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {selectedBranch ? (
            <Card className="bg-gray-900/50 border-gray-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center justify-between">
                  Branch Details
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setSelectedBranch(null)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-sm text-gray-400 mb-1">Repository</p>
                  <p className="text-white font-medium">{selectedBranch.repository}</p>
                </div>

                <div>
                  <p className="text-sm text-gray-400 mb-2">Protection Rules</p>
                  <div className="space-y-2 text-xs">
                    {Object.entries(selectedBranch.protectionRules).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span className="text-gray-400">{key}:</span>
                        <span className={value ? 'text-green-400' : 'text-gray-500'}>
                          {value ? 'Enabled' : 'Disabled'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-sm text-gray-400 mb-2">Reviewers</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedBranch.reviewers.map((reviewer: string) => (
                      <Badge key={reviewer} variant="outline" className="text-xs">
                        @{reviewer}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-sm text-gray-400 mb-2">Status Checks</p>
                  <div className="space-y-1">
                    {selectedBranch.statusChecks.map((check: string) => (
                      <div key={check} className="flex items-center gap-2 text-xs">
                        <CheckCircle className="w-3 h-3 text-green-400" />
                        <span className="text-gray-300">{check}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex gap-2 pt-4">
                  <Button
                    size="sm"
                    className="flex-1"
                  >
                    <Edit2 className="w-4 h-4 mr-1" />
                    Edit Rules
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="flex-1 border-gray-700"
                  >
                    <Copy className="w-4 h-4 mr-1" />
                    Clone
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="bg-gray-900/50 border-gray-800">
              <CardContent className="p-6 text-center">
                <GitBranch className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                <p className="text-gray-400">Select a branch to view details</p>
              </CardContent>
            </Card>
          )}

          {/* Protection Settings */}
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white text-base">Global Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                <div className="flex items-center gap-3">
                  <Shield className="w-4 h-4 text-blue-400" />
                  <span className="text-sm text-white">Auto-enforcement</span>
                </div>
                <Switch
                  checked={autoEnforcement}
                  onCheckedChange={setAutoEnforcement}
                  className="scale-75"
                />
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                <div className="flex items-center gap-3">
                  <Key className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-white">Signed Commits</span>
                </div>
                <Switch
                  checked={requireSignedCommits}
                  onCheckedChange={setRequireSignedCommits}
                  className="scale-75"
                />
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                <div className="flex items-center gap-3">
                  <UserCheck className="w-4 h-4 text-yellow-400" />
                  <span className="text-sm text-white">Admin Bypass</span>
                </div>
                <Switch
                  checked={allowBypass}
                  onCheckedChange={setAllowBypass}
                  className="scale-75"
                />
              </div>

              <div className="pt-3">
                <Label className="text-sm text-gray-400">Min Approvals</Label>
                <div className="flex items-center gap-3 mt-2">
                  <Slider
                    value={minApprovals}
                    onValueChange={setMinApprovals}
                    min={1}
                    max={5}
                    step={1}
                    className="flex-1"
                  />
                  <span className="text-sm text-white w-8">{minApprovals[0]}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white text-base">Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button className="w-full justify-start" variant="outline" size="sm">
                <RefreshCw className="w-4 h-4 mr-2" />
                Sync All Branches
              </Button>
              <Button className="w-full justify-start" variant="outline" size="sm">
                <AlertTriangle className="w-4 h-4 mr-2" />
                Review Violations
              </Button>
              <Button className="w-full justify-start" variant="outline" size="sm">
                <Archive className="w-4 h-4 mr-2" />
                Archive Stale Branches
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Protection Dialog */}
      <Dialog open={showProtectionDialog} onOpenChange={setShowProtectionDialog}>
        <DialogContent className="bg-gray-900 border-gray-800 max-w-2xl">
          <DialogHeader>
            <DialogTitle className="text-white">Create Protection Rule</DialogTitle>
            <DialogDescription className="text-gray-400">
              Configure branch protection settings
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label className="text-gray-300">Branch Pattern</Label>
              <Input className="mt-1 bg-gray-800 border-gray-700" placeholder="e.g., main, release/*" />
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-gray-300">Require pull request reviews</Label>
                <Switch />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-gray-300">Dismiss stale reviews</Label>
                <Switch />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-gray-300">Require status checks</Label>
                <Switch />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-gray-300">Include administrators</Label>
                <Switch />
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowProtectionDialog(false)} className="border-gray-700">
              Cancel
            </Button>
            <Button className="bg-gradient-to-r from-blue-600 to-cyan-600">
              Create Rule
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}