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
  Lock,
  Users,
  Shield,
  Key,
  UserCheck,
  UserX,
  Settings,
  AlertTriangle,
  CheckCircle,
  Eye,
  EyeOff,
  Plus,
  Edit,
  Trash2,
  Download,
  RefreshCw,
  Search,
  Filter,
  MoreVertical,
  Clock,
  Activity,
  TrendingUp,
  TrendingDown,
  Calendar,
  Globe,
  Server,
  Database,
  Cloud,
  Terminal,
  FileText,
  Bell,
  Zap,
  ChevronRight,
  ChevronDown,
  X,
  Check,
  AlertCircle,
  Info,
  ExternalLink,
  History,
  UserPlus,
  UserMinus
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
  ArcElement
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
  ArcElement
)

interface Permission {
  id: string
  name: string
  resource: string
  actions: string[]
  effect: 'Allow' | 'Deny'
}

interface Role {
  id: string
  name: string
  description: string
  permissions: Permission[]
  users: number
  createdAt: string
  lastModified: string
  isBuiltIn: boolean
}

interface AccessRequest {
  id: string
  user: string
  email: string
  role: string
  resource: string
  reason: string
  requestedAt: string
  status: 'pending' | 'approved' | 'denied'
  approver?: string
  priority: 'low' | 'medium' | 'high' | 'critical'
  expiresAt?: string
  ipAddress: string
  userAgent: string
}

interface AccessEvent {
  id: string
  user: string
  action: string
  resource: string
  timestamp: Date
  status: 'success' | 'failed' | 'blocked'
  ipAddress: string
  userAgent: string
  risk: 'low' | 'medium' | 'high'
}

interface PrivilegedAccess {
  id: string
  user: string
  role: string
  resource: string
  grantedAt: string
  expiresAt: string
  isActive: boolean
  sessionDuration: number
  accessCount: number
}

interface SecurityMetrics {
  totalUsers: number
  activeUsers: number
  privilegedUsers: number
  totalRoles: number
  totalPermissions: number
  pendingRequests: number
  failedAttempts24h: number
  mfaEnabled: number
  passwordExpiring: number
  riskScore: number
  complianceScore: number
}

export default function AccessControlPage() {
  const [roles, setRoles] = useState<Role[]>([])
  const [accessRequests, setAccessRequests] = useState<AccessRequest[]>([])
  const [accessEvents, setAccessEvents] = useState<AccessEvent[]>([])
  const [privilegedAccess, setPrivilegedAccess] = useState<PrivilegedAccess[]>([])
  const [metrics, setMetrics] = useState<SecurityMetrics | null>(null)
  const [selectedTab, setSelectedTab] = useState<'overview' | 'roles' | 'permissions' | 'requests' | 'events' | 'privileged'>('overview')
  const [loading, setLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedRole, setSelectedRole] = useState<Role | null>(null)
  const [showCreateRole, setShowCreateRole] = useState(false)
  const [realTimeData, setRealTimeData] = useState<any[]>([])

  useEffect(() => {
    loadAllData()
    const interval = autoRefresh ? setInterval(loadRealTimeData, 10000) : null
    return () => { if (interval) clearInterval(interval) }
  }, [autoRefresh])

  const loadAllData = () => {
    setLoading(true)
    setTimeout(() => {
      // Set metrics
      setMetrics({
        totalUsers: 2456,
        activeUsers: 1834,
        privilegedUsers: 89,
        totalRoles: 12,
        totalPermissions: 156,
        pendingRequests: 7,
        failedAttempts24h: 23,
        mfaEnabled: 87,
        passwordExpiring: 34,
        riskScore: 72,
        complianceScore: 94
      })

      setRoles([
        {
          id: 'role-001',
          name: 'Administrator',
          description: 'Full system access with all permissions',
          permissions: [
            { id: 'p1', name: 'All Access', resource: '*', actions: ['*'], effect: 'Allow' }
          ],
          users: 3,
          createdAt: '6 months ago',
          lastModified: '1 week ago',
          isBuiltIn: true
        },
        {
          id: 'role-002',
          name: 'Developer',
          description: 'Development environment access',
          permissions: [
            { id: 'p2', name: 'Read Resources', resource: 'dev/*', actions: ['read', 'list'], effect: 'Allow' },
            { id: 'p3', name: 'Deploy Code', resource: 'dev/deployments', actions: ['create', 'update'], effect: 'Allow' }
          ],
          users: 24,
          createdAt: '5 months ago',
          lastModified: '2 days ago',
          isBuiltIn: false
        },
        {
          id: 'role-003',
          name: 'Security Analyst',
          description: 'Security monitoring and audit access',
          permissions: [
            { id: 'p4', name: 'View Logs', resource: 'logs/*', actions: ['read'], effect: 'Allow' },
            { id: 'p5', name: 'Security Scan', resource: 'security/*', actions: ['read', 'execute'], effect: 'Allow' }
          ],
          users: 8,
          createdAt: '4 months ago',
          lastModified: '1 week ago',
          isBuiltIn: false
        },
        {
          id: 'role-004',
          name: 'Read Only',
          description: 'View-only access to all resources',
          permissions: [
            { id: 'p6', name: 'Read All', resource: '*', actions: ['read', 'list'], effect: 'Allow' },
            { id: 'p7', name: 'Deny Modifications', resource: '*', actions: ['create', 'update', 'delete'], effect: 'Deny' }
          ],
          users: 45,
          createdAt: '6 months ago',
          lastModified: '1 month ago',
          isBuiltIn: true
        },
        {
          id: 'role-005',
          name: 'Auditor',
          description: 'Audit and compliance monitoring access',
          permissions: [
            { id: 'p8', name: 'Audit Logs', resource: 'audit/*', actions: ['read'], effect: 'Allow' },
            { id: 'p9', name: 'Compliance Reports', resource: 'compliance/*', actions: ['read', 'export'], effect: 'Allow' }
          ],
          users: 12,
          createdAt: '3 months ago',
          lastModified: '2 weeks ago',
          isBuiltIn: false
        },
        {
          id: 'role-006',
          name: 'Support Agent',
          description: 'Customer support and troubleshooting access',
          permissions: [
            { id: 'p10', name: 'View Tickets', resource: 'support/*', actions: ['read', 'update'], effect: 'Allow' },
            { id: 'p11', name: 'User Lookup', resource: 'users/*', actions: ['read'], effect: 'Allow' }
          ],
          users: 67,
          createdAt: '2 months ago',
          lastModified: '5 days ago',
          isBuiltIn: false
        }
      ])

      setAccessRequests([
        {
          id: 'req-001',
          user: 'John Doe',
          email: 'john.doe@company.com',
          role: 'Developer',
          resource: 'production/database',
          reason: 'Need to debug production issue #1234',
          requestedAt: '2 hours ago',
          status: 'pending',
          priority: 'high',
          expiresAt: '4 hours',
          ipAddress: '192.168.1.100',
          userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        },
        {
          id: 'req-002',
          user: 'Jane Smith',
          email: 'jane.smith@company.com',
          role: 'Administrator',
          resource: 'billing/accounts',
          reason: 'Quarterly audit requirement',
          requestedAt: '1 day ago',
          status: 'approved',
          approver: 'admin@company.com',
          priority: 'medium',
          ipAddress: '192.168.1.101',
          userAgent: 'Mozilla/5.0 (macOS; Intel Mac OS X 10_15_7)'
        },
        {
          id: 'req-003',
          user: 'Bob Wilson',
          email: 'bob.wilson@company.com',
          role: 'Security Analyst',
          resource: 'logs/sensitive',
          reason: 'Security incident investigation',
          requestedAt: '3 days ago',
          status: 'denied',
          approver: 'security@company.com',
          priority: 'critical',
          ipAddress: '192.168.1.102',
          userAgent: 'Mozilla/5.0 (Linux; Ubuntu)'
        },
        {
          id: 'req-004',
          user: 'Alice Johnson',
          email: 'alice.johnson@company.com',
          role: 'Auditor',
          resource: 'financial/reports',
          reason: 'SOX compliance audit',
          requestedAt: '6 hours ago',
          status: 'pending',
          priority: 'medium',
          expiresAt: '24 hours',
          ipAddress: '192.168.1.103',
          userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
      ])

      setAccessEvents([
        { id: 'e1', user: 'john.doe@company.com', action: 'Login', resource: 'Portal', timestamp: new Date(Date.now() - 300000), status: 'success', ipAddress: '192.168.1.100', userAgent: 'Chrome/120.0', risk: 'low' },
        { id: 'e2', user: 'jane.smith@company.com', action: 'Access', resource: 'Admin Panel', timestamp: new Date(Date.now() - 600000), status: 'success', ipAddress: '192.168.1.101', userAgent: 'Safari/17.0', risk: 'medium' },
        { id: 'e3', user: 'unknown@external.com', action: 'Failed Login', resource: 'Portal', timestamp: new Date(Date.now() - 900000), status: 'blocked', ipAddress: '203.0.113.42', userAgent: 'Bot/1.0', risk: 'high' },
        { id: 'e4', user: 'bob.wilson@company.com', action: 'Export Data', resource: 'Database', timestamp: new Date(Date.now() - 1200000), status: 'success', ipAddress: '192.168.1.102', userAgent: 'Firefox/119.0', risk: 'medium' },
        { id: 'e5', user: 'alice.johnson@company.com', action: 'View Report', resource: 'Audit Logs', timestamp: new Date(Date.now() - 1800000), status: 'success', ipAddress: '192.168.1.103', userAgent: 'Edge/119.0', risk: 'low' }
      ])

      setPrivilegedAccess([
        { id: 'pa1', user: 'admin@company.com', role: 'Super Admin', resource: 'System/*', grantedAt: '2 hours ago', expiresAt: '6 hours', isActive: true, sessionDuration: 120, accessCount: 15 },
        { id: 'pa2', user: 'security@company.com', role: 'Security Admin', resource: 'Security/*', grantedAt: '1 day ago', expiresAt: '7 days', isActive: true, sessionDuration: 480, accessCount: 67 },
        { id: 'pa3', user: 'dba@company.com', role: 'Database Admin', resource: 'Database/*', grantedAt: '3 hours ago', expiresAt: '12 hours', isActive: false, sessionDuration: 45, accessCount: 3 }
      ])

      setRealTimeData(generateRealTimeData())
      setLoading(false)
    }, 1000)
  }

  const loadRealTimeData = () => {
    setRealTimeData(prev => {
      const newData = [...prev, {
        timestamp: new Date(),
        logins: Math.floor(Math.random() * 20),
        failures: Math.floor(Math.random() * 5),
        privileged: Math.floor(Math.random() * 10)
      }]
      return newData.slice(-20)
    })
  }

  const generateRealTimeData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 60000),
      logins: Math.floor(Math.random() * 20),
      failures: Math.floor(Math.random() * 5),
      privileged: Math.floor(Math.random() * 10)
    }))
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved': case 'success': return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'denied': case 'failed': case 'blocked': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'pending': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'low': return 'bg-green-500/20 text-green-400 border-green-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return 'text-red-500'
      case 'medium': return 'text-yellow-500'
      case 'low': return 'text-green-500'
      default: return 'text-gray-500'
    }
  }

  const accessTrendData = {
    labels: realTimeData.map(d => d.timestamp.toLocaleTimeString()),
    datasets: [
      {
        label: 'Successful Logins',
        data: realTimeData.map(d => d.logins),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4
      },
      {
        label: 'Failed Attempts',
        data: realTimeData.map(d => d.failures),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4
      },
      {
        label: 'Privileged Access',
        data: realTimeData.map(d => d.privileged),
        borderColor: 'rgb(168, 85, 247)',
        backgroundColor: 'rgba(168, 85, 247, 0.1)',
        tension: 0.4
      }
    ]
  }

  const roleDistributionData = {
    labels: roles.map(r => r.name),
    datasets: [{
      data: roles.map(r => r.users),
      backgroundColor: [
        'rgba(239, 68, 68, 0.8)',
        'rgba(251, 146, 60, 0.8)',
        'rgba(251, 191, 36, 0.8)',
        'rgba(34, 197, 94, 0.8)',
        'rgba(59, 130, 246, 0.8)',
        'rgba(168, 85, 247, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const filteredRoles = roles.filter(role => 
    searchTerm === '' || 
    role.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    role.description.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const filteredRequests = accessRequests.filter(request =>
    searchTerm === '' ||
    request.user.toLowerCase().includes(searchTerm.toLowerCase()) ||
    request.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
    request.role.toLowerCase().includes(searchTerm.toLowerCase()) ||
    request.resource.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const filteredEvents = accessEvents.filter(event =>
    searchTerm === '' ||
    event.user.toLowerCase().includes(searchTerm.toLowerCase()) ||
    event.action.toLowerCase().includes(searchTerm.toLowerCase()) ||
    event.resource.toLowerCase().includes(searchTerm.toLowerCase())
  )

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading Access Control Center...</p>
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
              <Lock className="w-8 h-8 text-purple-500" />
              <div>
                <h1 className="text-2xl font-bold">Access Control Center</h1>
                <p className="text-sm text-gray-500">Role-based access control and identity management</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">IAM OPERATIONAL</span>
              </div>
              <button 
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
                onClick={() => setShowCreateRole(true)}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded transition-colors flex items-center space-x-2"
              >
                <Plus className="w-4 h-4" />
                <span>Create Role</span>
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'roles', 'permissions', 'requests', 'events', 'privileged'].map((tab) => (
            <button
              key={tab}
              onClick={() => setSelectedTab(tab as any)}
              className={`py-3 px-1 border-b-2 transition-colors capitalize ${selectedTab === tab
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
        {selectedTab === 'overview' && metrics && (
          <>
            {/* Overview Metrics */}
            <div className="grid grid-cols-6 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Users className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Users</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.totalUsers}</p>
                <p className="text-xs text-gray-500 mt-1">{metrics.activeUsers} active</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Shield className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Roles</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.totalRoles}</p>
                <p className="text-xs text-gray-500 mt-1">{roles.filter(r => !r.isBuiltIn).length} custom</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Key className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Permissions</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.totalPermissions}</p>
                <p className="text-xs text-gray-500 mt-1">across all roles</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Requests</span>
                </div>
                <p className="text-2xl font-bold font-mono text-yellow-500">{metrics.pendingRequests}</p>
                <p className="text-xs text-gray-500 mt-1">pending approval</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <UserCheck className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">MFA Enabled</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.mfaEnabled}%</p>
                <div className="mt-2 h-1 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500 rounded-full" style={{ width: `${metrics.mfaEnabled}%` }} />
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <AlertCircle className="w-5 h-5 text-red-500" />
                  <span className="text-xs text-gray-500">Failed Attempts</span>
                </div>
                <p className="text-2xl font-bold font-mono text-red-500">{metrics.failedAttempts24h}</p>
                <p className="text-xs text-gray-500 mt-1">last 24h</p>
              </motion.div>
            </div>

            {/* Charts and Data Grid */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {/* Access Trend Chart */}
              <div className="col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">ACCESS ACTIVITY TRENDS</h3>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      <span className="text-xs text-gray-500">Logins</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-red-500 rounded-full" />
                      <span className="text-xs text-gray-500">Failures</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-purple-500 rounded-full" />
                      <span className="text-xs text-gray-500">Privileged</span>
                    </div>
                  </div>
                </div>
                <div className="h-64">
                  <Line data={accessTrendData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false }
                    },
                    scales: {
                      x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                      },
                      y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                      }
                    }
                  }} />
                </div>
              </div>

              {/* Role Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">ROLE DISTRIBUTION</h3>
                <div className="h-64 flex items-center justify-center">
                  <Doughnut data={roleDistributionData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: { 
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 10 }
                        }
                      }
                    }
                  }} />
                </div>
              </div>
            </div>

            {/* Quick Stats Grid */}
            <div className="grid grid-cols-4 gap-6 mb-6">
              {/* Risk Score */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">RISK ASSESSMENT</h3>
                </div>
                <div className="p-4">
                  <div className="flex items-center space-x-4">
                    <div className="relative w-16 h-16">
                      <svg className="w-full h-full transform -rotate-90">
                        <circle cx="32" cy="32" r="28" stroke="rgba(255,255,255,0.1)" strokeWidth="4" fill="none" />
                        <circle
                          cx="32" cy="32" r="28"
                          stroke={metrics.riskScore > 75 ? 'rgb(239, 68, 68)' : metrics.riskScore > 50 ? 'rgb(251, 191, 36)' : 'rgb(34, 197, 94)'}
                          strokeWidth="4"
                          fill="none"
                          strokeDasharray={`${2 * Math.PI * 28}`}
                          strokeDashoffset={`${2 * Math.PI * 28 * (1 - metrics.riskScore / 100)}`}
                          className="transition-all duration-1000"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-lg font-bold">{metrics.riskScore}%</span>
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Current Risk Level</p>
                      <p className={`text-sm font-medium ${metrics.riskScore > 75 ? 'text-red-500' : metrics.riskScore > 50 ? 'text-yellow-500' : 'text-green-500'}`}>
                        {metrics.riskScore > 75 ? 'High Risk' : metrics.riskScore > 50 ? 'Medium Risk' : 'Low Risk'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Compliance Score */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">COMPLIANCE</h3>
                </div>
                <div className="p-4">
                  <div className="flex items-center space-x-4">
                    <div className="relative w-16 h-16">
                      <svg className="w-full h-full transform -rotate-90">
                        <circle cx="32" cy="32" r="28" stroke="rgba(255,255,255,0.1)" strokeWidth="4" fill="none" />
                        <circle
                          cx="32" cy="32" r="28"
                          stroke="rgb(34, 197, 94)"
                          strokeWidth="4"
                          fill="none"
                          strokeDasharray={`${2 * Math.PI * 28}`}
                          strokeDashoffset={`${2 * Math.PI * 28 * (1 - metrics.complianceScore / 100)}`}
                          className="transition-all duration-1000"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-lg font-bold">{metrics.complianceScore}%</span>
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Compliance Score</p>
                      <p className="text-sm font-medium text-green-500">Compliant</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Privileged Users */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">PRIVILEGED ACCESS</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Total Privileged</span>
                    <span className="font-mono text-yellow-500">{metrics.privilegedUsers}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Active Sessions</span>
                    <span className="font-mono text-green-500">{privilegedAccess.filter(pa => pa.isActive).length}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Expired Access</span>
                    <span className="font-mono text-red-500">{privilegedAccess.filter(pa => !pa.isActive).length}</span>
                  </div>
                  <div className="pt-2 border-t border-gray-800">
                    <button className="w-full px-3 py-2 bg-yellow-600 hover:bg-yellow-700 rounded text-sm">
                      Review Access
                    </button>
                  </div>
                </div>
              </div>

              {/* Password Security */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">PASSWORD SECURITY</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Expiring Soon</span>
                    <span className="font-mono text-orange-500">{metrics.passwordExpiring}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Strong Passwords</span>
                    <span className="font-mono text-green-500">92%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Password Resets</span>
                    <span className="font-mono">12 today</span>
                  </div>
                  <div className="pt-2 border-t border-gray-800">
                    <button className="w-full px-3 py-2 bg-orange-600 hover:bg-orange-700 rounded text-sm">
                      Force Reset
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {selectedTab === 'roles' && (
          <>
            {/* Search and Filters */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <input
                    type="text"
                    placeholder="Search roles..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                <button className="p-2 bg-gray-800 border border-gray-700 rounded-lg hover:bg-gray-700">
                  <Filter className="w-4 h-4 text-gray-400" />
                </button>
              </div>
              <div className="flex items-center space-x-2">
                <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg flex items-center space-x-2">
                  <Download className="w-4 h-4" />
                  <span>Export</span>
                </button>
              </div>
            </div>

            {/* Roles Grid */}
            <div className="space-y-4">
              {filteredRoles.map((role, index) => (
                <motion.div
                  key={role.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-gray-900 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors"
                >
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                          {role.name}
                          {role.isBuiltIn && (
                            <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 text-xs rounded">
                              Built-in
                            </span>
                          )}
                        </h3>
                        <p className="text-sm text-gray-400 mt-1">{role.description}</p>
                      </div>
                      <div className="flex gap-2">
                        <button 
                          onClick={() => setSelectedRole(role)}
                          className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                        >
                          <Eye className="w-4 h-4 text-gray-400" />
                        </button>
                        <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
                          <Edit className="w-4 h-4 text-gray-400" />
                        </button>
                        {!role.isBuiltIn && (
                          <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
                            <Trash2 className="w-4 h-4 text-red-400" />
                          </button>
                        )}
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-4 mb-4">
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Users Assigned</p>
                        <p className="text-lg font-semibold text-white">{role.users}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Permissions</p>
                        <p className="text-lg font-semibold text-white">{role.permissions.length}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Created</p>
                        <p className="text-sm text-white">{role.createdAt}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Last Modified</p>
                        <p className="text-sm text-white">{role.lastModified}</p>
                      </div>
                    </div>

                    <div className="border-t border-gray-700 pt-4">
                      <p className="text-xs text-gray-400 mb-2">Key Permissions</p>
                      <div className="flex flex-wrap gap-2">
                        {role.permissions.slice(0, 3).map((perm) => (
                          <span
                            key={perm.id}
                            className={`px-2 py-1 rounded text-xs ${perm.effect === 'Allow'
                                ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                                : 'bg-red-500/20 text-red-400 border border-red-500/30'
                              }`}
                          >
                            {perm.name}
                          </span>
                        ))}
                        {role.permissions.length > 3 && (
                          <span className="px-2 py-1 rounded text-xs bg-gray-700 text-gray-400">
                            +{role.permissions.length - 3} more
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        )}

        {selectedTab === 'requests' && (
          <>
            {/* Search and Filters */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <input
                    type="text"
                    placeholder="Search requests..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                <select className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                  <option value="">All Status</option>
                  <option value="pending">Pending</option>
                  <option value="approved">Approved</option>
                  <option value="denied">Denied</option>
                </select>
                <select className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                  <option value="">All Priority</option>
                  <option value="critical">Critical</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
              </div>
            </div>

            {/* Requests Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                <h3 className="text-sm font-bold text-gray-400 uppercase">ACCESS REQUESTS</h3>
                <div className="flex items-center space-x-2">
                  <button className="p-1.5 hover:bg-gray-800 rounded">
                    <Download className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">User</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Role</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Resource</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Priority</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Requested</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {filteredRequests.map((request) => (
                      <motion.tr
                        key={request.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div>
                            <div className="font-medium text-white">{request.user}</div>
                            <div className="text-xs text-gray-500">{request.email}</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="px-2 py-1 bg-purple-500/20 text-purple-400 text-xs rounded">
                            {request.role}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <code className="text-xs bg-gray-800 px-2 py-1 rounded text-blue-400">{request.resource}</code>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 text-xs rounded border ${getPriorityColor(request.priority)}`}>
                            {request.priority.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 text-xs rounded border ${getStatusColor(request.status)}`}>
                            {request.status.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm text-white">{request.requestedAt}</div>
                          {request.expiresAt && (
                            <div className="text-xs text-gray-500">Expires in {request.expiresAt}</div>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            {request.status === 'pending' ? (
                              <>
                                <button className="p-1 hover:bg-green-700 rounded text-green-400">
                                  <Check className="w-4 h-4" />
                                </button>
                                <button className="p-1 hover:bg-red-700 rounded text-red-400">
                                  <X className="w-4 h-4" />
                                </button>
                              </>
                            ) : (
                              <button className="p-1 hover:bg-gray-700 rounded">
                                <Eye className="w-4 h-4 text-gray-400" />
                              </button>
                            )}
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

        {selectedTab === 'events' && (
          <>
            {/* Search and Filters */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <input
                    type="text"
                    placeholder="Search events..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                <select className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                  <option value="">All Events</option>
                  <option value="login">Login</option>
                  <option value="logout">Logout</option>
                  <option value="access">Access</option>
                  <option value="failed">Failed</option>
                </select>
                <select className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                  <option value="">All Risk</option>
                  <option value="high">High Risk</option>
                  <option value="medium">Medium Risk</option>
                  <option value="low">Low Risk</option>
                </select>
              </div>
            </div>

            {/* Events Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                <h3 className="text-sm font-bold text-gray-400 uppercase">ACCESS EVENTS</h3>
                <div className="flex items-center space-x-2">
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    <span className="text-xs text-gray-500">Live Feed</span>
                  </div>
                  <button className="p-1.5 hover:bg-gray-800 rounded">
                    <Download className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Timestamp</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">User</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Action</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Resource</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Risk</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">IP Address</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {filteredEvents.map((event) => (
                      <motion.tr
                        key={event.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3 text-xs text-gray-400">
                          {event.timestamp.toLocaleString()}
                        </td>
                        <td className="px-4 py-3">
                          <div className="font-medium text-white">{event.user}</div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs rounded">
                            {event.action}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <code className="text-xs bg-gray-800 px-2 py-1 rounded text-green-400">{event.resource}</code>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 text-xs rounded border ${getStatusColor(event.status)}`}>
                            {event.status.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`text-xs font-medium ${getRiskColor(event.risk)}`}>
                            {event.risk.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <code className="text-xs bg-gray-800 px-2 py-1 rounded">{event.ipAddress}</code>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {selectedTab === 'privileged' && (
          <>
            {/* Privileged Access Overview */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <UserCheck className="w-5 h-5 text-yellow-500" />
                  <span className="text-2xl font-bold text-white">{privilegedAccess.length}</span>
                </div>
                <p className="text-gray-400 text-sm">Total Privileged</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <Activity className="w-5 h-5 text-green-500" />
                  <span className="text-2xl font-bold text-green-500">{privilegedAccess.filter(pa => pa.isActive).length}</span>
                </div>
                <p className="text-gray-400 text-sm">Active Sessions</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <Clock className="w-5 h-5 text-orange-500" />
                  <span className="text-2xl font-bold text-orange-500">{privilegedAccess.filter(pa => !pa.isActive).length}</span>
                </div>
                <p className="text-gray-400 text-sm">Expired Access</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-red-500" />
                  <span className="text-2xl font-bold text-red-500">2</span>
                </div>
                <p className="text-gray-400 text-sm">Requires Review</p>
              </div>
            </div>

            {/* Privileged Access Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                <h3 className="text-sm font-bold text-gray-400 uppercase">PRIVILEGED ACCESS SESSIONS</h3>
                <div className="flex items-center space-x-2">
                  <button className="px-3 py-1.5 bg-yellow-600 hover:bg-yellow-700 text-white text-sm rounded">
                    Review All
                  </button>
                  <button className="p-1.5 hover:bg-gray-800 rounded">
                    <Download className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">User</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Role</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Resource</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Duration</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Access Count</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Expires</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {privilegedAccess.map((access) => (
                      <motion.tr
                        key={access.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div className="font-medium text-white">{access.user}</div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="px-2 py-1 bg-red-500/20 text-red-400 text-xs rounded border border-red-500/30">
                            {access.role}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <code className="text-xs bg-gray-800 px-2 py-1 rounded text-yellow-400">{access.resource}</code>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-2">
                            <div className={`w-2 h-2 rounded-full ${access.isActive ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
                            <span className={`text-xs ${access.isActive ? 'text-green-400' : 'text-gray-400'}`}>
                              {access.isActive ? 'ACTIVE' : 'EXPIRED'}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm text-white">{Math.floor(access.sessionDuration / 60)}h {access.sessionDuration % 60}m</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm font-mono text-white">{access.accessCount}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-xs text-gray-400">{access.expiresAt}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            {access.isActive ? (
                              <button className="p-1 hover:bg-red-700 rounded text-red-400" title="Revoke Access">
                                <UserX className="w-4 h-4" />
                              </button>
                            ) : (
                              <button className="p-1 hover:bg-green-700 rounded text-green-400" title="Restore Access">
                                <UserPlus className="w-4 h-4" />
                              </button>
                            )}
                            <button className="p-1 hover:bg-gray-700 rounded" title="View Details">
                              <Eye className="w-4 h-4 text-gray-400" />
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

        {selectedTab === 'permissions' && (
          <>
            {/* Permissions Overview */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 mb-6">
              <h2 className="text-xl font-bold mb-4">Permission Management</h2>
              <div className="grid grid-cols-6 gap-4">
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-400 mb-1">Total Permissions</p>
                  <p className="text-2xl font-bold">156</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-400 mb-1">Allow Rules</p>
                  <p className="text-2xl font-bold text-green-500">134</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-400 mb-1">Deny Rules</p>
                  <p className="text-2xl font-bold text-red-500">22</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-400 mb-1">Resources</p>
                  <p className="text-2xl font-bold">67</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-400 mb-1">Actions</p>
                  <p className="text-2xl font-bold">23</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-400 mb-1">Custom</p>
                  <p className="text-2xl font-bold">89</p>
                </div>
              </div>
            </div>

            {/* Permissions by Category */}
            <div className="space-y-6">
              {[
                { category: 'System Administration', permissions: ['Full System Access', 'User Management', 'System Configuration'], count: 24 },
                { category: 'Development', permissions: ['Code Repository Access', 'Deployment Rights', 'Environment Access'], count: 31 },
                { category: 'Security', permissions: ['Security Monitoring', 'Audit Log Access', 'Incident Response'], count: 18 },
                { category: 'Data Access', permissions: ['Database Read', 'Export Data', 'Sensitive Data View'], count: 27 },
                { category: 'Financial', permissions: ['Billing Access', 'Cost Reports', 'Budget Management'], count: 15 },
                { category: 'Compliance', permissions: ['Compliance Reports', 'Audit Trail', 'Policy Management'], count: 21 }
              ].map((category, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-gray-900 border border-gray-800 rounded-lg"
                >
                  <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                    <h3 className="font-semibold text-white">{category.category}</h3>
                    <span className="text-sm text-gray-400">{category.count} permissions</span>
                  </div>
                  <div className="p-4">
                    <div className="grid grid-cols-3 gap-2">
                      {category.permissions.map((perm, idx) => (
                        <div key={idx} className="bg-gray-800 rounded-lg p-3">
                          <p className="text-sm text-white">{perm}</p>
                          <p className="text-xs text-gray-500 mt-1">
                            {Math.floor(Math.random() * 30) + 10} roles assigned
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}