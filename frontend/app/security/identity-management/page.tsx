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
  Users,
  UserPlus,
  UserCheck,
  UserX,
  Shield,
  Key,
  Lock,
  Smartphone,
  Mail,
  Globe,
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Settings,
  RefreshCw,
  Download,
  Search,
  Filter,
  Eye,
  Edit,
  Trash2,
  MoreVertical,
  TrendingUp,
  TrendingDown,
  Calendar,
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
  Fingerprint,
  CreditCard,
  Phone,
  MapPin,
  Building,
  UserMinus,
  RotateCcw
} from 'lucide-react'
import { Line, Bar, Doughnut, Radar } from 'react-chartjs-2'
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
  RadialLinearScale
)

interface User {
  id: string
  email: string
  name: string
  department: string
  role: string
  status: 'active' | 'inactive' | 'suspended' | 'pending' | 'locked'
  mfaEnabled: boolean
  lastLogin: string
  createdAt: string
  riskScore: number
  groups: string[]
  permissions: number
  avatar?: string
  phone?: string
  location?: string
  manager?: string
  passwordExpiry: string
  loginAttempts: number
  deviceCount: number
  sessionCount: number
  complianceScore: number
  lastPasswordChange: string
  accountAge: number
}

interface IdentityProvider {
  id: string
  name: string
  type: string
  status: 'connected' | 'disconnected' | 'error' | 'syncing'
  users: number
  lastSync: string
  domain: string
  protocol: string
  healthScore: number
  errorCount: number
  syncFrequency: string
}

interface AuthenticationEvent {
  id: string
  user: string
  event: string
  timestamp: Date
  status: 'success' | 'failure' | 'blocked'
  ipAddress: string
  userAgent: string
  location: string
  riskLevel: 'low' | 'medium' | 'high'
  mfaUsed: boolean
}

interface IdentityMetrics {
  totalUsers: number
  activeUsers: number
  inactiveUsers: number
  suspendedUsers: number
  mfaEnabledUsers: number
  highRiskUsers: number
  passwordExpiring: number
  totalProviders: number
  connectedProviders: number
  failedLogins24h: number
  successfulLogins24h: number
  newUsers30d: number
  avgRiskScore: number
  complianceScore: number
}

export default function IdentityManagementPage() {
  const [users, setUsers] = useState<User[]>([])
  const [providers, setProviders] = useState<IdentityProvider[]>([])
  const [authEvents, setAuthEvents] = useState<AuthenticationEvent[]>([])
  const [metrics, setMetrics] = useState<IdentityMetrics | null>(null)
  const [selectedTab, setSelectedTab] = useState<'overview' | 'users' | 'providers' | 'authentication' | 'lifecycle' | 'analytics'>('overview')
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [realTimeData, setRealTimeData] = useState<any[]>([])

  useEffect(() => {
    loadAllData()
    const interval = autoRefresh ? setInterval(loadRealTimeData, 15000) : null
    return () => { if (interval) clearInterval(interval) }
  }, [autoRefresh])

  const loadAllData = () => {
    setLoading(true)
    setTimeout(() => {
      // Set metrics
      setMetrics({
        totalUsers: 2847,
        activeUsers: 2156,
        inactiveUsers: 456,
        suspendedUsers: 89,
        mfaEnabledUsers: 2123,
        highRiskUsers: 67,
        passwordExpiring: 156,
        totalProviders: 7,
        connectedProviders: 5,
        failedLogins24h: 234,
        successfulLogins24h: 8945,
        newUsers30d: 89,
        avgRiskScore: 23,
        complianceScore: 94
      })

      setUsers([
        {
          id: 'usr-001',
          email: 'admin@company.com',
          name: 'System Administrator',
          department: 'IT',
          role: 'Administrator',
          status: 'active',
          mfaEnabled: true,
          lastLogin: '2 hours ago',
          createdAt: '1 year ago',
          riskScore: 5,
          groups: ['Admins', 'Security', 'DevOps'],
          permissions: 156,
          phone: '+1-555-0101',
          location: 'San Francisco, CA',
          manager: 'CTO Office',
          passwordExpiry: '45 days',
          loginAttempts: 0,
          deviceCount: 3,
          sessionCount: 2,
          complianceScore: 98,
          lastPasswordChange: '30 days ago',
          accountAge: 365
        },
        {
          id: 'usr-002',
          email: 'john.doe@company.com',
          name: 'John Doe',
          department: 'Engineering',
          role: 'Senior Developer',
          status: 'active',
          mfaEnabled: true,
          lastLogin: '1 day ago',
          createdAt: '6 months ago',
          riskScore: 12,
          groups: ['Developers', 'Project-Alpha', 'Code-Review'],
          permissions: 48,
          phone: '+1-555-0102',
          location: 'Austin, TX',
          manager: 'Jane Smith',
          passwordExpiry: '23 days',
          loginAttempts: 0,
          deviceCount: 4,
          sessionCount: 1,
          complianceScore: 89,
          lastPasswordChange: '67 days ago',
          accountAge: 183
        },
        {
          id: 'usr-003',
          email: 'jane.smith@company.com',
          name: 'Jane Smith',
          department: 'Security',
          role: 'Security Analyst',
          status: 'active',
          mfaEnabled: true,
          lastLogin: '5 minutes ago',
          createdAt: '8 months ago',
          riskScore: 8,
          groups: ['Security', 'Compliance', 'Audit', 'IR-Team'],
          permissions: 89,
          phone: '+1-555-0103',
          location: 'New York, NY',
          manager: 'CISO Office',
          passwordExpiry: '67 days',
          loginAttempts: 0,
          deviceCount: 2,
          sessionCount: 3,
          complianceScore: 96,
          lastPasswordChange: '23 days ago',
          accountAge: 245
        },
        {
          id: 'usr-004',
          email: 'bob.wilson@company.com',
          name: 'Bob Wilson',
          department: 'Finance',
          role: 'Financial Analyst',
          status: 'suspended',
          mfaEnabled: false,
          lastLogin: '1 week ago',
          createdAt: '3 months ago',
          riskScore: 75,
          groups: ['Finance', 'Reporting'],
          permissions: 12,
          phone: '+1-555-0104',
          location: 'Chicago, IL',
          manager: 'CFO Office',
          passwordExpiry: '2 days',
          loginAttempts: 5,
          deviceCount: 1,
          sessionCount: 0,
          complianceScore: 34,
          lastPasswordChange: '88 days ago',
          accountAge: 92
        },
        {
          id: 'usr-005',
          email: 'alice.johnson@company.com',
          name: 'Alice Johnson',
          department: 'HR',
          role: 'HR Manager',
          status: 'active',
          mfaEnabled: false,
          lastLogin: '3 days ago',
          createdAt: '1 year ago',
          riskScore: 35,
          groups: ['HR', 'Management', 'Payroll'],
          permissions: 34,
          phone: '+1-555-0105',
          location: 'Seattle, WA',
          manager: 'VP People',
          passwordExpiry: '12 days',
          loginAttempts: 1,
          deviceCount: 2,
          sessionCount: 1,
          complianceScore: 67,
          lastPasswordChange: '78 days ago',
          accountAge: 365
        },
        {
          id: 'usr-006',
          email: 'mike.chen@company.com',
          name: 'Mike Chen',
          department: 'Engineering',
          role: 'DevOps Engineer',
          status: 'locked',
          mfaEnabled: true,
          lastLogin: '2 days ago',
          createdAt: '4 months ago',
          riskScore: 85,
          groups: ['DevOps', 'Infrastructure'],
          permissions: 67,
          phone: '+1-555-0106',
          location: 'Denver, CO',
          manager: 'Engineering Lead',
          passwordExpiry: 'Expired',
          loginAttempts: 8,
          deviceCount: 3,
          sessionCount: 0,
          complianceScore: 23,
          lastPasswordChange: '95 days ago',
          accountAge: 123
        }
      ])

      setProviders([
        {
          id: 'idp-001',
          name: 'Azure Active Directory',
          type: 'Identity Provider',
          status: 'connected',
          users: 1850,
          lastSync: '5 minutes ago',
          domain: 'company.onmicrosoft.com',
          protocol: 'SAML 2.0',
          healthScore: 98,
          errorCount: 2,
          syncFrequency: 'Every 15 minutes'
        },
        {
          id: 'idp-002',
          name: 'Okta',
          type: 'Identity Provider',
          status: 'connected',
          users: 650,
          lastSync: '1 hour ago',
          domain: 'company.okta.com',
          protocol: 'OAuth 2.0',
          healthScore: 95,
          errorCount: 0,
          syncFrequency: 'Every 30 minutes'
        },
        {
          id: 'idp-003',
          name: 'Google Workspace',
          type: 'Identity Provider',
          status: 'connected',
          users: 347,
          lastSync: '20 minutes ago',
          domain: 'company.google.com',
          protocol: 'OAuth 2.0',
          healthScore: 92,
          errorCount: 1,
          syncFrequency: 'Every 1 hour'
        },
        {
          id: 'idp-004',
          name: 'On-Premise AD',
          type: 'Directory Service',
          status: 'error',
          users: 0,
          lastSync: '3 days ago',
          domain: 'internal.company.local',
          protocol: 'LDAP',
          healthScore: 12,
          errorCount: 45,
          syncFrequency: 'Every 6 hours'
        },
        {
          id: 'idp-005',
          name: 'GitHub Enterprise',
          type: 'Code Repository',
          status: 'connected',
          users: 234,
          lastSync: '10 minutes ago',
          domain: 'github.company.com',
          protocol: 'OAuth 2.0',
          healthScore: 88,
          errorCount: 3,
          syncFrequency: 'Every 2 hours'
        }
      ])

      setAuthEvents([
        { id: 'e1', user: 'john.doe@company.com', event: 'Login Success', timestamp: new Date(Date.now() - 180000), status: 'success', ipAddress: '192.168.1.100', userAgent: 'Chrome 120.0', location: 'Austin, TX', riskLevel: 'low', mfaUsed: true },
        { id: 'e2', user: 'jane.smith@company.com', event: 'MFA Challenge', timestamp: new Date(Date.now() - 300000), status: 'success', ipAddress: '192.168.1.101', userAgent: 'Safari 17.0', location: 'New York, NY', riskLevel: 'low', mfaUsed: true },
        { id: 'e3', user: 'unknown@external.com', event: 'Failed Login', timestamp: new Date(Date.now() - 450000), status: 'blocked', ipAddress: '203.0.113.42', userAgent: 'Bot/1.0', location: 'Unknown', riskLevel: 'high', mfaUsed: false },
        { id: 'e4', user: 'bob.wilson@company.com', event: 'Account Locked', timestamp: new Date(Date.now() - 600000), status: 'failure', ipAddress: '192.168.1.102', userAgent: 'Firefox 119.0', location: 'Chicago, IL', riskLevel: 'high', mfaUsed: false },
        { id: 'e5', user: 'alice.johnson@company.com', event: 'Password Reset', timestamp: new Date(Date.now() - 900000), status: 'success', ipAddress: '192.168.1.103', userAgent: 'Edge 119.0', location: 'Seattle, WA', riskLevel: 'medium', mfaUsed: false },
        { id: 'e6', user: 'admin@company.com', event: 'Privileged Access', timestamp: new Date(Date.now() - 1200000), status: 'success', ipAddress: '192.168.1.104', userAgent: 'Chrome 120.0', location: 'San Francisco, CA', riskLevel: 'medium', mfaUsed: true }
      ])

      setRealTimeData(generateRealTimeData())
      setLoading(false)
    }, 1000)
  }

  const loadRealTimeData = () => {
    setRealTimeData(prev => {
      const newData = [...prev, {
        timestamp: new Date(),
        logins: Math.floor(Math.random() * 50) + 10,
        failures: Math.floor(Math.random() * 10),
        mfaAuth: Math.floor(Math.random() * 30) + 5
      }]
      return newData.slice(-20)
    })
  }

  const generateRealTimeData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 60000),
      logins: Math.floor(Math.random() * 50) + 10,
      failures: Math.floor(Math.random() * 10),
      mfaAuth: Math.floor(Math.random() * 30) + 5
    }))
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': case 'connected': case 'success': return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'inactive': case 'disconnected': return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
      case 'suspended': case 'error': case 'failure': case 'blocked': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'pending': case 'syncing': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'locked': return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  const getRiskColor = (score: number) => {
    if (score < 20) return 'text-green-400'
    if (score < 50) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getRiskLevel = (score: number) => {
    if (score < 20) return 'Low'
    if (score < 50) return 'Medium'
    return 'High'
  }

  const filteredUsers = users.filter(user =>
    user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.department.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.role.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const authTrendData = {
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
        label: 'MFA Authentications',
        data: realTimeData.map(d => d.mfaAuth),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4
      }
    ]
  }

  const userStatusData = {
    labels: ['Active', 'Inactive', 'Suspended', 'Locked'],
    datasets: [{
      data: [
        metrics?.activeUsers || 0,
        metrics?.inactiveUsers || 0,
        metrics?.suspendedUsers || 0,
        users.filter(u => u.status === 'locked').length
      ],
      backgroundColor: [
        'rgba(34, 197, 94, 0.8)',
        'rgba(156, 163, 175, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(251, 146, 60, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const riskDistributionData = {
    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
    datasets: [{
      data: [
        users.filter(u => u.riskScore < 20).length,
        users.filter(u => u.riskScore >= 20 && u.riskScore < 50).length,
        users.filter(u => u.riskScore >= 50).length
      ],
      backgroundColor: [
        'rgba(34, 197, 94, 0.8)',
        'rgba(251, 191, 36, 0.8)',
        'rgba(239, 68, 68, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading Identity Management Center...</p>
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
              <Users className="w-8 h-8 text-blue-500" />
              <div>
                <h1 className="text-2xl font-bold">Identity Management Center</h1>
                <p className="text-sm text-gray-500">User lifecycle and authentication management</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">IDENTITY SERVICES ONLINE</span>
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
              <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors flex items-center space-x-2">
                <UserPlus className="w-4 h-4" />
                <span>Add User</span>
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'users', 'providers', 'authentication', 'lifecycle', 'analytics'].map((tab) => (
            <button
              key={tab}
              onClick={() => setSelectedTab(tab as any)}
              className={`py-3 px-1 border-b-2 transition-colors capitalize ${selectedTab === tab
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
                  <span className="text-xs text-gray-500">Total Users</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.totalUsers}</p>
                <p className="text-xs text-gray-500 mt-1">{metrics.newUsers30d} new this month</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <UserCheck className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Active</span>
                </div>
                <p className="text-2xl font-bold font-mono text-green-500">{metrics.activeUsers}</p>
                <p className="text-xs text-gray-500 mt-1">{Math.round((metrics.activeUsers / metrics.totalUsers) * 100)}% of total</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Smartphone className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">MFA Enabled</span>
                </div>
                <p className="text-2xl font-bold font-mono">{Math.round((metrics.mfaEnabledUsers / metrics.totalUsers) * 100)}%</p>
                <div className="mt-2 h-1 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500 rounded-full" style={{ width: `${(metrics.mfaEnabledUsers / metrics.totalUsers) * 100}%` }} />
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Globe className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">Providers</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.connectedProviders}/{metrics.totalProviders}</p>
                <p className="text-xs text-gray-500 mt-1">{Math.round((metrics.connectedProviders / metrics.totalProviders) * 100)}% healthy</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">High Risk</span>
                </div>
                <p className="text-2xl font-bold font-mono text-yellow-500">{metrics.highRiskUsers}</p>
                <p className="text-xs text-gray-500 mt-1">{Math.round((metrics.highRiskUsers / metrics.totalUsers) * 100)}% of users</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Clock className="w-5 h-5 text-red-500" />
                  <span className="text-xs text-gray-500">Password Expiry</span>
                </div>
                <p className="text-2xl font-bold font-mono text-red-500">{metrics.passwordExpiring}</p>
                <p className="text-xs text-gray-500 mt-1">expiring soon</p>
              </motion.div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {/* Authentication Trends */}
              <div className="col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">AUTHENTICATION ACTIVITY</h3>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      <span className="text-xs text-gray-500">Success</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-red-500 rounded-full" />
                      <span className="text-xs text-gray-500">Failed</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full" />
                      <span className="text-xs text-gray-500">MFA</span>
                    </div>
                  </div>
                </div>
                <div className="h-64">
                  <Line data={authTrendData} options={{
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

              {/* User Status Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">USER STATUS</h3>
                <div className="h-64 flex items-center justify-center">
                  <Doughnut data={userStatusData} options={{
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

            {/* Detailed Stats Grid */}
            <div className="grid grid-cols-4 gap-6 mb-6">
              {/* Risk Assessment */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">RISK ASSESSMENT</h3>
                </div>
                <div className="p-4">
                  <div className="flex items-center space-x-4 mb-4">
                    <div className="relative w-16 h-16">
                      <svg className="w-full h-full transform -rotate-90">
                        <circle cx="32" cy="32" r="28" stroke="rgba(255,255,255,0.1)" strokeWidth="4" fill="none" />
                        <circle
                          cx="32" cy="32" r="28"
                          stroke={metrics.avgRiskScore > 50 ? 'rgb(239, 68, 68)' : metrics.avgRiskScore > 25 ? 'rgb(251, 191, 36)' : 'rgb(34, 197, 94)'}
                          strokeWidth="4"
                          fill="none"
                          strokeDasharray={`${2 * Math.PI * 28}`}
                          strokeDashoffset={`${2 * Math.PI * 28 * (1 - metrics.avgRiskScore / 100)}`}
                          className="transition-all duration-1000"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-lg font-bold">{metrics.avgRiskScore}%</span>
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Average Risk Score</p>
                      <p className={`text-sm font-medium ${getRiskColor(metrics.avgRiskScore)}`}>
                        {getRiskLevel(metrics.avgRiskScore)} Risk
                      </p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Low Risk</span>
                      <span className="text-green-400">{users.filter(u => u.riskScore < 20).length}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Medium Risk</span>
                      <span className="text-yellow-400">{users.filter(u => u.riskScore >= 20 && u.riskScore < 50).length}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">High Risk</span>
                      <span className="text-red-400">{users.filter(u => u.riskScore >= 50).length}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Authentication Stats */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">AUTHENTICATION (24H)</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Successful Logins</span>
                    <span className="font-mono text-green-500">{metrics.successfulLogins24h}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Failed Attempts</span>
                    <span className="font-mono text-red-500">{metrics.failedLogins24h}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Success Rate</span>
                    <span className="font-mono text-blue-400">
                      {Math.round((metrics.successfulLogins24h / (metrics.successfulLogins24h + metrics.failedLogins24h)) * 100)}%
                    </span>
                  </div>
                  <div className="pt-2 border-t border-gray-800">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-400">MFA Coverage</span>
                      <span className="text-sm text-blue-500">{Math.round((metrics.mfaEnabledUsers / metrics.totalUsers) * 100)}%</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full">
                      <div className="h-2 bg-blue-500 rounded-full" style={{ width: `${(metrics.mfaEnabledUsers / metrics.totalUsers) * 100}%` }} />
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
                  <div className="flex items-center space-x-4 mb-4">
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
                      <p className="text-xs text-gray-500">Identity Compliance</p>
                      <p className="text-sm font-medium text-green-500">Excellent</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Password Policy</span>
                      <span className="text-green-400">98%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">MFA Enforcement</span>
                      <span className="text-yellow-400">{Math.round((metrics.mfaEnabledUsers / metrics.totalUsers) * 100)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Access Reviews</span>
                      <span className="text-green-400">96%</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">QUICK ACTIONS</h3>
                </div>
                <div className="p-4 space-y-2">
                  <button className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm text-left flex items-center justify-between group">
                    <span>Bulk Password Reset</span>
                    <RotateCcw className="w-4 h-4 text-blue-300 group-hover:text-white" />
                  </button>
                  <button className="w-full px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm text-left flex items-center justify-between group">
                    <span>MFA Enforcement</span>
                    <Smartphone className="w-4 h-4 text-purple-300 group-hover:text-white" />
                  </button>
                  <button className="w-full px-3 py-2 bg-yellow-600 hover:bg-yellow-700 rounded text-sm text-left flex items-center justify-between group">
                    <span>Risk Assessment</span>
                    <AlertTriangle className="w-4 h-4 text-yellow-300 group-hover:text-white" />
                  </button>
                  <button className="w-full px-3 py-2 bg-green-600 hover:bg-green-700 rounded text-sm text-left flex items-center justify-between group">
                    <span>Compliance Report</span>
                    <Download className="w-4 h-4 text-green-300 group-hover:text-white" />
                  </button>
                </div>
              </div>
            </div>
          </>
        )}

        {selectedTab === 'users' && (
          <>
            {/* Search and Filters */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <input
                    type="text"
                    placeholder="Search users..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <select className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                  <option value="">All Status</option>
                  <option value="active">Active</option>
                  <option value="inactive">Inactive</option>
                  <option value="suspended">Suspended</option>
                  <option value="locked">Locked</option>
                </select>
                <select className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                  <option value="">All Departments</option>
                  <option value="engineering">Engineering</option>
                  <option value="security">Security</option>
                  <option value="finance">Finance</option>
                  <option value="hr">HR</option>
                </select>
                <button className="p-2 bg-gray-800 border border-gray-700 rounded-lg hover:bg-gray-700">
                  <Filter className="w-4 h-4 text-gray-400" />
                </button>
              </div>
              <div className="flex items-center space-x-2">
                <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg flex items-center space-x-2">
                  <Download className="w-4 h-4" />
                  <span>Export</span>
                </button>
                <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center space-x-2">
                  <UserPlus className="w-4 h-4" />
                  <span>Add User</span>
                </button>
              </div>
            </div>

            {/* Users Grid */}
            <div className="space-y-4">
              {filteredUsers.map((user, index) => (
                <motion.div
                  key={user.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-gray-900 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors"
                >
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-start gap-4">
                        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
                          <span className="text-white font-semibold">
                            {user.name.split(' ').map(n => n[0]).join('')}
                          </span>
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold text-white">{user.name}</h3>
                          <p className="text-sm text-gray-400">{user.email}</p>
                          <p className="text-sm text-gray-500 mt-1">
                            {user.department} • {user.role}
                          </p>
                          {user.location && (
                            <p className="text-xs text-gray-600 flex items-center gap-1 mt-1">
                              <MapPin className="w-3 h-3" />
                              {user.location}
                            </p>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(user.status)}`}>
                          {user.status.toUpperCase()}
                        </span>
                        {user.mfaEnabled && (
                          <span className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs rounded border border-blue-500/30">
                            MFA
                          </span>
                        )}
                        <span className={`px-2 py-1 text-xs rounded border ${user.riskScore < 20 ? 'bg-green-500/20 text-green-400 border-green-500/30' : 
                          user.riskScore < 50 ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' : 
                          'bg-red-500/20 text-red-400 border-red-500/30'}`}>
                          {getRiskLevel(user.riskScore)} Risk
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-6 gap-4 mb-4">
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Last Login</p>
                        <p className="text-sm text-white">{user.lastLogin}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Risk Score</p>
                        <p className={`text-sm font-semibold ${getRiskColor(user.riskScore)}`}>
                          {user.riskScore}%
                        </p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Sessions</p>
                        <p className="text-sm text-white">{user.sessionCount}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Devices</p>
                        <p className="text-sm text-white">{user.deviceCount}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Compliance</p>
                        <p className={`text-sm font-semibold ${user.complianceScore > 80 ? 'text-green-400' : user.complianceScore > 60 ? 'text-yellow-400' : 'text-red-400'}`}>
                          {user.complianceScore}%
                        </p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Password</p>
                        <p className={`text-sm ${user.passwordExpiry === 'Expired' ? 'text-red-400' : user.passwordExpiry.includes('days') && parseInt(user.passwordExpiry) < 7 ? 'text-orange-400' : 'text-white'}`}>
                          {user.passwordExpiry}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-4 border-t border-gray-700">
                      <div className="flex flex-wrap gap-2">
                        {user.groups.slice(0, 3).map((group) => (
                          <span key={group} className="px-2 py-1 bg-purple-500/20 text-purple-400 text-xs rounded border border-purple-500/30">
                            {group}
                          </span>
                        ))}
                        {user.groups.length > 3 && (
                          <span className="px-2 py-1 bg-gray-500/20 text-gray-400 text-xs rounded">
                            +{user.groups.length - 3} more
                          </span>
                        )}
                      </div>
                      <div className="flex gap-2">
                        <button 
                          onClick={() => setSelectedUser(user)}
                          className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white">
                          <Edit className="w-4 h-4" />
                        </button>
                        {user.status === 'suspended' || user.status === 'locked' ? (
                          <button className="p-1.5 hover:bg-green-700 rounded text-green-400">
                            <UserCheck className="w-4 h-4" />
                          </button>
                        ) : (
                          <button className="p-1.5 hover:bg-red-700 rounded text-red-400">
                            <UserX className="w-4 h-4" />
                          </button>
                        )}
                        <button className="p-1.5 hover:bg-gray-700 rounded">
                          <MoreVertical className="w-4 h-4 text-gray-400" />
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        )}

        {selectedTab === 'providers' && (
          <>
            {/* Provider Stats */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <Globe className="w-5 h-5 text-blue-500" />
                  <span className="text-2xl font-bold text-white">{providers.length}</span>
                </div>
                <p className="text-gray-400 text-sm">Total Providers</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-2xl font-bold text-green-500">{providers.filter(p => p.status === 'connected').length}</span>
                </div>
                <p className="text-gray-400 text-sm">Connected</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-red-500" />
                  <span className="text-2xl font-bold text-red-500">{providers.filter(p => p.status === 'error').length}</span>
                </div>
                <p className="text-gray-400 text-sm">Errors</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <Users className="w-5 h-5 text-purple-500" />
                  <span className="text-2xl font-bold text-white">{providers.reduce((sum, p) => sum + p.users, 0)}</span>
                </div>
                <p className="text-gray-400 text-sm">Total Users</p>
              </div>
            </div>

            {/* Providers Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {providers.map((provider, index) => (
                <motion.div
                  key={provider.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-gray-900 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors"
                >
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-white">{provider.name}</h3>
                        <p className="text-sm text-gray-400 mt-1">{provider.type}</p>
                        <p className="text-xs text-gray-500 mt-1">{provider.protocol}</p>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(provider.status)}`}>
                        {provider.status.toUpperCase()}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400">Users</p>
                        <p className="text-xl font-semibold text-white">{provider.users}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400">Health Score</p>
                        <p className={`text-xl font-semibold ${provider.healthScore > 90 ? 'text-green-400' : 
                          provider.healthScore > 70 ? 'text-yellow-400' : 'text-red-400'}`}>
                          {provider.healthScore}%
                        </p>
                      </div>
                    </div>

                    <div className="space-y-2 mb-4">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Domain</span>
                        <span className="text-white font-mono text-xs">{provider.domain}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Last Sync</span>
                        <span className="text-white">{provider.lastSync}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Sync Frequency</span>
                        <span className="text-white">{provider.syncFrequency}</span>
                      </div>
                      {provider.errorCount > 0 && (
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Errors</span>
                          <span className="text-red-400">{provider.errorCount}</span>
                        </div>
                      )}
                    </div>

                    <div className="flex gap-2">
                      <button className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm flex items-center justify-center space-x-1">
                        <RefreshCw className="w-3 h-3" />
                        <span>Sync</span>
                      </button>
                      <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-white text-sm">
                        <Settings className="w-3 h-3" />
                      </button>
                      <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-white text-sm">
                        <MoreVertical className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        )}

        {selectedTab === 'authentication' && (
          <>
            {/* Authentication Overview */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-2xl font-bold text-green-500">{metrics?.successfulLogins24h || 0}</span>
                </div>
                <p className="text-gray-400 text-sm">Successful Logins</p>
                <p className="text-xs text-gray-600 mt-1">Last 24 hours</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <X className="w-5 h-5 text-red-500" />
                  <span className="text-2xl font-bold text-red-500">{metrics?.failedLogins24h || 0}</span>
                </div>
                <p className="text-gray-400 text-sm">Failed Attempts</p>
                <p className="text-xs text-gray-600 mt-1">Last 24 hours</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <Smartphone className="w-5 h-5 text-blue-500" />
                  <span className="text-2xl font-bold text-blue-500">{authEvents.filter(e => e.mfaUsed).length}</span>
                </div>
                <p className="text-gray-400 text-sm">MFA Authentications</p>
                <p className="text-xs text-gray-600 mt-1">Recent events</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  <span className="text-2xl font-bold text-yellow-500">{authEvents.filter(e => e.riskLevel === 'high').length}</span>
                </div>
                <p className="text-gray-400 text-sm">High Risk Events</p>
                <p className="text-xs text-gray-600 mt-1">Requires attention</p>
              </div>
            </div>

            {/* Authentication Events Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                <h3 className="text-sm font-bold text-gray-400 uppercase">RECENT AUTHENTICATION EVENTS</h3>
                <div className="flex items-center space-x-2">
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    <span className="text-xs text-gray-500">Live</span>
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
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Event</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Location</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Risk</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">MFA</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">IP Address</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {authEvents.map((event) => (
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
                            {event.event}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 text-xs rounded border ${getStatusColor(event.status)}`}>
                            {event.status.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm text-white flex items-center space-x-1">
                            <MapPin className="w-3 h-3 text-gray-500" />
                            <span>{event.location}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`text-xs font-medium ${event.riskLevel === 'high' ? 'text-red-400' : 
                            event.riskLevel === 'medium' ? 'text-yellow-400' : 'text-green-400'}`}>
                            {event.riskLevel.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          {event.mfaUsed ? (
                            <CheckCircle className="w-4 h-4 text-green-400" />
                          ) : (
                            <X className="w-4 h-4 text-gray-500" />
                          )}
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

        {selectedTab === 'lifecycle' && (
          <>
            {/* User Lifecycle Overview */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 mb-6">
              <h2 className="text-xl font-bold mb-4">User Lifecycle Management</h2>
              <div className="grid grid-cols-5 gap-4">
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-400 mb-1">Onboarding</p>
                  <p className="text-2xl font-bold text-blue-500">12</p>
                  <p className="text-xs text-gray-500">in progress</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-400 mb-1">Provisioned</p>
                  <p className="text-2xl font-bold text-green-500">89</p>
                  <p className="text-xs text-gray-500">this month</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-400 mb-1">Role Changes</p>
                  <p className="text-2xl font-bold text-yellow-500">34</p>
                  <p className="text-xs text-gray-500">pending approval</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-400 mb-1">Offboarding</p>
                  <p className="text-2xl font-bold text-orange-500">7</p>
                  <p className="text-xs text-gray-500">in progress</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-400 mb-1">Deprovisioned</p>
                  <p className="text-2xl font-bold text-gray-400">23</p>
                  <p className="text-xs text-gray-500">this month</p>
                </div>
              </div>
            </div>

            {/* Lifecycle Workflows */}
            <div className="space-y-6">
              {[
                { stage: 'Onboarding', users: 12, color: 'blue', description: 'New employees being set up with accounts and access' },
                { stage: 'Active Management', users: 2156, color: 'green', description: 'Regular access reviews and permission updates' },
                { stage: 'Role Transitions', users: 34, color: 'yellow', description: 'Users changing roles or departments' },
                { stage: 'Offboarding', users: 7, color: 'orange', description: 'Employees leaving, access being revoked' },
                { stage: 'Archived', users: 245, color: 'gray', description: 'Former employees with deprovisioned accounts' }
              ].map((workflow, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-gray-900 border border-gray-800 rounded-lg"
                >
                  <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold text-white">{workflow.stage}</h3>
                      <p className="text-sm text-gray-400 mt-1">{workflow.description}</p>
                    </div>
                    <div className="text-right">
                      <p className={`text-2xl font-bold text-${workflow.color}-500`}>{workflow.users}</p>
                      <p className="text-xs text-gray-500">users</p>
                    </div>
                  </div>
                  <div className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex space-x-2">
                        <button className={`px-3 py-1 bg-${workflow.color}-600 hover:bg-${workflow.color}-700 rounded text-white text-sm`}>
                          Manage
                        </button>
                        <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-white text-sm">
                          View All
                        </button>
                      </div>
                      <span className={`text-xs px-2 py-1 bg-${workflow.color}-500/20 text-${workflow.color}-400 rounded`}>
                        {workflow.stage === 'Active Management' ? 'Ongoing' : 'In Progress'}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        )}

        {selectedTab === 'analytics' && (
          <>
            {/* Analytics Overview */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              {/* Risk Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">RISK DISTRIBUTION</h3>
                <div className="h-64 flex items-center justify-center">
                  <Doughnut data={riskDistributionData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: { 
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12 }
                        }
                      }
                    }
                  }} />
                </div>
              </div>

              {/* Authentication Trends */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">AUTHENTICATION TRENDS</h3>
                <div className="h-64">
                  <Line data={authTrendData} options={{
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
            </div>

            {/* Detailed Analytics */}
            <div className="grid grid-cols-3 gap-6">
              {/* Top Risk Users */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">HIGHEST RISK USERS</h3>
                </div>
                <div className="p-4 space-y-3">
                  {users.filter(u => u.riskScore > 50).slice(0, 5).map((user, index) => (
                    <div key={user.id} className="flex items-center justify-between p-2 bg-gray-800 rounded">
                      <div>
                        <p className="text-sm font-medium text-white">{user.name}</p>
                        <p className="text-xs text-gray-400">{user.department}</p>
                      </div>
                      <span className="text-sm font-bold text-red-400">{user.riskScore}%</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Department Breakdown */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">USERS BY DEPARTMENT</h3>
                </div>
                <div className="p-4 space-y-3">
                  {Object.entries(
                    users.reduce((acc: { [key: string]: number }, user) => {
                      acc[user.department] = (acc[user.department] || 0) + 1
                      return acc
                    }, {})
                  ).map(([dept, count]) => (
                    <div key={dept} className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">{dept}</span>
                      <span className="text-sm font-mono text-white">{count}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recent Activity */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">RECENT ACTIVITY</h3>
                </div>
                <div className="p-4 space-y-3">
                  {authEvents.slice(0, 5).map((event, index) => (
                    <div key={event.id} className="flex items-start justify-between p-2 bg-gray-800 rounded">
                      <div>
                        <p className="text-xs font-medium text-white">{event.event}</p>
                        <p className="text-xs text-gray-400">{event.user}</p>
                        <p className="text-xs text-gray-500">{event.timestamp.toLocaleTimeString()}</p>
                      </div>
                      <span className={`px-2 py-1 text-xs rounded border ${getStatusColor(event.status)}`}>
                        {event.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}