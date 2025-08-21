'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import {
  Users, Shield, Lock, Key, UserCheck, UserX,
  ShieldCheck, ShieldAlert, AlertTriangle, CheckCircle,
  Activity, Clock, TrendingUp, Settings, RefreshCw,
  Download, Search, Filter, Mail, Phone, Building,
  Globe, Laptop, Smartphone, ArrowRight, ChevronRight,
  Eye, EyeOff, UserPlus, LogIn, LogOut, AlertCircle
} from 'lucide-react'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar, AreaChart, Area
} from 'recharts'

// TypeScript Types
interface User {
  id: string
  displayName: string
  userPrincipalName: string
  email: string
  department: string
  jobTitle: string
  accountEnabled: boolean
  createdDateTime: string
  lastSignIn: string
  mfaStatus: 'enabled' | 'disabled' | 'enforced'
  riskLevel: 'none' | 'low' | 'medium' | 'high'
  licenses: string[]
  groups: Group[]
  applications: Application[]
}

interface Group {
  id: string
  displayName: string
  description: string
  memberCount: number
  groupType: 'security' | 'office365' | 'distribution'
  dynamicMembership: boolean
  owners: string[]
}

interface Application {
  id: string
  displayName: string
  appId: string
  publisherName: string
  permissions: string[]
  consentType: 'admin' | 'user'
  riskScore: number
}

interface ConditionalAccessPolicy {
  id: string
  displayName: string
  state: 'enabled' | 'disabled' | 'report'
  conditions: {
    users: string[]
    applications: string[]
    locations: string[]
    platforms: string[]
  }
  grantControls: {
    requireMfa: boolean
    requireCompliantDevice: boolean
    requireApprovedApp: boolean
    blockAccess: boolean
  }
  sessionControls: {
    signInFrequency: string
    persistentBrowser: boolean
  }
}

interface IdentityRisk {
  id: string
  userId: string
  userDisplayName: string
  riskEventType: string
  riskLevel: 'low' | 'medium' | 'high'
  riskState: 'atRisk' | 'confirmedSafe' | 'remediated' | 'dismissed'
  riskDetail: string
  detectedDateTime: string
  activity: string
  location: string
  ipAddress: string
}

interface SignInLog {
  id: string
  userPrincipalName: string
  appDisplayName: string
  ipAddress: string
  location: string
  deviceDetail: {
    browser: string
    operatingSystem: string
    deviceId: string
  }
  status: {
    errorCode: number
    failureReason: string | null
  }
  createdDateTime: string
  conditionalAccessStatus: 'success' | 'failure' | 'notApplied'
  mfaDetail: {
    authMethod: string | null
    authDetail: string | null
  }
}

interface ServicePrincipal {
  id: string
  displayName: string
  appId: string
  servicePrincipalType: 'application' | 'managedIdentity'
  appOwnerOrganizationId: string
  permissions: {
    application: string[]
    delegated: string[]
  }
  certificateExpiry: string | null
  secretExpiry: string | null
  lastUsed: string
}

export default function IAMPage() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterRisk, setFilterRisk] = useState<string>('all')
  const [loading, setLoading] = useState(false)

  // Mock metrics
  const metrics = {
    totalUsers: 2847,
    activeUsers: 2543,
    guestUsers: 156,
    mfaEnabled: 89,
    conditionalAccessPolicies: 24,
    riskySignIns: 47,
    servicePrincipals: 68,
    applications: 234
  }

  // MFA enrollment trend
  const mfaTrend = [
    { month: 'Jan', enrolled: 75, target: 95 },
    { month: 'Feb', enrolled: 78, target: 95 },
    { month: 'Mar', enrolled: 82, target: 95 },
    { month: 'Apr', enrolled: 85, target: 95 },
    { month: 'May', enrolled: 87, target: 95 },
    { month: 'Jun', enrolled: 89, target: 95 }
  ]

  // User distribution by department
  const departmentDistribution = [
    { name: 'Engineering', value: 845, color: '#3b82f6' },
    { name: 'Sales', value: 523, color: '#10b981' },
    { name: 'Marketing', value: 312, color: '#f59e0b' },
    { name: 'Finance', value: 234, color: '#8b5cf6' },
    { name: 'HR', value: 156, color: '#ef4444' },
    { name: 'Other', value: 777, color: '#6b7280' }
  ]

  // Risk detection data
  const riskDetections: IdentityRisk[] = [
    {
      id: '1',
      userId: 'user1',
      userDisplayName: 'John Doe',
      riskEventType: 'Impossible travel',
      riskLevel: 'high',
      riskState: 'atRisk',
      riskDetail: 'User signed in from two distant locations within 1 hour',
      detectedDateTime: '2024-03-01T10:30:00Z',
      activity: 'Sign-in',
      location: 'New York, US → Tokyo, JP',
      ipAddress: '192.168.1.1'
    },
    {
      id: '2',
      userId: 'user2',
      userDisplayName: 'Jane Smith',
      riskEventType: 'Leaked credentials',
      riskLevel: 'high',
      riskState: 'atRisk',
      riskDetail: 'User credentials found in dark web dump',
      detectedDateTime: '2024-03-01T09:15:00Z',
      activity: 'Credential leak',
      location: 'N/A',
      ipAddress: 'N/A'
    },
    {
      id: '3',
      userId: 'user3',
      userDisplayName: 'Bob Wilson',
      riskEventType: 'Suspicious IP',
      riskLevel: 'medium',
      riskState: 'atRisk',
      riskDetail: 'Sign-in from known malicious IP address',
      detectedDateTime: '2024-02-29T22:45:00Z',
      activity: 'Sign-in',
      location: 'Unknown',
      ipAddress: '10.0.0.1'
    }
  ]

  // Conditional Access policies
  const conditionalAccessPolicies: ConditionalAccessPolicy[] = [
    {
      id: '1',
      displayName: 'Require MFA for all users',
      state: 'enabled',
      conditions: {
        users: ['All users'],
        applications: ['All cloud apps'],
        locations: ['All locations'],
        platforms: ['All platforms']
      },
      grantControls: {
        requireMfa: true,
        requireCompliantDevice: false,
        requireApprovedApp: false,
        blockAccess: false
      },
      sessionControls: {
        signInFrequency: '90 days',
        persistentBrowser: true
      }
    },
    {
      id: '2',
      displayName: 'Block legacy authentication',
      state: 'enabled',
      conditions: {
        users: ['All users'],
        applications: ['All cloud apps'],
        locations: ['All locations'],
        platforms: ['Other clients']
      },
      grantControls: {
        requireMfa: false,
        requireCompliantDevice: false,
        requireApprovedApp: false,
        blockAccess: true
      },
      sessionControls: {
        signInFrequency: 'N/A',
        persistentBrowser: false
      }
    },
    {
      id: '3',
      displayName: 'Require compliant device for sensitive apps',
      state: 'enabled',
      conditions: {
        users: ['All users'],
        applications: ['Office 365', 'Azure Portal'],
        locations: ['All locations'],
        platforms: ['iOS', 'Android', 'Windows']
      },
      grantControls: {
        requireMfa: true,
        requireCompliantDevice: true,
        requireApprovedApp: true,
        blockAccess: false
      },
      sessionControls: {
        signInFrequency: '1 day',
        persistentBrowser: false
      }
    }
  ]

  // Guest user activity
  const guestActivity = [
    { day: 'Mon', internal: 2345, guest: 156 },
    { day: 'Tue', internal: 2456, guest: 178 },
    { day: 'Wed', internal: 2398, guest: 165 },
    { day: 'Thu', internal: 2487, guest: 189 },
    { day: 'Fri', internal: 2234, guest: 145 },
    { day: 'Sat', internal: 987, guest: 45 },
    { day: 'Sun', internal: 765, guest: 34 }
  ]

  // Service principals
  const servicePrincipals: ServicePrincipal[] = [
    {
      id: '1',
      displayName: 'Production API',
      appId: 'app-001',
      servicePrincipalType: 'application',
      appOwnerOrganizationId: 'org-001',
      permissions: {
        application: ['User.Read.All', 'Directory.Read.All'],
        delegated: ['User.Read', 'Profile.Read']
      },
      certificateExpiry: '2024-12-31',
      secretExpiry: '2024-06-30',
      lastUsed: '2024-03-01'
    },
    {
      id: '2',
      displayName: 'Backup Service',
      appId: 'app-002',
      servicePrincipalType: 'managedIdentity',
      appOwnerOrganizationId: 'org-001',
      permissions: {
        application: ['Storage.Read.All'],
        delegated: []
      },
      certificateExpiry: null,
      secretExpiry: '2024-09-30',
      lastUsed: '2024-03-01'
    }
  ]

  // Sign-in success rate
  const signInSuccessRate = [
    { time: '00:00', success: 98, failure: 2 },
    { time: '04:00', success: 97, failure: 3 },
    { time: '08:00', success: 95, failure: 5 },
    { time: '12:00', success: 96, failure: 4 },
    { time: '16:00', success: 94, failure: 6 },
    { time: '20:00', success: 97, failure: 3 }
  ]

  const handleRiskRemediation = (riskId: string) => {
    alert(`Starting remediation for risk ${riskId}`)
  }

  const handleUserAction = (userId: string, action: string) => {
    alert(`Performing ${action} for user ${userId}`)
  }

  const exportReport = (type: string) => {
    alert(`Exporting ${type} report...`)
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Identity & Access Management</h1>
              <p className="text-sm text-gray-400 mt-1">
                Manage users, groups, applications, and security policies
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button type="button"
                onClick={() => exportReport('iam')}
                className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
              >
                <Download className="w-4 h-4" />
                Export
              </button>
              <button type="button"
                onClick={() => router.push('/security/iam/users/new')}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
              >
                <UserPlus className="w-4 h-4" />
                Add User
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-800 bg-gray-900/30">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-6 overflow-x-auto">
            {['overview', 'users', 'groups', 'applications', 'conditional-access', 'risks', 'sign-ins', 'service-principals'].map((tab) => (
              <button type="button"
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-3 border-b-2 transition-colors capitalize whitespace-nowrap ${
                  activeTab === tab
                    ? 'border-blue-500 text-white'
                    : 'border-transparent text-gray-400 hover:text-white'
                }`}
              >
                {tab.replace('-', ' ')}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm text-gray-400">Total Users</h3>
                  <Users className="w-5 h-5 text-blue-400" />
                </div>
                <p className="text-3xl font-bold">{metrics.totalUsers.toLocaleString()}</p>
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-sm text-green-400">{metrics.activeUsers} active</span>
                  <span className="text-sm text-gray-500">•</span>
                  <span className="text-sm text-yellow-400">{metrics.guestUsers} guests</span>
                </div>
              </div>

              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm text-gray-400">MFA Coverage</h3>
                  <ShieldCheck className="w-5 h-5 text-green-400" />
                </div>
                <p className="text-3xl font-bold">{metrics.mfaEnabled}%</p>
                <div className="mt-2">
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-green-500 rounded-full"
                      style={{ width: `${metrics.mfaEnabled}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-500 mt-1">Target: 95%</p>
                </div>
              </div>

              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm text-gray-400">Risky Sign-ins</h3>
                  <ShieldAlert className="w-5 h-5 text-red-400" />
                </div>
                <p className="text-3xl font-bold">{metrics.riskySignIns}</p>
                <p className="text-sm text-red-400 mt-2">Requires attention</p>
              </div>

              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm text-gray-400">CA Policies</h3>
                  <Shield className="w-5 h-5 text-purple-400" />
                </div>
                <p className="text-3xl font-bold">{metrics.conditionalAccessPolicies}</p>
                <p className="text-sm text-gray-500 mt-2">Active policies</p>
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-2 gap-6">
              {/* MFA Enrollment Trend */}
              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
                <h3 className="text-lg font-semibold mb-4">MFA Enrollment Progress</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={mfaTrend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="month" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="target"
                      stroke="#6b7280"
                      fill="none"
                      strokeDasharray="5 5"
                      name="Target"
                    />
                    <Area
                      type="monotone"
                      dataKey="enrolled"
                      stroke="#10b981"
                      fill="url(#colorEnrolled)"
                      name="Enrolled %"
                    />
                    <defs>
                      <linearGradient id="colorEnrolled" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0.1} />
                      </linearGradient>
                    </defs>
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* User Distribution */}
              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
                <h3 className="text-lg font-semibold mb-4">Users by Department</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={departmentDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {departmentDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex flex-wrap justify-center gap-3 mt-4">
                  {departmentDistribution.map((item) => (
                    <div key={item.name} className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-xs text-gray-400">
                        {item.name} ({item.value})
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Guest vs Internal Activity */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">User Activity (Internal vs Guest)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={guestActivity}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="day" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  />
                  <Legend />
                  <Bar dataKey="internal" fill="#3b82f6" name="Internal Users" />
                  <Bar dataKey="guest" fill="#f59e0b" name="Guest Users" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'risks' && (
          <div className="space-y-6">
            {/* Risk Detections */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Identity Risk Detections</h3>
                <span className="px-3 py-1 bg-red-900/50 text-red-400 rounded-full text-sm">
                  {riskDetections.filter(r => r.riskState === 'atRisk').length} active risks
                </span>
              </div>
              <div className="space-y-3">
                {riskDetections.map((risk) => (
                  <div key={risk.id} className="p-4 bg-gray-800/50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <AlertTriangle className={`w-5 h-5 ${
                          risk.riskLevel === 'high' ? 'text-red-400' :
                          risk.riskLevel === 'medium' ? 'text-yellow-400' :
                          'text-blue-400'
                        }`} />
                        <div>
                          <div className="flex items-center gap-3">
                            <p className="font-medium">{risk.userDisplayName}</p>
                            <span className={`px-2 py-1 rounded-full text-xs ${
                              risk.riskLevel === 'high' ? 'bg-red-900/50 text-red-400' :
                              risk.riskLevel === 'medium' ? 'bg-yellow-900/50 text-yellow-400' :
                              'bg-blue-900/50 text-blue-400'
                            }`}>
                              {risk.riskLevel} risk
                            </span>
                          </div>
                          <p className="text-sm text-gray-400 mt-1">
                            {risk.riskEventType}: {risk.riskDetail}
                          </p>
                          {risk.location !== 'N/A' && (
                            <p className="text-xs text-gray-500 mt-1">
                              Location: {risk.location} • IP: {risk.ipAddress}
                            </p>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-gray-500">
                          {new Date(risk.detectedDateTime).toLocaleString()}
                        </span>
                        <button type="button"
                          onClick={() => handleRiskRemediation(risk.id)}
                          className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                        >
                          Remediate
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Risk Summary */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-4">
                <h4 className="text-sm text-gray-400 mb-2">High Risk Users</h4>
                <p className="text-2xl font-bold text-red-400">12</p>
                <p className="text-xs text-gray-500 mt-1">Immediate action required</p>
              </div>
              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-4">
                <h4 className="text-sm text-gray-400 mb-2">Medium Risk Users</h4>
                <p className="text-2xl font-bold text-yellow-400">28</p>
                <p className="text-xs text-gray-500 mt-1">Monitor closely</p>
              </div>
              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-4">
                <h4 className="text-sm text-gray-400 mb-2">Remediated Today</h4>
                <p className="text-2xl font-bold text-green-400">7</p>
                <p className="text-xs text-gray-500 mt-1">Successfully resolved</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'conditional-access' && (
          <div className="space-y-6">
            {/* Conditional Access Policies */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Conditional Access Policies</h3>
                <button type="button" className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                  New Policy
                </button>
              </div>
              <div className="space-y-3">
                {conditionalAccessPolicies.map((policy) => (
                  <div key={policy.id} className="p-4 bg-gray-800/50 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <Shield className={`w-5 h-5 ${
                          policy.state === 'enabled' ? 'text-green-400' :
                          policy.state === 'disabled' ? 'text-gray-400' :
                          'text-yellow-400'
                        }`} />
                        <h4 className="font-medium">{policy.displayName}</h4>
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          policy.state === 'enabled' ? 'bg-green-900/50 text-green-400' :
                          policy.state === 'disabled' ? 'bg-gray-800 text-gray-400' :
                          'bg-yellow-900/50 text-yellow-400'
                        }`}>
                          {policy.state}
                        </span>
                      </div>
                      <button type="button" className="text-sm text-blue-400 hover:text-blue-300">
                        Edit
                      </button>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400 mb-1">Conditions:</p>
                        <ul className="text-xs space-y-1">
                          <li>• Users: {policy.conditions.users.join(', ')}</li>
                          <li>• Apps: {policy.conditions.applications.join(', ')}</li>
                          <li>• Platforms: {policy.conditions.platforms.join(', ')}</li>
                        </ul>
                      </div>
                      <div>
                        <p className="text-gray-400 mb-1">Grant Controls:</p>
                        <ul className="text-xs space-y-1">
                          {policy.grantControls.requireMfa && <li>• Require MFA</li>}
                          {policy.grantControls.requireCompliantDevice && <li>• Require compliant device</li>}
                          {policy.grantControls.requireApprovedApp && <li>• Require approved app</li>}
                          {policy.grantControls.blockAccess && <li className="text-red-400">• Block access</li>}
                        </ul>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Policy Impact */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Policy Impact Analysis</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <p className="text-3xl font-bold text-green-400">98.2%</p>
                  <p className="text-sm text-gray-400 mt-1">Sign-ins protected</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-yellow-400">1,234</p>
                  <p className="text-sm text-gray-400 mt-1">Blocked attempts today</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-blue-400">45ms</p>
                  <p className="text-sm text-gray-400 mt-1">Avg. evaluation time</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'sign-ins' && (
          <div className="space-y-6">
            {/* Sign-in Success Rate */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Sign-in Success Rate (24h)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={signInSuccessRate}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="time" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="success"
                    stroke="#10b981"
                    strokeWidth={2}
                    name="Success Rate %"
                  />
                  <Line
                    type="monotone"
                    dataKey="failure"
                    stroke="#ef4444"
                    strokeWidth={2}
                    name="Failure Rate %"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Recent Sign-in Activity */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Recent Sign-in Activity</h3>
              <div className="space-y-2">
                <div className="grid grid-cols-6 gap-2 text-xs text-gray-400 font-medium uppercase pb-2 border-b border-gray-800">
                  <div>User</div>
                  <div>Application</div>
                  <div>Location</div>
                  <div>Device</div>
                  <div>Status</div>
                  <div>Time</div>
                </div>
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="grid grid-cols-6 gap-2 text-sm py-2 hover:bg-gray-800/50 rounded">
                    <div className="truncate">user{i}@company.com</div>
                    <div>Office 365</div>
                    <div>New York, US</div>
                    <div>Windows/Chrome</div>
                    <div>
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        i % 3 === 0 ? 'bg-red-900/50 text-red-400' : 'bg-green-900/50 text-green-400'
                      }`}>
                        {i % 3 === 0 ? 'Failed' : 'Success'}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">2 min ago</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'service-principals' && (
          <div className="space-y-6">
            {/* Service Principals */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Service Principals & Apps</h3>
                <button type="button" className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                  Register App
                </button>
              </div>
              <div className="space-y-3">
                {servicePrincipals.map((sp) => (
                  <div key={sp.id} className="p-4 bg-gray-800/50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="flex items-center gap-3 mb-2">
                          <Key className="w-5 h-5 text-purple-400" />
                          <p className="font-medium">{sp.displayName}</p>
                          <span className="px-2 py-1 bg-purple-900/50 text-purple-400 rounded text-xs">
                            {sp.servicePrincipalType}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <p className="text-gray-400">App ID: {sp.appId}</p>
                            <p className="text-gray-400 mt-1">
                              Permissions: {sp.permissions.application.length} app, {sp.permissions.delegated.length} delegated
                            </p>
                          </div>
                          <div>
                            {sp.certificateExpiry && (
                              <p className="text-gray-400">
                                Cert expires: <span className={new Date(sp.certificateExpiry) < new Date('2024-06-01') ? 'text-yellow-400' : ''}>
                                  {sp.certificateExpiry}
                                </span>
                              </p>
                            )}
                            {sp.secretExpiry && (
                              <p className="text-gray-400 mt-1">
                                Secret expires: <span className={new Date(sp.secretExpiry) < new Date('2024-06-01') ? 'text-yellow-400' : ''}>
                                  {sp.secretExpiry}
                                </span>
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                      <button type="button" className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm">
                        Manage
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}