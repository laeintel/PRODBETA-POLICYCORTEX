'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { 
  Users, Shield, Lock, Key, AlertTriangle, CheckCircle,
  UserPlus, UserMinus, Settings, Eye, EyeOff, Clock,
  Activity, BarChart3, TrendingUp, AlertCircle, ChevronRight,
  RefreshCw, Download, Search, Filter, Zap, UserCheck
} from 'lucide-react'
import { 
  Treemap, BarChart, Bar, LineChart, Line, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, PieChart, Pie, Cell, RadialBarChart, 
  RadialBar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

// TypeScript Types
interface User {
  id: string
  email: string
  name: string
  department: string
  roles: Role[]
  permissions: Permission[]
  lastActive: string
  riskScore: number
  mfaEnabled: boolean
}

interface Role {
  id: string
  name: string
  description: string
  permissions: Permission[]
  users: number
  critical: boolean
  custom: boolean
}

interface Permission {
  id: string
  resource: string
  action: string
  effect: 'Allow' | 'Deny'
  conditions?: string[]
  lastUsed?: string
  riskLevel: 'low' | 'medium' | 'high' | 'critical'
}

interface AccessRequest {
  id: string
  user: string
  role: string
  reason: string
  duration: string
  status: 'pending' | 'approved' | 'denied' | 'expired'
  requestedAt: string
  approver?: string
}

interface SegregationConflict {
  id: string
  user: string
  conflictingRoles: string[]
  severity: 'high' | 'medium' | 'low'
  description: string
}

export default function RBACPage() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [selectedRole, setSelectedRole] = useState<Role | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [permissionMatrix, setPermissionMatrix] = useState<any[]>([])

  // Mock data
  const users: User[] = [
    {
      id: '1',
      email: 'admin@company.com',
      name: 'John Admin',
      department: 'IT',
      roles: [
        { id: '1', name: 'Global Administrator', description: 'Full access', permissions: [], users: 3, critical: true, custom: false }
      ],
      permissions: [],
      lastActive: '2024-03-01T10:30:00Z',
      riskScore: 95,
      mfaEnabled: true
    },
    {
      id: '2',
      email: 'developer@company.com',
      name: 'Jane Developer',
      department: 'Engineering',
      roles: [
        { id: '2', name: 'Developer', description: 'Development resources', permissions: [], users: 45, critical: false, custom: false },
        { id: '3', name: 'Reader', description: 'Read-only access', permissions: [], users: 120, critical: false, custom: false }
      ],
      permissions: [],
      lastActive: '2024-03-01T11:00:00Z',
      riskScore: 45,
      mfaEnabled: true
    },
    {
      id: '3',
      email: 'analyst@company.com',
      name: 'Bob Analyst',
      department: 'Finance',
      roles: [
        { id: '4', name: 'Billing Administrator', description: 'Billing access', permissions: [], users: 8, critical: true, custom: false }
      ],
      permissions: [],
      lastActive: '2024-02-28T15:45:00Z',
      riskScore: 65,
      mfaEnabled: false
    }
  ]

  const roles: Role[] = [
    {
      id: '1',
      name: 'Global Administrator',
      description: 'Full administrative access to all resources',
      permissions: [
        { id: '1', resource: '*', action: '*', effect: 'Allow', riskLevel: 'critical' }
      ],
      users: 3,
      critical: true,
      custom: false
    },
    {
      id: '2',
      name: 'Security Administrator',
      description: 'Manage security policies and configurations',
      permissions: [
        { id: '2', resource: 'Security/*', action: '*', effect: 'Allow', riskLevel: 'high' },
        { id: '3', resource: 'Policies/*', action: '*', effect: 'Allow', riskLevel: 'high' }
      ],
      users: 5,
      critical: true,
      custom: false
    },
    {
      id: '3',
      name: 'Developer',
      description: 'Access to development resources',
      permissions: [
        { id: '4', resource: 'Compute/VMs', action: 'Create,Read,Update', effect: 'Allow', riskLevel: 'medium' },
        { id: '5', resource: 'Storage/*', action: 'Read,Write', effect: 'Allow', riskLevel: 'medium' }
      ],
      users: 45,
      critical: false,
      custom: false
    },
    {
      id: '4',
      name: 'Reader',
      description: 'Read-only access to resources',
      permissions: [
        { id: '6', resource: '*', action: 'Read', effect: 'Allow', riskLevel: 'low' }
      ],
      users: 120,
      critical: false,
      custom: false
    }
  ]

  // Permission usage heatmap data
  const heatmapData = [
    { hour: '00', monday: 10, tuesday: 15, wednesday: 8, thursday: 12, friday: 5, saturday: 2, sunday: 1 },
    { hour: '06', monday: 25, tuesday: 30, wednesday: 28, thursday: 35, friday: 20, saturday: 5, sunday: 3 },
    { hour: '09', monday: 85, tuesday: 90, wednesday: 88, thursday: 92, friday: 80, saturday: 15, sunday: 10 },
    { hour: '12', monday: 70, tuesday: 75, wednesday: 72, thursday: 78, friday: 65, saturday: 20, sunday: 15 },
    { hour: '15', monday: 80, tuesday: 85, wednesday: 82, thursday: 88, friday: 75, saturday: 12, sunday: 8 },
    { hour: '18', monday: 45, tuesday: 50, wednesday: 48, thursday: 55, friday: 30, saturday: 8, sunday: 5 },
    { hour: '21', monday: 20, tuesday: 25, wednesday: 22, thursday: 28, friday: 15, saturday: 10, sunday: 8 }
  ]

  // Role hierarchy data
  const hierarchyData = [
    {
      name: 'Organization Admin',
      value: 100,
      children: [
        {
          name: 'Security Admin',
          value: 80,
          children: [
            { name: 'Security Reader', value: 30 }
          ]
        },
        {
          name: 'Billing Admin',
          value: 70,
          children: [
            { name: 'Cost Reader', value: 25 }
          ]
        },
        {
          name: 'Resource Admin',
          value: 75,
          children: [
            { name: 'Developer', value: 40 },
            { name: 'Operator', value: 35 }
          ]
        }
      ]
    }
  ]

  // Over-provisioned permissions data
  const overProvisionedData = [
    { user: 'john.doe@company.com', unused: 45, used: 55, department: 'IT' },
    { user: 'jane.smith@company.com', unused: 68, used: 32, department: 'Finance' },
    { user: 'bob.wilson@company.com', unused: 72, used: 28, department: 'HR' },
    { user: 'alice.brown@company.com', unused: 38, used: 62, department: 'Engineering' },
    { user: 'charlie.davis@company.com', unused: 85, used: 15, department: 'Marketing' }
  ]

  // Access review campaigns
  const accessReviews = [
    {
      id: '1',
      name: 'Q1 2024 Privileged Access Review',
      status: 'in-progress',
      progress: 65,
      totalUsers: 150,
      reviewed: 98,
      findings: 12,
      dueDate: '2024-03-31'
    },
    {
      id: '2',
      name: 'Annual Compliance Review',
      status: 'scheduled',
      progress: 0,
      totalUsers: 500,
      reviewed: 0,
      findings: 0,
      dueDate: '2024-06-30'
    }
  ]

  // Segregation of duties conflicts
  const sodConflicts: SegregationConflict[] = [
    {
      id: '1',
      user: 'admin@company.com',
      conflictingRoles: ['Billing Admin', 'Payment Approver'],
      severity: 'high',
      description: 'User can both create and approve payments'
    },
    {
      id: '2',
      user: 'developer@company.com',
      conflictingRoles: ['Developer', 'Production Deployer'],
      severity: 'medium',
      description: 'User can deploy own code to production without review'
    }
  ]

  // JIT access requests
  const jitRequests: AccessRequest[] = [
    {
      id: '1',
      user: 'contractor@company.com',
      role: 'Database Administrator',
      reason: 'Emergency database maintenance',
      duration: '4 hours',
      status: 'pending',
      requestedAt: '2024-03-01T12:00:00Z'
    },
    {
      id: '2',
      user: 'support@company.com',
      role: 'Customer Data Access',
      reason: 'Customer support ticket #12345',
      duration: '2 hours',
      status: 'approved',
      requestedAt: '2024-03-01T10:00:00Z',
      approver: 'manager@company.com'
    }
  ]

  useEffect(() => {
    generatePermissionMatrix()
  }, [])

  const generatePermissionMatrix = () => {
    const resources = ['VMs', 'Storage', 'Network', 'Database', 'KeyVault']
    const actions = ['Read', 'Write', 'Delete', 'Admin']
    const matrix: any[] = []

    roles.forEach(role => {
      const row: any = { role: role.name }
      resources.forEach(resource => {
        actions.forEach(action => {
          const key = `${resource}-${action}`
          row[key] = Math.random() > 0.6 ? 1 : 0
        })
      })
      matrix.push(row)
    })

    setPermissionMatrix(matrix)
  }

  const handleJITApproval = (requestId: string, approved: boolean) => {
    alert(`JIT request ${requestId} ${approved ? 'approved' : 'denied'}`)
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
              <h1 className="text-2xl font-bold">Role-Based Access Control</h1>
              <p className="text-sm text-gray-400 mt-1">
                Manage user permissions, roles, and access policies
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => exportReport('permissions')}
                className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
              >
                <Download className="w-4 h-4" />
                Export
              </button>
              <button
                onClick={() => router.push('/security/rbac/review')}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
              >
                <UserCheck className="w-4 h-4" />
                Start Review
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-800 bg-gray-900/30">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-6">
            {['overview', 'users', 'roles', 'permissions', 'jit-access', 'segregation', 'reviews'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-3 border-b-2 transition-colors capitalize ${
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
                <p className="text-3xl font-bold">1,247</p>
                <p className="text-sm text-gray-500 mt-2">+52 this month</p>
              </div>

              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm text-gray-400">Active Roles</h3>
                  <Shield className="w-5 h-5 text-green-400" />
                </div>
                <p className="text-3xl font-bold">68</p>
                <p className="text-sm text-gray-500 mt-2">12 custom</p>
              </div>

              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm text-gray-400">Over-provisioned</h3>
                  <AlertTriangle className="w-5 h-5 text-yellow-400" />
                </div>
                <p className="text-3xl font-bold">234</p>
                <p className="text-sm text-red-400 mt-2">Needs review</p>
              </div>

              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm text-gray-400">MFA Enabled</h3>
                  <Lock className="w-5 h-5 text-purple-400" />
                </div>
                <p className="text-3xl font-bold">89%</p>
                <p className="text-sm text-gray-500 mt-2">137 without MFA</p>
              </div>
            </div>

            {/* Permission Usage Heatmap */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Permission Usage Patterns</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr>
                      <th className="text-left px-2 py-1 text-gray-400">Hour</th>
                      {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map(day => (
                        <th key={day} className="px-2 py-1 text-gray-400">{day}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {heatmapData.map((row) => (
                      <tr key={row.hour}>
                        <td className="px-2 py-1 text-gray-400">{row.hour}:00</td>
                        {['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'].map(day => {
                          const value = row[day as keyof typeof row] as number
                          const intensity = value / 100
                          return (
                            <td key={day} className="px-2 py-1">
                              <div 
                                className="w-12 h-8 rounded flex items-center justify-center text-xs"
                                style={{
                                  backgroundColor: `rgba(59, 130, 246, ${intensity})`,
                                  color: intensity > 0.5 ? 'white' : '#9ca3af'
                                }}
                              >
                                {value}
                              </div>
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Over-provisioned Users Chart */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Over-provisioned Permissions</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={overProvisionedData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="user" stroke="#9ca3af" angle={-45} textAnchor="end" height={100} />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                  <Legend />
                  <Bar dataKey="used" stackId="a" fill="#10b981" name="Used Permissions" />
                  <Bar dataKey="unused" stackId="a" fill="#ef4444" name="Unused Permissions" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-3 gap-4">
              <button className="p-4 bg-gray-900/50 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors">
                <h3 className="font-medium mb-2">Review Privileged Accounts</h3>
                <p className="text-sm text-gray-400">23 accounts need review</p>
              </button>
              <button className="p-4 bg-gray-900/50 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors">
                <h3 className="font-medium mb-2">Pending JIT Requests</h3>
                <p className="text-sm text-gray-400">5 requests awaiting approval</p>
              </button>
              <button className="p-4 bg-gray-900/50 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors">
                <h3 className="font-medium mb-2">SoD Conflicts</h3>
                <p className="text-sm text-gray-400">8 conflicts detected</p>
              </button>
            </div>
          </div>
        )}

        {activeTab === 'users' && (
          <div className="space-y-4">
            {/* Search */}
            <div className="flex gap-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search users..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-gray-900/50 border border-gray-800 rounded-lg focus:outline-none focus:border-blue-500"
                />
              </div>
              <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors">
                <Filter className="w-4 h-4" />
              </button>
            </div>

            {/* Users Table */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 overflow-hidden">
              <table className="w-full">
                <thead className="bg-gray-800/50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">User</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Department</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Roles</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Risk Score</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">MFA</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Last Active</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {users.map((user) => (
                    <tr key={user.id} className="hover:bg-gray-800/30 transition-colors">
                      <td className="px-4 py-4">
                        <div>
                          <p className="font-medium">{user.name}</p>
                          <p className="text-xs text-gray-400">{user.email}</p>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-sm">{user.department}</td>
                      <td className="px-4 py-4">
                        <div className="flex flex-wrap gap-1">
                          {user.roles.map((role) => (
                            <span 
                              key={role.id}
                              className={`px-2 py-1 rounded-full text-xs ${
                                role.critical 
                                  ? 'bg-red-900/50 text-red-400' 
                                  : 'bg-blue-900/50 text-blue-400'
                              }`}
                            >
                              {role.name}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="px-4 py-4">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-gray-800 rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${
                                user.riskScore > 70 ? 'bg-red-500' :
                                user.riskScore > 40 ? 'bg-yellow-500' :
                                'bg-green-500'
                              }`}
                              style={{ width: `${user.riskScore}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-400">{user.riskScore}</span>
                        </div>
                      </td>
                      <td className="px-4 py-4">
                        {user.mfaEnabled ? (
                          <CheckCircle className="w-4 h-4 text-green-400" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-red-400" />
                        )}
                      </td>
                      <td className="px-4 py-4 text-sm text-gray-400">
                        {new Date(user.lastActive).toLocaleDateString()}
                      </td>
                      <td className="px-4 py-4">
                        <div className="flex items-center gap-2">
                          <button className="p-1 hover:bg-gray-700 rounded">
                            <Eye className="w-4 h-4" />
                          </button>
                          <button className="p-1 hover:bg-gray-700 rounded">
                            <Settings className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'roles' && (
          <div className="space-y-6">
            {/* Role Hierarchy Visualization */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Role Hierarchy</h3>
              <div className="text-sm text-gray-400">
                <div className="space-y-2">
                  {roles.map((role) => (
                    <div 
                      key={role.id}
                      className="flex items-center justify-between p-3 bg-gray-800/50 rounded hover:bg-gray-800/70 transition-colors cursor-pointer"
                      onClick={() => setSelectedRole(role)}
                    >
                      <div className="flex items-center gap-3">
                        <Shield className={`w-5 h-5 ${role.critical ? 'text-red-400' : 'text-blue-400'}`} />
                        <div>
                          <p className="font-medium">{role.name}</p>
                          <p className="text-xs text-gray-400">{role.description}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-sm">{role.users} users</span>
                        <span className="text-sm">{role.permissions.length} permissions</span>
                        {role.custom && (
                          <span className="px-2 py-1 bg-purple-900/50 text-purple-400 rounded text-xs">Custom</span>
                        )}
                        <ChevronRight className="w-4 h-4 text-gray-400" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Selected Role Details */}
            {selectedRole && (
              <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">{selectedRole.name} - Permissions</h3>
                  <button 
                    onClick={() => setSelectedRole(null)}
                    className="text-sm text-gray-400 hover:text-white"
                  >
                    Close
                  </button>
                </div>
                <div className="space-y-2">
                  {selectedRole.permissions.map((perm) => (
                    <div key={perm.id} className="flex items-center justify-between p-2 bg-gray-800/50 rounded">
                      <div className="flex items-center gap-3">
                        <Key className={`w-4 h-4 ${
                          perm.riskLevel === 'critical' ? 'text-red-400' :
                          perm.riskLevel === 'high' ? 'text-orange-400' :
                          perm.riskLevel === 'medium' ? 'text-yellow-400' :
                          'text-green-400'
                        }`} />
                        <div>
                          <p className="text-sm">
                            <span className="font-medium">{perm.resource}</span>
                            <span className="text-gray-400 mx-2">→</span>
                            <span className="text-blue-400">{perm.action}</span>
                          </p>
                        </div>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs ${
                        perm.effect === 'Allow' 
                          ? 'bg-green-900/50 text-green-400' 
                          : 'bg-red-900/50 text-red-400'
                      }`}>
                        {perm.effect}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'permissions' && (
          <div className="space-y-6">
            {/* Permission Matrix Grid */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Permission Matrix</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr>
                      <th className="text-left px-2 py-1 text-gray-400">Role</th>
                      {['VMs', 'Storage', 'Network', 'Database', 'KeyVault'].map(resource => (
                        ['Read', 'Write', 'Delete', 'Admin'].map(action => (
                          <th key={`${resource}-${action}`} className="px-1 py-1 text-xs text-gray-400">
                            <div className="writing-mode-vertical">{`${resource.slice(0,3)}-${action.slice(0,1)}`}</div>
                          </th>
                        ))
                      )).flat()}
                    </tr>
                  </thead>
                  <tbody>
                    {permissionMatrix.map((row, idx) => (
                      <tr key={idx} className="hover:bg-gray-800/30">
                        <td className="px-2 py-2 font-medium">{row.role}</td>
                        {Object.keys(row).filter(k => k !== 'role').map(key => (
                          <td key={key} className="px-1 py-2 text-center">
                            {row[key] === 1 ? (
                              <CheckCircle className="w-4 h-4 text-green-400 mx-auto" />
                            ) : (
                              <div className="w-4 h-4 mx-auto" />
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Permission Recommendations */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">AI-Powered Permission Recommendations</h3>
              <div className="space-y-3">
                <div className="p-3 bg-green-900/20 border border-green-800 rounded">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-green-400">Remove unused permission</p>
                      <p className="text-sm text-gray-400 mt-1">
                        User 'developer@company.com' hasn't used 'Database/Delete' in 90 days
                      </p>
                    </div>
                    <button className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm">
                      Apply
                    </button>
                  </div>
                </div>
                <div className="p-3 bg-yellow-900/20 border border-yellow-800 rounded">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-yellow-400">Suggest role assignment</p>
                      <p className="text-sm text-gray-400 mt-1">
                        5 users have similar permissions - consider creating 'Data Analyst' role
                      </p>
                    </div>
                    <button className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-sm">
                      Review
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'jit-access' && (
          <div className="space-y-6">
            {/* JIT Requests */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Just-In-Time Access Requests</h3>
              <div className="space-y-3">
                {jitRequests.map((request) => (
                  <div key={request.id} className="p-4 bg-gray-800/50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="flex items-center gap-3 mb-2">
                          <p className="font-medium">{request.user}</p>
                          <span className={`px-2 py-1 rounded-full text-xs ${
                            request.status === 'pending' ? 'bg-yellow-900/50 text-yellow-400' :
                            request.status === 'approved' ? 'bg-green-900/50 text-green-400' :
                            request.status === 'denied' ? 'bg-red-900/50 text-red-400' :
                            'bg-gray-800 text-gray-400'
                          }`}>
                            {request.status}
                          </span>
                        </div>
                        <p className="text-sm text-gray-400">
                          Requesting: <span className="text-white">{request.role}</span> for {request.duration}
                        </p>
                        <p className="text-sm text-gray-400 mt-1">
                          Reason: {request.reason}
                        </p>
                        {request.approver && (
                          <p className="text-xs text-gray-500 mt-2">
                            Approved by: {request.approver}
                          </p>
                        )}
                      </div>
                      {request.status === 'pending' && (
                        <div className="flex gap-2">
                          <button 
                            onClick={() => handleJITApproval(request.id, true)}
                            className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm"
                          >
                            Approve
                          </button>
                          <button 
                            onClick={() => handleJITApproval(request.id, false)}
                            className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm"
                          >
                            Deny
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* JIT Configuration */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">JIT Access Configuration</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-gray-800/50 rounded">
                  <p className="text-sm text-gray-400">Max Duration</p>
                  <p className="font-medium">8 hours</p>
                </div>
                <div className="p-3 bg-gray-800/50 rounded">
                  <p className="text-sm text-gray-400">Auto-expire</p>
                  <p className="font-medium">Enabled</p>
                </div>
                <div className="p-3 bg-gray-800/50 rounded">
                  <p className="text-sm text-gray-400">Approval Required</p>
                  <p className="font-medium">Manager + Security</p>
                </div>
                <div className="p-3 bg-gray-800/50 rounded">
                  <p className="text-sm text-gray-400">MFA Required</p>
                  <p className="font-medium">Yes</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'segregation' && (
          <div className="space-y-6">
            {/* SoD Conflicts */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Segregation of Duties Conflicts</h3>
              <div className="space-y-3">
                {sodConflicts.map((conflict) => (
                  <div key={conflict.id} className="p-4 bg-gray-800/50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="flex items-center gap-3 mb-2">
                          <AlertTriangle className={`w-5 h-5 ${
                            conflict.severity === 'high' ? 'text-red-400' :
                            conflict.severity === 'medium' ? 'text-yellow-400' :
                            'text-blue-400'
                          }`} />
                          <p className="font-medium">{conflict.user}</p>
                          <span className={`px-2 py-1 rounded-full text-xs ${
                            conflict.severity === 'high' ? 'bg-red-900/50 text-red-400' :
                            conflict.severity === 'medium' ? 'bg-yellow-900/50 text-yellow-400' :
                            'bg-blue-900/50 text-blue-400'
                          }`}>
                            {conflict.severity} risk
                          </span>
                        </div>
                        <p className="text-sm text-gray-400">{conflict.description}</p>
                        <div className="flex gap-2 mt-2">
                          {conflict.conflictingRoles.map((role) => (
                            <span key={role} className="px-2 py-1 bg-gray-800 rounded text-xs">
                              {role}
                            </span>
                          ))}
                        </div>
                      </div>
                      <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                        Resolve
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* SoD Rules */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Segregation Rules</h3>
              <div className="space-y-2">
                <div className="p-3 bg-gray-800/50 rounded flex items-center justify-between">
                  <div>
                    <p className="font-medium">Financial Approval Separation</p>
                    <p className="text-sm text-gray-400">Users cannot both create and approve financial transactions</p>
                  </div>
                  <span className="text-xs px-2 py-1 bg-green-900/50 text-green-400 rounded">Active</span>
                </div>
                <div className="p-3 bg-gray-800/50 rounded flex items-center justify-between">
                  <div>
                    <p className="font-medium">Code Deployment Separation</p>
                    <p className="text-sm text-gray-400">Developers cannot deploy to production without review</p>
                  </div>
                  <span className="text-xs px-2 py-1 bg-green-900/50 text-green-400 rounded">Active</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'reviews' && (
          <div className="space-y-6">
            {/* Access Review Campaigns */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Access Review Campaigns</h3>
              <div className="space-y-4">
                {accessReviews.map((review) => (
                  <div key={review.id} className="p-4 bg-gray-800/50 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <p className="font-medium">{review.name}</p>
                        <p className="text-sm text-gray-400">Due: {review.dueDate}</p>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs ${
                        review.status === 'in-progress' ? 'bg-yellow-900/50 text-yellow-400' :
                        'bg-gray-800 text-gray-400'
                      }`}>
                        {review.status}
                      </span>
                    </div>
                    <div className="mb-3">
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-400">Progress</span>
                        <span>{review.reviewed}/{review.totalUsers} reviewed</span>
                      </div>
                      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-blue-500 rounded-full"
                          style={{ width: `${review.progress}%` }}
                        />
                      </div>
                    </div>
                    {review.findings > 0 && (
                      <p className="text-sm text-orange-400">{review.findings} findings identified</p>
                    )}
                    <button className="mt-3 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                      Continue Review
                    </button>
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