'use client'

import { useEffect, useState } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { 
  Lock, Users, Shield, Key, AlertTriangle, CheckCircle,
  UserCheck, Clock, Globe, RefreshCw, AlertCircle,
  ChevronRight, ExternalLink, ArrowLeft, TrendingUp,
  TrendingDown, BarChart3, Activity, Zap, FileCheck,
  ShieldCheck, UserPlus, Settings, Eye, Building
} from 'lucide-react'

interface SecurityCard {
  id: string
  title: string
  description: string
  icon: any
  href: string
  color: string
  stats?: {
    label: string
    value: string | number
    trend?: 'up' | 'down' | 'stable'
    status?: 'good' | 'warning' | 'critical'
  }[]
  quickActions?: {
    label: string
    href: string
  }[]
}

export default function SecurityAccessHub() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    const tab = searchParams.get('tab')
    if (tab && ['overview','iam','rbac','pim','conditional','zerotrust','entitlements','reviews'].includes(tab)) {
      setActiveTab(tab)
    }
  }, [searchParams])

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Shield },
    { id: 'iam', label: 'Identity & Access', icon: Users },
    { id: 'rbac', label: 'Role Management', icon: UserCheck },
    { id: 'pim', label: 'Privileged Identity', icon: Key },
    { id: 'conditional', label: 'Conditional Access', icon: Lock },
    { id: 'zerotrust', label: 'Zero Trust', icon: Globe },
    { id: 'entitlements', label: 'Entitlements', icon: Building },
    { id: 'reviews', label: 'Access Reviews', icon: RefreshCw }
  ]

  const securityCards: SecurityCard[] = [
    {
      id: 'iam',
      title: 'Identity & Access Management',
      description: 'Manage user identities, authentication methods, and access controls',
      icon: Users,
      href: '/security/iam',
      color: 'blue',
      stats: [
        { label: 'Total Users', value: '2,451', trend: 'up' },
        { label: 'MFA Enabled', value: '94%', status: 'good' },
        { label: 'Guest Users', value: 234 },
        { label: 'Service Principals', value: 567 }
      ],
      quickActions: [
        { label: 'Add User', href: '/security/iam#add-user' },
        { label: 'Manage Groups', href: '/security/iam#groups' }
      ]
    },
    {
      id: 'rbac',
      title: 'Role-Based Access Control',
      description: 'Configure and manage role assignments and permissions',
      icon: UserCheck,
      href: '/security/rbac',
      color: 'purple',
      stats: [
        { label: 'Custom Roles', value: 89 },
        { label: 'Role Assignments', value: '1,234', trend: 'up' },
        { label: 'Orphaned Roles', value: 5, status: 'warning' },
        { label: 'Over-privileged', value: 12, status: 'critical' }
      ],
      quickActions: [
        { label: 'Create Role', href: '/security/rbac#create' },
        { label: 'Audit Permissions', href: '/security/rbac#audit' }
      ]
    },
    {
      id: 'pim',
      title: 'Privileged Identity Management',
      description: 'Just-in-time access for privileged roles and administrative tasks',
      icon: Key,
      href: '/security/pim',
      color: 'orange',
      stats: [
        { label: 'Eligible Roles', value: 127 },
        { label: 'Active Elevations', value: 7, status: 'warning' },
        { label: 'Pending Requests', value: 3 },
        { label: 'Avg Duration', value: '2.5h' }
      ],
      quickActions: [
        { label: 'Request Access', href: '/security/pim#request' },
        { label: 'Approve Requests', href: '/security/pim#approve' }
      ]
    },
    {
      id: 'conditional',
      title: 'Conditional Access Policies',
      description: 'Configure context-aware access controls and security policies',
      icon: Lock,
      href: '/security/conditional-access',
      color: 'red',
      stats: [
        { label: 'Active Policies', value: 23 },
        { label: 'Block Events', value: 156, trend: 'down', status: 'good' },
        { label: 'MFA Challenges', value: '1.2K' },
        { label: 'Success Rate', value: '98.5%', status: 'good' }
      ],
      quickActions: [
        { label: 'Create Policy', href: '/security/conditional-access#create' },
        { label: 'View Insights', href: '/security/conditional-access#insights' }
      ]
    },
    {
      id: 'zerotrust',
      title: 'Zero Trust Implementation',
      description: 'Implement and monitor zero trust security principles',
      icon: Globe,
      href: '/security/zero-trust',
      color: 'green',
      stats: [
        { label: 'Trust Score', value: '82%', trend: 'up', status: 'good' },
        { label: 'Verified Devices', value: 892 },
        { label: 'Secure Apps', value: 145 },
        { label: 'Compliance', value: '91%', status: 'good' }
      ],
      quickActions: [
        { label: 'View Dashboard', href: '/security/zero-trust#dashboard' },
        { label: 'Security Posture', href: '/security/zero-trust#posture' }
      ]
    },
    {
      id: 'entitlements',
      title: 'Entitlement Management',
      description: 'Manage access packages and entitlement lifecycle',
      icon: Building,
      href: '/security/entitlements',
      color: 'indigo',
      stats: [
        { label: 'Access Packages', value: 45 },
        { label: 'Active Assignments', value: 678 },
        { label: 'Pending Approvals', value: 12, status: 'warning' },
        { label: 'Expiring Soon', value: 23, status: 'warning' }
      ],
      quickActions: [
        { label: 'Create Package', href: '/security/entitlements#create' },
        { label: 'Review Requests', href: '/security/entitlements#requests' }
      ]
    },
    {
      id: 'reviews',
      title: 'Access Reviews',
      description: 'Periodic reviews of user access and permissions',
      icon: RefreshCw,
      href: '/security/access-reviews',
      color: 'yellow',
      stats: [
        { label: 'Active Reviews', value: 8 },
        { label: 'Completion Rate', value: '67%', trend: 'up' },
        { label: 'Overdue', value: 3, status: 'critical' },
        { label: 'Decisions Made', value: 234 }
      ],
      quickActions: [
        { label: 'Start Review', href: '/security/access-reviews#start' },
        { label: 'View History', href: '/security/access-reviews#history' }
      ]
    }
  ]

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
      <div className="border-b border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-gray-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold flex items-center space-x-3">
                <Shield className="w-8 h-8 text-red-500" />
                <span>Security & Access Control</span>
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Identity, RBAC, PIM, conditional access, and zero trust management
              </p>
            </div>
            <button
              onClick={() => router.push('/tactical')}
              className="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors flex items-center space-x-2"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Command Center</span>
            </button>
          </div>
        </div>
      </div>

      <div className="border-b border-gray-200 dark:border-gray-800 bg-white/30 dark:bg-gray-900/30 overflow-x-auto">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-4">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button type="button"
                  key={tab.id}
                  onClick={() => {
                    setActiveTab(tab.id)
                    const params = new URLSearchParams(searchParams.toString())
                    params.set('tab', tab.id)
                    router.replace(`/security?${params.toString()}`)
                  }}
                  className={`
                    flex items-center gap-2 px-3 py-3 border-b-2 transition-colors whitespace-nowrap
                    ${activeTab === tab.id 
                      ? 'border-red-500 text-gray-900 dark:text-white' 
                      : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'}
                  `}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6">
        {activeTab === 'overview' && <SecurityOverview cards={securityCards} router={router} />}
        {activeTab === 'iam' && <IAMView router={router} />}
        {activeTab === 'rbac' && <RBACView router={router} />}
        {activeTab === 'pim' && <PIMView router={router} />}
        {activeTab === 'conditional' && <ConditionalAccessView router={router} />}
        {activeTab === 'zerotrust' && <ZeroTrustView router={router} />}
        {activeTab === 'entitlements' && <EntitlementsView router={router} />}
        {activeTab === 'reviews' && <AccessReviewsView router={router} />}
      </div>
    </div>
  )
}

function SecurityOverview({ cards, router }: { cards: SecurityCard[], router: any }) {
  return (
    <div className="space-y-6">
      {/* Key Security Metrics - Clickable */}
      <div className="grid grid-cols-4 gap-4">
        <div onClick={() => router.push('/security/iam')} className="cursor-pointer hover:shadow-lg transition-all">
          <MetricCard title="Security Score" value="87/100" icon={Shield} status="good" />
        </div>
        <div onClick={() => router.push('/security/iam')} className="cursor-pointer hover:shadow-lg transition-all">
          <MetricCard title="Active Users" value="2,451" icon={Users} />
        </div>
        <div onClick={() => router.push('/security/pim')} className="cursor-pointer hover:shadow-lg transition-all">
          <MetricCard title="Privileged Accounts" value="47" icon={Key} status="warning" />
        </div>
        <div onClick={() => router.push('/security/rbac')} className="cursor-pointer hover:shadow-lg transition-all">
          <MetricCard title="Policy Violations" value="12" icon={AlertTriangle} status="critical" />
        </div>
      </div>

      {/* Critical Security Alert */}
      <div 
        onClick={() => router.push('/security/pim')}
        className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6 cursor-pointer hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors"
      >
        <h2 className="text-lg font-semibold text-red-600 dark:text-red-400 mb-4 flex items-center justify-between">
          <span>Critical Security Items</span>
          <ChevronRight className="w-5 h-5" />
        </h2>
        <div className="space-y-3">
          <AlertItem title="3 accounts with expired MFA" action="Enable MFA" onClick={() => router.push('/security/iam#mfa')} />
          <AlertItem title="5 standing privileged access" action="Configure PIM" onClick={() => router.push('/security/pim')} />
          <AlertItem title="17 stale access reviews" action="Complete Reviews" onClick={() => router.push('/security/access-reviews')} />
        </div>
      </div>

      {/* Security Dashboard Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {cards.map((card) => {
          const Icon = card.icon
          return (
            <div
              key={card.id}
              className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 hover:shadow-xl transition-all"
            >
              <div
                className="p-6 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/30 transition-colors"
                onClick={() => router.push(card.href)}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`p-3 rounded-lg bg-${card.color}-500/10`}>
                      <Icon className={`w-6 h-6 text-${card.color}-500`} />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold">{card.title}</h3>
                    </div>
                  </div>
                  <ExternalLink className="w-4 h-4 text-gray-400" />
                </div>
                
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                  {card.description}
                </p>

                {/* Stats Grid */}
                {card.stats && (
                  <div className="grid grid-cols-2 gap-3 mb-4">
                    {card.stats.map((stat, idx) => (
                      <div key={idx}>
                        <div className="text-lg font-bold flex items-center space-x-1">
                          <span className={
                            stat.status === 'warning' ? 'text-yellow-500' : 
                            stat.status === 'critical' ? 'text-red-500' :
                            stat.status === 'good' ? 'text-green-500' : ''
                          }>
                            {stat.value}
                          </span>
                          {stat.trend === 'up' && <TrendingUp className="w-3 h-3 text-green-500" />}
                          {stat.trend === 'down' && <TrendingDown className="w-3 h-3 text-red-500" />}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">{stat.label}</div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Quick Actions */}
                {card.quickActions && (
                  <div className="flex space-x-2">
                    {card.quickActions.map((action, idx) => (
                      <button
                        key={idx}
                        onClick={(e) => {
                          e.stopPropagation()
                          router.push(action.href)
                        }}
                        className="flex-1 px-3 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg transition-colors text-sm"
                      >
                        {action.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Quick Security Actions */}
      <div className="grid grid-cols-4 gap-4">
        <button
          onClick={() => router.push('/security/iam#add-user')}
          className="p-4 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors text-center"
        >
          <UserPlus className="w-6 h-6 mx-auto mb-2" />
          <div className="font-medium">Add User</div>
        </button>
        <button
          onClick={() => router.push('/security/pim#request')}
          className="p-4 bg-orange-600 hover:bg-orange-700 rounded-lg transition-colors text-center"
        >
          <Key className="w-6 h-6 mx-auto mb-2" />
          <div className="font-medium">Request PIM</div>
        </button>
        <button
          onClick={() => router.push('/security/conditional-access#create')}
          className="p-4 bg-red-600 hover:bg-red-700 rounded-lg transition-colors text-center"
        >
          <Lock className="w-6 h-6 mx-auto mb-2" />
          <div className="font-medium">New Policy</div>
        </button>
        <button
          onClick={() => router.push('/security/access-reviews#start')}
          className="p-4 bg-green-600 hover:bg-green-700 rounded-lg transition-colors text-center"
        >
          <RefreshCw className="w-6 h-6 mx-auto mb-2" />
          <div className="font-medium">Start Review</div>
        </button>
      </div>
    </div>
  )
}

function IAMView({ router }: { router: any }) {
  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Identity Management</h2>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">User Accounts</h3>
            <div className="space-y-2">
              <UserRow type="Employees" count={847} onClick={() => router.push('/security/iam#employees')} />
              <UserRow type="Contractors" count={124} onClick={() => router.push('/security/iam#contractors')} />
              <UserRow type="Service Principals" count={276} onClick={() => router.push('/security/iam#service')} />
            </div>
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">Authentication</h3>
            <div className="space-y-2">
              <AuthRow method="Password + MFA" percentage={94} onClick={() => router.push('/security/iam#mfa')} />
              <AuthRow method="Passwordless" percentage={20} onClick={() => router.push('/security/iam#passwordless')} />
              <AuthRow method="FIDO2" percentage={5} onClick={() => router.push('/security/iam#fido2')} />
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-3 gap-4">
        <button
          onClick={() => router.push('/security/iam#add-user')}
          className="p-4 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
        >
          <UserPlus className="w-6 h-6 mx-auto mb-2" />
          <div>Add New User</div>
        </button>
        <button
          onClick={() => router.push('/security/iam#groups')}
          className="p-4 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
        >
          <Users className="w-6 h-6 mx-auto mb-2" />
          <div>Manage Groups</div>
        </button>
        <button
          onClick={() => router.push('/security/iam#mfa')}
          className="p-4 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
        >
          <ShieldCheck className="w-6 h-6 mx-auto mb-2" />
          <div>Configure MFA</div>
        </button>
      </div>
    </div>
  )
}

function RBACView({ router }: { router: any }) {
  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Role Assignments</h2>
        <div className="grid grid-cols-3 gap-4">
          <RoleCard role="Owner" count={3} risk="Critical" onClick={() => router.push('/security/rbac#owner')} />
          <RoleCard role="Contributor" count={47} risk="High" onClick={() => router.push('/security/rbac#contributor')} />
          <RoleCard role="Reader" count={523} risk="Low" onClick={() => router.push('/security/rbac#reader')} />
        </div>
      </div>

      {/* Custom Roles */}
      <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <div 
          onClick={() => router.push('/security/rbac#custom')}
          className="flex items-center justify-between mb-4 cursor-pointer hover:text-blue-500 transition-colors"
        >
          <h2 className="text-lg font-semibold">Custom Roles</h2>
          <ChevronRight className="w-4 h-4" />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-blue-500 transition-colors cursor-pointer"
               onClick={() => router.push('/security/rbac#create')}>
            <div className="text-2xl font-bold">89</div>
            <div className="text-sm text-gray-500 dark:text-gray-400">Custom roles defined</div>
          </div>
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-red-500 transition-colors cursor-pointer"
               onClick={() => router.push('/security/rbac#audit')}>
            <div className="text-2xl font-bold text-red-500">12</div>
            <div className="text-sm text-gray-500 dark:text-gray-400">Over-privileged accounts</div>
          </div>
        </div>
      </div>
    </div>
  )
}

function PIMView({ router }: { router: any }) {
  return (
    <div className="space-y-6">
      <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold text-orange-600 dark:text-orange-400 mb-4">
          Privileged Identity Management (JIT Access)
        </h2>
        <div className="grid grid-cols-3 gap-4">
          <PIMCard title="Eligible" value="127" onClick={() => router.push('/security/pim#eligible')} />
          <PIMCard title="Active" value="8" onClick={() => router.push('/security/pim#active')} />
          <PIMCard title="Pending" value="3" onClick={() => router.push('/security/pim#pending')} />
        </div>
        <div className="mt-6 space-y-3">
          <ElevationItem 
            user="admin@company.com" 
            role="Global Admin" 
            expires="2h 15m" 
            onRevoke={() => router.push('/security/pim#revoke')}
          />
          <ElevationItem 
            user="devops@company.com" 
            role="Contributor" 
            expires="45m"
            onRevoke={() => router.push('/security/pim#revoke')}
          />
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 gap-4">
        <button
          onClick={() => router.push('/security/pim#request')}
          className="p-4 bg-orange-600 hover:bg-orange-700 rounded-lg transition-colors"
        >
          <Zap className="w-6 h-6 mx-auto mb-2" />
          <div>Request Elevation</div>
        </button>
        <button
          onClick={() => router.push('/security/pim#approve')}
          className="p-4 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
        >
          <CheckCircle className="w-6 h-6 mx-auto mb-2" />
          <div>Approve Requests</div>
        </button>
      </div>
    </div>
  )
}

function ConditionalAccessView({ router }: { router: any }) {
  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <div 
          onClick={() => router.push('/security/conditional-access#policies')}
          className="flex items-center justify-between mb-4 cursor-pointer hover:text-blue-500 transition-colors"
        >
          <h2 className="text-lg font-semibold">Conditional Access Policies</h2>
          <ChevronRight className="w-4 h-4" />
        </div>
        <div className="space-y-3">
          <PolicyRow name="Require MFA for admins" status="enabled" onClick={() => router.push('/security/conditional-access#mfa-admins')} />
          <PolicyRow name="Block legacy authentication" status="enabled" onClick={() => router.push('/security/conditional-access#block-legacy')} />
          <PolicyRow name="Location-based access" status="enabled" onClick={() => router.push('/security/conditional-access#location')} />
          <PolicyRow name="Risk-based MFA" status="enabled" onClick={() => router.push('/security/conditional-access#risk-mfa')} />
          <PolicyRow name="Require compliant devices" status="enabled" onClick={() => router.push('/security/conditional-access#compliant')} />
        </div>
      </div>

      {/* Policy Insights */}
      <div className="grid grid-cols-3 gap-4">
        <button
          onClick={() => router.push('/security/conditional-access#create')}
          className="p-4 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
        >
          <Lock className="w-6 h-6 mx-auto mb-2" />
          <div>Create Policy</div>
        </button>
        <button
          onClick={() => router.push('/security/conditional-access#insights')}
          className="p-4 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
        >
          <BarChart3 className="w-6 h-6 mx-auto mb-2" />
          <div>View Insights</div>
        </button>
        <button
          onClick={() => router.push('/security/conditional-access#test')}
          className="p-4 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
        >
          <Activity className="w-6 h-6 mx-auto mb-2" />
          <div>Test Policies</div>
        </button>
      </div>
    </div>
  )
}

function ZeroTrustView({ router }: { router: any }) {
  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <div 
          onClick={() => router.push('/security/zero-trust#maturity')}
          className="flex items-center justify-between mb-4 cursor-pointer hover:text-blue-500 transition-colors"
        >
          <h2 className="text-lg font-semibold">Zero Trust Maturity</h2>
          <ChevronRight className="w-4 h-4" />
        </div>
        <div className="space-y-4">
          <MaturityBar category="Identity" level={3} onClick={() => router.push('/security/zero-trust#identity')} />
          <MaturityBar category="Devices" level={2} onClick={() => router.push('/security/zero-trust#devices')} />
          <MaturityBar category="Networks" level={2} onClick={() => router.push('/security/zero-trust#networks')} />
          <MaturityBar category="Applications" level={3} onClick={() => router.push('/security/zero-trust#apps')} />
          <MaturityBar category="Data" level={2} onClick={() => router.push('/security/zero-trust#data')} />
        </div>
      </div>

      {/* Zero Trust Actions */}
      <div className="grid grid-cols-2 gap-4">
        <button
          onClick={() => router.push('/security/zero-trust#dashboard')}
          className="p-4 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
        >
          <Eye className="w-6 h-6 mx-auto mb-2" />
          <div>View Dashboard</div>
        </button>
        <button
          onClick={() => router.push('/security/zero-trust#posture')}
          className="p-4 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
        >
          <Shield className="w-6 h-6 mx-auto mb-2" />
          <div>Security Posture</div>
        </button>
      </div>
    </div>
  )
}

function EntitlementsView({ router }: { router: any }) {
  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Entitlement Management</h2>
        <div className="grid grid-cols-2 gap-4">
          <div 
            onClick={() => router.push('/security/entitlements#packages')}
            className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-blue-500 transition-colors cursor-pointer"
          >
            <div className="text-2xl font-bold">45</div>
            <div className="text-sm text-gray-500 dark:text-gray-400">Access Packages</div>
          </div>
          <div 
            onClick={() => router.push('/security/entitlements#assignments')}
            className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-green-500 transition-colors cursor-pointer"
          >
            <div className="text-2xl font-bold">678</div>
            <div className="text-sm text-gray-500 dark:text-gray-400">Active Assignments</div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 gap-4">
        <button
          onClick={() => router.push('/security/entitlements#create')}
          className="p-4 bg-indigo-600 hover:bg-indigo-700 rounded-lg transition-colors"
        >
          <Building className="w-6 h-6 mx-auto mb-2" />
          <div>Create Package</div>
        </button>
        <button
          onClick={() => router.push('/security/entitlements#requests')}
          className="p-4 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
        >
          <FileCheck className="w-6 h-6 mx-auto mb-2" />
          <div>Review Requests</div>
        </button>
      </div>
    </div>
  )
}

function AccessReviewsView({ router }: { router: any }) {
  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <div 
          onClick={() => router.push('/security/access-reviews#active')}
          className="flex items-center justify-between mb-4 cursor-pointer hover:text-blue-500 transition-colors"
        >
          <h2 className="text-lg font-semibold">Active Access Reviews</h2>
          <ChevronRight className="w-4 h-4" />
        </div>
        <div className="space-y-3">
          <ReviewRow 
            name="Q1 Privileged Review" 
            progress={67} 
            dueDate="7 days" 
            onClick={() => router.push('/security/access-reviews#q1-review')}
          />
          <ReviewRow 
            name="Monthly Guest Review" 
            progress={45} 
            dueDate="14 days"
            onClick={() => router.push('/security/access-reviews#guest-review')}
          />
          <ReviewRow 
            name="Service Principal Audit" 
            progress={23} 
            dueDate="21 days"
            onClick={() => router.push('/security/access-reviews#sp-audit')}
          />
        </div>
      </div>

      {/* Review Actions */}
      <div className="grid grid-cols-2 gap-4">
        <button
          onClick={() => router.push('/security/access-reviews#start')}
          className="p-4 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition-colors"
        >
          <RefreshCw className="w-6 h-6 mx-auto mb-2" />
          <div>Start New Review</div>
        </button>
        <button
          onClick={() => router.push('/security/access-reviews#history')}
          className="p-4 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <Clock className="w-6 h-6 mx-auto mb-2" />
          <div>View History</div>
        </button>
      </div>
    </div>
  )
}

// Component library
function MetricCard({ title, value, icon: Icon, status }: {
  title: string
  value: string | number
  icon: React.ElementType
  status?: 'good' | 'warning' | 'critical'
}) {
  const statusColors = {
    good: 'text-green-500',
    warning: 'text-yellow-500',
    critical: 'text-red-500'
  }

  return (
    <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-4 hover:shadow-md transition-all">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">{title}</p>
          <p className={`text-2xl font-bold mt-1 ${status ? statusColors[status] : ''}`}>{value}</p>
        </div>
        <Icon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
      </div>
    </div>
  )
}

function AlertItem({ title, action, onClick }: {
  title: string
  action: string
  onClick: () => void
}) {
  return (
    <div className="flex justify-between items-center p-3 bg-gray-100 dark:bg-gray-900/50 rounded">
      <span>{title}</span>
      <button 
        type="button" 
        onClick={onClick}
        className="text-xs px-3 py-1 bg-red-600 hover:bg-red-700 rounded transition-colors"
      >
        {action}
      </button>
    </div>
  )
}

function UserRow({ type, count, onClick }: {
  type: string
  count: number
  onClick: () => void
}) {
  return (
    <div 
      onClick={onClick}
      className="flex justify-between p-2 hover:bg-gray-200 dark:hover:bg-gray-800/50 rounded cursor-pointer transition-colors"
    >
      <span>{type}</span>
      <span className="font-bold">{count}</span>
    </div>
  )
}

function AuthRow({ method, percentage, onClick }: {
  method: string
  percentage: number
  onClick: () => void
}) {
  return (
    <div 
      onClick={onClick}
      className="flex justify-between p-2 hover:bg-gray-200 dark:hover:bg-gray-800/50 rounded cursor-pointer transition-colors"
    >
      <span>{method}</span>
      <span className="font-bold">{percentage}%</span>
    </div>
  )
}

function RoleCard({ role, count, risk, onClick }: {
  role: string
  count: number
  risk: 'Critical' | 'High' | 'Low'
  onClick: () => void
}) {
  const riskColors = {
    Critical: 'border-red-500',
    High: 'border-orange-500',
    Low: 'border-green-500'
  }
  
  return (
    <div 
      onClick={onClick}
      className={`p-4 rounded-lg border ${riskColors[risk]} bg-white dark:bg-gray-900/50 cursor-pointer hover:shadow-md transition-all`}
    >
      <h3 className="font-medium">{role}</h3>
      <p className="text-2xl font-bold mt-1">{count} users</p>
      <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">Risk: {risk}</p>
    </div>
  )
}

function PIMCard({ title, value, onClick }: {
  title: string
  value: string | number
  onClick: () => void
}) {
  return (
    <div 
      onClick={onClick}
      className="bg-gray-100 dark:bg-gray-900/50 rounded-lg p-4 cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 transition-colors"
    >
      <p className="text-sm text-gray-600 dark:text-gray-400">{title}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
    </div>
  )
}

function ElevationItem({ user, role, expires, onRevoke }: {
  user: string
  role: string
  expires: string
  onRevoke: () => void
}) {
  return (
    <div className="flex justify-between p-3 bg-gray-200 dark:bg-gray-800/50 rounded">
      <div>
        <p className="font-medium">{user}</p>
        <p className="text-sm text-orange-600 dark:text-orange-400">{role}</p>
      </div>
      <div className="text-right">
        <p className="text-sm text-yellow-400">Expires: {expires}</p>
        <button 
          type="button" 
          onClick={onRevoke}
          className="text-xs px-2 py-1 bg-red-600 hover:bg-red-700 rounded mt-1 transition-colors"
        >
          Revoke
        </button>
      </div>
    </div>
  )
}

function PolicyRow({ name, status, onClick }: {
  name: string
  status: 'enabled' | 'disabled'
  onClick: () => void
}) {
  return (
    <div 
      onClick={onClick}
      className="flex justify-between p-3 bg-gray-100 dark:bg-gray-800/50 rounded cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
    >
      <span>{name}</span>
      <span className={`text-xs px-2 py-1 rounded ${
        status === 'enabled' ? 'bg-green-100 dark:bg-green-900/50 text-green-600 dark:text-green-400' : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
      }`}>
        {status}
      </span>
    </div>
  )
}

function MaturityBar({ category, level, onClick }: {
  category: string
  level: number
  onClick: () => void
}) {
  return (
    <div onClick={onClick} className="cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 p-2 rounded transition-colors">
      <div className="flex justify-between mb-2">
        <span>{category}</span>
        <span className="text-sm text-gray-600 dark:text-gray-400">Level {level}/5</span>
      </div>
      <div className="bg-gray-300 dark:bg-gray-800 rounded-full h-2">
        <div 
          className="bg-gradient-to-r from-red-500 to-green-500 h-2 rounded-full transition-all"
          style={{ width: `${(level / 5) * 100}%` }}
        />
      </div>
    </div>
  )
}

function ReviewRow({ name, progress, dueDate, onClick }: {
  name: string
  progress: number
  dueDate: string
  onClick: () => void
}) {
  return (
    <div 
      onClick={onClick}
      className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
    >
      <div className="flex justify-between mb-2">
        <span className="font-medium">{name}</span>
        <span className="text-sm text-yellow-600 dark:text-yellow-400">Due: {dueDate}</span>
      </div>
      <div className="bg-gray-300 dark:bg-gray-700 rounded-full h-2">
        <div className="bg-blue-500 h-2 rounded-full transition-all" style={{ width: `${progress}%` }} />
      </div>
      <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">{progress}% complete</p>
    </div>
  )
}