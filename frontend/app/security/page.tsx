'use client'

import { useEffect, useState } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { 
  Lock, Users, Shield, Key, AlertTriangle, CheckCircle,
  UserCheck, Clock, Globe, RefreshCw, AlertCircle
} from 'lucide-react'

export default function SecurityAccessHub() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    const tab = searchParams.get('tab')
    if (tab && ['overview','iam','rbac','pim','conditional','zerotrust','reviews'].includes(tab)) {
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
    { id: 'reviews', label: 'Access Reviews', icon: RefreshCw }
  ]

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-white">
      <div className="border-b border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-gray-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <h1 className="text-2xl font-bold">Security & Access Control</h1>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Identity, RBAC, PIM, conditional access, and zero trust management
          </p>
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
                      ? 'border-orange-500 text-gray-900 dark:text-white' 
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
        {activeTab === 'overview' && <SecurityOverview />}
        {activeTab === 'iam' && <IAMView />}
        {activeTab === 'rbac' && <RBACView />}
        {activeTab === 'pim' && <PIMView />}
        {activeTab === 'conditional' && <ConditionalAccessView />}
        {activeTab === 'zerotrust' && <ZeroTrustView />}
        {activeTab === 'reviews' && <AccessReviewsView />}
      </div>
    </div>
  )
}

function SecurityOverview() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-4 gap-4">
        <MetricCard title="Security Score" value="87/100" icon={Shield} />
        <MetricCard title="Active Users" value="1,247" icon={Users} />
        <MetricCard title="Privileged Accounts" value="47" icon={Key} />
        <MetricCard title="Policy Violations" value="12" icon={AlertTriangle} />
      </div>

      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold text-red-600 dark:text-red-400 mb-4">Critical Security Items</h2>
        <div className="space-y-3">
          <AlertItem title="3 accounts with expired MFA" action="Enable MFA" />
          <AlertItem title="5 standing privileged access" action="Configure PIM" />
          <AlertItem title="17 stale access reviews" action="Complete Reviews" />
        </div>
      </div>
    </div>
  )
}

function IAMView() {
  return (
    <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Identity Management</h2>
      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">User Accounts</h3>
          <div className="space-y-2">
            <UserRow type="Employees" count={847} />
            <UserRow type="Contractors" count={124} />
            <UserRow type="Service Principals" count={276} />
          </div>
        </div>
        <div>
          <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">Authentication</h3>
          <div className="space-y-2">
            <AuthRow method="Password + MFA" percentage={94} />
            <AuthRow method="Passwordless" percentage={20} />
            <AuthRow method="FIDO2" percentage={5} />
          </div>
        </div>
      </div>
    </div>
  )
}

function RBACView() {
  return (
    <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Role Assignments</h2>
      <div className="grid grid-cols-3 gap-4">
        <RoleCard role="Owner" count={3} risk="Critical" />
        <RoleCard role="Contributor" count={47} risk="High" />
        <RoleCard role="Reader" count={523} risk="Low" />
      </div>
    </div>
  )
}

function PIMView() {
  return (
    <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-6">
      <h2 className="text-lg font-semibold text-orange-600 dark:text-orange-400 mb-4">
        Privileged Identity Management (JIT Access)
      </h2>
      <div className="grid grid-cols-3 gap-4">
        <PIMCard title="Eligible" value="127" />
        <PIMCard title="Active" value="8" />
        <PIMCard title="Pending" value="3" />
      </div>
      <div className="mt-6 space-y-3">
        <ElevationItem user="admin@company.com" role="Global Admin" expires="2h 15m" />
        <ElevationItem user="devops@company.com" role="Contributor" expires="45m" />
      </div>
    </div>
  )
}

function ConditionalAccessView() {
  return (
    <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Conditional Access Policies</h2>
      <div className="space-y-3">
        <PolicyRow name="Require MFA for admins" status="enabled" />
        <PolicyRow name="Block legacy authentication" status="enabled" />
        <PolicyRow name="Location-based access" status="enabled" />
        <PolicyRow name="Risk-based MFA" status="enabled" />
        <PolicyRow name="Require compliant devices" status="enabled" />
      </div>
    </div>
  )
}

function ZeroTrustView() {
  return (
    <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Zero Trust Maturity</h2>
      <div className="space-y-4">
        <MaturityBar category="Identity" level={3} />
        <MaturityBar category="Devices" level={2} />
        <MaturityBar category="Networks" level={2} />
        <MaturityBar category="Applications" level={3} />
        <MaturityBar category="Data" level={2} />
      </div>
    </div>
  )
}

function AccessReviewsView() {
  return (
    <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Active Access Reviews</h2>
      <div className="space-y-3">
        <ReviewRow name="Q1 Privileged Review" progress={67} dueDate="7 days" />
        <ReviewRow name="Monthly Guest Review" progress={45} dueDate="14 days" />
        <ReviewRow name="Service Principal Audit" progress={23} dueDate="21 days" />
      </div>
    </div>
  )
}

// Component library
function MetricCard({ title, value, icon: Icon }: {
  title: string
  value: string | number
  icon: React.ElementType
}) {
  return (
    <div className="bg-white dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">{title}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
        </div>
        <Icon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
      </div>
    </div>
  )
}

function AlertItem({ title, action }: {
  title: string
  action: string
}) {
  return (
    <div className="flex justify-between items-center p-3 bg-gray-100 dark:bg-gray-900/50 rounded">
      <span>{title}</span>
      <button type="button" className="text-xs px-3 py-1 bg-orange-600 hover:bg-orange-700 rounded">
        {action}
      </button>
    </div>
  )
}

function UserRow({ type, count }: {
  type: string
  count: number
}) {
  return (
    <div className="flex justify-between p-2 hover:bg-gray-200 dark:hover:bg-gray-800/50 rounded">
      <span>{type}</span>
      <span className="font-bold">{count}</span>
    </div>
  )
}

function AuthRow({ method, percentage }: {
  method: string
  percentage: number
}) {
  return (
    <div className="flex justify-between p-2 hover:bg-gray-200 dark:hover:bg-gray-800/50 rounded">
      <span>{method}</span>
      <span className="font-bold">{percentage}%</span>
    </div>
  )
}

function RoleCard({ role, count, risk }: {
  role: string
  count: number
  risk: 'Critical' | 'High' | 'Low'
}) {
  const riskColors = {
    Critical: 'border-red-500',
    High: 'border-orange-500',
    Low: 'border-green-500'
  }
  
  return (
    <div className={`p-4 rounded-lg border ${riskColors[risk]} bg-white dark:bg-gray-900/50`}>
      <h3 className="font-medium">{role}</h3>
      <p className="text-2xl font-bold mt-1">{count} users</p>
      <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">Risk: {risk}</p>
    </div>
  )
}

function PIMCard({ title, value }: {
  title: string
  value: string | number
}) {
  return (
    <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg p-4">
      <p className="text-sm text-gray-600 dark:text-gray-400">{title}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
    </div>
  )
}

function ElevationItem({ user, role, expires }: {
  user: string
  role: string
  expires: string
}) {
  return (
    <div className="flex justify-between p-3 bg-gray-200 dark:bg-gray-800/50 rounded">
      <div>
        <p className="font-medium">{user}</p>
        <p className="text-sm text-orange-600 dark:text-orange-400">{role}</p>
      </div>
      <div className="text-right">
        <p className="text-sm text-yellow-400">Expires: {expires}</p>
        <button type="button" className="text-xs px-2 py-1 bg-red-600 hover:bg-red-700 rounded mt-1">
          Revoke
        </button>
      </div>
    </div>
  )
}

function PolicyRow({ name, status }: {
  name: string
  status: 'enabled' | 'disabled'
}) {
  return (
    <div className="flex justify-between p-3 bg-gray-100 dark:bg-gray-800/50 rounded">
      <span>{name}</span>
      <span className={`text-xs px-2 py-1 rounded ${
        status === 'enabled' ? 'bg-green-100 dark:bg-green-900/50 text-green-600 dark:text-green-400' : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
      }`}>
        {status}
      </span>
    </div>
  )
}

function MaturityBar({ category, level }: {
  category: string
  level: number
}) {
  return (
    <div>
      <div className="flex justify-between mb-2">
        <span>{category}</span>
        <span className="text-sm text-gray-600 dark:text-gray-400">Level {level}/5</span>
      </div>
      <div className="bg-gray-300 dark:bg-gray-800 rounded-full h-2">
        <div 
          className="bg-gradient-to-r from-orange-500 to-green-500 h-2 rounded-full"
          style={{ width: `${(level / 5) * 100}%` }}
        />
      </div>
    </div>
  )
}

function ReviewRow({ name, progress, dueDate }: {
  name: string
  progress: number
  dueDate: string
}) {
  return (
    <div className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded">
      <div className="flex justify-between mb-2">
        <span className="font-medium">{name}</span>
        <span className="text-sm text-yellow-600 dark:text-yellow-400">Due: {dueDate}</span>
      </div>
      <div className="bg-gray-300 dark:bg-gray-700 rounded-full h-2">
        <div className="bg-blue-500 h-2 rounded-full" style={{ width: `${progress}%` }} />
      </div>
      <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">{progress}% complete</p>
    </div>
  )
}