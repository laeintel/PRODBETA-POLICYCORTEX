'use client'

import { useEffect, useState } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { 
  Shield, AlertTriangle, DollarSign, FileCheck, 
  TrendingUp, TrendingDown, CheckCircle, XCircle,
  ArrowRight, BarChart3, AlertCircle, FileText
} from 'lucide-react'

export default function GovernanceHub() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    const tab = searchParams.get('tab')
    if (tab && ['overview','compliance','risk','cost'].includes(tab)) {
      setActiveTab(tab)
    }
  }, [searchParams])

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'compliance', label: 'Policies & Compliance', icon: FileCheck },
    { id: 'risk', label: 'Risk Management', icon: AlertTriangle },
    { id: 'cost', label: 'Cost Optimization', icon: DollarSign }
  ]

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-white">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-gray-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <h1 className="text-2xl font-bold">Governance Hub</h1>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Unified policies, compliance, risk management, and cost optimization
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-800 bg-gray-100/30 dark:bg-gray-900/30">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-6">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button type="button"
                  key={tab.id}
                  onClick={() => {
                    setActiveTab(tab.id)
                    const params = new URLSearchParams(searchParams.toString())
                    params.set('tab', tab.id)
                    router.replace(`/governance?${params.toString()}`)
                  }}
                  className={`
                    flex items-center gap-2 px-4 py-3 border-b-2 transition-colors
                    ${activeTab === tab.id 
                      ? 'border-blue-500 text-gray-900 dark:text-white' 
                      : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'}
                  `}>
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6">
        {activeTab === 'overview' && <GovernanceOverview />}
        {activeTab === 'compliance' && <ComplianceView />}
        {activeTab === 'risk' && <RiskView />}
        {activeTab === 'cost' && <CostView />}
      </div>
    </div>
  )
}

function GovernanceOverview() {
  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          title="Overall Compliance"
          value="94%"
          trend="+2%"
          status="good"
          icon={CheckCircle}
        />
        <MetricCard
          title="Active Policies"
          value="127"
          trend="+5"
          status="neutral"
          icon={FileText}
        />
        <MetricCard
          title="Risk Score"
          value="Medium"
          trend="Stable"
          status="warning"
          icon={AlertTriangle}
        />
        <MetricCard
          title="Monthly Savings"
          value="$45K"
          trend="+12%"
          status="good"
          icon={DollarSign}
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-3 gap-4">
        <QuickAction
          title="Review Expiring Policies"
          description="2 policies need renewal"
          urgency="high"
          href="/governance/compliance"
        />
        <QuickAction
          title="Address Critical Risks"
          description="3 high-priority risks"
          urgency="high"
          href="/governance/risk"
        />
        <QuickAction
          title="Optimize Costs"
          description="$12K potential savings"
          urgency="medium"
          href="/governance/cost"
        />
      </div>

      {/* Recent Activity */}
      <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Recent Governance Activity</h2>
        <div className="space-y-3">
          <ActivityItem
            title="Policy Updated"
            description="Data retention policy modified for GDPR compliance"
            time="2 hours ago"
            type="policy"
          />
          <ActivityItem
            title="Risk Detected"
            description="Elevated permissions on production resources"
            time="5 hours ago"
            type="risk"
          />
          <ActivityItem
            title="Cost Alert"
            description="Unexpected spike in compute costs (+$3K)"
            time="1 day ago"
            type="cost"
          />
        </div>
      </div>
    </div>
  )
}

function ComplianceView() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-6">
        {/* Compliance Frameworks */}
        <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">Compliance Frameworks</h2>
          <div className="space-y-3">
            <FrameworkItem name="SOC 2" status="compliant" coverage="98%" />
            <FrameworkItem name="ISO 27001" status="compliant" coverage="95%" />
            <FrameworkItem name="GDPR" status="compliant" coverage="100%" />
            <FrameworkItem name="HIPAA" status="partial" coverage="87%" />
            <FrameworkItem name="PCI DSS" status="compliant" coverage="92%" />
          </div>
        </div>

        {/* Policy Status */}
        <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">Policy Status</h2>
          <div className="space-y-3">
            <PolicyItem name="Data Encryption" status="active" violations={0} />
            <PolicyItem name="Access Control" status="active" violations={2} />
            <PolicyItem name="Network Security" status="active" violations={0} />
            <PolicyItem name="Backup Policy" status="expiring" violations={1} />
            <PolicyItem name="Incident Response" status="active" violations={0} />
          </div>
        </div>
      </div>

      {/* Violations */}
      <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Active Violations</h2>
        <div className="space-y-2">
          <ViolationItem
            policy="Access Control"
            resource="prod-db-01"
            severity="high"
            age="2 days"
          />
          <ViolationItem
            policy="Access Control"
            resource="staging-api"
            severity="medium"
            age="5 days"
          />
          <ViolationItem
            policy="Backup Policy"
            resource="analytics-cluster"
            severity="low"
            age="1 week"
          />
        </div>
      </div>
    </div>
  )
}

function RiskView() {
  return (
    <div className="space-y-6">
      {/* Risk Matrix */}
      <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Risk Assessment Matrix</h2>
        <div className="grid grid-cols-3 gap-4">
          <RiskCategory
            category="Security"
            level="High"
            risks={5}
            trend="increasing"
          />
          <RiskCategory
            category="Compliance"
            level="Low"
            risks={2}
            trend="stable"
          />
          <RiskCategory
            category="Operational"
            level="Medium"
            risks={8}
            trend="decreasing"
          />
        </div>
      </div>

      {/* Critical Risks */}
      <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Critical Risks Requiring Action</h2>
        <div className="space-y-3">
          <RiskItem
            title="Unencrypted data in transit"
            impact="High"
            likelihood="Medium"
            mitigation="Enable TLS on all endpoints"
          />
          <RiskItem
            title="Excessive permissions on service accounts"
            impact="High"
            likelihood="High"
            mitigation="Implement least privilege access"
          />
          <RiskItem
            title="Missing disaster recovery plan"
            impact="Critical"
            likelihood="Low"
            mitigation="Create and test DR procedures"
          />
        </div>
      </div>
    </div>
  )
}

function CostView() {
  return (
    <div className="space-y-6">
      {/* Cost Summary */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          title="Current Month"
          value="$127K"
          trend="-8%"
          status="good"
          icon={DollarSign}
        />
        <MetricCard
          title="Projected"
          value="$135K"
          trend="+6%"
          status="warning"
          icon={TrendingUp}
        />
        <MetricCard
          title="Budget"
          value="$150K"
          trend="Under"
          status="good"
          icon={BarChart3}
        />
        <MetricCard
          title="Savings YTD"
          value="$245K"
          trend="+32%"
          status="good"
          icon={TrendingDown}
        />
      </div>

      {/* Optimization Opportunities */}
      <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Cost Optimization Opportunities</h2>
        <div className="space-y-3">
          <OpportunityItem
            title="Rightsize underutilized VMs"
            savings="$12K/mo"
            effort="Low"
            impact="High"
          />
          <OpportunityItem
            title="Purchase reserved instances"
            savings="$8K/mo"
            effort="Medium"
            impact="High"
          />
          <OpportunityItem
            title="Delete unattached disks"
            savings="$3K/mo"
            effort="Low"
            impact="Medium"
          />
          <OpportunityItem
            title="Optimize data transfer costs"
            savings="$5K/mo"
            effort="High"
            impact="Medium"
          />
        </div>
      </div>
    </div>
  )
}

// Reusable Components
function MetricCard({ title, value, trend, status, icon: Icon }: {
  title: string
  value: string | number
  trend: string
  status: 'good' | 'warning' | 'critical' | 'neutral'
  icon: React.ElementType
}) {
  const statusColors = {
    good: 'text-green-400 bg-green-900/20',
    warning: 'text-yellow-400 bg-yellow-900/20',
    critical: 'text-red-400 bg-red-900/20',
    neutral: 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-800/50'
  }

  return (
    <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">{title}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{trend}</p>
        </div>
        <div className={`p-2 rounded-lg ${statusColors[status]}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </div>
  )
}

function QuickAction({ title, description, urgency, href }: {
  title: string
  description: string
  urgency: 'high' | 'medium' | 'low'
  href: string
}) {
  const urgencyColors = {
    high: 'border-red-800 bg-red-900/20',
    medium: 'border-yellow-800 bg-yellow-900/20',
    low: 'border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800/50'
  }

  return (
    <Link
      href={href}
      className={`block p-4 rounded-lg border ${urgencyColors[urgency]} hover:bg-gray-200 dark:hover:bg-gray-800/50 transition-colors`}>
      <h3 className="font-medium">{title}</h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{description}</p>
      <ArrowRight className="w-4 h-4 text-gray-500 dark:text-gray-400 mt-2" />
    </Link>
  )
}

function ActivityItem({ title, description, time, type }: {
  title: string
  description: string
  time: string
  type: 'policy' | 'risk' | 'cost'
}) {
  const typeIcons = {
    policy: FileCheck,
    risk: AlertTriangle,
    cost: DollarSign
  }
  const Icon = typeIcons[type]

  return (
    <div className="flex items-start gap-3 p-3 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-800/50 transition-colors">
      <Icon className="w-5 h-5 text-gray-600 dark:text-gray-400 mt-0.5" />
      <div className="flex-1">
        <p className="font-medium">{title}</p>
        <p className="text-sm text-gray-600 dark:text-gray-400">{description}</p>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{time}</p>
      </div>
    </div>
  )
}

function FrameworkItem({ name, status, coverage }: {
  name: string
  status: 'compliant' | 'partial'
  coverage: string
}) {
  return (
    <div className="flex items-center justify-between p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-800/50">
      <span>{name}</span>
      <div className="flex items-center gap-3">
        <span className="text-sm text-gray-600 dark:text-gray-400">{coverage}</span>
        <span className={`text-xs px-2 py-1 rounded ${
          status === 'compliant' ? 'bg-green-900/50 text-green-400' : 'bg-yellow-900/50 text-yellow-400'
        }`}>
          {status}
        </span>
      </div>
    </div>
  )
}

function PolicyItem({ name, status, violations }: {
  name: string
  status: 'active' | 'expiring' | 'expired'
  violations: number
}) {
  return (
    <div className="flex items-center justify-between p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-800/50">
      <span>{name}</span>
      <div className="flex items-center gap-3">
        {violations > 0 && (
          <span className="text-sm text-red-400">{violations} violations</span>
        )}
        <span className={`text-xs px-2 py-1 rounded ${
          status === 'active' ? 'bg-green-900/50 text-green-400' : 
          status === 'expiring' ? 'bg-yellow-900/50 text-yellow-400' :
          'bg-red-900/50 text-red-400'
        }`}>
          {status}
        </span>
      </div>
    </div>
  )
}

function ViolationItem({ policy, resource, severity, age }: {
  policy: string
  resource: string
  severity: 'high' | 'medium' | 'low'
  age: string
}) {
  const severityColors = {
    high: 'text-red-400',
    medium: 'text-yellow-400',
    low: 'text-blue-400'
  }

  return (
    <div className="flex items-center justify-between p-3 bg-gray-100 dark:bg-gray-800/50 rounded">
      <div className="flex items-center gap-4">
        <AlertCircle className={`w-4 h-4 ${severityColors[severity]}`} />
        <div>
          <span className="font-medium">{policy}</span>
          <span className="text-gray-600 dark:text-gray-400 mx-2">→</span>
          <span className="text-gray-700 dark:text-gray-300">{resource}</span>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <span className="text-sm text-gray-600 dark:text-gray-400">{age}</span>
        <button type="button" className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded">
          Remediate
        </button>
      </div>
    </div>
  )
}

function RiskCategory({ category, level, risks, trend }: {
  category: string
  level: 'High' | 'Medium' | 'Low'
  risks: number
  trend: string
}) {
  const levelColors = {
    High: 'border-red-500 bg-red-900/20',
    Medium: 'border-yellow-500 bg-yellow-900/20',
    Low: 'border-green-500 bg-green-900/20'
  }

  return (
    <div className={`p-4 rounded-lg border ${levelColors[level]}`}>
      <h3 className="font-medium">{category}</h3>
      <p className="text-2xl font-bold mt-2">{risks}</p>
      <p className="text-sm text-gray-600 dark:text-gray-400">risks</p>
      <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">Trend: {trend}</p>
    </div>
  )
}

function RiskItem({ title, impact, likelihood, mitigation }: {
  title: string
  impact: string
  likelihood: string
  mitigation: string
}) {
  return (
    <div className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg">
      <h3 className="font-medium">{title}</h3>
      <div className="flex gap-4 mt-2 text-sm">
        <span className="text-gray-600 dark:text-gray-400">Impact: <span className="text-red-400">{impact}</span></span>
        <span className="text-gray-600 dark:text-gray-400">Likelihood: <span className="text-yellow-400">{likelihood}</span></span>
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">Mitigation: {mitigation}</p>
    </div>
  )
}

function OpportunityItem({ title, savings, effort, impact }: {
  title: string
  savings: string
  effort: 'Low' | 'Medium' | 'High'
  impact: string
}) {
  const effortColors = {
    Low: 'text-green-400',
    Medium: 'text-yellow-400',
    High: 'text-red-400'
  }

  return (
    <div className="flex items-center justify-between p-3 bg-gray-100 dark:bg-gray-800/50 rounded hover:bg-gray-200 dark:hover:bg-gray-700/50 transition-colors">
      <div>
        <p className="font-medium">{title}</p>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Effort: <span className={effortColors[effort]}>{effort}</span> • 
          Impact: <span className="text-blue-400 ml-2">{impact}</span>
        </p>
      </div>
      <div className="text-right">
        <p className="text-lg font-bold text-green-400">{savings}</p>
        <button type="button" className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded mt-1">
          Implement
        </button>
      </div>
    </div>
  )
}