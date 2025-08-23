'use client'

import { useEffect, useState } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { 
  Shield, AlertTriangle, DollarSign, FileCheck, 
  TrendingUp, TrendingDown, CheckCircle, XCircle,
  ArrowRight, BarChart3, AlertCircle, FileText,
  ChevronRight, ExternalLink, Activity, Clock,
  Building, Scale, ShieldCheck, Target, Info,
  ArrowLeft, Users, Briefcase, Zap
} from 'lucide-react'
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, AreaChart, Area
} from 'recharts'
import ViewToggle from '@/components/ViewToggle'
import MetricCard from '@/components/MetricCard'
import ChartContainer from '@/components/ChartContainer'
import DataExport from '@/components/DataExport'
import { useViewPreference } from '@/hooks/useViewPreference'

interface GovernanceCard {
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

export default function GovernanceHub() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [activeTab, setActiveTab] = useState('overview')
  const [expandedCard, setExpandedCard] = useState<string | null>(null)

  useEffect(() => {
    const tab = searchParams.get('tab')
    if (tab && ['overview','compliance','risk','cost','policies'].includes(tab)) {
      setActiveTab(tab)
    }
  }, [searchParams])

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'compliance', label: 'Compliance', icon: FileCheck },
    { id: 'risk', label: 'Risk Management', icon: AlertTriangle },
    { id: 'cost', label: 'Cost Optimization', icon: DollarSign },
    { id: 'policies', label: 'Policy Management', icon: Shield }
  ]

  const governanceCards: GovernanceCard[] = [
    {
      id: 'compliance',
      title: 'Compliance Management',
      description: 'Track and manage regulatory compliance across all cloud resources',
      icon: FileCheck,
      href: '/governance/compliance',
      color: 'blue',
      stats: [
        { label: 'Overall Score', value: '94%', trend: 'up', status: 'good' },
        { label: 'Active Policies', value: 127 },
        { label: 'Violations', value: 3, status: 'warning' }
      ],
      quickActions: [
        { label: 'Run Compliance Check', href: '/governance/compliance' },
        { label: 'View Reports', href: '/governance/compliance#reports' }
      ]
    },
    {
      id: 'risk',
      title: 'Risk Management',
      description: 'Identify, assess, and mitigate risks across your infrastructure',
      icon: AlertTriangle,
      href: '/governance/risk',
      color: 'orange',
      stats: [
        { label: 'Risk Score', value: 'Low', status: 'good' },
        { label: 'Active Risks', value: 12, trend: 'down' },
        { label: 'Critical', value: 0, status: 'good' }
      ],
      quickActions: [
        { label: 'Risk Assessment', href: '/governance/risk#assessment' },
        { label: 'Mitigation Plans', href: '/governance/risk#mitigation' }
      ]
    },
    {
      id: 'cost',
      title: 'Cost Optimization',
      description: 'Monitor spending, identify savings, and optimize cloud costs',
      icon: DollarSign,
      href: '/governance/cost',
      color: 'green',
      stats: [
        { label: 'Monthly Spend', value: '$42,341', trend: 'down', status: 'good' },
        { label: 'Budget', value: '$50,000' },
        { label: 'Savings', value: '$4,523', trend: 'up' }
      ],
      quickActions: [
        { label: 'Cost Analysis', href: '/governance/cost#analysis' },
        { label: 'Recommendations', href: '/governance/cost#recommendations' }
      ]
    },
    {
      id: 'policies',
      title: 'Policy Management',
      description: 'Define, enforce, and audit governance policies',
      icon: Shield,
      href: '/governance/policies',
      color: 'purple',
      stats: [
        { label: 'Total Policies', value: 234 },
        { label: 'Enforced', value: 198, status: 'good' },
        { label: 'Pending Review', value: 12, status: 'warning' }
      ],
      quickActions: [
        { label: 'Create Policy', href: '/governance/policies#create' },
        { label: 'Policy Library', href: '/governance/policies#library' }
      ]
    }
  ]

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-gray-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold flex items-center space-x-3">
                <Shield className="w-8 h-8 text-blue-500" />
                <span>Governance Hub</span>
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Unified policies, compliance, risk management, and cost optimization
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
        {activeTab === 'overview' && <GovernanceOverview cards={governanceCards} router={router} />}
        {activeTab === 'compliance' && <ComplianceView router={router} />}
        {activeTab === 'risk' && <RiskView router={router} />}
        {activeTab === 'cost' && <CostView router={router} />}
        {activeTab === 'policies' && <PolicyView router={router} />}
      </div>
    </div>
  )
}

function GovernanceOverview({ cards, router }: { cards: GovernanceCard[], router: any }) {
  const { view, setView } = useViewPreference('governance-view', 'cards');

  // Mock data for charts
  const complianceData = [
    { month: 'Jan', compliance: 89, policies: 120, violations: 8 },
    { month: 'Feb', compliance: 92, policies: 123, violations: 6 },
    { month: 'Mar', compliance: 91, policies: 125, violations: 7 },
    { month: 'Apr', compliance: 94, policies: 127, violations: 3 },
    { month: 'May', compliance: 96, policies: 129, violations: 2 },
    { month: 'Jun', compliance: 94, policies: 127, violations: 3 }
  ];

  const riskData = [
    { category: 'Security', high: 2, medium: 8, low: 15 },
    { category: 'Compliance', high: 0, medium: 3, low: 9 },
    { category: 'Operational', high: 1, medium: 5, low: 12 },
    { category: 'Financial', high: 0, medium: 2, low: 7 }
  ];

  const costData = [
    { month: 'Jan', spend: 145000, budget: 150000, savings: 8000 },
    { month: 'Feb', spend: 138000, budget: 150000, savings: 12000 },
    { month: 'Mar', spend: 142000, budget: 150000, savings: 15000 },
    { month: 'Apr', spend: 135000, budget: 150000, savings: 18000 },
    { month: 'May', spend: 127000, budget: 150000, savings: 23000 },
    { month: 'Jun', spend: 127000, budget: 150000, savings: 23000 }
  ];

  const policyDistribution = [
    { name: 'Security', value: 89, color: '#3B82F6' },
    { name: 'Compliance', value: 67, color: '#10B981' },
    { name: 'Operational', value: 45, color: '#8B5CF6' },
    { name: 'Cost', value: 33, color: '#F59E0B' }
  ];

  return (
    <div className="space-y-6">
      {/* View Toggle */}
      <div className="flex justify-end">
        <ViewToggle view={view} onViewChange={setView} />
      </div>

      {view === 'cards' ? (
        <>
          {/* Key Metrics Cards */}
          <div className="grid grid-cols-4 gap-4">
            <MetricCard
              title="Overall Compliance"
              value="94%"
              change={2}
              changeLabel="vs last month"
              icon={<CheckCircle className="h-5 w-5 text-blue-600" />}
              sparklineData={complianceData.map(d => d.compliance)}
              onClick={() => router.push('/governance/compliance')}
              status="success"
            />
            <MetricCard
              title="Active Policies"
              value={127}
              change={4}
              changeLabel="this month"
              icon={<FileText className="h-5 w-5 text-gray-600" />}
              sparklineData={complianceData.map(d => d.policies)}
              onClick={() => router.push('/governance/policies')}
              status="neutral"
            />
            <MetricCard
              title="Risk Score"
              value="Low"
              change={-15}
              changeLabel="improvement"
              icon={<AlertTriangle className="h-5 w-5 text-orange-600" />}
              sparklineData={[25, 23, 28, 20, 18, 15]}
              onClick={() => router.push('/governance/risk')}
              status="warning"
            />
            <MetricCard
              title="Monthly Savings"
              value="$45K"
              change={12}
              changeLabel="vs target"
              icon={<DollarSign className="h-5 w-5 text-green-600" />}
              sparklineData={costData.map(d => d.savings / 1000)}
              onClick={() => router.push('/governance/cost')}
              status="success"
            />
          </div>
        </>
      ) : (
        <>
          {/* Visualizations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartContainer 
              title="Compliance Trends" 
              onExport={() => {}}
              onDrillIn={() => router.push('/governance/compliance')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={complianceData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="month" className="text-gray-600 dark:text-gray-400" />
                  <YAxis className="text-gray-600 dark:text-gray-400" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'var(--tw-bg-opacity)', 
                      border: '1px solid var(--tw-border-opacity)',
                      borderRadius: '0.5rem'
                    }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="compliance" stroke="#3B82F6" strokeWidth={2} name="Compliance %" />
                  <Line type="monotone" dataKey="violations" stroke="#EF4444" strokeWidth={2} name="Violations" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer 
              title="Risk Distribution" 
              onExport={() => {}}
              onDrillIn={() => router.push('/governance/risk')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={riskData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="category" className="text-gray-600 dark:text-gray-400" />
                  <YAxis className="text-gray-600 dark:text-gray-400" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="high" stackId="a" fill="#EF4444" name="High Risk" />
                  <Bar dataKey="medium" stackId="a" fill="#F59E0B" name="Medium Risk" />
                  <Bar dataKey="low" stackId="a" fill="#10B981" name="Low Risk" />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer 
              title="Cost Optimization" 
              onExport={() => {}}
              onDrillIn={() => router.push('/governance/cost')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={costData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="month" className="text-gray-600 dark:text-gray-400" />
                  <YAxis className="text-gray-600 dark:text-gray-400" />
                  <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, '']} />
                  <Legend />
                  <Area type="monotone" dataKey="budget" stackId="1" stroke="#6B7280" fill="#6B7280" fillOpacity={0.3} name="Budget" />
                  <Area type="monotone" dataKey="spend" stackId="2" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} name="Actual Spend" />
                  <Area type="monotone" dataKey="savings" stackId="3" stroke="#10B981" fill="#10B981" fillOpacity={0.8} name="Savings" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer 
              title="Policy Distribution" 
              onExport={() => {}}
              onDrillIn={() => router.push('/governance/policies')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={policyDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {policyDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        </>
      )}

      {view === 'cards' && (
        <>
          {/* Main Dashboard Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {cards.map((card) => {
              const Icon = card.icon
              return (
                <div
                  key={card.id}
                  className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 hover:shadow-xl transition-all"
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
                          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                            {card.description}
                          </p>
                        </div>
                      </div>
                      <ExternalLink className="w-4 h-4 text-gray-400" />
                    </div>

                    {/* Stats */}
                    {card.stats && (
                      <div className="grid grid-cols-3 gap-4 mb-4">
                        {card.stats.map((stat, idx) => (
                          <div key={idx} className="text-center">
                            <div className="text-lg font-bold flex items-center justify-center space-x-1">
                              <span className={stat.status === 'warning' ? 'text-yellow-500' : stat.status === 'critical' ? 'text-red-500' : ''}>
                                {stat.value}
                              </span>
                              {stat.trend === 'up' && <TrendingUp className="w-3 h-3 text-green-500" />}
                              {stat.trend === 'down' && <TrendingDown className="w-3 h-3 text-green-500" />}
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
        </>
      )}

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

      {/* Recent Activity - Clickable */}
      <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800">
        <div 
          className="p-6 border-b border-gray-200 dark:border-gray-700 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/30 transition-colors"
          onClick={() => router.push('/audit')}
        >
          <h2 className="text-lg font-semibold flex items-center justify-between">
            <span>Recent Governance Activity</span>
            <ChevronRight className="w-4 h-4 text-gray-400" />
          </h2>
        </div>
        <div className="p-6 space-y-3">
          <div onClick={() => router.push('/governance/policies')} className="cursor-pointer">
            <ActivityItem
              title="Policy Updated"
              description="Data retention policy modified for GDPR compliance"
              time="2 hours ago"
              type="policy"
            />
          </div>
          <div onClick={() => router.push('/governance/risk')} className="cursor-pointer">
            <ActivityItem
              title="Risk Detected"
              description="Elevated permissions on production resources"
              time="5 hours ago"
              type="risk"
            />
          </div>
          <div onClick={() => router.push('/governance/cost')} className="cursor-pointer">
            <ActivityItem
              title="Cost Alert"
              description="Unexpected spike in compute costs (+$3K)"
              time="1 day ago"
              type="cost"
            />
          </div>
          <button
            onClick={() => router.push('/audit')}
            className="w-full mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors text-sm"
          >
            View All Activities
          </button>
        </div>
      </div>
    </div>
  )
}

function ComplianceView({ router }: { router: any }) {
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
            router={router}
          />
          <ViolationItem
            policy="Access Control"
            resource="staging-api"
            severity="medium"
            age="5 days"
            router={router}
          />
          <ViolationItem
            policy="Backup Policy"
            resource="analytics-cluster"
            severity="low"
            age="1 week"
            router={router}
          />
        </div>
      </div>
    </div>
  )
}

function RiskView({ router }: { router: any }) {
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

function CostView({ router }: { router: any }) {
  return (
    <div className="space-y-6">
      {/* Cost Summary */}
      <div className="grid grid-cols-4 gap-4">
        <LegacyMetricCard
          title="Current Month"
          value="$127K"
          trend="-8%"
          status="good"
          icon={DollarSign}
        />
        <LegacyMetricCard
          title="Projected"
          value="$135K"
          trend="+6%"
          status="warning"
          icon={TrendingUp}
        />
        <LegacyMetricCard
          title="Budget"
          value="$150K"
          trend="Under"
          status="good"
          icon={BarChart3}
        />
        <LegacyMetricCard
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
            router={router}
          />
          <OpportunityItem
            title="Purchase reserved instances"
            savings="$8K/mo"
            effort="Medium"
            impact="High"
            router={router}
          />
          <OpportunityItem
            title="Delete unattached disks"
            savings="$3K/mo"
            effort="Low"
            impact="Medium"
            router={router}
          />
          <OpportunityItem
            title="Optimize data transfer costs"
            savings="$5K/mo"
            effort="High"
            impact="Medium"
            router={router}
          />
        </div>
      </div>
    </div>
  )
}

function PolicyView({ router }: { router: any }) {
  return (
    <div className="space-y-6">
      {/* Policy Summary */}
      <div className="grid grid-cols-4 gap-4">
        <LegacyMetricCard
          title="Total Policies"
          value="234"
          trend="+12"
          status="neutral"
          icon={Shield}
        />
        <LegacyMetricCard
          title="Enforced"
          value="198"
          trend="85%"
          status="good"
          icon={ShieldCheck}
        />
        <LegacyMetricCard
          title="Pending Review"
          value="12"
          trend="-3"
          status="warning"
          icon={Clock}
        />
        <LegacyMetricCard
          title="Violations Today"
          value="3"
          trend="-2"
          status="good"
          icon={AlertCircle}
        />
      </div>

      {/* Policy Categories */}
      <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Policy Categories</h2>
        <div className="grid grid-cols-2 gap-4">
          <div 
            onClick={() => router.push('/governance/policies#security')}
            className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:shadow-md hover:border-blue-500 transition-all cursor-pointer"
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium">Security Policies</h3>
              <Shield className="w-5 h-5 text-blue-500" />
            </div>
            <div className="text-2xl font-bold">89</div>
            <div className="text-sm text-gray-500 dark:text-gray-400">12 require attention</div>
          </div>
          <div 
            onClick={() => router.push('/governance/policies#compliance')}
            className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:shadow-md hover:border-green-500 transition-all cursor-pointer"
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium">Compliance Policies</h3>
              <FileCheck className="w-5 h-5 text-green-500" />
            </div>
            <div className="text-2xl font-bold">67</div>
            <div className="text-sm text-gray-500 dark:text-gray-400">All active</div>
          </div>
          <div 
            onClick={() => router.push('/governance/policies#operational')}
            className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:shadow-md hover:border-purple-500 transition-all cursor-pointer"
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium">Operational Policies</h3>
              <Activity className="w-5 h-5 text-purple-500" />
            </div>
            <div className="text-2xl font-bold">45</div>
            <div className="text-sm text-gray-500 dark:text-gray-400">3 pending review</div>
          </div>
          <div 
            onClick={() => router.push('/governance/policies#cost')}
            className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:shadow-md hover:border-yellow-500 transition-all cursor-pointer"
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium">Cost Management</h3>
              <DollarSign className="w-5 h-5 text-yellow-500" />
            </div>
            <div className="text-2xl font-bold">33</div>
            <div className="text-sm text-gray-500 dark:text-gray-400">5 optimizations available</div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-3 gap-4">
        <button
          onClick={() => router.push('/governance/policies#create')}
          className="p-4 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors text-center"
        >
          <Zap className="w-6 h-6 mx-auto mb-2" />
          <div className="font-medium">Create New Policy</div>
          <div className="text-sm opacity-90 mt-1">Define custom governance rules</div>
        </button>
        <button
          onClick={() => router.push('/governance/policies#library')}
          className="p-4 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors text-center"
        >
          <Building className="w-6 h-6 mx-auto mb-2" />
          <div className="font-medium">Policy Library</div>
          <div className="text-sm opacity-90 mt-1">Browse pre-built templates</div>
        </button>
        <button
          onClick={() => router.push('/governance/policies#audit')}
          className="p-4 bg-green-600 hover:bg-green-700 rounded-lg transition-colors text-center"
        >
          <FileText className="w-6 h-6 mx-auto mb-2" />
          <div className="font-medium">Audit Reports</div>
          <div className="text-sm opacity-90 mt-1">View compliance history</div>
        </button>
      </div>
    </div>
  )
}

// Reusable Components
function LegacyMetricCard({ title, value, trend, status, icon: Icon }: {
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

function ViolationItem({ policy, resource, severity, age, router }: {
  policy: string
  resource: string
  severity: 'high' | 'medium' | 'low'
  age: string
  router: any
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
        <button 
          type="button" 
          onClick={() => router.push('/governance/compliance#remediation')}
          className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded"
        >
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

function OpportunityItem({ title, savings, effort, impact, router }: {
  title: string
  savings: string
  effort: 'Low' | 'Medium' | 'High'
  impact: string
  router: any
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
        <button 
          type="button"
          onClick={() => router.push('/governance/cost#implement')}
          className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded mt-1"
        >
          Implement
        </button>
      </div>
    </div>
  )
}