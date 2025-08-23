'use client'

import { useEffect, useState } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { 
  Activity, Server, Database, HardDrive, Network, Cloud,
  Cpu, MemoryStick, Zap, AlertCircle, CheckCircle, 
  BarChart3, GitBranch, Package, Settings, Play,
  RefreshCw, Terminal, Clock, TrendingUp, AlertTriangle
} from 'lucide-react'
import { toast } from '@/hooks/useToast'
import type {
  MetricCardProps,
  AlertItemProps,
  ResourceRowProps,
  AutomationRowProps,
  ResourceCardProps,
  ResourceTableRowProps,
  OptimizationItemProps,
  MonitoringCardProps,
  UtilizationBarProps,
  HealthItemProps,
  MonitorRowProps,
  AutomationCardProps,
  RunItemProps,
  PipelineItemProps,
  JobItemProps
} from './type-fixes'

export default function OperationsHub() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    const tab = searchParams.get('tab')
    if (tab && ['overview','resources','monitoring','automation'].includes(tab)) {
      setActiveTab(tab)
    }
  }, [searchParams])

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'resources', label: 'Resources', icon: Server },
    { id: 'monitoring', label: 'Monitoring', icon: BarChart3 },
    { id: 'automation', label: 'Automation', icon: Zap }
  ]

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-white">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-gray-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <h1 className="text-2xl font-bold">Operations Center</h1>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Infrastructure resources, monitoring, and automation in one place
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
                    router.replace(`/operations?${params.toString()}`)
                  }}
                  className={`
                    flex items-center gap-2 px-4 py-3 border-b-2 transition-colors
                    ${activeTab === tab.id 
                      ? 'border-blue-500 text-white' 
                      : 'border-transparent text-gray-400 hover:text-white'}
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
        {activeTab === 'overview' && <OperationsOverview />}
        {activeTab === 'resources' && <ResourcesView />}
        {activeTab === 'monitoring' && <MonitoringView />}
        {activeTab === 'automation' && <AutomationView />}
      </div>
    </div>
  )
}

function OperationsOverview() {
  return (
    <div className="space-y-6">
      {/* Health Status */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          title="System Health"
          value="98.5%"
          trend="+0.5%"
          status="good"
          icon={Activity}
        />
        <MetricCard
          title="Active Resources"
          value="342"
          trend="+12"
          status="neutral"
          icon={Server}
        />
        <MetricCard
          title="Active Alerts"
          value="7"
          trend="-3"
          status="warning"
          icon={AlertCircle}
        />
        <MetricCard
          title="Automation Runs"
          value="1,247"
          trend="+89"
          status="good"
          icon={Zap}
        />
      </div>

      {/* Critical Alerts */}
      <div className="bg-yellow-900/20 border border-yellow-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold text-yellow-400 mb-4 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5" />
          Active Operational Issues
        </h2>
        <div className="space-y-3">
          <AlertItem
            title="High CPU usage on prod-api-01"
            severity="high"
            time="10 min ago"
            action="Investigate"
          />
          <AlertItem
            title="Database connection pool exhausted"
            severity="medium"
            time="25 min ago"
            action="Scale"
          />
          <AlertItem
            title="Backup job failed for analytics-db"
            severity="low"
            time="2 hours ago"
            action="Retry"
          />
        </div>
      </div>

      {/* Resource Distribution */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">Resource Distribution</h2>
          <div className="space-y-3">
            <ResourceRow type="Virtual Machines" count={87} percentage={65} />
            <ResourceRow type="Databases" count={23} percentage={85} />
            <ResourceRow type="Storage Accounts" count={45} percentage={42} />
            <ResourceRow type="Networks" count={12} percentage={78} />
            <ResourceRow type="Containers" count={156} percentage={92} />
          </div>
        </div>

        <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">Automation Status</h2>
          <div className="space-y-3">
            <AutomationRow name="CI/CD Pipelines" status="running" count={12} />
            <AutomationRow name="Backup Jobs" status="scheduled" count={24} />
            <AutomationRow name="Scaling Rules" status="active" count={8} />
            <AutomationRow name="Runbooks" status="idle" count={31} />
          </div>
        </div>
      </div>
    </div>
  )
}

function ResourcesView() {
  return (
    <div className="space-y-6">
      {/* Resource Summary */}
      <div className="grid grid-cols-5 gap-4">
        <ResourceCard
          icon={Server}
          type="VMs"
          count={87}
          status="running"
          health="healthy"
        />
        <ResourceCard
          icon={Database}
          type="Databases"
          count={23}
          status="active"
          health="healthy"
        />
        <ResourceCard
          icon={HardDrive}
          type="Storage"
          count={45}
          status="online"
          health="warning"
        />
        <ResourceCard
          icon={Network}
          type="Networks"
          count={12}
          status="connected"
          health="healthy"
        />
        <ResourceCard
          icon={Cloud}
          type="Services"
          count={67}
          status="running"
          health="healthy"
        />
      </div>

      {/* Resource Details */}
      <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Resource Inventory</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-sm text-gray-400 border-b border-gray-800">
                <th className="pb-3">Name</th>
                <th className="pb-3">Type</th>
                <th className="pb-3">Region</th>
                <th className="pb-3">Status</th>
                <th className="pb-3">Cost/Month</th>
                <th className="pb-3">Actions</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              <ResourceTableRow
                name="prod-api-01"
                type="VM"
                region="East US"
                status="running"
                cost="$450"
              />
              <ResourceTableRow
                name="prod-db-primary"
                type="Database"
                region="East US"
                status="active"
                cost="$1,200"
              />
              <ResourceTableRow
                name="storage-backup-01"
                type="Storage"
                region="West US"
                status="online"
                cost="$89"
              />
              <ResourceTableRow
                name="prod-vnet-01"
                type="Network"
                region="East US"
                status="connected"
                cost="$25"
              />
            </tbody>
          </table>
        </div>
      </div>

      {/* Resource Optimization */}
      <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Optimization Recommendations</h2>
        <div className="space-y-3">
          <OptimizationItem
            resource="prod-api-02"
            issue="Oversized - using 20% CPU"
            recommendation="Downsize to D2v3"
            savings="$180/mo"
          />
          <OptimizationItem
            resource="test-db-01"
            issue="Idle for 30+ days"
            recommendation="Delete or archive"
            savings="$450/mo"
          />
          <OptimizationItem
            resource="storage-logs-01"
            issue="90% unused capacity"
            recommendation="Reduce storage tier"
            savings="$67/mo"
          />
        </div>
      </div>
    </div>
  )
}

function MonitoringView() {
  return (
    <div className="space-y-6">
      {/* Monitoring Overview */}
      <div className="grid grid-cols-4 gap-4">
        <MonitoringCard
          title="Uptime"
          value="99.95%"
          period="30 days"
          status="good"
        />
        <MonitoringCard
          title="Response Time"
          value="142ms"
          period="P95"
          status="good"
        />
        <MonitoringCard
          title="Error Rate"
          value="0.03%"
          period="24 hours"
          status="good"
        />
        <MonitoringCard
          title="Throughput"
          value="1.2M req/h"
          period="Average"
          status="neutral"
        />
      </div>

      {/* Performance Metrics */}
      <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Performance Metrics</h2>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <h3 className="text-sm font-medium text-gray-400 mb-3">Resource Utilization</h3>
            <div className="space-y-3">
              <UtilizationBar label="CPU" value={68} threshold={80} />
              <UtilizationBar label="Memory" value={72} threshold={85} />
              <UtilizationBar label="Disk I/O" value={45} threshold={70} />
              <UtilizationBar label="Network" value={52} threshold={75} />
            </div>
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-400 mb-3">Application Health</h3>
            <div className="space-y-3">
              <HealthItem service="API Gateway" status="healthy" latency="45ms" />
              <HealthItem service="Database" status="healthy" latency="12ms" />
              <HealthItem service="Cache" status="degraded" latency="89ms" />
              <HealthItem service="Queue" status="healthy" latency="23ms" />
            </div>
          </div>
        </div>
      </div>

      {/* Active Monitors */}
      <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Active Monitors</h2>
        <div className="space-y-3">
          <MonitorRow
            name="API Endpoint Health"
            type="HTTP"
            frequency="1 min"
            status="passing"
          />
          <MonitorRow
            name="Database Connection"
            type="TCP"
            frequency="30 sec"
            status="passing"
          />
          <MonitorRow
            name="SSL Certificate"
            type="Certificate"
            frequency="1 hour"
            status="warning"
          />
          <MonitorRow
            name="Disk Space"
            type="Metric"
            frequency="5 min"
            status="passing"
          />
        </div>
      </div>
    </div>
  )
}

function AutomationView() {
  return (
    <div className="space-y-6">
      {/* Automation Summary */}
      <div className="grid grid-cols-4 gap-4">
        <AutomationCard
          title="Active Workflows"
          value="47"
          icon={GitBranch}
          status="active"
        />
        <AutomationCard
          title="Runs Today"
          value="1,247"
          icon={Play}
          status="success"
        />
        <AutomationCard
          title="Success Rate"
          value="98.2%"
          icon={CheckCircle}
          status="good"
        />
        <AutomationCard
          title="Avg Duration"
          value="3.2 min"
          icon={Clock}
          status="neutral"
        />
      </div>

      {/* Recent Automation Runs */}
      <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Recent Automation Runs</h2>
        <div className="space-y-3">
          <RunItem
            workflow="Deploy to Production"
            trigger="Manual"
            status="success"
            duration="4m 32s"
            time="10 min ago"
          />
          <RunItem
            workflow="Database Backup"
            trigger="Schedule"
            status="success"
            duration="12m 18s"
            time="1 hour ago"
          />
          <RunItem
            workflow="Auto-scaling"
            trigger="Metric"
            status="success"
            duration="45s"
            time="2 hours ago"
          />
          <RunItem
            workflow="Security Scan"
            trigger="Schedule"
            status="failed"
            duration="8m 12s"
            time="3 hours ago"
          />
        </div>
      </div>

      {/* Automation Workflows */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">CI/CD Pipelines</h2>
          <div className="space-y-3">
            <PipelineItem
              name="Frontend Build"
              branch="main"
              status="running"
              progress={67}
            />
            <PipelineItem
              name="Backend API"
              branch="develop"
              status="queued"
              progress={0}
            />
            <PipelineItem
              name="Infrastructure"
              branch="main"
              status="success"
              progress={100}
            />
          </div>
        </div>

        <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">Scheduled Jobs</h2>
          <div className="space-y-3">
            <JobItem
              name="Daily Backup"
              schedule="0 2 * * *"
              nextRun="in 8 hours"
              enabled={true}
            />
            <JobItem
              name="Weekly Report"
              schedule="0 9 * * MON"
              nextRun="in 3 days"
              enabled={true}
            />
            <JobItem
              name="Cleanup Logs"
              schedule="0 0 * * *"
              nextRun="in 6 hours"
              enabled={false}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

// Reusable Components
function MetricCard({ title, value, trend, status, icon: Icon }: MetricCardProps) {
  const statusColors = {
    good: 'text-green-400 bg-green-900/20',
    warning: 'text-yellow-400 bg-yellow-900/20',
    critical: 'text-red-400 bg-red-900/20',
    neutral: 'text-gray-400 bg-gray-800/50'
  }

  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-400">{title}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
          <p className="text-sm text-gray-500 mt-1">{trend}</p>
        </div>
        <div className={`p-2 rounded-lg ${statusColors[status]}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </div>
  )
}

function AlertItem({ title, severity, time, action }: AlertItemProps) {
  const severityColors = {
    high: 'text-red-400',
    medium: 'text-yellow-400',
    low: 'text-blue-400'
  }

  return (
    <div className="flex items-center justify-between p-3 bg-gray-900/50 rounded">
      <div className="flex items-center gap-3">
        <AlertCircle className={`w-4 h-4 ${severityColors[severity]}`} />
        <div>
          <p className="font-medium">{title}</p>
          <p className="text-xs text-gray-400">{time}</p>
        </div>
      </div>
      <button type="button" className="text-xs px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded">
        {action}
      </button>
    </div>
  )
}

function ResourceRow({ type, count, percentage }: ResourceRowProps) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <span className="font-medium">{type}</span>
        <span className="text-sm text-gray-400">({count})</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-24 bg-gray-800 rounded-full h-2">
          <div 
            className="bg-blue-500 h-2 rounded-full"
            style={{ width: `${percentage}%` }}
          />
        </div>
        <span className="text-sm text-gray-400 w-10 text-right">{percentage}%</span>
      </div>
    </div>
  )
}

function AutomationRow({ name, status, count }: AutomationRowProps) {
  const statusColors = {
    running: 'text-green-400',
    scheduled: 'text-blue-400',
    active: 'text-yellow-400',
    idle: 'text-gray-400'
  }

  return (
    <div className="flex items-center justify-between p-2 hover:bg-gray-800/50 rounded">
      <span className="font-medium">{name}</span>
      <div className="flex items-center gap-3">
        <span className="text-sm text-gray-400">{count}</span>
        <span className={`text-sm ${statusColors[status]}`}>{status}</span>
      </div>
    </div>
  )
}

function ResourceCard({ icon: Icon, type, count, status, health }: ResourceCardProps) {
  const healthColors = {
    healthy: 'border-green-500',
    warning: 'border-yellow-500',
    critical: 'border-red-500'
  }

  return (
    <div className={`bg-gray-900/50 rounded-lg border ${healthColors[health]} p-4`}>
      <div className="flex items-center justify-between mb-2">
        <Icon className="w-5 h-5 text-gray-400" />
        <span className={`text-xs px-2 py-1 rounded bg-gray-800 text-gray-300`}>
          {status}
        </span>
      </div>
      <p className="text-2xl font-bold">{count}</p>
      <p className="text-sm text-gray-400">{type}</p>
    </div>
  )
}

function ResourceTableRow({ name, type, region, status, cost }: ResourceTableRowProps) {
  return (
    <tr className="border-b border-gray-800/50 hover:bg-gray-900/50">
      <td className="py-3 font-medium">{name}</td>
      <td className="py-3 text-gray-400">{type}</td>
      <td className="py-3 text-gray-400">{region}</td>
      <td className="py-3">
        <span className={`text-xs px-2 py-1 rounded ${
          status === 'running' || status === 'active' || status === 'connected' || status === 'online'
            ? 'bg-green-900/50 text-green-400' 
            : 'bg-gray-800 text-gray-400'
        }`}>
          {status}
        </span>
      </td>
      <td className="py-3 text-gray-400">{cost}</td>
      <td className="py-3">
        <button
          type="button"
          className="text-xs px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded mr-2"
          onClick={() => toast({ title: 'Resource', description: `Viewing ${name}` })}
        >
          View
        </button>
        <button
          type="button"
          className="text-xs px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded"
          onClick={() => toast({ title: 'Manage', description: `Managing ${name}` })}
        >
          Manage
        </button>
      </td>
    </tr>
  )
}

function OptimizationItem({ resource, issue, recommendation, savings }: OptimizationItemProps) {
  return (
    <div className="flex items-start justify-between p-3 bg-gray-800/50 rounded">
      <div>
        <p className="font-medium">{resource}</p>
        <p className="text-sm text-gray-400 mt-1">{issue}</p>
        <p className="text-sm text-blue-400 mt-1">{recommendation}</p>
      </div>
      <div className="text-right">
        <p className="text-lg font-bold text-green-400">{savings}</p>
        <button
          type="button"
          className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded mt-1"
          onClick={() => toast({ title: 'Optimize', description: `Optimizing ${resource}` })}
        >
          Optimize
        </button>
      </div>
    </div>
  )
}

function MonitoringCard({ title, value, period, status }: MonitoringCardProps) {
  const statusColors = {
    good: 'text-green-400',
    warning: 'text-yellow-400',
    critical: 'text-red-400',
    neutral: 'text-gray-400'
  }

  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-4">
      <p className="text-sm text-gray-400">{title}</p>
      <p className={`text-2xl font-bold mt-1 ${statusColors[status]}`}>{value}</p>
      <p className="text-xs text-gray-500 mt-1">{period}</p>
    </div>
  )
}

function UtilizationBar({ label, value, threshold }: UtilizationBarProps) {
  const isWarning = value > threshold

  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="text-sm">{label}</span>
        <span className={`text-sm ${isWarning ? 'text-yellow-400' : 'text-gray-400'}`}>
          {value}%
        </span>
      </div>
      <div className="bg-gray-800 rounded-full h-2">
        <div 
          className={`h-2 rounded-full ${isWarning ? 'bg-yellow-500' : 'bg-blue-500'}`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  )
}

function HealthItem({ service, status, latency }: HealthItemProps) {
  const statusColors = {
    healthy: 'bg-green-900/50 text-green-400',
    degraded: 'bg-yellow-900/50 text-yellow-400',
    unhealthy: 'bg-red-900/50 text-red-400'
  }

  return (
    <div className="flex items-center justify-between p-2 hover:bg-gray-800/50 rounded">
      <span>{service}</span>
      <div className="flex items-center gap-3">
        <span className="text-sm text-gray-400">{latency}</span>
        <span className={`text-xs px-2 py-1 rounded ${statusColors[status]}`}>
          {status}
        </span>
      </div>
    </div>
  )
}

function MonitorRow({ name, type, frequency, status }: MonitorRowProps) {
  const statusColors = {
    passing: 'text-green-400',
    warning: 'text-yellow-400',
    failing: 'text-red-400'
  }

  return (
    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
      <div>
        <p className="font-medium">{name}</p>
        <p className="text-xs text-gray-400">{type} • {frequency}</p>
      </div>
      <span className={`text-sm ${statusColors[status]}`}>{status}</span>
    </div>
  )
}

function AutomationCard({ title, value, icon: Icon, status }: AutomationCardProps) {
  const statusColors = {
    active: 'text-blue-400',
    success: 'text-green-400',
    good: 'text-green-400',
    neutral: 'text-gray-400'
  }

  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-2">
        <Icon className="w-5 h-5 text-gray-400" />
        <span className={`text-sm ${statusColors[status]}`}>{status}</span>
      </div>
      <p className="text-2xl font-bold">{value}</p>
      <p className="text-sm text-gray-400">{title}</p>
    </div>
  )
}

function RunItem({ workflow, trigger, status, duration, time }: RunItemProps) {
  const statusColors = {
    success: 'bg-green-900/50 text-green-400',
    failed: 'bg-red-900/50 text-red-400',
    running: 'bg-blue-900/50 text-blue-400'
  }

  return (
    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
      <div>
        <p className="font-medium">{workflow}</p>
        <p className="text-xs text-gray-400">Trigger: {trigger} • Duration: {duration}</p>
      </div>
      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-500">{time}</span>
        <span className={`text-xs px-2 py-1 rounded ${statusColors[status]}`}>
          {status}
        </span>
      </div>
    </div>
  )
}

function PipelineItem({ name, branch, status, progress }: PipelineItemProps) {
  const statusColors = {
    running: 'text-blue-400',
    queued: 'text-yellow-400',
    success: 'text-green-400',
    failed: 'text-red-400'
  }

  return (
    <div className="p-3 bg-gray-800/50 rounded">
      <div className="flex justify-between mb-2">
        <div>
          <p className="font-medium">{name}</p>
          <p className="text-xs text-gray-400">Branch: {branch}</p>
        </div>
        <span className={`text-sm ${statusColors[status]}`}>{status}</span>
      </div>
      {status === 'running' && (
        <div className="bg-gray-700 rounded-full h-2">
          <div className="bg-blue-500 h-2 rounded-full" style={{ width: `${progress}%` }} />
        </div>
      )}
    </div>
  )
}

function JobItem({ name, schedule, nextRun, enabled }: JobItemProps) {
  return (
    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
      <div>
        <p className="font-medium">{name}</p>
        <p className="text-xs text-gray-400 font-mono">{schedule}</p>
        <p className="text-xs text-gray-500">{nextRun}</p>
      </div>
      <button type="button" className={`text-xs px-3 py-1 rounded ${
        enabled 
          ? 'bg-green-900/50 text-green-400' 
          : 'bg-gray-800 text-gray-400'
      }`}>
        {enabled ? 'Enabled' : 'Disabled'}
      </button>
    </div>
  )
}