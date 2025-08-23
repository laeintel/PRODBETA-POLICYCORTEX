'use client'

import { useEffect, useState } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { 
  GitBranch, Package, Rocket, Clock, CheckCircle, 
  XCircle, AlertCircle, TrendingUp, Server, Database,
  Activity, BarChart3, GitCommit, Timer, Target, Shield,
  Users, Zap, FileCode, GitPullRequest, GitMerge, Settings
} from 'lucide-react'
import { 
  LineChart, Line, BarChart, Bar, PieChart, Pie,
  AreaChart, Area, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, Cell,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts'

export default function DevOpsHub() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    const tab = searchParams.get('tab')
    if (tab && ['overview','pipelines','releases','artifacts','deployments','builds','repos'].includes(tab)) {
      setActiveTab(tab)
    }
  }, [searchParams])

  const tabs = [
    { id: 'overview', label: 'Overview', icon: GitBranch },
    { id: 'pipelines', label: 'Pipelines', icon: GitBranch },
    { id: 'releases', label: 'Releases', icon: Rocket },
    { id: 'artifacts', label: 'Artifacts', icon: Package },
    { id: 'deployments', label: 'Deployments', icon: Server },
    { id: 'builds', label: 'Build Status', icon: CheckCircle },
    { id: 'repos', label: 'Repositories', icon: Database }
  ]

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-white">
      <div className="border-b border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-gray-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <h1 className="text-2xl font-bold">DevOps & CI/CD</h1>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Continuous integration, deployment pipelines, and release management
          </p>
        </div>
      </div>

      <div className="border-b border-gray-200 dark:border-gray-800 bg-gray-100/30 dark:bg-gray-900/30 overflow-x-auto">
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
                    router.replace(`/devops?${params.toString()}`)
                  }}
                  className={`
                    flex items-center gap-2 px-3 py-3 border-b-2 transition-colors whitespace-nowrap
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
        {activeTab === 'overview' && <DevOpsOverview />}
        {activeTab === 'pipelines' && <PipelinesView />}
        {activeTab === 'releases' && <ReleasesView />}
        {activeTab === 'artifacts' && <ArtifactsView />}
        {activeTab === 'deployments' && <DeploymentsView />}
        {activeTab === 'builds' && <BuildsView />}
        {activeTab === 'repos' && <ReposView />}
      </div>
    </div>
  )
}

function DevOpsOverview() {
  const [buildMetrics, setBuildMetrics] = useState<any[]>([])
  const [velocityData, setVelocityData] = useState<any[]>([])
  const [repoHealth, setRepoHealth] = useState<any[]>([])
  const [leadTimeData, setLeadTimeData] = useState<any[]>([])

  useEffect(() => {
    // Mock data - replace with actual API calls
    setBuildMetrics([
      { date: 'Mon', success: 45, failed: 5, total: 50, successRate: 90 },
      { date: 'Tue', success: 52, failed: 3, total: 55, successRate: 94.5 },
      { date: 'Wed', success: 48, failed: 7, total: 55, successRate: 87.3 },
      { date: 'Thu', success: 56, failed: 4, total: 60, successRate: 93.3 },
      { date: 'Fri', success: 61, failed: 2, total: 63, successRate: 96.8 },
      { date: 'Sat', success: 32, failed: 1, total: 33, successRate: 97 },
      { date: 'Sun', success: 28, failed: 2, total: 30, successRate: 93.3 }
    ])

    setVelocityData([
      { week: 'Week 1', commits: 125, deployments: 15, pullRequests: 42, issues: 38 },
      { week: 'Week 2', commits: 142, deployments: 18, pullRequests: 48, issues: 35 },
      { week: 'Week 3', commits: 138, deployments: 22, pullRequests: 51, issues: 29 },
      { week: 'Week 4', commits: 156, deployments: 25, pullRequests: 56, issues: 24 }
    ])

    setRepoHealth([
      { name: 'Core API', score: 92, coverage: 85, bugs: 3, vulnerabilities: 0 },
      { name: 'Frontend', score: 88, coverage: 78, bugs: 5, vulnerabilities: 1 },
      { name: 'AI Engine', score: 95, coverage: 92, bugs: 1, vulnerabilities: 0 },
      { name: 'GraphQL', score: 86, coverage: 73, bugs: 4, vulnerabilities: 2 },
      { name: 'Edge Functions', score: 90, coverage: 81, bugs: 2, vulnerabilities: 0 }
    ])

    setLeadTimeData([
      { stage: 'Commit', time: 0 },
      { stage: 'Build', time: 5 },
      { stage: 'Test', time: 15 },
      { stage: 'Deploy to Dev', time: 20 },
      { stage: 'Deploy to Staging', time: 45 },
      { stage: 'Deploy to Prod', time: 120 }
    ])
  }, [])

  const radarData = repoHealth.map(repo => ({
    repository: repo.name,
    health: repo.score,
    coverage: repo.coverage,
    quality: 100 - (repo.bugs + repo.vulnerabilities * 2)
  }))

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard 
          title="Build Success Rate" 
          value="94.2%" 
          trend="+5% from last week"
          icon={Target} 
          status="good"
        />
        <MetricCard 
          title="Deployment Frequency" 
          value="3.4/day" 
          trend="+0.5 from average"
          icon={Rocket} 
          status="success"
        />
        <MetricCard 
          title="Lead Time" 
          value="2.5h" 
          trend="-30m improvement"
          icon={Timer} 
          status="improved"
        />
        <MetricCard 
          title="MTTR" 
          value="45m" 
          trend="-15m from last month"
          icon={Shield} 
          status="improved"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Build Metrics Chart */}
        <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-blue-500" />
            Build Success/Failure Trends
          </h2>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={buildMetrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Area type="monotone" dataKey="success" stackId="1" stroke="#10b981" fill="#10b981" fillOpacity={0.6} />
              <Area type="monotone" dataKey="failed" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.6} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Team Velocity Chart */}
        <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-500" />
            Team Velocity
          </h2>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={velocityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="week" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend />
              <Line type="monotone" dataKey="commits" stroke="#3b82f6" strokeWidth={2} />
              <Line type="monotone" dataKey="deployments" stroke="#10b981" strokeWidth={2} />
              <Line type="monotone" dataKey="pullRequests" stroke="#f59e0b" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Repository Health Radar */}
      <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-purple-500" />
          Repository Health Metrics
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="repository" stroke="#9ca3af" />
              <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#9ca3af" />
              <Radar name="Health Score" dataKey="health" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
              <Radar name="Coverage" dataKey="coverage" stroke="#10b981" fill="#10b981" fillOpacity={0.6} />
              <Radar name="Code Quality" dataKey="quality" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.6} />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
          <div className="space-y-3">
            {repoHealth.map((repo, index) => (
              <div key={index} className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-gray-900 dark:text-white font-medium">{repo.name}</h3>
                  <span className={`text-lg font-bold ${
                    repo.score >= 90 ? 'text-green-500' : 
                    repo.score >= 80 ? 'text-yellow-500' : 'text-red-500'
                  }`}>
                    {repo.score}%
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="text-gray-600 dark:text-gray-400">Coverage: <span className="text-gray-900 dark:text-white">{repo.coverage}%</span></div>
                  <div className="text-gray-600 dark:text-gray-400">Bugs: <span className="text-gray-900 dark:text-white">{repo.bugs}</span></div>
                  <div className="text-gray-600 dark:text-gray-400">Vulnerabilities: <span className="text-gray-900 dark:text-white">{repo.vulnerabilities}</span></div>
                  <div className="text-gray-600 dark:text-gray-400">Quality Score: <span className="text-gray-900 dark:text-white">A</span></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Lead Time Analysis */}
      <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Timer className="w-5 h-5 text-orange-500" />
          Lead Time Analysis (Commit to Production)
        </h2>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={leadTimeData} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis type="number" stroke="#9ca3af" />
            <YAxis type="category" dataKey="stage" stroke="#9ca3af" />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
              labelStyle={{ color: '#f3f4f6' }}
              formatter={(value: any) => `${value} minutes`}
            />
            <Bar dataKey="time" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Recent Activity */}
      <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Recent Pipeline Activity</h2>
        <div className="space-y-3">
          <ActivityItem 
            name="Frontend Build"
            branch="main"
            status="success"
            time="2 minutes ago"
            duration="3m 42s"
          />
          <ActivityItem 
            name="API Tests"
            branch="feature/auth"
            status="running"
            time="5 minutes ago"
            duration="--"
          />
          <ActivityItem 
            name="Production Deploy"
            branch="release/v2.1"
            status="failed"
            time="15 minutes ago"
            duration="1m 23s"
          />
          <ActivityItem 
            name="Security Scan"
            branch="main"
            status="success"
            time="1 hour ago"
            duration="8m 15s"
          />
        </div>
      </div>

      {/* Deployment Stats */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <h3 className="text-lg font-semibold mb-4">Deployment Frequency</h3>
          <div className="space-y-3">
            <FrequencyBar env="Production" count={12} percentage={30} />
            <FrequencyBar env="Staging" count={28} percentage={70} />
            <FrequencyBar env="Development" count={156} percentage={100} />
          </div>
        </div>
        <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <h3 className="text-lg font-semibold mb-4">Build Performance</h3>
          <div className="space-y-3">
            <PerformanceRow metric="Average Build Time" value="4m 32s" trend="improving" />
            <PerformanceRow metric="Test Coverage" value="87%" trend="stable" />
            <PerformanceRow metric="Code Quality Score" value="A" trend="improving" />
          </div>
        </div>
      </div>
    </div>
  )
}

function PipelinesView() {
  const [pipelines] = useState([
    {
      id: 'pipe-1',
      name: 'Main Branch CI/CD',
      status: 'running',
      branch: 'main',
      commit: 'abc123f',
      author: 'John Doe',
      duration: '5m 23s',
      stages: 5,
      completedStages: 3,
      successRate: 98
    },
    {
      id: 'pipe-2',
      name: 'PR Validation',
      status: 'succeeded',
      branch: 'feature/auth',
      commit: 'def456g',
      author: 'Jane Smith',
      duration: '3m 45s',
      stages: 4,
      completedStages: 4,
      successRate: 92
    },
    {
      id: 'pipe-3',
      name: 'Security Scanning',
      status: 'failed',
      branch: 'hotfix/security',
      commit: 'ghi789h',
      author: 'Bob Johnson',
      duration: '2m 12s',
      stages: 3,
      completedStages: 2,
      successRate: 100
    },
    {
      id: 'pipe-4',
      name: 'Integration Tests',
      status: 'pending',
      branch: 'develop',
      commit: 'jkl012i',
      author: 'Alice Chen',
      duration: '--',
      stages: 6,
      completedStages: 0,
      successRate: 95
    }
  ])

  const statusColors = {
    running: '#3b82f6',
    succeeded: '#10b981',
    failed: '#ef4444',
    pending: '#f59e0b'
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Clock className="w-4 h-4 animate-spin" />
      case 'succeeded':
        return <CheckCircle className="w-4 h-4" />
      case 'failed':
        return <XCircle className="w-4 h-4" />
      case 'pending':
        return <AlertCircle className="w-4 h-4" />
      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      {/* Pipeline Status Summary */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
          <div className="flex items-center gap-2 text-green-400 mb-2">
            <CheckCircle className="w-5 h-5" />
            <span className="text-sm font-medium">Successful</span>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">12</p>
          <p className="text-xs text-gray-600 dark:text-gray-400">Last 24 hours</p>
        </div>
        <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
          <div className="flex items-center gap-2 text-blue-400 mb-2">
            <Clock className="w-5 h-5" />
            <span className="text-sm font-medium">Running</span>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">3</p>
          <p className="text-xs text-gray-600 dark:text-gray-400">Active now</p>
        </div>
        <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
          <div className="flex items-center gap-2 text-red-400 mb-2">
            <XCircle className="w-5 h-5" />
            <span className="text-sm font-medium">Failed</span>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">2</p>
          <p className="text-xs text-gray-600 dark:text-gray-400">Requires attention</p>
        </div>
        <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
          <div className="flex items-center gap-2 text-yellow-400 mb-2">
            <AlertCircle className="w-5 h-5" />
            <span className="text-sm font-medium">Queued</span>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">5</p>
          <p className="text-xs text-gray-600 dark:text-gray-400">Waiting to run</p>
        </div>
      </div>

      {/* Active Pipelines */}
      <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <GitCommit className="w-5 h-5 text-blue-500" />
          Active Pipelines
        </h2>
        <div className="space-y-3">
          {pipelines.map((pipeline) => (
            <div key={pipeline.id} className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className={`flex items-center gap-2 px-2 py-1 rounded-full text-xs font-medium`}
                       style={{ 
                         backgroundColor: `${statusColors[pipeline.status as keyof typeof statusColors]}20`,
                         color: statusColors[pipeline.status as keyof typeof statusColors]
                       }}>
                    {getStatusIcon(pipeline.status)}
                    {pipeline.status}
                  </div>
                  <h3 className="text-gray-900 dark:text-white font-medium">{pipeline.name}</h3>
                </div>
                <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                  <span className="flex items-center gap-1">
                    <GitBranch className="w-3 h-3" />
                    {pipeline.branch}
                  </span>
                  <span>{pipeline.commit}</span>
                  <span>{pipeline.author}</span>
                  <span>{pipeline.duration}</span>
                </div>
              </div>
              <div className="w-full bg-gray-300 dark:bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${(pipeline.completedStages / pipeline.stages) * 100}%` }}
                ></div>
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-gray-600 dark:text-gray-400">
                <span>Stage {pipeline.completedStages} of {pipeline.stages}</span>
                <span>Success Rate: {pipeline.successRate}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Pipeline Templates */}
      <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Pipeline Templates</h2>
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 hover:bg-gray-200 dark:hover:bg-gray-700/50 cursor-pointer">
            <h3 className="font-medium mb-2">Node.js Application</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">Build, test, and deploy Node.js apps</p>
          </div>
          <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 hover:bg-gray-200 dark:hover:bg-gray-700/50 cursor-pointer">
            <h3 className="font-medium mb-2">Docker Container</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">Build and push Docker images</p>
          </div>
          <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 hover:bg-gray-200 dark:hover:bg-gray-700/50 cursor-pointer">
            <h3 className="font-medium mb-2">Kubernetes Deploy</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">Deploy to Kubernetes clusters</p>
          </div>
        </div>
      </div>
    </div>
  )
}

function ReleasesView() {
  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Recent Releases</h2>
      <div className="space-y-3">
        <ReleaseItem version="v2.1.0" date="Today" status="deployed" environment="Production" />
        <ReleaseItem version="v2.1.0-rc.2" date="Yesterday" status="staging" environment="Staging" />
        <ReleaseItem version="v2.0.9" date="3 days ago" status="deployed" environment="Production" />
      </div>
    </div>
  )
}

function ArtifactsView() {
  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Build Artifacts</h2>
      <div className="space-y-3">
        <ArtifactItem name="frontend-v2.1.0.tar.gz" size="24.5 MB" created="2 hours ago" />
        <ArtifactItem name="api-v2.1.0.zip" size="18.2 MB" created="2 hours ago" />
        <ArtifactItem name="docs-v2.1.0.pdf" size="2.8 MB" created="3 hours ago" />
      </div>
    </div>
  )
}

function DeploymentsView() {
  return (
    <div className="space-y-6">
      <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Active Deployments</h2>
        <table className="w-full">
          <thead>
            <tr className="text-left text-sm text-gray-400 border-b border-gray-800">
              <th className="pb-2">Environment</th>
              <th className="pb-2">Version</th>
              <th className="pb-2">Status</th>
              <th className="pb-2">Deployed</th>
            </tr>
          </thead>
          <tbody className="text-sm">
            <DeploymentRow env="Production" version="v2.1.0" status="healthy" deployed="2 hours ago" />
            <DeploymentRow env="Staging" version="v2.1.0-rc.2" status="healthy" deployed="1 day ago" />
            <DeploymentRow env="Development" version="v2.2.0-dev" status="updating" deployed="10 minutes ago" />
          </tbody>
        </table>
      </div>
    </div>
  )
}

function BuildsView() {
  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Build History</h2>
      <div className="space-y-3">
        <BuildItem number="#1234" branch="main" status="success" time="5 minutes ago" />
        <BuildItem number="#1233" branch="feature/auth" status="failed" time="1 hour ago" />
        <BuildItem number="#1232" branch="main" status="success" time="2 hours ago" />
      </div>
    </div>
  )
}

function ReposView() {
  return (
    <div className="grid grid-cols-2 gap-6">
      <RepoCard name="policycortex-frontend" language="TypeScript" stars={42} lastCommit="2 hours ago" />
      <RepoCard name="policycortex-backend" language="Rust" stars={38} lastCommit="3 hours ago" />
      <RepoCard name="policycortex-ml" language="Python" stars={15} lastCommit="1 day ago" />
      <RepoCard name="policycortex-infra" language="Terraform" stars={8} lastCommit="3 days ago" />
    </div>
  )
}

// Component library
function MetricCard({ title, value, trend, icon: Icon, status }: {
  title: string
  value: string | number
  trend: string
  icon: React.ElementType
  status: string
}) {
  const statusColors: Record<string, string> = {
    active: 'text-blue-400',
    success: 'text-green-400',
    good: 'text-green-400',
    improved: 'text-purple-400'
  }

  return (
    <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">{title}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{trend}</p>
        </div>
        <Icon className={`w-5 h-5 ${statusColors[status] || 'text-gray-600 dark:text-gray-400'}`} />
      </div>
    </div>
  )
}

function ActivityItem({ name, branch, status, time, duration }: {
  name: string
  branch: string
  status: 'success' | 'running' | 'failed'
  time: string
  duration: string
}) {
  const statusConfig = {
    success: { color: 'text-green-400', icon: CheckCircle },
    running: { color: 'text-blue-400', icon: Clock },
    failed: { color: 'text-red-400', icon: XCircle }
  }
  
  const config = statusConfig[status]
  const StatusIcon = config.icon

  return (
    <div className="flex items-center justify-between p-3 bg-gray-100/50 dark:bg-gray-800/50 rounded">
      <div className="flex items-center gap-3">
        <StatusIcon className={`w-5 h-5 ${config.color}`} />
        <div>
          <p className="font-medium">{name}</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Branch: {branch} • Duration: {duration}</p>
        </div>
      </div>
      <span className="text-sm text-gray-500 dark:text-gray-400">{time}</span>
    </div>
  )
}

function FrequencyBar({ env, count, percentage }: {
  env: string
  count: number
  percentage: number
}) {
  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="text-sm">{env}</span>
        <span className="text-sm text-gray-600 dark:text-gray-400">{count} deploys</span>
      </div>
      <div className="bg-gray-300 dark:bg-gray-800 rounded-full h-2">
        <div 
          className="bg-blue-500 h-2 rounded-full"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}

function PerformanceRow({ metric, value, trend }: {
  metric: string
  value: string
  trend: 'improving' | 'stable' | 'declining'
}) {
  const trendColors = {
    improving: 'text-green-400',
    stable: 'text-gray-600 dark:text-gray-400',
    declining: 'text-red-400'
  }

  return (
    <div className="flex justify-between p-2 hover:bg-gray-100 dark:hover:bg-gray-800/50 rounded">
      <span className="text-sm">{metric}</span>
      <div className="flex items-center gap-2">
        <span className="font-medium">{value}</span>
        <TrendingUp className={`w-4 h-4 ${trendColors[trend]}`} />
      </div>
    </div>
  )
}

function PipelineCard({ name, status, lastRun, successRate }: {
  name: string
  status: 'running' | 'idle' | 'queued'
  lastRun: string
  successRate: number
}) {
  const statusColors = {
    running: 'bg-blue-900/50 text-blue-400',
    idle: 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
    queued: 'bg-yellow-900/50 text-yellow-400'
  }

  return (
    <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
      <h3 className="font-medium mb-2">{name}</h3>
      <div className="space-y-2">
        <span className={`text-xs px-2 py-1 rounded ${statusColors[status]}`}>
          {status}
        </span>
        <p className="text-sm text-gray-600 dark:text-gray-400">Last run: {lastRun}</p>
        <p className="text-sm">Success rate: {successRate}%</p>
      </div>
    </div>
  )
}

function ReleaseItem({ version, date, status, environment }: {
  version: string
  date: string
  status: string
  environment: string
}) {
  return (
    <div className="flex justify-between p-3 bg-gray-100/50 dark:bg-gray-800/50 rounded">
      <div>
        <p className="font-medium">{version}</p>
        <p className="text-sm text-gray-600 dark:text-gray-400">{environment}</p>
      </div>
      <div className="text-right">
        <p className="text-sm text-gray-600 dark:text-gray-400">{date}</p>
        <span className="text-xs px-2 py-1 bg-green-900/50 text-green-400 rounded">
          {status}
        </span>
      </div>
    </div>
  )
}

function ArtifactItem({ name, size, created }: {
  name: string
  size: string
  created: string
}) {
  return (
    <div className="flex justify-between p-3 bg-gray-100/50 dark:bg-gray-800/50 rounded hover:bg-gray-200 dark:hover:bg-gray-700/50">
      <div className="flex items-center gap-3">
        <Package className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        <div>
          <p className="font-medium">{name}</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">{size}</p>
        </div>
      </div>
      <span className="text-sm text-gray-500 dark:text-gray-400">{created}</span>
    </div>
  )
}

function DeploymentRow({ env, version, status, deployed }: {
  env: string
  version: string
  status: string
  deployed: string
}) {
  return (
    <tr className="border-b border-gray-200/50 dark:border-gray-800/50">
      <td className="py-3">{env}</td>
      <td className="py-3 font-mono text-sm">{version}</td>
      <td className="py-3">
        <span className={`text-xs px-2 py-1 rounded ${
          status === 'healthy' ? 'bg-green-900/50 text-green-400' : 
          status === 'updating' ? 'bg-blue-900/50 text-blue-400' :
          'bg-red-900/50 text-red-400'
        }`}>
          {status}
        </span>
      </td>
      <td className="py-3 text-gray-600 dark:text-gray-400">{deployed}</td>
    </tr>
  )
}

function BuildItem({ number, branch, status, time }: {
  number: string
  branch: string
  status: 'success' | 'failed'
  time: string
}) {
  return (
    <div className="flex justify-between p-3 bg-gray-100/50 dark:bg-gray-800/50 rounded">
      <div className="flex items-center gap-3">
        {status === 'success' ? (
          <CheckCircle className="w-5 h-5 text-green-400" />
        ) : (
          <XCircle className="w-5 h-5 text-red-400" />
        )}
        <div>
          <p className="font-medium">{number}</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Branch: {branch}</p>
        </div>
      </div>
      <span className="text-sm text-gray-500 dark:text-gray-400">{time}</span>
    </div>
  )
}

function RepoCard({ name, language, stars, lastCommit }: {
  name: string
  language: string
  stars: number
  lastCommit: string
}) {
  return (
    <div className="bg-gray-100 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
      <h3 className="font-medium mb-2">{name}</h3>
      <div className="space-y-1 text-sm">
        <p className="text-gray-600 dark:text-gray-400">Language: {language}</p>
        <p className="text-gray-600 dark:text-gray-400">⭐ {stars} stars</p>
        <p className="text-gray-600 dark:text-gray-400">Last commit: {lastCommit}</p>
      </div>
    </div>
  )
}