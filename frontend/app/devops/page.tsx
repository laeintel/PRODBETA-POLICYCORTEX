'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import {
  Code2,
  Rocket,
  Box,
  Upload,
  Cpu,
  Database,
  ArrowLeft,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  BarChart3,
  RefreshCw,
  Play,
  Beaker,
  ShieldCheck,
  Zap
} from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, AreaChart, Area, ComposedChart
} from 'recharts';
import ViewToggle from '@/components/ViewToggle';
import MetricCard from '@/components/MetricCard';
import ChartContainer from '@/components/ChartContainer';
import DataExport from '@/components/DataExport';
import { useViewPreference } from '@/hooks/useViewPreference';

interface DevOpsCard {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  stats: {
    label: string;
    value: string | number;
    trend?: 'up' | 'down' | 'stable';
    status?: 'success' | 'warning' | 'error';
  }[];
  route: string;
  color: string;
  actions?: { label: string; onClick: () => void }[];
}

export default function DevOpsPage() {
  const router = useRouter();
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);
  const { view, setView } = useViewPreference('devops-view', 'cards');

  // Mock data for DevOps charts
  const pipelineData = [
    { date: '2024-01-01', success: 45, failed: 3, duration: 4.2 },
    { date: '2024-01-02', success: 52, failed: 2, duration: 3.8 },
    { date: '2024-01-03', success: 48, failed: 4, duration: 4.5 },
    { date: '2024-01-04', success: 59, failed: 1, duration: 3.9 },
    { date: '2024-01-05', success: 63, failed: 2, duration: 3.7 },
    { date: '2024-01-06', success: 67, failed: 1, duration: 3.5 }
  ];

  const deploymentFrequency = [
    { environment: 'Development', daily: 12, weekly: 84, monthly: 336 },
    { environment: 'Staging', daily: 8, weekly: 56, monthly: 224 },
    { environment: 'Production', daily: 3, weekly: 21, monthly: 84 },
    { environment: 'Testing', daily: 15, weekly: 105, monthly: 420 }
  ];

  const buildMetrics = [
    { week: 'W1', builds: 234, success: 221, failed: 13 },
    { week: 'W2', builds: 267, success: 251, failed: 16 },
    { week: 'W3', builds: 245, success: 235, failed: 10 },
    { week: 'W4', builds: 289, success: 273, failed: 16 }
  ];

  const leadTimeData = [
    { stage: 'Code Commit', time: 0.5 },
    { stage: 'Build', time: 4.2 },
    { stage: 'Test', time: 8.7 },
    { stage: 'Deploy', time: 2.3 },
    { stage: 'Verification', time: 1.8 }
  ];

  const devopsCards: DevOpsCard[] = [
    {
      id: 'pipelines',
      title: 'CI/CD Pipelines',
      description: 'Manage build and deployment pipelines',
      icon: Code2,
      stats: [
        { label: 'Active Pipelines', value: '24', status: 'success' },
        { label: 'Running Now', value: '3', trend: 'stable' },
        { label: 'Success Rate', value: '94.2%', trend: 'up' },
        { label: 'Avg Duration', value: '4.2 min', trend: 'down' }
      ],
      route: '/devops/pipelines',
      color: 'blue',
      actions: [
        { label: 'View Pipelines', onClick: () => router.push('/devops/pipelines') },
        { label: 'Create Pipeline', onClick: () => router.push('/devops/pipelines?action=create') }
      ]
    },
    {
      id: 'releases',
      title: 'Release Management',
      description: 'Track and manage software releases',
      icon: Rocket,
      stats: [
        { label: 'Latest Release', value: 'v2.18.0', status: 'success' },
        { label: 'Pending', value: '2', trend: 'stable' },
        { label: 'This Month', value: '8', trend: 'up' },
        { label: 'Rollback Rate', value: '0.5%', status: 'success' }
      ],
      route: '/devops/releases',
      color: 'green',
      actions: [
        { label: 'View Releases', onClick: () => router.push('/devops/releases') },
        { label: 'Create Release', onClick: () => router.push('/devops/releases?action=create') }
      ]
    },
    {
      id: 'artifacts',
      title: 'Artifact Repository',
      description: 'Store and manage build artifacts',
      icon: Box,
      stats: [
        { label: 'Total Artifacts', value: '1,234', trend: 'up' },
        { label: 'Storage Used', value: '458 GB', trend: 'up' },
        { label: 'Downloads Today', value: '847', status: 'success' },
        { label: 'Repositories', value: '12', trend: 'stable' }
      ],
      route: '/devops/artifacts',
      color: 'purple',
      actions: [
        { label: 'Browse Artifacts', onClick: () => router.push('/devops/artifacts') },
        { label: 'Upload Artifact', onClick: () => router.push('/devops/artifacts?action=upload') }
      ]
    },
    {
      id: 'deployments',
      title: 'Deployment History',
      description: 'Track deployment activities and status',
      icon: Upload,
      stats: [
        { label: 'Today', value: '12', trend: 'up' },
        { label: 'This Week', value: '68', status: 'success' },
        { label: 'Failed', value: '2', status: 'warning' },
        { label: 'Environments', value: '4', trend: 'stable' }
      ],
      route: '/devops/deployments',
      color: 'orange',
      actions: [
        { label: 'View History', onClick: () => router.push('/devops/deployments') },
        { label: 'Deploy Now', onClick: () => router.push('/devops/deployments?action=deploy') }
      ]
    },
    {
      id: 'builds',
      title: 'Build Status',
      description: 'Monitor build jobs and results',
      icon: Cpu,
      stats: [
        { label: 'Queued', value: '5', status: 'warning' },
        { label: 'Building', value: '3', trend: 'stable' },
        { label: 'Completed', value: '142', status: 'success' },
        { label: 'Failed Today', value: '4', status: 'error' }
      ],
      route: '/devops/builds',
      color: 'red',
      actions: [
        { label: 'View Builds', onClick: () => router.push('/devops/builds') },
        { label: 'Build Queue', onClick: () => router.push('/devops/builds?tab=queue') }
      ]
    },
    {
      id: 'repos',
      title: 'Repository Management',
      description: 'Manage source code repositories',
      icon: Database,
      stats: [
        { label: 'Repositories', value: '18', trend: 'up' },
        { label: 'Branches', value: '124', trend: 'up' },
        { label: 'Pull Requests', value: '23', status: 'warning' },
        { label: 'Contributors', value: '47', trend: 'up' }
      ],
      route: '/devops/repos',
      color: 'indigo',
      actions: [
        { label: 'View Repos', onClick: () => router.push('/devops/repos') },
        { label: 'Code Review', onClick: () => router.push('/devops/repos?tab=reviews') }
      ]
    }
  ];

  const recentActivities = [
    { id: 1, type: 'deployment', message: 'Production deployment v2.18.0 completed', time: '10 min ago', status: 'success' },
    { id: 2, type: 'build', message: 'Build #1234 failed on main branch', time: '25 min ago', status: 'error' },
    { id: 3, type: 'release', message: 'Release v2.17.11 created and tagged', time: '1 hour ago', status: 'success' },
    { id: 4, type: 'pipeline', message: 'CI pipeline for feature/auth completed', time: '2 hours ago', status: 'success' },
    { id: 5, type: 'artifact', message: 'New artifact uploaded: frontend-v2.18.0.zip', time: '3 hours ago', status: 'info' }
  ];

  const quickStats = [
    { label: 'Pipeline Success', value: '94.2%', icon: CheckCircle, color: 'green' },
    { label: 'Active Builds', value: '8', icon: Cpu, color: 'blue' },
    { label: 'Deploy Frequency', value: '3.4/day', icon: Rocket, color: 'purple' },
    { label: 'Lead Time', value: '2.5 hrs', icon: Clock, color: 'orange' }
  ];

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'success': return 'text-green-600 dark:text-green-400';
      case 'warning': return 'text-yellow-600 dark:text-yellow-400';
      case 'error': return 'text-red-600 dark:text-red-400';
      default: return 'text-blue-600 dark:text-blue-400';
    }
  };

  const getTrendSymbol = (trend?: string) => {
    if (trend === 'up') return '↑';
    if (trend === 'down') return '↓';
    return '→';
  };

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'deployment': return Upload;
      case 'build': return Cpu;
      case 'release': return Rocket;
      case 'pipeline': return Code2;
      case 'artifact': return Box;
      default: return Database;
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <button
            onClick={() => router.push('/tactical')}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            aria-label="Back to Command Center"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div>
            <h1 className="text-4xl font-bold">DevOps & CI/CD</h1>
            <p className="text-gray-600 dark:text-gray-400 mt-2">
              Continuous integration, deployment pipelines, and release management
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <ViewToggle view={view} onViewChange={setView} />
          <button
            onClick={() => router.push('/devops/pipelines?action=create')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
          >
            <Zap className="h-5 w-5" />
            Create Pipeline
          </button>
          <button
            onClick={() => router.push('/devops/deployments?action=deploy')}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
          >
            <Rocket className="h-5 w-5" />
            Deploy Now
          </button>
        </div>
      </div>

      {view === 'cards' ? (
        <>
          {/* Quick Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <MetricCard
              title="Pipeline Success"
              value="94.2%"
              change={2.1}
              changeLabel="improvement"
              icon={<CheckCircle className="h-5 w-5 text-green-600" />}
              sparklineData={[92.1, 92.8, 93.2, 93.8, 94.0, 94.2]}
              onClick={() => router.push('/devops/pipelines')}
              status="success"
            />
            <MetricCard
              title="Active Builds"
              value={8}
              change={-20}
              changeLabel="from yesterday"
              icon={<Cpu className="h-5 w-5 text-blue-600" />}
              sparklineData={[12, 10, 9, 8, 10, 8]}
              onClick={() => router.push('/devops/builds')}
              status="neutral"
            />
            <MetricCard
              title="Deploy Frequency"
              value="3.4/day"
              change={13}
              changeLabel="increase"
              icon={<Rocket className="h-5 w-5 text-purple-600" />}
              sparklineData={[2.8, 3.0, 3.1, 3.3, 3.4, 3.4]}
              onClick={() => router.push('/devops/deployments')}
              status="success"
            />
            <MetricCard
              title="Lead Time"
              value="2.5 hrs"
              change={-15}
              changeLabel="faster"
              icon={<Clock className="h-5 w-5 text-orange-600" />}
              sparklineData={[3.2, 3.0, 2.8, 2.6, 2.5, 2.5]}
              onClick={() => router.push('/devops/pipelines')}
              status="success"
            />
          </div>
        </>
      ) : (
        <>
          {/* DevOps Visualizations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <ChartContainer 
              title="Pipeline Success Rate" 
              onExport={() => {}}
              onDrillIn={() => router.push('/devops/pipelines')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={pipelineData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="date" className="text-gray-600 dark:text-gray-400" />
                  <YAxis className="text-gray-600 dark:text-gray-400" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="success" fill="#10B981" name="Successful" />
                  <Bar dataKey="failed" fill="#EF4444" name="Failed" />
                  <Line type="monotone" dataKey="duration" stroke="#F59E0B" strokeWidth={2} name="Avg Duration (min)" />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer 
              title="Deployment Frequency" 
              onExport={() => {}}
              onDrillIn={() => router.push('/devops/deployments')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={deploymentFrequency}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="environment" className="text-gray-600 dark:text-gray-400" />
                  <YAxis className="text-gray-600 dark:text-gray-400" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="daily" fill="#3B82F6" name="Daily" />
                  <Bar dataKey="weekly" fill="#8B5CF6" name="Weekly" />
                  <Bar dataKey="monthly" fill="#10B981" name="Monthly" />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer 
              title="Build Success Trends" 
              onExport={() => {}}
              onDrillIn={() => router.push('/devops/builds')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={buildMetrics}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="week" className="text-gray-600 dark:text-gray-400" />
                  <YAxis className="text-gray-600 dark:text-gray-400" />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="builds" stackId="1" stroke="#6B7280" fill="#6B7280" fillOpacity={0.3} name="Total Builds" />
                  <Area type="monotone" dataKey="success" stackId="2" stroke="#10B981" fill="#10B981" fillOpacity={0.6} name="Successful" />
                  <Area type="monotone" dataKey="failed" stackId="3" stroke="#EF4444" fill="#EF4444" fillOpacity={0.8} name="Failed" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer 
              title="Lead Time Breakdown" 
              onExport={() => {}}
              onDrillIn={() => router.push('/devops/pipelines')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart layout="horizontal" data={leadTimeData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis type="number" className="text-gray-600 dark:text-gray-400" />
                  <YAxis dataKey="stage" type="category" width={100} className="text-gray-600 dark:text-gray-400" />
                  <Tooltip formatter={(value) => [`${value} hours`, 'Time']} />
                  <Bar dataKey="time" fill="#F59E0B" name="Hours" />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        </>
      )}

      {view === 'cards' && (
        <>
          {/* DevOps Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {devopsCards.map((card) => {
              const Icon = card.icon;
              return (
                <div
                  key={card.id}
                  className="bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-lg transition-all cursor-pointer transform hover:scale-[1.02]"
                  onMouseEnter={() => setHoveredCard(card.id)}
                  onMouseLeave={() => setHoveredCard(null)}
                  onClick={() => router.push(card.route)}
                >
                  <div className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className={`p-3 rounded-lg bg-${card.color}-50 dark:bg-${card.color}-900/20`}>
                        <Icon className={`h-8 w-8 text-${card.color}-600 dark:text-${card.color}-400`} />
                      </div>
                      {hoveredCard === card.id && (
                        <ArrowLeft className="h-5 w-5 rotate-180 text-gray-400" />
                      )}
                    </div>
                    
                    <h3 className="text-xl font-semibold mb-2">{card.title}</h3>
                    <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                      {card.description}
                    </p>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      {card.stats.map((stat, index) => (
                        <div key={index} className="text-sm">
                          <p className="text-gray-500 dark:text-gray-400">{stat.label}</p>
                          <p className={`font-semibold flex items-center gap-1 ${
                            stat.status ? getStatusColor(stat.status) : ''
                          }`}>
                            {stat.value}
                            {stat.trend && (
                              <span className={`text-xs ${
                                stat.trend === 'up' ? 'text-green-500' :
                                stat.trend === 'down' ? 'text-red-500' :
                                'text-gray-500'
                              }`}>
                                {getTrendSymbol(stat.trend)}
                              </span>
                            )}
                          </p>
                        </div>
                      ))}
                    </div>

                    {/* Action Buttons */}
                    {card.actions && (
                      <div className="flex gap-2 pt-3 border-t dark:border-gray-700">
                        {card.actions.map((action, index) => (
                          <button
                            key={index}
                            onClick={(e) => {
                              e.stopPropagation();
                              action.onClick();
                            }}
                            className="flex-1 px-3 py-1.5 text-xs font-medium bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-lg transition-colors"
                          >
                            {action.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Recent Activity */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Recent DevOps Activity</h2>
          <button
            onClick={() => router.push('/devops/pipelines')}
            className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
          >
            View All Activity
          </button>
        </div>
        
        <div className="space-y-3">
          {recentActivities.map((activity) => {
            const Icon = getActivityIcon(activity.type);
            return (
              <div
                key={activity.id}
                className="flex items-center justify-between p-3 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg cursor-pointer transition-colors"
                onClick={() => {
                  if (activity.type === 'deployment') router.push('/devops/deployments');
                  else if (activity.type === 'build') router.push('/devops/builds');
                  else if (activity.type === 'release') router.push('/devops/releases');
                  else if (activity.type === 'pipeline') router.push('/devops/pipelines');
                  else if (activity.type === 'artifact') router.push('/devops/artifacts');
                  else router.push('/devops/repos');
                }}
              >
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg bg-gray-100 dark:bg-gray-700`}>
                    <Icon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
                  </div>
                  <div>
                    <p className="font-medium">{activity.message}</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400 flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {activity.time}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {activity.status === 'success' && <CheckCircle className="h-5 w-5 text-green-500" />}
                  {activity.status === 'warning' && <AlertTriangle className="h-5 w-5 text-yellow-500" />}
                  {activity.status === 'error' && <XCircle className="h-5 w-5 text-red-500" />}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}