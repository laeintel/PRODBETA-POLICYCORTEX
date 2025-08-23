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
        <div className="flex gap-3">
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

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        {quickStats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div
              key={index}
              className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm hover:shadow-md transition-all cursor-pointer"
              onClick={() => router.push('/devops/pipelines')}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{stat.label}</p>
                  <p className="text-2xl font-bold mt-1">{stat.value}</p>
                </div>
                <div className={`p-3 rounded-lg bg-${stat.color}-50 dark:bg-${stat.color}-900/20`}>
                  <Icon className={`h-6 w-6 text-${stat.color}-600 dark:text-${stat.color}-400`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

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