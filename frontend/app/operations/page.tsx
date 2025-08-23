'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import {
  Server,
  BarChart3,
  Settings,
  Bell,
  AlertTriangle,
  ArrowLeft,
  Clock,
  Cpu,
  Upload,
  RefreshCw,
  CheckCircle,
  XCircle,
  ShieldCheck,
  FileBarChart,
  Sparkles,
  Terminal,
  Box,
  Database
} from 'lucide-react';

interface OperationCard {
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

export default function OperationsPage() {
  const router = useRouter();
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  const operationCards: OperationCard[] = [
    {
      id: 'resources',
      title: 'Resource Management',
      description: 'Monitor and manage cloud infrastructure resources',
      icon: Server,
      stats: [
        { label: 'Total Resources', value: '2,847', trend: 'up' },
        { label: 'Active VMs', value: '342', status: 'success' },
        { label: 'Storage Used', value: '78.4 TB', trend: 'up' },
        { label: 'Networks', value: '48', status: 'success' }
      ],
      route: '/operations/resources',
      color: 'blue',
      actions: [
        { label: 'View Inventory', onClick: () => router.push('/operations/resources') },
        { label: 'Resource Health', onClick: () => router.push('/operations/resources?tab=health') }
      ]
    },
    {
      id: 'monitoring',
      title: 'Real-time Monitoring',
      description: 'Track performance metrics and system health',
      icon: BarChart3,
      stats: [
        { label: 'Metrics Tracked', value: '15,234', trend: 'stable' },
        { label: 'Dashboards', value: '42', status: 'success' },
        { label: 'Uptime', value: '99.98%', status: 'success' },
        { label: 'Response Time', value: '124ms', trend: 'down' }
      ],
      route: '/operations/monitoring',
      color: 'green',
      actions: [
        { label: 'View Dashboards', onClick: () => router.push('/operations/monitoring') },
        { label: 'Custom Metrics', onClick: () => router.push('/operations/monitoring?tab=custom') }
      ]
    },
    {
      id: 'automation',
      title: 'Automation Workflows',
      description: 'Manage automated processes and runbooks',
      icon: Settings,
      stats: [
        { label: 'Active Workflows', value: '186', trend: 'up' },
        { label: 'Executions Today', value: '3,421', status: 'success' },
        { label: 'Success Rate', value: '96.8%', status: 'success' },
        { label: 'Time Saved', value: '427 hrs', trend: 'up' }
      ],
      route: '/operations/automation',
      color: 'purple',
      actions: [
        { label: 'View Workflows', onClick: () => router.push('/operations/automation') },
        { label: 'Create Workflow', onClick: () => router.push('/operations/automation?action=create') }
      ]
    },
    {
      id: 'notifications',
      title: 'Notification Center',
      description: 'Configure and manage system notifications',
      icon: Bell,
      stats: [
        { label: 'Active Channels', value: '12', status: 'success' },
        { label: 'Sent Today', value: '847', trend: 'up' },
        { label: 'Delivery Rate', value: '99.2%', status: 'success' },
        { label: 'Subscribers', value: '234', trend: 'stable' }
      ],
      route: '/operations/notifications',
      color: 'yellow',
      actions: [
        { label: 'View Notifications', onClick: () => router.push('/operations/notifications') },
        { label: 'Configure Channels', onClick: () => router.push('/operations/notifications?tab=channels') }
      ]
    },
    {
      id: 'alerts',
      title: 'Alert Management',
      description: 'Monitor and respond to system alerts',
      icon: AlertTriangle,
      stats: [
        { label: 'Active Alerts', value: '23', status: 'warning' },
        { label: 'Critical', value: '3', status: 'error' },
        { label: 'Resolved Today', value: '142', status: 'success' },
        { label: 'MTTR', value: '14 min', trend: 'down' }
      ],
      route: '/operations/alerts',
      color: 'red',
      actions: [
        { label: 'View Alerts', onClick: () => router.push('/operations/alerts') },
        { label: 'Alert Rules', onClick: () => router.push('/operations/alerts?tab=rules') }
      ]
    }
  ];

  const recentActivities = [
    { id: 1, type: 'deployment', message: 'Production deployment completed', time: '5 min ago', status: 'success' },
    { id: 2, type: 'alert', message: 'High CPU usage on app-server-03', time: '12 min ago', status: 'warning' },
    { id: 3, type: 'automation', message: 'Backup workflow executed successfully', time: '23 min ago', status: 'success' },
    { id: 4, type: 'resource', message: 'New VM provisioned in West US 2', time: '45 min ago', status: 'info' },
    { id: 5, type: 'monitoring', message: 'Dashboard "Sales Metrics" updated', time: '1 hour ago', status: 'info' }
  ];

  const quickStats = [
    { label: 'System Health', value: '98.5%', icon: ShieldCheck, color: 'green' },
    { label: 'Active Incidents', value: '4', icon: AlertTriangle, color: 'yellow' },
    { label: 'Automation Rate', value: '87%', icon: RefreshCw, color: 'purple' },
    { label: 'Cost Savings', value: '$42.3K', icon: FileBarChart, color: 'blue' }
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
      case 'alert': return AlertTriangle;
      case 'automation': return Settings;
      case 'resource': return Server;
      case 'monitoring': return BarChart3;
      default: return Bell;
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
            <h1 className="text-4xl font-bold">Operations Center</h1>
            <p className="text-gray-600 dark:text-gray-400 mt-2">
              Monitor and manage cloud infrastructure operations
            </p>
          </div>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => router.push('/operations/automation?action=create')}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2"
          >
            <Sparkles className="h-5 w-5" />
            Create Automation
          </button>
          <button
            onClick={() => router.push('/operations/monitoring')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
          >
            <BarChart3 className="h-5 w-5" />
            View Dashboards
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
              onClick={() => router.push('/operations/monitoring')}
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

      {/* Operation Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {operationCards.map((card) => {
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
          <h2 className="text-xl font-semibold">Recent Activity</h2>
          <button
            onClick={() => router.push('/operations/monitoring')}
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
                  if (activity.type === 'alert') router.push('/operations/alerts');
                  else if (activity.type === 'automation') router.push('/operations/automation');
                  else if (activity.type === 'resource') router.push('/operations/resources');
                  else router.push('/operations/monitoring');
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