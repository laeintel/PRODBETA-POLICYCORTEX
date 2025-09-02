'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import {
  Server,
  HardDrive,
  Network,
  Users,
  AlertTriangle,
  CheckCircle,
  Clock,
  ArrowLeft,
  BarChart3,
  Settings,
  Wrench,
  Activity,
  Shield,
  Database,
  Cpu,
  Globe,
  RefreshCw,
  TrendingUp,
  Archive
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

interface ITSMCard {
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

export default function ITSMPage() {
  const router = useRouter();
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);
  const { view, setView } = useViewPreference('itsm-view', 'cards');

  // Mock data for ITSM charts
  const resourceHealthData = [
    { date: '2024-01-01', healthy: 847, warning: 23, critical: 5, offline: 2 },
    { date: '2024-01-02', healthy: 851, warning: 21, critical: 4, offline: 1 },
    { date: '2024-01-03', healthy: 856, warning: 19, critical: 3, offline: 1 },
    { date: '2024-01-04', healthy: 862, warning: 17, critical: 2, offline: 0 },
    { date: '2024-01-05', healthy: 859, warning: 20, critical: 3, offline: 1 },
    { date: '2024-01-06', healthy: 863, warning: 18, critical: 2, offline: 1 }
  ];

  const incidentTrends = [
    { week: 'W1', incidents: 42, resolved: 38, sla_met: 89 },
    { week: 'W2', incidents: 38, resolved: 35, sla_met: 92 },
    { week: 'W3', incidents: 45, resolved: 43, sla_met: 87 },
    { week: 'W4', incidents: 35, resolved: 33, sla_met: 94 }
  ];

  const assetDistribution = [
    { name: 'Servers', value: 342, color: '#3B82F6' },
    { name: 'Workstations', value: 1247, color: '#10B981' },
    { name: 'Network Devices', value: 89, color: '#F59E0B' },
    { name: 'Storage Systems', value: 56, color: '#8B5CF6' }
  ];

  const changeRequests = [
    { priority: 'Emergency', approved: 2, pending: 1, rejected: 0 },
    { priority: 'High', approved: 12, pending: 4, rejected: 1 },
    { priority: 'Medium', approved: 34, pending: 8, rejected: 2 },
    { priority: 'Low', approved: 67, pending: 15, rejected: 5 }
  ];

  const itsmCards: ITSMCard[] = [
    {
      id: 'assets',
      title: 'Asset Management',
      description: 'Track and manage IT assets throughout their lifecycle',
      icon: Server,
      stats: [
        { label: 'Total Assets', value: '1,734', trend: 'up' },
        { label: 'Active', value: '1,689', status: 'success' },
        { label: 'Deprecated', value: '34', status: 'warning' },
        { label: 'Compliance', value: '97.3%', status: 'success' }
      ],
      route: '/itsm/assets',
      color: 'blue',
      actions: [
        { label: 'View Assets', onClick: () => router.push('/itsm/assets') },
        { label: 'Asset Discovery', onClick: () => router.push('/itsm/assets?action=discover') }
      ]
    },
    {
      id: 'incidents',
      title: 'Incident Management',
      description: 'Track and resolve IT service incidents efficiently',
      icon: AlertTriangle,
      stats: [
        { label: 'Open Incidents', value: '23', status: 'warning' },
        { label: 'Resolved Today', value: '47', status: 'success' },
        { label: 'MTTR', value: '2.4 hrs', trend: 'down' },
        { label: 'SLA Compliance', value: '94.2%', status: 'success' }
      ],
      route: '/itsm/incidents',
      color: 'red',
      actions: [
        { label: 'View Incidents', onClick: () => router.push('/itsm/incidents') },
        { label: 'Create Incident', onClick: () => router.push('/itsm/incidents?action=create') }
      ]
    },
    {
      id: 'changes',
      title: 'Change Management',
      description: 'Plan, approve, and track IT infrastructure changes',
      icon: RefreshCw,
      stats: [
        { label: 'Pending Approval', value: '12', status: 'warning' },
        { label: 'In Progress', value: '8', trend: 'stable' },
        { label: 'Success Rate', value: '96.7%', status: 'success' },
        { label: 'This Month', value: '156', trend: 'up' }
      ],
      route: '/itsm/changes',
      color: 'purple',
      actions: [
        { label: 'View Changes', onClick: () => router.push('/itsm/changes') },
        { label: 'Request Change', onClick: () => router.push('/itsm/changes?action=request') }
      ]
    },
    {
      id: 'problems',
      title: 'Problem Management',
      description: 'Identify root causes and prevent recurring incidents',
      icon: Activity,
      stats: [
        { label: 'Active Problems', value: '5', status: 'warning' },
        { label: 'Root Cause Found', value: '8', status: 'success' },
        { label: 'Prevention Rate', value: '89.4%', status: 'success' },
        { label: 'Avg Resolution', value: '5.2 days', trend: 'down' }
      ],
      route: '/itsm/problems',
      color: 'orange',
      actions: [
        { label: 'View Problems', onClick: () => router.push('/itsm/problems') },
        { label: 'Create Problem', onClick: () => router.push('/itsm/problems?action=create') }
      ]
    },
    {
      id: 'configuration',
      title: 'Configuration Items',
      description: 'Maintain comprehensive CMDB of IT components',
      icon: Database,
      stats: [
        { label: 'Total CIs', value: '2,847', trend: 'up' },
        { label: 'Relationships', value: '8,934', trend: 'up' },
        { label: 'Accuracy', value: '98.1%', status: 'success' },
        { label: 'Auto-Discovery', value: '87%', status: 'success' }
      ],
      route: '/itsm/configuration',
      color: 'green',
      actions: [
        { label: 'Browse CMDB', onClick: () => router.push('/itsm/configuration') },
        { label: 'Import CIs', onClick: () => router.push('/itsm/configuration?action=import') }
      ]
    },
    {
      id: 'service-catalog',
      title: 'Service Catalog',
      description: 'Self-service portal for IT service requests',
      icon: Archive,
      stats: [
        { label: 'Available Services', value: '67', trend: 'stable' },
        { label: 'Requests Today', value: '89', trend: 'up' },
        { label: 'Fulfillment Rate', value: '97.8%', status: 'success' },
        { label: 'User Satisfaction', value: '4.6/5', status: 'success' }
      ],
      route: '/itsm/service-catalog',
      color: 'indigo',
      actions: [
        { label: 'Browse Catalog', onClick: () => router.push('/itsm/service-catalog') },
        { label: 'Request Service', onClick: () => router.push('/itsm/service-catalog?action=request') }
      ]
    }
  ];

  const recentActivities = [
    { id: 1, type: 'incident', message: 'Critical incident INC-2024-001 resolved', time: '15 min ago', status: 'success' },
    { id: 2, type: 'change', message: 'Change request CHG-2024-045 approved', time: '30 min ago', status: 'success' },
    { id: 3, type: 'asset', message: 'New server SRV-2024-089 added to inventory', time: '1 hour ago', status: 'info' },
    { id: 4, type: 'problem', message: 'Problem PRB-2024-012 root cause identified', time: '2 hours ago', status: 'success' },
    { id: 5, type: 'service', message: 'New service request fulfilled for HR department', time: '3 hours ago', status: 'success' }
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
      case 'incident': return AlertTriangle;
      case 'change': return RefreshCw;
      case 'asset': return Server;
      case 'problem': return Activity;
      case 'service': return Archive;
      default: return Settings;
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
            <h1 className="text-4xl font-bold">IT Service Management</h1>
            <p className="text-gray-600 dark:text-gray-400 mt-2">
              Comprehensive ITSM platform for asset, incident, change, and service management
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <ViewToggle view={view} onViewChange={setView} />
          <button
            onClick={() => router.push('/itsm/incidents?action=create')}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2"
          >
            <AlertTriangle className="h-5 w-5" />
            Report Incident
          </button>
          <button
            onClick={() => router.push('/itsm/service-catalog')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
          >
            <Archive className="h-5 w-5" />
            Service Catalog
          </button>
        </div>
      </div>

      {view === 'cards' ? (
        <>
          {/* Quick Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <MetricCard
              title="System Health"
              value="98.7%"
              change={0.3}
              changeLabel="improvement"
              icon={<CheckCircle className="h-5 w-5 text-green-600" />}
              sparklineData={[98.2, 98.4, 98.5, 98.6, 98.7, 98.7]}
              onClick={() => router.push('/itsm/assets')}
              status="success"
            />
            <MetricCard
              title="Open Incidents"
              value={23}
              change={-15}
              changeLabel="reduction"
              icon={<AlertTriangle className="h-5 w-5 text-red-600" />}
              sparklineData={[35, 30, 28, 26, 25, 23]}
              onClick={() => router.push('/itsm/incidents')}
              status="warning"
            />
            <MetricCard
              title="SLA Compliance"
              value="94.2%"
              change={2.1}
              changeLabel="improvement"
              icon={<Clock className="h-5 w-5 text-blue-600" />}
              sparklineData={[91.8, 92.1, 92.8, 93.4, 93.8, 94.2]}
              onClick={() => router.push('/itsm/incidents')}
              status="success"
            />
            <MetricCard
              title="Change Success Rate"
              value="96.7%"
              change={1.4}
              changeLabel="increase"
              icon={<RefreshCw className="h-5 w-5 text-purple-600" />}
              sparklineData={[94.8, 95.1, 95.6, 96.0, 96.3, 96.7]}
              onClick={() => router.push('/itsm/changes')}
              status="success"
            />
          </div>
        </>
      ) : (
        <>
          {/* ITSM Visualizations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <ChartContainer 
              title="Resource Health Status" 
              onExport={() => {}}
              onDrillIn={() => router.push('/itsm/assets')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={resourceHealthData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="date" className="text-gray-600 dark:text-gray-400" />
                  <YAxis className="text-gray-600 dark:text-gray-400" />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="healthy" stackId="1" stroke="#10B981" fill="#10B981" fillOpacity={0.8} name="Healthy" />
                  <Area type="monotone" dataKey="warning" stackId="1" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.6} name="Warning" />
                  <Area type="monotone" dataKey="critical" stackId="1" stroke="#EF4444" fill="#EF4444" fillOpacity={0.8} name="Critical" />
                  <Area type="monotone" dataKey="offline" stackId="1" stroke="#6B7280" fill="#6B7280" fillOpacity={0.6} name="Offline" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer 
              title="Incident Resolution Trends" 
              onExport={() => {}}
              onDrillIn={() => router.push('/itsm/incidents')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={incidentTrends}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="week" className="text-gray-600 dark:text-gray-400" />
                  <YAxis className="text-gray-600 dark:text-gray-400" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="incidents" fill="#EF4444" name="Total Incidents" />
                  <Bar dataKey="resolved" fill="#10B981" name="Resolved" />
                  <Line type="monotone" dataKey="sla_met" stroke="#3B82F6" strokeWidth={2} name="SLA Met %" />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer 
              title="Asset Distribution" 
              onExport={() => {}}
              onDrillIn={() => router.push('/itsm/configuration')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={assetDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {assetDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer 
              title="Change Request Status" 
              onExport={() => {}}
              onDrillIn={() => router.push('/itsm/changes')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={changeRequests}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="priority" className="text-gray-600 dark:text-gray-400" />
                  <YAxis className="text-gray-600 dark:text-gray-400" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="approved" fill="#10B981" name="Approved" />
                  <Bar dataKey="pending" fill="#F59E0B" name="Pending" />
                  <Bar dataKey="rejected" fill="#EF4444" name="Rejected" />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        </>
      )}

      {view === 'cards' && (
        <>
          {/* ITSM Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {itsmCards.map((card) => {
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

      {/* Recent ITSM Activity */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Recent ITSM Activity</h2>
          <button
            onClick={() => router.push('/itsm/dashboard')}
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
                  if (activity.type === 'incident') router.push('/itsm/incidents');
                  else if (activity.type === 'change') router.push('/itsm/changes');
                  else if (activity.type === 'asset') router.push('/itsm/assets');
                  else if (activity.type === 'problem') router.push('/itsm/problems');
                  else if (activity.type === 'service') router.push('/itsm/service-catalog');
                  else router.push('/itsm/configuration');
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
                  {activity.status === 'error' && <AlertTriangle className="h-5 w-5 text-red-500" />}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}