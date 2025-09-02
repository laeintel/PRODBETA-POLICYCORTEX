'use client';

import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  Shield, 
  DollarSign,
  Activity,
  Clock,
  CheckCircle,
  XCircle,
  Info,
  ArrowUpRight,
  ArrowDownRight,
  BarChart3,
  Users,
  Server,
  Brain,
  Zap,
  Target,
  GitBranch,
  Database,
  Globe,
  Cpu,
  ChevronRight
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import CloudIntegrationStatus from '@/components/CloudIntegrationStatus';
import PageContainer from '@/components/PageContainer';
import { useRouter } from 'next/navigation';
import { toast } from '@/hooks/useToast';

// KPI Card Component
const KPICard = ({ 
  title, 
  value, 
  change, 
  changeType, 
  icon: Icon, 
  trend,
  subtitle,
  onClick 
}: any) => {
  const isPositive = changeType === 'positive';
  const TrendIcon = isPositive ? TrendingUp : TrendingDown;
  const changeColor = isPositive ? 'text-green-600' : 'text-red-600';
  const bgColor = isPositive ? 'bg-green-50' : 'bg-red-50';

  return (
    <div 
      className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-all cursor-pointer"
      onClick={onClick}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <div className="p-2 bg-blue-50 rounded-lg">
              <Icon className="h-5 w-5 text-blue-600" />
            </div>
            <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</p>
          </div>
          <div className="flex items-baseline gap-2">
            <h3 className="font-bold tabular-nums text-[var(--font-kpi)] text-gray-900 dark:text-gray-100">{value}</h3>
            {subtitle && (
              <span className="text-[var(--font-kpi-label)] text-gray-500 dark:text-gray-400">{subtitle}</span>
            )}
          </div>
          {change && (
            <div className={`flex items-center gap-1 mt-2 ${changeColor}`}>
              <TrendIcon className="h-4 w-4" />
              <span className="text-sm font-medium">{change}</span>
              <span className="text-xs text-gray-500 dark:text-gray-400">vs last period</span>
            </div>
          )}
        </div>
        {trend && (
          <div className="w-24 h-12">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trend}>
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  stroke={isPositive ? '#10b981' : '#ef4444'}
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};

// Alert Item Component
const AlertItem = ({ alert }: any) => {
  const router = useRouter();
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-300 border-red-200 dark:border-red-800';
      case 'high': return 'bg-orange-100 dark:bg-orange-900/20 text-orange-800 dark:text-orange-300 border-orange-200 dark:border-orange-800';
      case 'medium': return 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-300 border-yellow-200 dark:border-yellow-800';
      case 'low': return 'bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300 border-blue-200 dark:border-blue-800';
      default: return 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 border-gray-200 dark:border-gray-600';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return XCircle;
      case 'high': return AlertTriangle;
      case 'medium': return Info;
      default: return CheckCircle;
    }
  };

  const Icon = getSeverityIcon(alert.severity);

  return (
    <div className={`rounded-lg border p-4 ${getSeverityColor(alert.severity)}`}>
      <div className="flex items-start gap-3">
        <Icon className="h-5 w-5 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <h4 className="font-medium">{alert.title}</h4>
          <p className="text-sm mt-1 opacity-90">{alert.description}</p>
          <div className="flex items-center gap-4 mt-2 text-xs">
            <span>{alert.resource}</span>
            <span>•</span>
            <span>{alert.time}</span>
          </div>
        </div>
        <button
          type="button"
          className="text-sm font-medium hover:underline"
          onClick={() => router.push('/operations/alerts')}>
          View →
        </button>
      </div>
    </div>
  );
};

// Predictive Insights Widget
const PredictiveInsights = ({ insights }: any) => {
  return (
    <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg p-6 border border-purple-200">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="h-5 w-5 text-purple-600" />
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">AI Predictions</h3>
      </div>
      <div className="space-y-3">
        {insights.map((insight: any, index: number) => (
          <div key={index} className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <div className={`w-2 h-2 rounded-full mt-2 ${
                insight.type === 'warning' ? 'bg-yellow-500' : 
                insight.type === 'danger' ? 'bg-red-500' : 'bg-green-500'
              }`} />
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{insight.title}</p>
                <p className="text-xs text-gray-600 mt-1">{insight.prediction}</p>
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
                    {insight.confidence}% confidence
                  </span>
                  <span className="text-xs text-gray-500">{insight.timeline}</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default function DashboardPage(): JSX.Element {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('7d');
  const [selectedMetric, setSelectedMetric] = useState('all');

  // Mock data - would come from API
  const kpiData = [
    {
      title: 'Compliance Score',
      value: '94.5%',
      change: '+2.3%',
      changeType: 'positive',
      icon: Shield,
      subtitle: 'Excellent',
      trend: [
        { value: 92 }, { value: 91 }, { value: 93 }, { value: 94 }, { value: 94.5 }
      ]
    },
    {
      title: 'Total Resources',
      value: '2,847',
      change: '+145',
      changeType: 'positive',
      icon: Server,
      subtitle: 'Active',
      trend: [
        { value: 2702 }, { value: 2750 }, { value: 2780 }, { value: 2820 }, { value: 2847 }
      ]
    },
    {
      title: 'Monthly Cost',
      value: '$124.5K',
      change: '-8.2%',
      changeType: 'positive',
      icon: DollarSign,
      subtitle: 'Optimized',
      trend: [
        { value: 135 }, { value: 132 }, { value: 128 }, { value: 126 }, { value: 124.5 }
      ]
    },
    {
      title: 'Risk Score',
      value: '32',
      change: '-5 points',
      changeType: 'positive',
      icon: AlertTriangle,
      subtitle: 'Low',
      trend: [
        { value: 37 }, { value: 36 }, { value: 34 }, { value: 33 }, { value: 32 }
      ]
    }
  ];

  const complianceTrendData = [
    { name: 'Mon', compliance: 92, target: 95, violations: 12 },
    { name: 'Tue', compliance: 93, target: 95, violations: 10 },
    { name: 'Wed', compliance: 91, target: 95, violations: 15 },
    { name: 'Thu', compliance: 94, target: 95, violations: 8 },
    { name: 'Fri', compliance: 94.5, target: 95, violations: 7 },
    { name: 'Sat', compliance: 95, target: 95, violations: 5 },
    { name: 'Sun', compliance: 94.5, target: 95, violations: 6 }
  ];

  const costBreakdownData = [
    { name: 'Compute', value: 45000, percentage: 36 },
    { name: 'Storage', value: 28000, percentage: 22 },
    { name: 'Network', value: 20000, percentage: 16 },
    { name: 'Database', value: 18000, percentage: 14 },
    { name: 'AI/ML', value: 8500, percentage: 7 },
    { name: 'Other', value: 5000, percentage: 5 }
  ];

  const riskDistributionData = [
    { category: 'Security', A: 85, B: 90, fullMark: 100 },
    { category: 'Compliance', A: 94, B: 95, fullMark: 100 },
    { category: 'Cost', A: 78, B: 85, fullMark: 100 },
    { category: 'Performance', A: 88, B: 92, fullMark: 100 },
    { category: 'Availability', A: 96, B: 98, fullMark: 100 },
    { category: 'Identity', A: 91, B: 93, fullMark: 100 }
  ];

  const recentAlerts = [
    {
      severity: 'critical',
      title: 'Unencrypted Storage Detected',
      description: 'Storage account "proddata001" has encryption disabled',
      resource: 'Storage Account',
      time: '5 minutes ago'
    },
    {
      severity: 'high',
      title: 'Cost Anomaly Detected',
      description: 'Unusual spike in compute costs (+45% from baseline)',
      resource: 'Virtual Machines',
      time: '1 hour ago'
    },
    {
      severity: 'medium',
      title: 'Policy Non-Compliance',
      description: '12 resources missing required tags',
      resource: 'Multiple',
      time: '2 hours ago'
    },
    {
      severity: 'low',
      title: 'Certificate Expiring Soon',
      description: 'SSL certificate expires in 14 days',
      resource: 'App Service',
      time: '3 hours ago'
    }
  ];

  const predictiveInsights = [
    {
      type: 'warning',
      title: 'Compliance Score Prediction',
      prediction: 'Expected to drop below 90% in 3 days if current trend continues',
      confidence: 87,
      timeline: 'Next 72 hours'
    },
    {
      type: 'danger',
      title: 'Cost Overrun Risk',
      prediction: 'VM scaling pattern indicates 23% budget overrun by month end',
      confidence: 92,
      timeline: 'End of month'
    },
    {
      type: 'success',
      title: 'Security Posture Improvement',
      prediction: 'Implementation of recommended policies will improve score by 15%',
      confidence: 94,
      timeline: 'Next 7 days'
    }
  ];

  const resourceUtilization = [
    { name: 'VM-Prod', cpu: 78, memory: 65, storage: 82 },
    { name: 'VM-Dev', cpu: 45, memory: 52, storage: 38 },
    { name: 'VM-Test', cpu: 62, memory: 71, storage: 55 },
    { name: 'AKS-Cluster', cpu: 89, memory: 84, storage: 76 },
    { name: 'SQL-DB', cpu: 54, memory: 68, storage: 91 },
    { name: 'App-Service', cpu: 42, memory: 38, storage: 45 }
  ];

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#6b7280'];

  useEffect(() => {
    // Simulate loading
    setTimeout(() => setLoading(false), 1000);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <PageContainer className="min-h-screen py-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
        <div>
          <h1 className="font-bold text-[var(--font-title)] text-gray-900 dark:text-gray-100">Executive Dashboard</h1>
          <p className="text-gray-500 mt-1">Real-time governance insights powered by 4 patented AI technologies</p>
        </div>
        <div className="flex items-center gap-3">
          <select 
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last Quarter</option>
          </select>
          <button
            type="button"
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
            onClick={() => toast({ title: 'Export started', description: `Generating ${timeRange} report...` })}>
            Export Report
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 lg:gap-4">
        {kpiData.map((kpi, index) => (
          <KPICard 
            key={index} 
            {...kpi}
            onClick={() => {
              if (kpi.title === 'Compliance Score') router.push('/governance/compliance');
              if (kpi.title === 'Monthly Cost') router.push('/governance/cost');
              if (kpi.title === 'Risk Score') router.push('/governance/risk');
            }}
          />
        ))}
      </div>

      {/* Patent #3: Unified Platform Metrics */}
      <div className="bg-gradient-to-r from-blue-900/10 to-purple-900/10 rounded-lg border border-blue-500/30 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Brain className="w-6 h-6 text-blue-400" />
            <div>
              <h2 className="text-xl font-bold">Unified AI-Driven Governance Platform</h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">Patent #3: Cross-Domain Metrics & Real-time Insights</p>
            </div>
          </div>
          <span className="text-xs bg-green-500/20 text-green-400 px-3 py-1 rounded-full">Live</span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <div className="bg-white/5 dark:bg-gray-900/50 rounded-lg p-3 text-center">
            <Cpu className="w-5 h-5 text-purple-400 mx-auto mb-2" />
            <p className="text-2xl font-bold">12</p>
            <p className="text-xs text-gray-400">AI Models Active</p>
          </div>
          <div className="bg-white/5 dark:bg-gray-900/50 rounded-lg p-3 text-center">
            <Activity className="w-5 h-5 text-green-400 mx-auto mb-2" />
            <p className="text-2xl font-bold">99.8%</p>
            <p className="text-xs text-gray-400">Service Uptime</p>
          </div>
          <div className="bg-white/5 dark:bg-gray-900/50 rounded-lg p-3 text-center">
            <Zap className="w-5 h-5 text-yellow-400 mx-auto mb-2" />
            <p className="text-2xl font-bold">&lt;50ms</p>
            <p className="text-xs text-gray-400">Avg Response</p>
          </div>
          <div className="bg-white/5 dark:bg-gray-900/50 rounded-lg p-3 text-center">
            <Database className="w-5 h-5 text-blue-400 mx-auto mb-2" />
            <p className="text-2xl font-bold">458TB</p>
            <p className="text-xs text-gray-400">Data Processed</p>
          </div>
          <div className="bg-white/5 dark:bg-gray-900/50 rounded-lg p-3 text-center">
            <Globe className="w-5 h-5 text-cyan-400 mx-auto mb-2" />
            <p className="text-2xl font-bold">3</p>
            <p className="text-xs text-gray-400">Cloud Providers</p>
          </div>
          <div className="bg-white/5 dark:bg-gray-900/50 rounded-lg p-3 text-center">
            <Users className="w-5 h-5 text-pink-400 mx-auto mb-2" />
            <p className="text-2xl font-bold">847</p>
            <p className="text-xs text-gray-400">Active Users</p>
          </div>
        </div>

        <div className="mt-4 p-3 bg-gray-800/50 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-medium">Cross-Domain Correlation Engine</p>
            <span className="text-xs text-green-400">Active</span>
          </div>
          <div className="grid grid-cols-4 gap-2 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-gray-400">Security: 98%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-gray-400">Compliance: 94%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
              <span className="text-gray-400">Cost: 87%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-gray-400">Performance: 96%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Compliance Trend */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Compliance Trend</h3>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 bg-blue-500 rounded-full"></span>
              <span className="text-sm text-gray-600 dark:text-gray-400">Score</span>
              <span className="w-3 h-3 bg-green-500 rounded-full ml-3"></span>
              <span className="text-sm text-gray-600 dark:text-gray-400">Target</span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={complianceTrendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} domain={[85, 100]} />
              <Tooltip />
              <Area 
                type="monotone" 
                dataKey="compliance" 
                stroke="#3b82f6" 
                fill="#3b82f6" 
                fillOpacity={0.1}
                strokeWidth={2}
              />
              <Area 
                type="monotone" 
                dataKey="target" 
                stroke="#10b981" 
                fill="transparent"
                strokeDasharray="5 5"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Cost Breakdown */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Cost Distribution</h3>
            <span className="text-sm text-gray-500 dark:text-gray-400">Total: $124.5K</span>
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={costBreakdownData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percentage }) => `${name} ${percentage}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {costBreakdownData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value: any) => `$${(value / 1000).toFixed(1)}K`} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Risk Distribution and Alerts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Radar */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Risk Assessment</h3>
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={riskDistributionData}>
              <PolarGrid stroke="#e5e7eb" />
              <PolarAngleAxis dataKey="category" fontSize={12} />
              <PolarRadiusAxis angle={90} domain={[0, 100]} fontSize={10} />
              <Radar 
                name="Current" 
                dataKey="A" 
                stroke="#3b82f6" 
                fill="#3b82f6" 
                fillOpacity={0.3}
              />
              <Radar 
                name="Target" 
                dataKey="B" 
                stroke="#10b981" 
                fill="#10b981" 
                fillOpacity={0.2}
              />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Alerts */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Recent Alerts</h3>
            <button type="button" 
              onClick={() => router.push('/operations/alerts')}
              className="text-sm text-blue-600 hover:text-blue-700 font-medium">
              View All →
            </button>
          </div>
          <div className="space-y-3 max-h-[300px] overflow-y-auto">
            {recentAlerts.map((alert, index) => (
              <AlertItem key={index} alert={alert} />
            ))}
          </div>
        </div>

        {/* Predictive Insights */}
        <PredictiveInsights insights={predictiveInsights} />
      </div>

      {/* Resource Utilization */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Resource Utilization</h3>
          <div className="flex items-center gap-4 text-sm">
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 bg-blue-500 rounded"></span>
              CPU
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 bg-green-500 rounded"></span>
              Memory
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 bg-purple-500 rounded"></span>
              Storage
            </span>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={resourceUtilization}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="name" fontSize={12} />
            <YAxis fontSize={12} />
            <Tooltip />
            <Bar dataKey="cpu" fill="#3b82f6" />
            <Bar dataKey="memory" fill="#10b981" />
            <Bar dataKey="storage" fill="#8b5cf6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Patent Technologies Performance */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="h-5 w-5 text-purple-600" />
          <h3 className="text-lg font-semibold text-gray-900">Patent Technologies Performance</h3>
          <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full">4 Patents</span>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-medium text-gray-600 mb-3">AI Model Accuracy</h4>
            <div className="space-y-3">
              {[
                { label: 'Predictive Compliance (Patent #4)', value: 99.2, color: '#8b5cf6' },
                { label: 'Cross-Domain Correlation (Patent #1)', value: 96.8, color: '#3b82f6' },
                { label: 'Conversational AI (Patent #2)', value: 98.7, color: '#10b981' },
                { label: 'Unified Platform (Patent #3)', value: 97.5, color: '#f59e0b' }
              ].map((item, idx) => (
                <div key={idx}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-700">{item.label}</span>
                    <span className="font-medium">{item.value}%</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full transition-all duration-500"
                      style={{ width: `${item.value}%`, backgroundColor: item.color }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h4 className="text-sm font-medium text-gray-600 mb-3">System Performance</h4>
            <div className="space-y-3">
              {[
                { label: 'API Response Time', value: 98, suffix: 'ms avg', color: '#10b981' },
                { label: 'Database Queries', value: 45, suffix: 'ms avg', color: '#3b82f6' },
                { label: 'ML Inference', value: 89, suffix: 'ms avg', color: '#8b5cf6' },
                { label: 'Cache Hit Rate', value: 94, suffix: '%', color: '#f59e0b' }
              ].map((item, idx) => (
                <div key={idx}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-700">{item.label}</span>
                    <span className="font-medium">{item.value}{item.suffix}</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full transition-all duration-500"
                      style={{ width: `${Math.min(item.value, 100)}%`, backgroundColor: item.color }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Cloud Integration Status */}
      <CloudIntegrationStatus />

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <button type="button" 
          onClick={() => router.push('/ai/chat')}
          className="p-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all shadow-sm"
        >
          <div className="flex items-center gap-3">
            <Brain className="h-5 w-5" />
            <div className="text-left">
              <p className="font-semibold">AI Assistant</p>
              <p className="text-xs opacity-90">Get instant help</p>
            </div>
            <ChevronRight className="h-4 w-4 ml-auto" />
          </div>
        </button>
        <button type="button" 
          onClick={() => router.push('/governance/policies')}
          className="p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:shadow-md transition-all"
        >
          <div className="flex items-center gap-3">
            <Shield className="h-5 w-5 text-gray-700" />
            <div className="text-left">
              <p className="font-semibold text-gray-900">Policies</p>
              <p className="text-xs text-gray-500">Manage rules</p>
            </div>
            <ChevronRight className="h-4 w-4 ml-auto text-gray-400" />
          </div>
        </button>
        <button type="button" 
          onClick={() => router.push('/operations/remediation')}
          className="p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:shadow-md transition-all"
        >
          <div className="flex items-center gap-3">
            <AlertTriangle className="h-5 w-5 text-gray-700" />
            <div className="text-left">
              <p className="font-semibold text-gray-900">Remediation</p>
              <p className="text-xs text-gray-500">Fix issues</p>
            </div>
            <ChevronRight className="h-4 w-4 ml-auto text-gray-400" />
          </div>
        </button>
        <button type="button" 
          onClick={() => router.push('/governance/cost')}
          className="p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:shadow-md transition-all"
        >
          <div className="flex items-center gap-3">
            <DollarSign className="h-5 w-5 text-gray-700" />
            <div className="text-left">
              <p className="font-semibold text-gray-900">Cost Analysis</p>
              <p className="text-xs text-gray-500">View spending</p>
            </div>
            <ChevronRight className="h-4 w-4 ml-auto text-gray-400" />
          </div>
        </button>
      </div>

      {/* System Status Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { icon: Users, title: 'Identity & Access', value: '1,247', subtitle: 'Active users', link: '/security/iam' },
          { icon: Shield, title: 'Security Policies', value: '89', subtitle: 'Active policies', link: '/security/rbac' },
          { icon: Server, title: 'Resources', value: '342', subtitle: 'Total resources', link: '/operations/resources' },
          { icon: GitBranch, title: 'Deployments', value: '12', subtitle: 'This week', link: '/devops/deployments' }
        ].map((item, idx) => (
          <button type="button"
            key={idx}
            onClick={() => router.push(item.link)}
            className="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-all text-left"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">{item.title}</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{item.value}</p>
                <p className="text-xs text-gray-500 mt-1">{item.subtitle}</p>
              </div>
              <item.icon className="h-5 w-5 text-gray-400" />
            </div>
          </button>
        ))}
      </div>
    </PageContainer>
  );
}