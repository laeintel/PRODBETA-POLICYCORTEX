'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  DollarSign,
  TrendingDown,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  Zap,
  Database,
  Cloud,
  Server,
  HardDrive,
  Network,
  Shield,
  BarChart3,
  PieChart,
  Target,
  Clock,
  Filter,
  Download,
  Settings,
  ChevronRight,
  Info,
  AlertCircle
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
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

interface CostAlert {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  title: string;
  description: string;
  cost: number;
  impact: string;
  recommendation: string;
  resource: string;
  timestamp: Date;
}

interface ResourceOptimization {
  id: string;
  resourceType: string;
  resourceName: string;
  currentCost: number;
  optimizedCost: number;
  savings: number;
  savingsPercentage: number;
  recommendation: string;
  effort: 'low' | 'medium' | 'high';
  impact: 'low' | 'medium' | 'high';
  status: 'identified' | 'in-progress' | 'completed';
}

interface CostTrend {
  date: string;
  actual: number;
  budget: number;
  forecast: number;
}

interface ServiceCost {
  service: string;
  cost: number;
  change: number;
  percentage: number;
  trend: 'up' | 'down' | 'stable';
}

export default function CostGovernanceTactical() {
  const [loading, setLoading] = useState(true);
  const [selectedView, setSelectedView] = useState<'overview' | 'optimization' | 'alerts' | 'analysis'>('overview');
  const [alerts, setAlerts] = useState<CostAlert[]>([]);
  const [optimizations, setOptimizations] = useState<ResourceOptimization[]>([]);
  const [costTrends, setCostTrends] = useState<CostTrend[]>([]);
  const [serviceCosts, setServiceCosts] = useState<ServiceCost[]>([]);
  const [radarData, setRadarData] = useState<any[]>([]);

  useEffect(() => {
    // Simulate loading data
    setTimeout(() => {
      // Mock alerts
      setAlerts([
        {
          id: '1',
          severity: 'critical',
          title: 'Unexpected spike in compute costs',
          description: 'VM instances in West US 2 region showing 150% cost increase',
          cost: 45000,
          impact: 'High - Affecting monthly budget',
          recommendation: 'Review and rightsize VM instances, consider reserved instances',
          resource: 'Virtual Machines',
          timestamp: new Date('2024-01-09T10:30:00')
        },
        {
          id: '2',
          severity: 'warning',
          title: 'Unattached storage volumes detected',
          description: '23 storage volumes not attached to any instances',
          cost: 8500,
          impact: 'Medium - Monthly recurring waste',
          recommendation: 'Delete or archive unattached volumes',
          resource: 'Storage',
          timestamp: new Date('2024-01-09T09:15:00')
        },
        {
          id: '3',
          severity: 'warning',
          title: 'Idle database instances',
          description: '5 database instances with < 5% utilization',
          cost: 12000,
          impact: 'Medium - Underutilized resources',
          recommendation: 'Consolidate or downsize database instances',
          resource: 'Databases',
          timestamp: new Date('2024-01-09T08:45:00')
        },
        {
          id: '4',
          severity: 'info',
          title: 'Reserved instance recommendations',
          description: 'Potential 30% savings with 1-year reserved instances',
          cost: 25000,
          impact: 'Low - Cost optimization opportunity',
          recommendation: 'Purchase reserved instances for steady-state workloads',
          resource: 'Compute',
          timestamp: new Date('2024-01-09T07:30:00')
        }
      ]);

      // Mock optimizations
      setOptimizations([
        {
          id: '1',
          resourceType: 'Virtual Machine',
          resourceName: 'prod-web-server-01',
          currentCost: 1500,
          optimizedCost: 900,
          savings: 600,
          savingsPercentage: 40,
          recommendation: 'Downsize from D8s_v3 to D4s_v3',
          effort: 'low',
          impact: 'low',
          status: 'identified'
        },
        {
          id: '2',
          resourceType: 'Storage Account',
          resourceName: 'backupstorageacct',
          currentCost: 3200,
          optimizedCost: 1800,
          savings: 1400,
          savingsPercentage: 43.75,
          recommendation: 'Move to Archive tier for old backups',
          effort: 'medium',
          impact: 'low',
          status: 'in-progress'
        },
        {
          id: '3',
          resourceType: 'SQL Database',
          resourceName: 'analytics-db-prod',
          currentCost: 4500,
          optimizedCost: 2800,
          savings: 1700,
          savingsPercentage: 37.78,
          recommendation: 'Switch to Serverless compute tier',
          effort: 'high',
          impact: 'medium',
          status: 'identified'
        },
        {
          id: '4',
          resourceType: 'App Service',
          resourceName: 'api-gateway-service',
          currentCost: 2100,
          optimizedCost: 1400,
          savings: 700,
          savingsPercentage: 33.33,
          recommendation: 'Consolidate to shared App Service Plan',
          effort: 'medium',
          impact: 'low',
          status: 'completed'
        },
        {
          id: '5',
          resourceType: 'Load Balancer',
          resourceName: 'lb-frontend-prod',
          currentCost: 800,
          optimizedCost: 500,
          savings: 300,
          savingsPercentage: 37.5,
          recommendation: 'Switch to Basic SKU for internal traffic',
          effort: 'low',
          impact: 'low',
          status: 'identified'
        }
      ]);

      // Mock cost trends
      setCostTrends([
        { date: 'Jan', actual: 125000, budget: 130000, forecast: 128000 },
        { date: 'Feb', actual: 118000, budget: 130000, forecast: 125000 },
        { date: 'Mar', actual: 135000, budget: 130000, forecast: 132000 },
        { date: 'Apr', actual: 142000, budget: 135000, forecast: 140000 },
        { date: 'May', actual: 138000, budget: 135000, forecast: 138000 },
        { date: 'Jun', actual: 145000, budget: 140000, forecast: 143000 },
        { date: 'Jul', actual: 152000, budget: 140000, forecast: 150000 },
        { date: 'Aug', actual: 148000, budget: 145000, forecast: 148000 },
        { date: 'Sep', actual: 155000, budget: 145000, forecast: 153000 },
        { date: 'Oct', actual: 162000, budget: 150000, forecast: 160000 }
      ]);

      // Mock service costs
      setServiceCosts([
        { service: 'Compute', cost: 45000, change: 12, percentage: 28, trend: 'up' },
        { service: 'Storage', cost: 32000, change: -5, percentage: 20, trend: 'down' },
        { service: 'Networking', cost: 28000, change: 8, percentage: 17, trend: 'up' },
        { service: 'Databases', cost: 25000, change: 3, percentage: 15, trend: 'stable' },
        { service: 'Analytics', cost: 18000, change: -10, percentage: 11, trend: 'down' },
        { service: 'Security', cost: 14000, change: 15, percentage: 9, trend: 'up' }
      ]);

      // Mock radar data for cost efficiency
      setRadarData([
        { metric: 'Resource Utilization', current: 65, target: 85 },
        { metric: 'Reserved Instances', current: 40, target: 70 },
        { metric: 'Spot Usage', current: 25, target: 50 },
        { metric: 'Rightsizing', current: 55, target: 80 },
        { metric: 'Tagging Compliance', current: 75, target: 95 },
        { metric: 'Budget Adherence', current: 70, target: 90 }
      ]);

      setLoading(false);
    }, 1000);
  }, []);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0
    }).format(value);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30';
      case 'warning':
        return 'text-amber-600 bg-amber-100 dark:text-amber-400 dark:bg-amber-900/30';
      case 'info':
        return 'text-blue-600 bg-blue-100 dark:text-blue-400 dark:bg-blue-900/30';
      default:
        return 'text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-900/30';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <XCircle className="h-5 w-5" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5" />;
      case 'info':
        return <Info className="h-5 w-5" />;
      default:
        return <AlertCircle className="h-5 w-5" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading cost governance data...</p>
        </div>
      </div>
    );
  }

  const totalCurrentCost = optimizations.reduce((sum, opt) => sum + opt.currentCost, 0);
  const totalSavings = optimizations.reduce((sum, opt) => sum + opt.savings, 0);
  const totalAlertCost = alerts.reduce((sum, alert) => sum + alert.cost, 0);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 pb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
              <DollarSign className="h-8 w-8 text-green-600" />
              Cost Governance Tactical View
            </h1>
            <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
              Real-time cost optimization and anomaly detection
            </p>
          </div>
          <div className="flex gap-2">
            <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
              <Download className="h-4 w-4" />
              Export Report
            </button>
            <button className="px-4 py-2 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Configure
            </button>
          </div>
        </div>
      </div>

      {/* View Selector */}
      <div className="flex gap-2 p-1 bg-gray-100 dark:bg-gray-800 rounded-lg w-fit">
        {(['overview', 'optimization', 'alerts', 'analysis'] as const).map((view) => (
          <button
            key={view}
            onClick={() => setSelectedView(view)}
            className={`px-4 py-2 rounded-md font-medium transition-colors ${
              selectedView === view
                ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            {view.charAt(0).toUpperCase() + view.slice(1)}
          </button>
        ))}
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-red-500 to-pink-600 text-white">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-red-100">Alert Impact</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCurrency(totalAlertCost)}</div>
            <div className="text-sm text-red-100 mt-1">{alerts.length} active alerts</div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-500 to-emerald-600 text-white">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-green-100">Potential Savings</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCurrency(totalSavings)}</div>
            <div className="text-sm text-green-100 mt-1">
              {optimizations.length} optimizations available
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-blue-500 to-cyan-600 text-white">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-blue-100">Current Spend</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCurrency(162000)}</div>
            <div className="text-sm text-blue-100 mt-1">+8% from budget</div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-500 to-indigo-600 text-white">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-purple-100">Efficiency Score</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">72%</div>
            <div className="text-sm text-purple-100 mt-1">+5% from last month</div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content based on selected view */}
      {selectedView === 'overview' && (
        <>
          {/* Cost Trends Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Cost Trends & Forecast</CardTitle>
              <CardDescription>Monthly actual vs budget with forecast</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={costTrends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip formatter={(value: number) => formatCurrency(value)} />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="budget"
                    stroke="#94A3B8"
                    fill="#E2E8F0"
                    fillOpacity={0.3}
                    name="Budget"
                  />
                  <Area
                    type="monotone"
                    dataKey="actual"
                    stroke="#3B82F6"
                    fill="#93C5FD"
                    fillOpacity={0.6}
                    name="Actual"
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    stroke="#10B981"
                    strokeDasharray="5 5"
                    name="Forecast"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Service Costs */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Service Cost Breakdown</CardTitle>
                <CardDescription>Cost distribution across Azure services</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {serviceCosts.map((service) => (
                    <div key={service.service} className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
                          {service.service === 'Compute' && <Server className="h-5 w-5 text-blue-600" />}
                          {service.service === 'Storage' && <HardDrive className="h-5 w-5 text-green-600" />}
                          {service.service === 'Networking' && <Network className="h-5 w-5 text-purple-600" />}
                          {service.service === 'Databases' && <Database className="h-5 w-5 text-orange-600" />}
                          {service.service === 'Analytics' && <BarChart3 className="h-5 w-5 text-indigo-600" />}
                          {service.service === 'Security' && <Shield className="h-5 w-5 text-red-600" />}
                        </div>
                        <div>
                          <div className="font-medium">{service.service}</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {service.percentage}% of total
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">{formatCurrency(service.cost)}</div>
                        <div className={`text-sm flex items-center gap-1 justify-end ${
                          service.trend === 'up' ? 'text-red-600 dark:text-red-400' :
                          service.trend === 'down' ? 'text-green-600 dark:text-green-400' :
                          'text-gray-600 dark:text-gray-400'
                        }`}>
                          {service.trend === 'up' && <TrendingUp className="h-4 w-4" />}
                          {service.trend === 'down' && <TrendingDown className="h-4 w-4" />}
                          {service.change > 0 ? '+' : ''}{service.change}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Cost Efficiency Metrics</CardTitle>
                <CardDescription>Current vs target performance</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <RadarChart data={radarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" className="text-xs" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar
                      name="Current"
                      dataKey="current"
                      stroke="#3B82F6"
                      fill="#3B82F6"
                      fillOpacity={0.6}
                    />
                    <Radar
                      name="Target"
                      dataKey="target"
                      stroke="#10B981"
                      fill="#10B981"
                      fillOpacity={0.3}
                    />
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </>
      )}

      {selectedView === 'optimization' && (
        <Card>
          <CardHeader>
            <CardTitle>Resource Optimization Opportunities</CardTitle>
            <CardDescription>Identified cost savings with implementation recommendations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {optimizations.map((opt) => (
                <div key={opt.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <h4 className="font-semibold text-lg">{opt.resourceName}</h4>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                          opt.status === 'completed' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                          opt.status === 'in-progress' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                          'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
                        }`}>
                          {opt.status}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{opt.resourceType}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                        {formatCurrency(opt.savings)}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {opt.savingsPercentage}% savings
                      </div>
                    </div>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 mb-3">
                    <p className="text-sm">{opt.recommendation}</p>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex gap-4">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Current:</span>
                        <span className="font-medium">{formatCurrency(opt.currentCost)}</span>
                      </div>
                      <ChevronRight className="h-4 w-4 text-gray-400" />
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Optimized:</span>
                        <span className="font-medium text-green-600 dark:text-green-400">
                          {formatCurrency(opt.optimizedCost)}
                        </span>
                      </div>
                    </div>
                    <div className="flex gap-3">
                      <div className="flex items-center gap-1">
                        <Zap className="h-4 w-4 text-gray-400" />
                        <span className="text-sm">Effort: {opt.effort}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Target className="h-4 w-4 text-gray-400" />
                        <span className="text-sm">Impact: {opt.impact}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {selectedView === 'alerts' && (
        <Card>
          <CardHeader>
            <CardTitle>Active Cost Alerts</CardTitle>
            <CardDescription>Real-time anomalies and threshold violations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {alerts.map((alert) => (
                <div key={alert.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-lg ${getSeverityColor(alert.severity)}`}>
                      {getSeverityIcon(alert.severity)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-semibold text-lg">{alert.title}</h4>
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          {alert.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-gray-600 dark:text-gray-400 mb-3">{alert.description}</p>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3">
                        <div>
                          <span className="text-sm text-gray-500 dark:text-gray-400">Cost Impact</span>
                          <div className="font-semibold text-red-600 dark:text-red-400">
                            {formatCurrency(alert.cost)}
                          </div>
                        </div>
                        <div>
                          <span className="text-sm text-gray-500 dark:text-gray-400">Resource</span>
                          <div className="font-medium">{alert.resource}</div>
                        </div>
                        <div>
                          <span className="text-sm text-gray-500 dark:text-gray-400">Impact Level</span>
                          <div className="font-medium">{alert.impact}</div>
                        </div>
                      </div>
                      <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                        <div className="flex items-start gap-2">
                          <CheckCircle className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                          <div>
                            <div className="font-medium text-blue-900 dark:text-blue-300 mb-1">
                              Recommendation
                            </div>
                            <p className="text-sm text-blue-800 dark:text-blue-400">
                              {alert.recommendation}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {selectedView === 'analysis' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Cost Distribution by Service</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={serviceCosts}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="service" />
                  <YAxis />
                  <Tooltip formatter={(value: number) => formatCurrency(value)} />
                  <Bar dataKey="cost" fill="#3B82F6" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Optimization Impact Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={optimizations.slice(0, 5)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="resourceName" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip formatter={(value: number) => formatCurrency(value)} />
                  <Legend />
                  <Bar dataKey="currentCost" fill="#EF4444" name="Current" />
                  <Bar dataKey="optimizedCost" fill="#10B981" name="Optimized" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}