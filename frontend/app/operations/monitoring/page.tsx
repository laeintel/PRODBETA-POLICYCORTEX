'use client';

import React, { useState, useEffect } from 'react';
import {
  Activity,
  BarChart3,
  LineChart as LineChartIcon,
  TrendingUp,
  AlertCircle,
  Gauge,
  Zap,
  Clock,
  Filter,
  Plus,
  Settings,
  Download,
  RefreshCw,
  Maximize2,
  Eye,
  Database,
  Cpu,
  MemoryStick,
  Network,
  HardDrive,
  Shield,
  Search
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  RadialBarChart,
  RadialBar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
} from 'recharts';

interface Metric {
  id: string;
  name: string;
  category: string;
  value: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  threshold: {
    warning: number;
    critical: number;
  };
  status: 'normal' | 'warning' | 'critical';
  lastUpdated: string;
}

interface Dashboard {
  id: string;
  name: string;
  description: string;
  widgets: Widget[];
  createdBy: string;
  lastModified: string;
}

interface Widget {
  id: string;
  type: 'line' | 'area' | 'bar' | 'gauge' | 'number' | 'heatmap';
  title: string;
  metrics: string[];
  timeRange: string;
  refreshInterval: number;
  position: { x: number; y: number; w: number; h: number };
}

interface AlertRule {
  id: string;
  name: string;
  metric: string;
  condition: string;
  threshold: number;
  severity: 'info' | 'warning' | 'critical';
  enabled: boolean;
  notifications: string[];
}

export default function MonitoringObservabilityPage() {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [dashboards, setDashboards] = useState<Dashboard[]>([]);
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [selectedDashboard, setSelectedDashboard] = useState<Dashboard | null>(null);
  const [timeRange, setTimeRange] = useState('1h');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [anomalyData, setAnomalyData] = useState<any[]>([]);
  const [baselineData, setBaselineData] = useState<any[]>([]);

  useEffect(() => {
    fetchMonitoringData();
    const interval = autoRefresh ? setInterval(fetchMonitoringData, 30000) : null;
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [timeRange, autoRefresh]);

  const fetchMonitoringData = async () => {
    try {
      setLoading(true);

      // Mock metrics data
      const mockMetrics: Metric[] = [
        {
          id: 'cpu-usage',
          name: 'CPU Usage',
          category: 'Compute',
          value: 68.5,
          unit: '%',
          trend: 'up',
          threshold: { warning: 70, critical: 90 },
          status: 'normal',
          lastUpdated: new Date().toISOString()
        },
        {
          id: 'memory-usage',
          name: 'Memory Usage',
          category: 'Compute',
          value: 82.3,
          unit: '%',
          trend: 'up',
          threshold: { warning: 80, critical: 95 },
          status: 'warning',
          lastUpdated: new Date().toISOString()
        },
        {
          id: 'disk-iops',
          name: 'Disk IOPS',
          category: 'Storage',
          value: 4523,
          unit: 'ops/s',
          trend: 'stable',
          threshold: { warning: 5000, critical: 8000 },
          status: 'normal',
          lastUpdated: new Date().toISOString()
        },
        {
          id: 'network-throughput',
          name: 'Network Throughput',
          category: 'Network',
          value: 125.8,
          unit: 'MB/s',
          trend: 'down',
          threshold: { warning: 200, critical: 300 },
          status: 'normal',
          lastUpdated: new Date().toISOString()
        },
        {
          id: 'response-time',
          name: 'Response Time',
          category: 'Performance',
          value: 245,
          unit: 'ms',
          trend: 'up',
          threshold: { warning: 300, critical: 500 },
          status: 'normal',
          lastUpdated: new Date().toISOString()
        },
        {
          id: 'error-rate',
          name: 'Error Rate',
          category: 'Reliability',
          value: 0.12,
          unit: '%',
          trend: 'down',
          threshold: { warning: 1, critical: 5 },
          status: 'normal',
          lastUpdated: new Date().toISOString()
        }
      ];

      // Mock performance data
      const mockPerformanceData = [];
      const now = Date.now();
      for (let i = 59; i >= 0; i--) {
        mockPerformanceData.push({
          time: new Date(now - i * 60000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          cpu: Math.random() * 30 + 50,
          memory: Math.random() * 20 + 70,
          disk: Math.random() * 40 + 30,
          network: Math.random() * 50 + 25,
          responseTime: Math.random() * 100 + 150
        });
      }

      // Mock anomaly data
      const mockAnomalyData = [
        { time: '10:15', metric: 'CPU', actual: 95, predicted: 65, anomalyScore: 0.92 },
        { time: '11:30', metric: 'Memory', actual: 98, predicted: 75, anomalyScore: 0.88 },
        { time: '14:45', metric: 'Network', actual: 285, predicted: 120, anomalyScore: 0.85 },
        { time: '16:20', metric: 'Disk', actual: 7500, predicted: 4000, anomalyScore: 0.79 }
      ];

      // Mock baseline data
      const mockBaselineData = [];
      for (let i = 0; i < 24; i++) {
        mockBaselineData.push({
          hour: `${i}:00`,
          currentWeek: Math.random() * 30 + 50,
          lastWeek: Math.random() * 30 + 45,
          baseline: Math.random() * 30 + 48
        });
      }

      // Mock dashboards
      const mockDashboards: Dashboard[] = [
        {
          id: 'dash-1',
          name: 'Infrastructure Overview',
          description: 'Overall system health and performance metrics',
          widgets: [],
          createdBy: 'admin',
          lastModified: new Date().toISOString()
        },
        {
          id: 'dash-2',
          name: 'Application Performance',
          description: 'Application-level metrics and KPIs',
          widgets: [],
          createdBy: 'devops',
          lastModified: new Date().toISOString()
        }
      ];

      // Mock alert rules
      const mockAlertRules: AlertRule[] = [
        {
          id: 'alert-1',
          name: 'High CPU Usage',
          metric: 'cpu-usage',
          condition: 'greater_than',
          threshold: 90,
          severity: 'critical',
          enabled: true,
          notifications: ['email', 'slack']
        },
        {
          id: 'alert-2',
          name: 'Memory Pressure',
          metric: 'memory-usage',
          condition: 'greater_than',
          threshold: 85,
          severity: 'warning',
          enabled: true,
          notifications: ['email']
        },
        {
          id: 'alert-3',
          name: 'High Error Rate',
          metric: 'error-rate',
          condition: 'greater_than',
          threshold: 1,
          severity: 'critical',
          enabled: true,
          notifications: ['email', 'pagerduty']
        }
      ];

      setMetrics(mockMetrics);
      setPerformanceData(mockPerformanceData);
      setAnomalyData(mockAnomalyData);
      setBaselineData(mockBaselineData);
      setDashboards(mockDashboards);
      setAlertRules(mockAlertRules);
    } catch (error) {
      console.error('Error fetching monitoring data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'normal': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'critical': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getTrendIcon = (trend: string) => {
    if (trend === 'up') return <TrendingUp className="w-3 h-3 text-green-400" />;
    if (trend === 'down') return <TrendingUp className="w-3 h-3 text-red-400 rotate-180" />;
    return <span className="w-3 h-3 text-gray-400">-</span>;
  };

  const filteredMetrics = metrics.filter(metric =>
    metric.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    metric.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const criticalMetrics = metrics.filter(m => m.status === 'critical').length;
  const warningMetrics = metrics.filter(m => m.status === 'warning').length;
  const normalMetrics = metrics.filter(m => m.status === 'normal').length;

  const gaugeData = [
    { name: 'CPU', value: 68, fill: '#3b82f6' },
    { name: 'Memory', value: 82, fill: '#10b981' },
    { name: 'Disk', value: 45, fill: '#f59e0b' },
    { name: 'Network', value: 62, fill: '#8b5cf6' }
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-400">Loading monitoring dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                <Activity className="w-8 h-8 text-blue-500" />
                Operations Monitoring
              </h1>
              <p className="text-gray-400 mt-2">Real-time metrics and performance monitoring</p>
            </div>
            <div className="flex items-center gap-4">
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="bg-gray-900 text-white px-4 py-2 rounded-lg border border-gray-800 focus:border-blue-500 focus:outline-none"
              >
                <option value="5m">Last 5 minutes</option>
                <option value="15m">Last 15 minutes</option>
                <option value="1h">Last hour</option>
                <option value="6h">Last 6 hours</option>
                <option value="24h">Last 24 hours</option>
                <option value="7d">Last 7 days</option>
              </select>
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
                  autoRefresh ? 'bg-green-600 hover:bg-green-700 text-white' : 'bg-gray-800 hover:bg-gray-700 text-gray-300'
                }`}
              >
                <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
                Auto Refresh
              </button>
              <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors">
                <Plus className="w-4 h-4" />
                New Dashboard
              </button>
            </div>
          </div>
        </div>

        {/* Metrics Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-between mb-4">
              <Gauge className="w-8 h-8 text-blue-500" />
              <span className="text-2xl font-bold text-white">{metrics.length}</span>
            </div>
            <h3 className="text-gray-400 text-sm">Total Metrics</h3>
            <p className="text-xs text-gray-500 mt-1">Being monitored</p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-between mb-4">
              <Shield className="w-8 h-8 text-green-500" />
              <span className="text-2xl font-bold text-green-400">{normalMetrics}</span>
            </div>
            <h3 className="text-gray-400 text-sm">Healthy</h3>
            <p className="text-xs text-gray-500 mt-1">Within thresholds</p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-between mb-4">
              <AlertCircle className="w-8 h-8 text-yellow-500" />
              <span className="text-2xl font-bold text-yellow-400">{warningMetrics}</span>
            </div>
            <h3 className="text-gray-400 text-sm">Warnings</h3>
            <p className="text-xs text-gray-500 mt-1">Approaching limits</p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-between mb-4">
              <Zap className="w-8 h-8 text-red-500" />
              <span className="text-2xl font-bold text-red-400">{criticalMetrics}</span>
            </div>
            <h3 className="text-gray-400 text-sm">Critical</h3>
            <p className="text-xs text-gray-500 mt-1">Requires attention</p>
          </div>
        </div>

        {/* Performance Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Real-time Performance */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <LineChartIcon className="w-5 h-5 text-blue-500" />
              Real-time Performance
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  labelStyle={{ color: '#f3f4f6' }}
                />
                <Legend />
                <Line type="monotone" dataKey="cpu" stroke="#3b82f6" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="memory" stroke="#10b981" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="disk" stroke="#f59e0b" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="network" stroke="#8b5cf6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Resource Utilization Gauges */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <Gauge className="w-5 h-5 text-green-500" />
              Resource Utilization
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <RadialBarChart cx="50%" cy="50%" innerRadius="10%" outerRadius="90%" data={gaugeData}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis type="number" domain={[0, 100]} />
                <RadialBar dataKey="value" cornerRadius={10} fill="#8884d8" />
                <Legend />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  formatter={(value: any) => `${value}%`}
                />
              </RadialBarChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-2 gap-4 mt-4">
              {gaugeData.map((item, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-gray-800 rounded">
                  <span className="text-sm text-gray-300">{item.name}</span>
                  <span className="text-sm font-medium text-white">{item.value}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Baseline Comparison */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 mb-8">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-purple-500" />
            Performance Baseline Comparison
          </h2>
          <ResponsiveContainer width="100%" height={250}>
            <ComposedChart data={baselineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="hour" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend />
              <Area type="monotone" dataKey="baseline" fill="#8b5cf6" stroke="#8b5cf6" fillOpacity={0.2} />
              <Line type="monotone" dataKey="currentWeek" stroke="#3b82f6" strokeWidth={2} />
              <Line type="monotone" dataKey="lastWeek" stroke="#10b981" strokeWidth={2} strokeDasharray="5 5" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Metrics Grid */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-white">Metrics Details</h2>
            <div className="flex items-center gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search metrics..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-gray-800 text-white pl-10 pr-4 py-2 rounded-lg border border-gray-700 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <button className="text-gray-400 hover:text-white">
                <Filter className="w-5 h-5" />
              </button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gray-800">
                  <th className="pb-3">Metric</th>
                  <th className="pb-3">Category</th>
                  <th className="pb-3">Value</th>
                  <th className="pb-3">Status</th>
                  <th className="pb-3">Trend</th>
                  <th className="pb-3">Threshold</th>
                  <th className="pb-3">Last Updated</th>
                </tr>
              </thead>
              <tbody>
                {filteredMetrics.map((metric) => (
                  <tr key={metric.id} className="border-b border-gray-800 hover:bg-gray-800/50">
                    <td className="py-3 text-white font-medium">{metric.name}</td>
                    <td className="py-3 text-gray-300">{metric.category}</td>
                    <td className="py-3">
                      <span className="text-white font-medium">
                        {metric.value} {metric.unit}
                      </span>
                    </td>
                    <td className="py-3">
                      <span className={`capitalize ${getStatusColor(metric.status)}`}>
                        {metric.status}
                      </span>
                    </td>
                    <td className="py-3">{getTrendIcon(metric.trend)}</td>
                    <td className="py-3 text-sm text-gray-400">
                      W: {metric.threshold.warning} / C: {metric.threshold.critical}
                    </td>
                    <td className="py-3 text-sm text-gray-400">
                      {new Date(metric.lastUpdated).toLocaleTimeString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Alert Rules */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-white">Alert Rules</h2>
            <button className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm flex items-center gap-2">
              <Plus className="w-4 h-4" />
              Add Rule
            </button>
          </div>
          <div className="space-y-3">
            {alertRules.map((rule) => (
              <div key={rule.id} className="bg-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`w-2 h-2 rounded-full ${rule.enabled ? 'bg-green-400' : 'bg-gray-400'}`}></div>
                    <div>
                      <h3 className="text-white font-medium">{rule.name}</h3>
                      <p className="text-sm text-gray-400">
                        {rule.metric} {rule.condition} {rule.threshold}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`text-xs px-2 py-1 rounded ${
                      rule.severity === 'critical' ? 'bg-red-900/50 text-red-400' :
                      rule.severity === 'warning' ? 'bg-yellow-900/50 text-yellow-400' :
                      'bg-blue-900/50 text-blue-400'
                    }`}>
                      {rule.severity}
                    </span>
                    <button className="text-gray-400 hover:text-white">
                      <Settings className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}