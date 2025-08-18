'use client';

import React, { useState, useEffect } from 'react';
import { 
  Gauge, Activity, TrendingUp, TrendingDown, Clock, Timer, Zap,
  Server, Cpu, HardDrive, Network, Database, Cloud, BarChart,
  LineChart, AlertCircle, CheckCircle, XCircle, Info, RefreshCw,
  Download, Settings, Filter, ChevronRight, ArrowUp, ArrowDown
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface PerformanceMetric {
  id: string;
  name: string;
  category: 'latency' | 'throughput' | 'cpu' | 'memory' | 'disk' | 'network';
  current: number;
  average: number;
  peak: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  trendPercent: number;
  status: 'optimal' | 'acceptable' | 'warning' | 'critical';
  threshold: {
    optimal: number;
    acceptable: number;
    warning: number;
    critical: number;
  };
  history: {
    timestamp: string;
    value: number;
  }[];
}

interface ServicePerformance {
  id: string;
  name: string;
  status: 'healthy' | 'degraded' | 'down';
  responseTime: number;
  errorRate: number;
  requestRate: number;
  uptime: number;
  sla: number;
  dependencies: string[];
  metrics: PerformanceMetric[];
}

interface PerformanceAlert {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  metric: string;
  service: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

export default function PerformanceMonitoring() {
  const [metrics, setMetrics] = useState<PerformanceMetric[]>([]);
  const [services, setServices] = useState<ServicePerformance[]>([]);
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [viewMode, setViewMode] = useState<'overview' | 'detailed'>('overview');

  useEffect(() => {
    // Initialize with mock performance data
    setMetrics([
      {
        id: 'PERF-001',
        name: 'API Response Time',
        category: 'latency',
        current: 145,
        average: 132,
        peak: 450,
        unit: 'ms',
        trend: 'down',
        trendPercent: -8.5,
        status: 'optimal',
        threshold: {
          optimal: 200,
          acceptable: 300,
          warning: 500,
          critical: 1000
        },
        history: [
          { timestamp: '1h ago', value: 160 },
          { timestamp: '45m ago', value: 155 },
          { timestamp: '30m ago', value: 150 },
          { timestamp: '15m ago', value: 148 },
          { timestamp: 'Now', value: 145 }
        ]
      },
      {
        id: 'PERF-002',
        name: 'Database Query Time',
        category: 'latency',
        current: 25,
        average: 22,
        peak: 85,
        unit: 'ms',
        trend: 'stable',
        trendPercent: 1.2,
        status: 'optimal',
        threshold: {
          optimal: 50,
          acceptable: 100,
          warning: 200,
          critical: 500
        },
        history: [
          { timestamp: '1h ago', value: 23 },
          { timestamp: '45m ago', value: 24 },
          { timestamp: '30m ago', value: 22 },
          { timestamp: '15m ago', value: 24 },
          { timestamp: 'Now', value: 25 }
        ]
      },
      {
        id: 'PERF-003',
        name: 'Request Throughput',
        category: 'throughput',
        current: 1250,
        average: 1180,
        peak: 2100,
        unit: 'req/s',
        trend: 'up',
        trendPercent: 5.9,
        status: 'optimal',
        threshold: {
          optimal: 800,
          acceptable: 600,
          warning: 400,
          critical: 200
        },
        history: [
          { timestamp: '1h ago', value: 1100 },
          { timestamp: '45m ago', value: 1150 },
          { timestamp: '30m ago', value: 1200 },
          { timestamp: '15m ago', value: 1230 },
          { timestamp: 'Now', value: 1250 }
        ]
      },
      {
        id: 'PERF-004',
        name: 'CPU Utilization',
        category: 'cpu',
        current: 68,
        average: 65,
        peak: 92,
        unit: '%',
        trend: 'up',
        trendPercent: 4.6,
        status: 'acceptable',
        threshold: {
          optimal: 60,
          acceptable: 75,
          warning: 85,
          critical: 95
        },
        history: [
          { timestamp: '1h ago', value: 62 },
          { timestamp: '45m ago', value: 64 },
          { timestamp: '30m ago', value: 66 },
          { timestamp: '15m ago', value: 67 },
          { timestamp: 'Now', value: 68 }
        ]
      },
      {
        id: 'PERF-005',
        name: 'Memory Usage',
        category: 'memory',
        current: 72,
        average: 70,
        peak: 88,
        unit: '%',
        trend: 'stable',
        trendPercent: 0.8,
        status: 'acceptable',
        threshold: {
          optimal: 65,
          acceptable: 80,
          warning: 90,
          critical: 95
        },
        history: [
          { timestamp: '1h ago', value: 70 },
          { timestamp: '45m ago', value: 71 },
          { timestamp: '30m ago', value: 71 },
          { timestamp: '15m ago', value: 72 },
          { timestamp: 'Now', value: 72 }
        ]
      },
      {
        id: 'PERF-006',
        name: 'Disk IOPS',
        category: 'disk',
        current: 8500,
        average: 7800,
        peak: 12000,
        unit: 'ops/s',
        trend: 'up',
        trendPercent: 8.9,
        status: 'warning',
        threshold: {
          optimal: 6000,
          acceptable: 8000,
          warning: 10000,
          critical: 15000
        },
        history: [
          { timestamp: '1h ago', value: 7500 },
          { timestamp: '45m ago', value: 7800 },
          { timestamp: '30m ago', value: 8100 },
          { timestamp: '15m ago', value: 8300 },
          { timestamp: 'Now', value: 8500 }
        ]
      },
      {
        id: 'PERF-007',
        name: 'Network Bandwidth',
        category: 'network',
        current: 850,
        average: 780,
        peak: 1200,
        unit: 'Mbps',
        trend: 'up',
        trendPercent: 8.9,
        status: 'optimal',
        threshold: {
          optimal: 1000,
          acceptable: 800,
          warning: 600,
          critical: 400
        },
        history: [
          { timestamp: '1h ago', value: 750 },
          { timestamp: '45m ago', value: 780 },
          { timestamp: '30m ago', value: 810 },
          { timestamp: '15m ago', value: 830 },
          { timestamp: 'Now', value: 850 }
        ]
      }
    ]);

    setServices([
      {
        id: 'SVC-001',
        name: 'API Gateway',
        status: 'healthy',
        responseTime: 145,
        errorRate: 0.02,
        requestRate: 1250,
        uptime: 99.98,
        sla: 99.95,
        dependencies: ['auth-service', 'database'],
        metrics: []
      },
      {
        id: 'SVC-002',
        name: 'Auth Service',
        status: 'healthy',
        responseTime: 85,
        errorRate: 0.01,
        requestRate: 450,
        uptime: 99.99,
        sla: 99.9,
        dependencies: ['database', 'cache'],
        metrics: []
      },
      {
        id: 'SVC-003',
        name: 'Database',
        status: 'degraded',
        responseTime: 25,
        errorRate: 0.05,
        requestRate: 2100,
        uptime: 99.5,
        sla: 99.9,
        dependencies: [],
        metrics: []
      },
      {
        id: 'SVC-004',
        name: 'Cache Service',
        status: 'healthy',
        responseTime: 2,
        errorRate: 0.001,
        requestRate: 5000,
        uptime: 100,
        sla: 99.5,
        dependencies: [],
        metrics: []
      }
    ]);

    setAlerts([
      {
        id: 'ALERT-001',
        severity: 'warning',
        metric: 'Disk IOPS',
        service: 'Database',
        message: 'Disk IOPS approaching warning threshold',
        timestamp: '5 minutes ago',
        acknowledged: false
      },
      {
        id: 'ALERT-002',
        severity: 'info',
        metric: 'CPU Utilization',
        service: 'API Gateway',
        message: 'CPU usage trending upward',
        timestamp: '15 minutes ago',
        acknowledged: true
      }
    ]);

    // Simulate real-time updates
    if (autoRefresh) {
      const interval = setInterval(() => {
        setMetrics(prevMetrics => 
          prevMetrics.map(metric => ({
            ...metric,
            current: metric.current + (Math.random() - 0.5) * 10,
            history: [
              ...metric.history.slice(1),
              { timestamp: 'Now', value: metric.current + (Math.random() - 0.5) * 10 }
            ]
          }))
        );
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'optimal':
      case 'healthy': return 'text-green-500 bg-green-900/20';
      case 'acceptable': return 'text-blue-500 bg-blue-900/20';
      case 'warning':
      case 'degraded': return 'text-yellow-500 bg-yellow-900/20';
      case 'critical':
      case 'down': return 'text-red-500 bg-red-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch(category) {
      case 'latency': return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'throughput': return <Zap className="w-4 h-4 text-blue-500" />;
      case 'cpu': return <Cpu className="w-4 h-4 text-purple-500" />;
      case 'memory': return <Server className="w-4 h-4 text-green-500" />;
      case 'disk': return <HardDrive className="w-4 h-4 text-orange-500" />;
      case 'network': return <Network className="w-4 h-4 text-cyan-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const getTrendIcon = (trend: string, percent: number) => {
    if (trend === 'up') return <ArrowUp className={`w-3 h-3 ${percent > 5 ? 'text-red-500' : 'text-yellow-500'}`} />;
    if (trend === 'down') return <ArrowDown className={`w-3 h-3 ${percent < -5 ? 'text-green-500' : 'text-blue-500'}`} />;
    return <span className="w-3 h-3 text-gray-500">→</span>;
  };

  const filteredMetrics = metrics.filter(metric => {
    if (selectedCategory !== 'all' && metric.category !== selectedCategory) return false;
    return true;
  });

  const stats = {
    avgResponseTime: Math.round(metrics.filter(m => m.category === 'latency').reduce((sum, m) => sum + m.current, 0) / metrics.filter(m => m.category === 'latency').length || 0),
    avgThroughput: Math.round(metrics.filter(m => m.category === 'throughput').reduce((sum, m) => sum + m.current, 0) / metrics.filter(m => m.category === 'throughput').length || 0),
    avgCpu: Math.round(metrics.filter(m => m.category === 'cpu').reduce((sum, m) => sum + m.current, 0) / metrics.filter(m => m.category === 'cpu').length || 0),
    healthyServices: services.filter(s => s.status === 'healthy').length,
    degradedServices: services.filter(s => s.status === 'degraded').length,
    activeAlerts: alerts.filter(a => !a.acknowledged).length
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Performance Monitoring</h1>
            <p className="text-sm text-gray-400 mt-1">Real-time performance metrics and analysis</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                autoRefresh ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              <span>{autoRefresh ? 'Auto Refresh' : 'Paused'}</span>
            </button>
            
            <button
              onClick={() => setViewMode(viewMode === 'overview' ? 'detailed' : 'overview')}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
            >
              {viewMode === 'overview' ? 'Detailed View' : 'Overview'}
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <Clock className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Avg Response</p>
              <p className="text-xl font-bold">{stats.avgResponseTime}ms</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Zap className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Throughput</p>
              <p className="text-xl font-bold">{stats.avgThroughput}/s</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Cpu className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">CPU Usage</p>
              <p className="text-xl font-bold">{stats.avgCpu}%</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Healthy</p>
              <p className="text-xl font-bold">{stats.healthyServices}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Degraded</p>
              <p className="text-xl font-bold text-yellow-500">{stats.degradedServices}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XCircle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Alerts</p>
              <p className="text-xl font-bold text-red-500">{stats.activeAlerts}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Filters */}
        <div className="flex items-center space-x-3 mb-6">
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="5m">Last 5 Minutes</option>
            <option value="15m">Last 15 Minutes</option>
            <option value="1h">Last Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="24h">Last 24 Hours</option>
          </select>
          
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Categories</option>
            <option value="latency">Latency</option>
            <option value="throughput">Throughput</option>
            <option value="cpu">CPU</option>
            <option value="memory">Memory</option>
            <option value="disk">Disk</option>
            <option value="network">Network</option>
          </select>
        </div>

        {viewMode === 'overview' ? (
          <>
            {/* Service Status Grid */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              {services.map(service => (
                <div key={service.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold">{service.name}</h3>
                    <span className={`px-2 py-1 text-xs rounded ${getStatusColor(service.status)}`}>
                      {service.status.toUpperCase()}
                    </span>
                  </div>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Response Time</span>
                      <span>{service.responseTime}ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Error Rate</span>
                      <span className={service.errorRate > 0.02 ? 'text-yellow-500' : ''}>
                        {(service.errorRate * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Request Rate</span>
                      <span>{service.requestRate}/s</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Uptime</span>
                      <span className={service.uptime < service.sla ? 'text-red-500' : 'text-green-500'}>
                        {service.uptime}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-4">
              {filteredMetrics.map(metric => (
                <div key={metric.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      {getCategoryIcon(metric.category)}
                      <h3 className="text-sm font-bold">{metric.name}</h3>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded ${getStatusColor(metric.status)}`}>
                      {metric.status.toUpperCase()}
                    </span>
                  </div>
                  
                  <div className="flex items-end justify-between mb-3">
                    <div>
                      <p className="text-2xl font-bold">
                        {metric.current.toFixed(metric.unit === '%' ? 0 : 0)}
                        <span className="text-sm text-gray-500 ml-1">{metric.unit}</span>
                      </p>
                      <div className="flex items-center space-x-2 mt-1">
                        {getTrendIcon(metric.trend, metric.trendPercent)}
                        <span className="text-xs text-gray-400">
                          {metric.trendPercent > 0 ? '+' : ''}{metric.trendPercent.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    
                    {/* Mini Chart */}
                    <div className="flex items-end space-x-1 h-12">
                      {metric.history.map((point, idx) => (
                        <div
                          key={idx}
                          className="w-3 bg-blue-500"
                          style={{
                            height: `${(point.value / Math.max(...metric.history.map(h => h.value))) * 48}px`
                          }}
                        />
                      ))}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                      <p className="text-gray-400">Average</p>
                      <p className="font-bold">{metric.average}{metric.unit}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Peak</p>
                      <p className="font-bold">{metric.peak}{metric.unit}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Threshold</p>
                      <p className="font-bold">{metric.threshold.warning}{metric.unit}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : (
          /* Detailed View */
          <div className="space-y-4">
            {filteredMetrics.map(metric => (
              <div key={metric.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    {getCategoryIcon(metric.category)}
                    <div>
                      <h3 className="text-sm font-bold">{metric.name}</h3>
                      <p className="text-xs text-gray-400">ID: {metric.id}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className={`px-2 py-1 text-xs rounded ${getStatusColor(metric.status)}`}>
                      {metric.status.toUpperCase()}
                    </span>
                    <ChevronRight className="w-5 h-5 text-gray-500" />
                  </div>
                </div>
                
                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Current Value</p>
                    <p className="text-2xl font-bold">
                      {metric.current.toFixed(0)}
                      <span className="text-sm text-gray-500 ml-1">{metric.unit}</span>
                    </p>
                    <div className="flex items-center space-x-1 mt-1">
                      {getTrendIcon(metric.trend, metric.trendPercent)}
                      <span className="text-xs">
                        {metric.trendPercent > 0 ? '+' : ''}{metric.trendPercent.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Average</p>
                    <p className="text-lg font-bold">{metric.average}{metric.unit}</p>
                  </div>
                  
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Peak</p>
                    <p className="text-lg font-bold">{metric.peak}{metric.unit}</p>
                  </div>
                  
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Thresholds</p>
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full" />
                        <span className="text-xs">Optimal: {metric.threshold.optimal}{metric.unit}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-yellow-500 rounded-full" />
                        <span className="text-xs">Warning: {metric.threshold.warning}{metric.unit}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-red-500 rounded-full" />
                        <span className="text-xs">Critical: {metric.threshold.critical}{metric.unit}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Active Alerts */}
        {alerts.length > 0 && (
          <div className="mt-6">
            <h3 className="text-sm font-bold mb-3">Active Alerts</h3>
            <div className="space-y-2">
              {alerts.filter(a => !a.acknowledged).map(alert => (
                <div key={alert.id} className="bg-gray-900 border border-gray-800 rounded-lg p-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <AlertCircle className={`w-4 h-4 ${
                        alert.severity === 'critical' ? 'text-red-500' :
                        alert.severity === 'warning' ? 'text-yellow-500' :
                        'text-blue-500'
                      }`} />
                      <div>
                        <p className="text-sm font-bold">{alert.metric} - {alert.service}</p>
                        <p className="text-xs text-gray-400">{alert.message}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-gray-500">{alert.timestamp}</span>
                      <button className="px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                        Acknowledge
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}