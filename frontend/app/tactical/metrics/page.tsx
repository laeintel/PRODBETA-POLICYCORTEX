'use client';

import React, { useState, useEffect } from 'react';
import { 
  BarChart, LineChart, TrendingUp, TrendingDown, Activity, Clock,
  Database, Server, Cloud, Cpu, HardDrive, Network, Zap, AlertCircle,
  CheckCircle, XCircle, Info, RefreshCw, Download, Settings, Filter,
  Calendar, ChevronRight, MoreVertical, ArrowUp, ArrowDown, Gauge
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface Metric {
  id: string;
  name: string;
  category: 'performance' | 'availability' | 'resource' | 'business' | 'custom';
  value: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  trendValue: number;
  status: 'healthy' | 'warning' | 'critical' | 'unknown';
  timestamp: string;
  history: {
    time: string;
    value: number;
  }[];
  threshold: {
    warning: number;
    critical: number;
  };
  metadata: {
    source: string;
    frequency: string;
    lastUpdated: string;
  };
}

interface MetricGroup {
  id: string;
  name: string;
  description: string;
  metrics: string[];
  aggregation: 'sum' | 'avg' | 'max' | 'min';
  visualization: 'line' | 'bar' | 'gauge' | 'number';
}

interface Dashboard {
  id: string;
  name: string;
  widgets: Widget[];
  refreshInterval: number;
  filters: {
    timeRange: string;
    environment: string;
    tags: string[];
  };
}

interface Widget {
  id: string;
  type: 'chart' | 'gauge' | 'number' | 'table' | 'heatmap';
  title: string;
  metricIds: string[];
  position: { x: number; y: number; w: number; h: number };
  config: any;
}

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [metricGroups, setMetricGroups] = useState<MetricGroup[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  useEffect(() => {
    // Initialize with mock metrics data
    setMetrics([
      {
        id: 'MTR-001',
        name: 'CPU Utilization',
        category: 'performance',
        value: 72.5,
        unit: '%',
        trend: 'up',
        trendValue: 5.2,
        status: 'warning',
        timestamp: 'Now',
        history: [
          { time: '1h ago', value: 68 },
          { time: '45m ago', value: 70 },
          { time: '30m ago', value: 69 },
          { time: '15m ago', value: 71 },
          { time: 'Now', value: 72.5 }
        ],
        threshold: {
          warning: 70,
          critical: 85
        },
        metadata: {
          source: 'CloudWatch',
          frequency: '1 min',
          lastUpdated: '10 seconds ago'
        }
      },
      {
        id: 'MTR-002',
        name: 'Memory Usage',
        category: 'performance',
        value: 8.2,
        unit: 'GB',
        trend: 'stable',
        trendValue: 0.1,
        status: 'healthy',
        timestamp: 'Now',
        history: [
          { time: '1h ago', value: 8.1 },
          { time: '45m ago', value: 8.2 },
          { time: '30m ago', value: 8.1 },
          { time: '15m ago', value: 8.2 },
          { time: 'Now', value: 8.2 }
        ],
        threshold: {
          warning: 12,
          critical: 14
        },
        metadata: {
          source: 'Prometheus',
          frequency: '30 sec',
          lastUpdated: '5 seconds ago'
        }
      },
      {
        id: 'MTR-003',
        name: 'Request Rate',
        category: 'performance',
        value: 1250,
        unit: 'req/s',
        trend: 'up',
        trendValue: 12.5,
        status: 'healthy',
        timestamp: 'Now',
        history: [
          { time: '1h ago', value: 1100 },
          { time: '45m ago', value: 1150 },
          { time: '30m ago', value: 1200 },
          { time: '15m ago', value: 1225 },
          { time: 'Now', value: 1250 }
        ],
        threshold: {
          warning: 2000,
          critical: 2500
        },
        metadata: {
          source: 'Application Insights',
          frequency: '10 sec',
          lastUpdated: '2 seconds ago'
        }
      },
      {
        id: 'MTR-004',
        name: 'Error Rate',
        category: 'availability',
        value: 0.02,
        unit: '%',
        trend: 'down',
        trendValue: -0.01,
        status: 'healthy',
        timestamp: 'Now',
        history: [
          { time: '1h ago', value: 0.05 },
          { time: '45m ago', value: 0.04 },
          { time: '30m ago', value: 0.03 },
          { time: '15m ago', value: 0.03 },
          { time: 'Now', value: 0.02 }
        ],
        threshold: {
          warning: 1,
          critical: 5
        },
        metadata: {
          source: 'Log Analytics',
          frequency: '1 min',
          lastUpdated: '15 seconds ago'
        }
      },
      {
        id: 'MTR-005',
        name: 'Service Availability',
        category: 'availability',
        value: 99.98,
        unit: '%',
        trend: 'stable',
        trendValue: 0,
        status: 'healthy',
        timestamp: 'Now',
        history: [
          { time: '1h ago', value: 99.98 },
          { time: '45m ago', value: 99.99 },
          { time: '30m ago', value: 99.98 },
          { time: '15m ago', value: 99.98 },
          { time: 'Now', value: 99.98 }
        ],
        threshold: {
          warning: 99.5,
          critical: 99
        },
        metadata: {
          source: 'Uptime Monitor',
          frequency: '30 sec',
          lastUpdated: '8 seconds ago'
        }
      },
      {
        id: 'MTR-006',
        name: 'Response Time',
        category: 'performance',
        value: 145,
        unit: 'ms',
        trend: 'down',
        trendValue: -12,
        status: 'healthy',
        timestamp: 'Now',
        history: [
          { time: '1h ago', value: 180 },
          { time: '45m ago', value: 170 },
          { time: '30m ago', value: 160 },
          { time: '15m ago', value: 150 },
          { time: 'Now', value: 145 }
        ],
        threshold: {
          warning: 300,
          critical: 500
        },
        metadata: {
          source: 'APM',
          frequency: '10 sec',
          lastUpdated: '3 seconds ago'
        }
      },
      {
        id: 'MTR-007',
        name: 'Disk I/O',
        category: 'resource',
        value: 450,
        unit: 'MB/s',
        trend: 'up',
        trendValue: 25,
        status: 'warning',
        timestamp: 'Now',
        history: [
          { time: '1h ago', value: 380 },
          { time: '45m ago', value: 400 },
          { time: '30m ago', value: 420 },
          { time: '15m ago', value: 430 },
          { time: 'Now', value: 450 }
        ],
        threshold: {
          warning: 400,
          critical: 500
        },
        metadata: {
          source: 'Infrastructure Monitor',
          frequency: '30 sec',
          lastUpdated: '12 seconds ago'
        }
      },
      {
        id: 'MTR-008',
        name: 'Network Throughput',
        category: 'resource',
        value: 850,
        unit: 'Mbps',
        trend: 'stable',
        trendValue: 5,
        status: 'healthy',
        timestamp: 'Now',
        history: [
          { time: '1h ago', value: 840 },
          { time: '45m ago', value: 845 },
          { time: '30m ago', value: 848 },
          { time: '15m ago', value: 850 },
          { time: 'Now', value: 850 }
        ],
        threshold: {
          warning: 1500,
          critical: 1800
        },
        metadata: {
          source: 'Network Monitor',
          frequency: '15 sec',
          lastUpdated: '6 seconds ago'
        }
      },
      {
        id: 'MTR-009',
        name: 'Active Users',
        category: 'business',
        value: 5234,
        unit: 'users',
        trend: 'up',
        trendValue: 234,
        status: 'healthy',
        timestamp: 'Now',
        history: [
          { time: '1h ago', value: 4800 },
          { time: '45m ago', value: 4950 },
          { time: '30m ago', value: 5100 },
          { time: '15m ago', value: 5180 },
          { time: 'Now', value: 5234 }
        ],
        threshold: {
          warning: 10000,
          critical: 15000
        },
        metadata: {
          source: 'Analytics',
          frequency: '1 min',
          lastUpdated: '20 seconds ago'
        }
      },
      {
        id: 'MTR-010',
        name: 'Transaction Volume',
        category: 'business',
        value: 8750,
        unit: 'txn/min',
        trend: 'up',
        trendValue: 15.5,
        status: 'healthy',
        timestamp: 'Now',
        history: [
          { time: '1h ago', value: 7500 },
          { time: '45m ago', value: 8000 },
          { time: '30m ago', value: 8300 },
          { time: '15m ago', value: 8500 },
          { time: 'Now', value: 8750 }
        ],
        threshold: {
          warning: 12000,
          critical: 15000
        },
        metadata: {
          source: 'Payment Gateway',
          frequency: '30 sec',
          lastUpdated: '10 seconds ago'
        }
      }
    ]);

    setMetricGroups([
      {
        id: 'GRP-001',
        name: 'Infrastructure Health',
        description: 'Core infrastructure metrics',
        metrics: ['MTR-001', 'MTR-002', 'MTR-007', 'MTR-008'],
        aggregation: 'avg',
        visualization: 'gauge'
      },
      {
        id: 'GRP-002',
        name: 'Application Performance',
        description: 'Application performance indicators',
        metrics: ['MTR-003', 'MTR-004', 'MTR-006'],
        aggregation: 'avg',
        visualization: 'line'
      },
      {
        id: 'GRP-003',
        name: 'Business Metrics',
        description: 'Business KPIs',
        metrics: ['MTR-009', 'MTR-010'],
        aggregation: 'sum',
        visualization: 'bar'
      }
    ]);

    // Simulate real-time updates
    if (autoRefresh) {
      const interval = setInterval(() => {
        setMetrics(prevMetrics => 
          prevMetrics.map(metric => ({
            ...metric,
            value: metric.value + (Math.random() - 0.5) * 10,
            timestamp: 'Now',
            metadata: {
              ...metric.metadata,
              lastUpdated: 'Just now'
            }
          }))
        );
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'healthy': return 'text-green-500 bg-green-900/20';
      case 'warning': return 'text-yellow-500 bg-yellow-900/20';
      case 'critical': return 'text-red-500 bg-red-900/20';
      case 'unknown': return 'text-gray-500 bg-gray-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getTrendIcon = (trend: string, value: number) => {
    if (trend === 'up') return <ArrowUp className={`w-3 h-3 ${value > 0 ? 'text-green-500' : 'text-red-500'}`} />;
    if (trend === 'down') return <ArrowDown className={`w-3 h-3 ${value < 0 ? 'text-green-500' : 'text-red-500'}`} />;
    return <span className="w-3 h-3 text-gray-500">-</span>;
  };

  const getCategoryIcon = (category: string) => {
    switch(category) {
      case 'performance': return <Zap className="w-4 h-4 text-yellow-500" />;
      case 'availability': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'resource': return <Server className="w-4 h-4 text-blue-500" />;
      case 'business': return <TrendingUp className="w-4 h-4 text-purple-500" />;
      case 'custom': return <Settings className="w-4 h-4 text-gray-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const filteredMetrics = metrics.filter(metric => {
    if (selectedCategory !== 'all' && metric.category !== selectedCategory) return false;
    return true;
  });

  const stats = {
    total: metrics.length,
    healthy: metrics.filter(m => m.status === 'healthy').length,
    warning: metrics.filter(m => m.status === 'warning').length,
    critical: metrics.filter(m => m.status === 'critical').length,
    avgPerformance: 95.5,
    dataPoints: metrics.reduce((sum, m) => sum + m.history.length, 0)
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Metrics Dashboard</h1>
            <p className="text-sm text-gray-400 mt-1">Real-time metrics and performance indicators</p>
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
              onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
            >
              {viewMode === 'grid' ? 'List View' : 'Grid View'}
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
            <BarChart className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Total Metrics</p>
              <p className="text-xl font-bold">{stats.total}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Healthy</p>
              <p className="text-xl font-bold">{stats.healthy}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Warning</p>
              <p className="text-xl font-bold text-yellow-500">{stats.warning}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XCircle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Critical</p>
              <p className="text-xl font-bold text-red-500">{stats.critical}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Gauge className="w-5 h-5 text-cyan-500" />
            <div>
              <p className="text-xs text-gray-400">Avg Performance</p>
              <p className="text-xl font-bold">{stats.avgPerformance}%</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Activity className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Data Points</p>
              <p className="text-xl font-bold">{stats.dataPoints}</p>
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
            <option value="7d">Last 7 Days</option>
          </select>
          
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Categories</option>
            <option value="performance">Performance</option>
            <option value="availability">Availability</option>
            <option value="resource">Resource</option>
            <option value="business">Business</option>
            <option value="custom">Custom</option>
          </select>
        </div>

        {/* Metric Groups Summary */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          {metricGroups.map(group => (
            <div key={group.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-2">{group.name}</h3>
              <p className="text-xs text-gray-400 mb-3">{group.description}</p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">{group.metrics.length} metrics</span>
                <span className="text-xs text-gray-500">{group.visualization}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Metrics Grid/List */}
        {viewMode === 'grid' ? (
          <div className="grid grid-cols-3 gap-4">
            {filteredMetrics.map(metric => (
              <div key={metric.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    {getCategoryIcon(metric.category)}
                    <span className="text-xs text-gray-500 font-mono">{metric.id}</span>
                  </div>
                  <span className={`px-2 py-1 text-xs rounded ${getStatusColor(metric.status)}`}>
                    {metric.status.toUpperCase()}
                  </span>
                </div>
                
                <h3 className="text-sm font-bold mb-2">{metric.name}</h3>
                
                <div className="flex items-end justify-between mb-3">
                  <div>
                    <p className="text-2xl font-bold">
                      {metric.value.toFixed(metric.unit === '%' ? 1 : 0)}
                      <span className="text-sm text-gray-500 ml-1">{metric.unit}</span>
                    </p>
                    <div className="flex items-center space-x-1 mt-1">
                      {getTrendIcon(metric.trend, metric.trendValue)}
                      <span className={`text-xs ${metric.trendValue > 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {Math.abs(metric.trendValue).toFixed(1)}{metric.unit === '%' ? '%' : ''}
                      </span>
                    </div>
                  </div>
                  
                  {/* Mini Chart */}
                  <div className="flex items-end space-x-1 h-12">
                    {metric.history.map((point, idx) => (
                      <div
                        key={idx}
                        className="w-2 bg-blue-500"
                        style={{
                          height: `${(point.value / Math.max(...metric.history.map(h => h.value))) * 48}px`
                        }}
                      />
                    ))}
                  </div>
                </div>
                
                <div className="pt-3 border-t border-gray-800 space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Source</span>
                    <span>{metric.metadata.source}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Frequency</span>
                    <span>{metric.metadata.frequency}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Updated</span>
                    <span>{metric.metadata.lastUpdated}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-3">
            {filteredMetrics.map(metric => (
              <div key={metric.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4 flex-1">
                    {getCategoryIcon(metric.category)}
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <span className="text-sm font-bold">{metric.name}</span>
                        <span className="text-xs text-gray-500 font-mono">{metric.id}</span>
                        <span className={`px-2 py-1 text-xs rounded ${getStatusColor(metric.status)}`}>
                          {metric.status.toUpperCase()}
                        </span>
                      </div>
                      <div className="flex items-center space-x-6 mt-1 text-xs text-gray-500">
                        <span>Source: {metric.metadata.source}</span>
                        <span>Updated: {metric.metadata.lastUpdated}</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-xl font-bold">
                        {metric.value.toFixed(metric.unit === '%' ? 1 : 0)}
                        <span className="text-sm text-gray-500 ml-1">{metric.unit}</span>
                      </p>
                      <div className="flex items-center justify-end space-x-1 mt-1">
                        {getTrendIcon(metric.trend, metric.trendValue)}
                        <span className={`text-xs ${metric.trendValue > 0 ? 'text-green-500' : 'text-red-500'}`}>
                          {Math.abs(metric.trendValue).toFixed(1)}{metric.unit === '%' ? '%' : ''}
                        </span>
                      </div>
                    </div>
                  </div>
                  <ChevronRight className="w-5 h-5 text-gray-500 ml-4" />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}