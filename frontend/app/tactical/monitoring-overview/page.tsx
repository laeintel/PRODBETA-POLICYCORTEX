'use client';

import React, { useState, useEffect } from 'react';
import { 
  Activity, AlertTriangle, CheckCircle, XCircle, TrendingUp, TrendingDown,
  Clock, Server, Database, Cloud, Shield, Globe, Cpu, HardDrive, 
  Network, Users, Zap, BarChart3, LineChart, PieChart, Settings,
  ChevronRight, MoreVertical, ExternalLink, RefreshCw, Download,
  Eye, EyeOff, Terminal, Code, GitBranch, Container, Layers,
  Gauge, Thermometer, Battery, Wifi, WifiOff, AlertCircle, Info
} from 'lucide-react';

interface ServiceHealth {
  id: string;
  name: string;
  status: 'healthy' | 'degraded' | 'down' | 'maintenance';
  uptime: number;
  responseTime: number;
  errorRate: number;
  region: string;
  lastCheck: string;
  dependencies: string[];
  incidents: number;
  metrics: {
    cpu: number;
    memory: number;
    disk: number;
    network: number;
  };
}

interface MetricCard {
  id: string;
  title: string;
  value: string | number;
  change: number;
  trend: 'up' | 'down' | 'stable';
  unit?: string;
  category: string;
  severity?: 'critical' | 'warning' | 'normal';
  sparkline?: number[];
}

interface Alert {
  id: string;
  service: string;
  message: string;
  severity: 'critical' | 'warning' | 'info';
  timestamp: string;
  acknowledged: boolean;
}

export default function MonitoringOverview() {
  const [services, setServices] = useState<ServiceHealth[]>([]);
  const [metrics, setMetrics] = useState<MetricCard[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedView, setSelectedView] = useState<'grid' | 'list' | 'map'>('grid');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30);
  const [showDetails, setShowDetails] = useState<string | null>(null);

  useEffect(() => {
    // Initialize with comprehensive monitoring data
    setServices([
      {
        id: 'SVC-001',
        name: 'API Gateway',
        status: 'healthy',
        uptime: 99.99,
        responseTime: 45,
        errorRate: 0.01,
        region: 'East US',
        lastCheck: '10 seconds ago',
        dependencies: ['Auth Service', 'Database', 'Cache'],
        incidents: 0,
        metrics: { cpu: 35, memory: 42, disk: 28, network: 65 }
      },
      {
        id: 'SVC-002',
        name: 'Database Cluster',
        status: 'healthy',
        uptime: 99.95,
        responseTime: 12,
        errorRate: 0.02,
        region: 'East US',
        lastCheck: '8 seconds ago',
        dependencies: ['Storage', 'Backup Service'],
        incidents: 0,
        metrics: { cpu: 48, memory: 68, disk: 72, network: 45 }
      },
      {
        id: 'SVC-003',
        name: 'Auth Service',
        status: 'degraded',
        uptime: 98.5,
        responseTime: 250,
        errorRate: 2.5,
        region: 'East US',
        lastCheck: '5 seconds ago',
        dependencies: ['Database', 'Cache', 'Identity Provider'],
        incidents: 2,
        metrics: { cpu: 78, memory: 85, disk: 45, network: 92 }
      },
      {
        id: 'SVC-004',
        name: 'Storage Service',
        status: 'healthy',
        uptime: 99.99,
        responseTime: 85,
        errorRate: 0.001,
        region: 'West US',
        lastCheck: '12 seconds ago',
        dependencies: ['CDN', 'Backup Service'],
        incidents: 0,
        metrics: { cpu: 22, memory: 35, disk: 88, network: 72 }
      },
      {
        id: 'SVC-005',
        name: 'Analytics Engine',
        status: 'maintenance',
        uptime: 95.0,
        responseTime: 150,
        errorRate: 0,
        region: 'Central US',
        lastCheck: '2 minutes ago',
        dependencies: ['Database', 'Message Queue', 'ML Service'],
        incidents: 0,
        metrics: { cpu: 0, memory: 0, disk: 45, network: 0 }
      },
      {
        id: 'SVC-006',
        name: 'Cache Layer',
        status: 'healthy',
        uptime: 99.99,
        responseTime: 2,
        errorRate: 0.001,
        region: 'East US',
        lastCheck: '3 seconds ago',
        dependencies: [],
        incidents: 0,
        metrics: { cpu: 15, memory: 78, disk: 12, network: 95 }
      },
      {
        id: 'SVC-007',
        name: 'Message Queue',
        status: 'healthy',
        uptime: 99.98,
        responseTime: 8,
        errorRate: 0.01,
        region: 'East US',
        lastCheck: '6 seconds ago',
        dependencies: ['Database'],
        incidents: 0,
        metrics: { cpu: 28, memory: 45, disk: 35, network: 68 }
      },
      {
        id: 'SVC-008',
        name: 'ML Service',
        status: 'down',
        uptime: 89.5,
        responseTime: 0,
        errorRate: 100,
        region: 'Central US',
        lastCheck: '30 seconds ago',
        dependencies: ['GPU Cluster', 'Storage Service'],
        incidents: 5,
        metrics: { cpu: 0, memory: 0, disk: 0, network: 0 }
      }
    ]);

    setMetrics([
      {
        id: 'M-001',
        title: 'Total Requests',
        value: '2.4M',
        change: 12.5,
        trend: 'up',
        unit: '/hour',
        category: 'traffic',
        sparkline: [20, 25, 30, 28, 35, 42, 45, 48, 52, 55, 58, 62]
      },
      {
        id: 'M-002',
        title: 'Error Rate',
        value: 0.12,
        change: -25,
        trend: 'down',
        unit: '%',
        category: 'reliability',
        severity: 'normal',
        sparkline: [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.18, 0.15, 0.14, 0.13, 0.12, 0.12]
      },
      {
        id: 'M-003',
        title: 'Avg Response Time',
        value: 142,
        change: -8,
        trend: 'down',
        unit: 'ms',
        category: 'performance',
        severity: 'normal',
        sparkline: [180, 175, 165, 160, 155, 150, 148, 145, 143, 142, 142, 142]
      },
      {
        id: 'M-004',
        title: 'System Uptime',
        value: 99.98,
        change: 0.02,
        trend: 'up',
        unit: '%',
        category: 'reliability',
        sparkline: [99.95, 99.96, 99.96, 99.97, 99.97, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98]
      },
      {
        id: 'M-005',
        title: 'Active Users',
        value: '12.8K',
        change: 18,
        trend: 'up',
        category: 'traffic',
        sparkline: [8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12300, 12500, 12800]
      },
      {
        id: 'M-006',
        title: 'CPU Usage',
        value: 68,
        change: 15,
        trend: 'up',
        unit: '%',
        category: 'infrastructure',
        severity: 'warning',
        sparkline: [45, 48, 52, 55, 58, 60, 62, 64, 65, 66, 67, 68]
      },
      {
        id: 'M-007',
        title: 'Memory Usage',
        value: 72,
        change: 8,
        trend: 'up',
        unit: '%',
        category: 'infrastructure',
        severity: 'warning',
        sparkline: [60, 62, 64, 65, 66, 68, 69, 70, 71, 71, 72, 72]
      },
      {
        id: 'M-008',
        title: 'Network Throughput',
        value: '8.2',
        change: 22,
        trend: 'up',
        unit: 'Gbps',
        category: 'infrastructure',
        sparkline: [5, 5.5, 6, 6.2, 6.5, 7, 7.2, 7.5, 7.8, 8, 8.1, 8.2]
      },
      {
        id: 'M-009',
        title: 'Database Connections',
        value: 2845,
        change: -5,
        trend: 'down',
        unit: 'active',
        category: 'database',
        sparkline: [3000, 2950, 2920, 2900, 2880, 2870, 2860, 2855, 2850, 2848, 2846, 2845]
      },
      {
        id: 'M-010',
        title: 'Cache Hit Rate',
        value: 94.5,
        change: 2.5,
        trend: 'up',
        unit: '%',
        category: 'performance',
        sparkline: [88, 89, 90, 91, 91.5, 92, 92.5, 93, 93.5, 94, 94.2, 94.5]
      },
      {
        id: 'M-011',
        title: 'SSL Certificates',
        value: 42,
        change: 0,
        trend: 'stable',
        unit: 'valid',
        category: 'security',
        sparkline: [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42]
      },
      {
        id: 'M-012',
        title: 'Failed Logins',
        value: 128,
        change: 45,
        trend: 'up',
        unit: '/hour',
        category: 'security',
        severity: 'critical',
        sparkline: [50, 55, 60, 65, 70, 80, 85, 95, 105, 115, 120, 128]
      }
    ]);

    setAlerts([
      {
        id: 'A-001',
        service: 'Auth Service',
        message: 'High response time detected (>200ms)',
        severity: 'warning',
        timestamp: '2 minutes ago',
        acknowledged: false
      },
      {
        id: 'A-002',
        service: 'ML Service',
        message: 'Service is down - connection timeout',
        severity: 'critical',
        timestamp: '30 seconds ago',
        acknowledged: false
      },
      {
        id: 'A-003',
        service: 'Security',
        message: 'Unusual number of failed login attempts detected',
        severity: 'critical',
        timestamp: '5 minutes ago',
        acknowledged: true
      },
      {
        id: 'A-004',
        service: 'Analytics Engine',
        message: 'Scheduled maintenance in progress',
        severity: 'info',
        timestamp: '10 minutes ago',
        acknowledged: true
      }
    ]);

    // Auto-refresh simulation
    if (autoRefresh) {
      const interval = setInterval(() => {
        // Update metrics with slight variations
        setMetrics(prev => prev.map(metric => ({
          ...metric,
          value: typeof metric.value === 'number' 
            ? metric.value + (Math.random() - 0.5) * 2
            : metric.value,
          change: metric.change + (Math.random() - 0.5) * 0.5
        })));
      }, refreshInterval * 1000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'healthy': return 'text-green-500 bg-green-900/20';
      case 'degraded': return 'text-yellow-500 bg-yellow-900/20';
      case 'down': return 'text-red-500 bg-red-900/20';
      case 'maintenance': return 'text-blue-500 bg-blue-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch(severity) {
      case 'critical': return 'text-red-500';
      case 'warning': return 'text-yellow-500';
      case 'normal': return 'text-green-500';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'healthy': return <CheckCircle className="w-4 h-4" />;
      case 'degraded': return <AlertCircle className="w-4 h-4" />;
      case 'down': return <XCircle className="w-4 h-4" />;
      case 'maintenance': return <Info className="w-4 h-4" />;
      default: return <AlertCircle className="w-4 h-4" />;
    }
  };

  const categories = [
    { id: 'all', label: 'All Systems', icon: <Globe className="w-4 h-4" /> },
    { id: 'traffic', label: 'Traffic', icon: <Users className="w-4 h-4" /> },
    { id: 'performance', label: 'Performance', icon: <Zap className="w-4 h-4" /> },
    { id: 'infrastructure', label: 'Infrastructure', icon: <Server className="w-4 h-4" /> },
    { id: 'database', label: 'Database', icon: <Database className="w-4 h-4" /> },
    { id: 'security', label: 'Security', icon: <Shield className="w-4 h-4" /> },
    { id: 'reliability', label: 'Reliability', icon: <Activity className="w-4 h-4" /> }
  ];

  const filteredMetrics = selectedCategory === 'all' 
    ? metrics 
    : metrics.filter(m => m.category === selectedCategory);

  const healthyServices = services.filter(s => s.status === 'healthy').length;
  const degradedServices = services.filter(s => s.status === 'degraded').length;
  const downServices = services.filter(s => s.status === 'down').length;
  const maintenanceServices = services.filter(s => s.status === 'maintenance').length;

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Monitoring Overview</h1>
            <p className="text-sm text-gray-400 mt-1">Real-time system health and performance monitoring</p>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* Auto Refresh Toggle */}
            <div className="flex items-center space-x-2 px-3 py-2 bg-gray-800 rounded">
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin text-green-500' : 'text-gray-500'}`} />
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className="text-sm"
              >
                {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
              </button>
              {autoRefresh && (
                <select
                  value={refreshInterval}
                  onChange={(e) => setRefreshInterval(Number(e.target.value))}
                  className="ml-2 px-2 py-1 bg-gray-700 rounded text-xs"
                >
                  <option value={10}>10s</option>
                  <option value={30}>30s</option>
                  <option value={60}>60s</option>
                </select>
              )}
            </div>

            {/* View Mode Toggle */}
            <div className="flex items-center bg-gray-800 rounded">
              <button
                onClick={() => setSelectedView('grid')}
                className={`px-3 py-2 text-sm ${selectedView === 'grid' ? 'bg-gray-700' : ''} rounded-l`}
              >
                Grid
              </button>
              <button
                onClick={() => setSelectedView('list')}
                className={`px-3 py-2 text-sm ${selectedView === 'list' ? 'bg-gray-700' : ''}`}
              >
                List
              </button>
              <button
                onClick={() => setSelectedView('map')}
                className={`px-3 py-2 text-sm ${selectedView === 'map' ? 'bg-gray-700' : ''} rounded-r`}
              >
                Map
              </button>
            </div>

            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export Report</span>
            </button>
          </div>
        </div>
      </header>

      {/* Service Health Summary Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <span className="text-sm">{healthyServices} Healthy</span>
            </div>
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-5 h-5 text-yellow-500" />
              <span className="text-sm">{degradedServices} Degraded</span>
            </div>
            <div className="flex items-center space-x-2">
              <XCircle className="w-5 h-5 text-red-500" />
              <span className="text-sm">{downServices} Down</span>
            </div>
            <div className="flex items-center space-x-2">
              <Info className="w-5 h-5 text-blue-500" />
              <span className="text-sm">{maintenanceServices} Maintenance</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-4 text-sm">
            <span className="text-gray-400">Overall Health:</span>
            <div className="flex items-center space-x-2">
              <div className="w-32 h-2 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-green-500 to-green-600" style={{ width: '92%' }}></div>
              </div>
              <span className="text-green-500 font-bold">92%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="flex h-full">
          {/* Left Sidebar - Categories */}
          <div className="w-48 bg-gray-900/50 border-r border-gray-800 p-4">
            <h3 className="text-xs uppercase text-gray-500 mb-3">Categories</h3>
            <div className="space-y-1">
              {categories.map(cat => (
                <button
                  key={cat.id}
                  onClick={() => setSelectedCategory(cat.id)}
                  className={`w-full flex items-center space-x-2 px-3 py-2 rounded text-sm ${
                    selectedCategory === cat.id ? 'bg-gray-800 text-white' : 'text-gray-400 hover:bg-gray-800/50'
                  }`}
                >
                  {cat.icon}
                  <span>{cat.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Main Content Area */}
          <div className="flex-1 p-6">
            {/* Alerts Section */}
            {alerts.filter(a => !a.acknowledged).length > 0 && (
              <div className="mb-6">
                <h2 className="text-sm font-bold mb-3">Active Alerts</h2>
                <div className="space-y-2">
                  {alerts.filter(a => !a.acknowledged).map(alert => (
                    <div key={alert.id} className={`bg-gray-900 border-l-4 ${
                      alert.severity === 'critical' ? 'border-red-500' :
                      alert.severity === 'warning' ? 'border-yellow-500' :
                      'border-blue-500'
                    } p-3 rounded`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <AlertTriangle className={`w-4 h-4 ${
                            alert.severity === 'critical' ? 'text-red-500' :
                            alert.severity === 'warning' ? 'text-yellow-500' :
                            'text-blue-500'
                          }`} />
                          <div>
                            <span className="text-sm font-medium">{alert.service}:</span>
                            <span className="text-sm text-gray-400 ml-2">{alert.message}</span>
                          </div>
                        </div>
                        <div className="flex items-center space-x-3">
                          <span className="text-xs text-gray-500">{alert.timestamp}</span>
                          <button className="text-xs px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded">
                            Acknowledge
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Metrics Grid */}
            {selectedView === 'grid' && (
              <div className="grid grid-cols-4 gap-4 mb-6">
                {filteredMetrics.map(metric => (
                  <div key={metric.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="text-xs text-gray-400">{metric.title}</h3>
                      {metric.trend === 'up' ? (
                        <TrendingUp className="w-4 h-4 text-green-500" />
                      ) : metric.trend === 'down' ? (
                        <TrendingDown className="w-4 h-4 text-red-500" />
                      ) : (
                        <BarChart3 className="w-4 h-4 text-gray-500" />
                      )}
                    </div>
                    <div className="flex items-baseline space-x-1 mb-2">
                      <span className={`text-2xl font-bold ${metric.severity ? getSeverityColor(metric.severity) : ''}`}>
                        {typeof metric.value === 'number' ? metric.value.toFixed(1) : metric.value}
                      </span>
                      {metric.unit && <span className="text-sm text-gray-500">{metric.unit}</span>}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className={`text-xs ${metric.change > 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {metric.change > 0 ? '+' : ''}{metric.change.toFixed(1)}%
                      </span>
                      {metric.sparkline && (
                        <div className="flex items-end space-x-1 h-8">
                          {metric.sparkline.slice(-8).map((value, idx) => (
                            <div
                              key={idx}
                              className="w-1 bg-blue-500 rounded-t"
                              style={{ height: `${(value / Math.max(...metric.sparkline)) * 100}%` }}
                            />
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Services Status */}
            <div>
              <h2 className="text-sm font-bold mb-3">Service Status</h2>
              <div className={selectedView === 'grid' ? 'grid grid-cols-2 gap-4' : 'space-y-3'}>
                {services.map(service => (
                  <div key={service.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <div className={`p-2 rounded ${getStatusColor(service.status)}`}>
                          {getStatusIcon(service.status)}
                        </div>
                        <div>
                          <h3 className="text-sm font-bold">{service.name}</h3>
                          <p className="text-xs text-gray-500">{service.region}</p>
                        </div>
                      </div>
                      <button 
                        onClick={() => setShowDetails(showDetails === service.id ? null : service.id)}
                        className="p-1 hover:bg-gray-800 rounded"
                      >
                        <ChevronRight className={`w-4 h-4 text-gray-500 transition-transform ${
                          showDetails === service.id ? 'rotate-90' : ''
                        }`} />
                      </button>
                    </div>

                    <div className="grid grid-cols-4 gap-2 text-xs">
                      <div>
                        <span className="text-gray-500">Uptime</span>
                        <p className="font-bold">{service.uptime}%</p>
                      </div>
                      <div>
                        <span className="text-gray-500">Response</span>
                        <p className="font-bold">{service.responseTime}ms</p>
                      </div>
                      <div>
                        <span className="text-gray-500">Errors</span>
                        <p className="font-bold">{service.errorRate}%</p>
                      </div>
                      <div>
                        <span className="text-gray-500">Incidents</span>
                        <p className={`font-bold ${service.incidents > 0 ? 'text-red-500' : 'text-green-500'}`}>
                          {service.incidents}
                        </p>
                      </div>
                    </div>

                    {showDetails === service.id && (
                      <div className="mt-4 pt-4 border-t border-gray-800">
                        <div className="space-y-2">
                          <div className="text-xs">
                            <span className="text-gray-500">Dependencies:</span>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {service.dependencies.map(dep => (
                                <span key={dep} className="px-2 py-1 bg-gray-800 rounded">
                                  {dep}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div className="grid grid-cols-4 gap-2 text-xs">
                            <div className="flex items-center space-x-1">
                              <Cpu className="w-3 h-3 text-gray-500" />
                              <span>{service.metrics.cpu}%</span>
                            </div>
                            <div className="flex items-center space-x-1">
                              <Thermometer className="w-3 h-3 text-gray-500" />
                              <span>{service.metrics.memory}%</span>
                            </div>
                            <div className="flex items-center space-x-1">
                              <HardDrive className="w-3 h-3 text-gray-500" />
                              <span>{service.metrics.disk}%</span>
                            </div>
                            <div className="flex items-center space-x-1">
                              <Network className="w-3 h-3 text-gray-500" />
                              <span>{service.metrics.network}%</span>
                            </div>
                          </div>
                          <p className="text-xs text-gray-500">Last checked: {service.lastCheck}</p>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}