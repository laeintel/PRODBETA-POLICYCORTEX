'use client';

import React, { useState, useEffect } from 'react';
import { 
  CheckCircle, XCircle, AlertTriangle, Activity, Clock, TrendingUp,
  TrendingDown, Shield, Server, Cloud, Database, Globe, Wifi, WifiOff,
  RefreshCw, Download, Calendar, Timer, BarChart, Info, ChevronRight,
  ArrowUp, ArrowDown, Zap, Settings, Target
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface ServiceAvailability {
  id: string;
  name: string;
  type: 'api' | 'database' | 'cache' | 'storage' | 'network' | 'compute';
  status: 'operational' | 'degraded' | 'partial_outage' | 'major_outage';
  uptime: {
    current: number;
    daily: number;
    weekly: number;
    monthly: number;
    yearly: number;
  };
  sla: {
    target: number;
    actual: number;
    compliant: boolean;
  };
  lastIncident?: {
    timestamp: string;
    duration: number;
    impact: string;
  };
  healthChecks: {
    endpoint: string;
    frequency: number;
    lastCheck: string;
    responseTime: number;
    successRate: number;
  };
  dependencies: string[];
  metrics: {
    mtbf: number; // Mean Time Between Failures
    mttr: number; // Mean Time To Recovery
    availability: number;
  };
}

interface Incident {
  id: string;
  serviceId: string;
  serviceName: string;
  startTime: string;
  endTime?: string;
  duration?: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  impact: string;
  status: 'active' | 'resolved' | 'investigating';
  affectedUsers: number;
  rootCause?: string;
}

interface AvailabilityTrend {
  timestamp: string;
  availability: number;
  incidents: number;
}

export default function AvailabilityTracking() {
  const [services, setServices] = useState<ServiceAvailability[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [trends, setTrends] = useState<AvailabilityTrend[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  const [selectedService, setSelectedService] = useState<string>('all');
  const [viewMode, setViewMode] = useState<'overview' | 'detailed' | 'timeline'>('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    // Initialize with mock availability data
    setServices([
      {
        id: 'SVC-001',
        name: 'API Gateway',
        type: 'api',
        status: 'operational',
        uptime: {
          current: 99.98,
          daily: 100,
          weekly: 99.95,
          monthly: 99.92,
          yearly: 99.89
        },
        sla: {
          target: 99.95,
          actual: 99.92,
          compliant: false
        },
        lastIncident: {
          timestamp: '3 days ago',
          duration: 15,
          impact: 'Minor service degradation'
        },
        healthChecks: {
          endpoint: '/health',
          frequency: 30,
          lastCheck: '30 seconds ago',
          responseTime: 45,
          successRate: 99.99
        },
        dependencies: ['database', 'cache'],
        metrics: {
          mtbf: 720, // hours
          mttr: 0.25, // hours
          availability: 99.97
        }
      },
      {
        id: 'SVC-002',
        name: 'Primary Database',
        type: 'database',
        status: 'operational',
        uptime: {
          current: 100,
          daily: 100,
          weekly: 99.99,
          monthly: 99.95,
          yearly: 99.93
        },
        sla: {
          target: 99.99,
          actual: 99.95,
          compliant: false
        },
        healthChecks: {
          endpoint: '/db/health',
          frequency: 60,
          lastCheck: '45 seconds ago',
          responseTime: 12,
          successRate: 100
        },
        dependencies: [],
        metrics: {
          mtbf: 1440,
          mttr: 0.5,
          availability: 99.97
        }
      },
      {
        id: 'SVC-003',
        name: 'Cache Service',
        type: 'cache',
        status: 'degraded',
        uptime: {
          current: 98.5,
          daily: 98.5,
          weekly: 99.2,
          monthly: 99.5,
          yearly: 99.7
        },
        sla: {
          target: 99.5,
          actual: 99.5,
          compliant: true
        },
        lastIncident: {
          timestamp: '2 hours ago',
          duration: 5,
          impact: 'Increased latency'
        },
        healthChecks: {
          endpoint: '/cache/health',
          frequency: 30,
          lastCheck: '15 seconds ago',
          responseTime: 85,
          successRate: 98.5
        },
        dependencies: [],
        metrics: {
          mtbf: 168,
          mttr: 0.08,
          availability: 99.95
        }
      },
      {
        id: 'SVC-004',
        name: 'Storage Service',
        type: 'storage',
        status: 'operational',
        uptime: {
          current: 100,
          daily: 100,
          weekly: 100,
          monthly: 99.99,
          yearly: 99.98
        },
        sla: {
          target: 99.95,
          actual: 99.99,
          compliant: true
        },
        healthChecks: {
          endpoint: '/storage/health',
          frequency: 120,
          lastCheck: '1 minute ago',
          responseTime: 25,
          successRate: 100
        },
        dependencies: [],
        metrics: {
          mtbf: 2160,
          mttr: 0.25,
          availability: 99.99
        }
      },
      {
        id: 'SVC-005',
        name: 'CDN',
        type: 'network',
        status: 'operational',
        uptime: {
          current: 100,
          daily: 100,
          weekly: 100,
          monthly: 100,
          yearly: 99.99
        },
        sla: {
          target: 99.9,
          actual: 100,
          compliant: true
        },
        healthChecks: {
          endpoint: '/cdn/health',
          frequency: 60,
          lastCheck: '30 seconds ago',
          responseTime: 5,
          successRate: 100
        },
        dependencies: ['storage'],
        metrics: {
          mtbf: 4320,
          mttr: 0.1,
          availability: 99.998
        }
      },
      {
        id: 'SVC-006',
        name: 'Compute Cluster',
        type: 'compute',
        status: 'partial_outage',
        uptime: {
          current: 85,
          daily: 92,
          weekly: 98.5,
          monthly: 99.3,
          yearly: 99.5
        },
        sla: {
          target: 99.5,
          actual: 99.3,
          compliant: false
        },
        lastIncident: {
          timestamp: 'Active',
          duration: 30,
          impact: '2 nodes offline'
        },
        healthChecks: {
          endpoint: '/compute/health',
          frequency: 30,
          lastCheck: '10 seconds ago',
          responseTime: 150,
          successRate: 85
        },
        dependencies: ['network'],
        metrics: {
          mtbf: 336,
          mttr: 1.5,
          availability: 99.55
        }
      }
    ]);

    setIncidents([
      {
        id: 'INC-001',
        serviceId: 'SVC-006',
        serviceName: 'Compute Cluster',
        startTime: '30 minutes ago',
        severity: 'high',
        impact: '2 compute nodes offline, reduced capacity',
        status: 'active',
        affectedUsers: 250
      },
      {
        id: 'INC-002',
        serviceId: 'SVC-003',
        serviceName: 'Cache Service',
        startTime: '2 hours ago',
        endTime: '1 hour 55 minutes ago',
        duration: 5,
        severity: 'medium',
        impact: 'Increased response times',
        status: 'resolved',
        affectedUsers: 1500,
        rootCause: 'Memory pressure from increased load'
      },
      {
        id: 'INC-003',
        serviceId: 'SVC-001',
        serviceName: 'API Gateway',
        startTime: '3 days ago',
        endTime: '3 days ago',
        duration: 15,
        severity: 'low',
        impact: 'Minor service degradation',
        status: 'resolved',
        affectedUsers: 50,
        rootCause: 'Configuration update'
      }
    ]);

    setTrends([
      { timestamp: '6h ago', availability: 99.8, incidents: 0 },
      { timestamp: '5h ago', availability: 99.9, incidents: 0 },
      { timestamp: '4h ago', availability: 99.95, incidents: 0 },
      { timestamp: '3h ago', availability: 99.5, incidents: 1 },
      { timestamp: '2h ago', availability: 98.8, incidents: 1 },
      { timestamp: '1h ago', availability: 99.2, incidents: 0 },
      { timestamp: 'Now', availability: 97.5, incidents: 1 }
    ]);

    // Simulate real-time updates
    if (autoRefresh) {
      const interval = setInterval(() => {
        setServices(prevServices => 
          prevServices.map(service => ({
            ...service,
            uptime: {
              ...service.uptime,
              current: Math.max(85, Math.min(100, service.uptime.current + (Math.random() - 0.5) * 2))
            },
            healthChecks: {
              ...service.healthChecks,
              responseTime: Math.max(1, service.healthChecks.responseTime + (Math.random() - 0.5) * 10),
              lastCheck: 'Just now'
            }
          }))
        );
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'operational': return 'text-green-500 bg-green-900/20';
      case 'degraded': return 'text-yellow-500 bg-yellow-900/20';
      case 'partial_outage': return 'text-orange-500 bg-orange-900/20';
      case 'major_outage': return 'text-red-500 bg-red-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'operational': return <CheckCircle className="w-4 h-4" />;
      case 'degraded': return <AlertTriangle className="w-4 h-4" />;
      case 'partial_outage': return <XCircle className="w-4 h-4" />;
      case 'major_outage': return <XCircle className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch(type) {
      case 'api': return <Globe className="w-4 h-4 text-blue-500" />;
      case 'database': return <Database className="w-4 h-4 text-purple-500" />;
      case 'cache': return <Zap className="w-4 h-4 text-yellow-500" />;
      case 'storage': return <Server className="w-4 h-4 text-green-500" />;
      case 'network': return <Wifi className="w-4 h-4 text-cyan-500" />;
      case 'compute': return <Cloud className="w-4 h-4 text-orange-500" />;
      default: return <Server className="w-4 h-4 text-gray-500" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch(severity) {
      case 'low': return 'text-blue-500';
      case 'medium': return 'text-yellow-500';
      case 'high': return 'text-orange-500';
      case 'critical': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const filteredServices = selectedService === 'all' 
    ? services 
    : services.filter(s => s.id === selectedService);

  const overallAvailability = services.reduce((sum, s) => sum + s.uptime.current, 0) / services.length;
  const operationalCount = services.filter(s => s.status === 'operational').length;
  const degradedCount = services.filter(s => s.status === 'degraded').length;
  const outageCount = services.filter(s => s.status === 'partial_outage' || s.status === 'major_outage').length;
  const activeIncidents = incidents.filter(i => i.status === 'active').length;
  const avgMTTR = services.reduce((sum, s) => sum + s.metrics.mttr, 0) / services.length;

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Availability Tracking</h1>
            <p className="text-sm text-gray-400 mt-1">Service uptime and availability monitoring</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                autoRefresh ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              <span>{autoRefresh ? 'Live' : 'Paused'}</span>
            </button>
            
            <button
              onClick={() => setViewMode(
                viewMode === 'overview' ? 'detailed' : 
                viewMode === 'detailed' ? 'timeline' : 'overview'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
            >
              {viewMode === 'overview' ? 'Detailed' : viewMode === 'detailed' ? 'Timeline' : 'Overview'}
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>SLA Report</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <Activity className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Overall</p>
              <p className="text-xl font-bold">{overallAvailability.toFixed(2)}%</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Operational</p>
              <p className="text-xl font-bold">{operationalCount}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Degraded</p>
              <p className="text-xl font-bold text-yellow-500">{degradedCount}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XCircle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Outages</p>
              <p className="text-xl font-bold text-red-500">{outageCount}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Shield className="w-5 h-5 text-orange-500" />
            <div>
              <p className="text-xs text-gray-400">Incidents</p>
              <p className="text-xl font-bold text-orange-500">{activeIncidents}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Timer className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Avg MTTR</p>
              <p className="text-xl font-bold">{avgMTTR.toFixed(1)}h</p>
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
            <option value="1h">Last Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          
          <select
            value={selectedService}
            onChange={(e) => setSelectedService(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Services</option>
            {services.map(service => (
              <option key={service.id} value={service.id}>{service.name}</option>
            ))}
          </select>
        </div>

        {viewMode === 'overview' && (
          <>
            {/* Service Status Grid */}
            <div className="grid grid-cols-3 gap-4 mb-6">
              {filteredServices.map(service => (
                <div key={service.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      {getTypeIcon(service.type)}
                      <h3 className="text-sm font-bold">{service.name}</h3>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded flex items-center space-x-1 ${getStatusColor(service.status)}`}>
                      {getStatusIcon(service.status)}
                      <span>{service.status.replace('_', ' ').toUpperCase()}</span>
                    </span>
                  </div>
                  
                  <div className="mb-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-400">Current Uptime</span>
                      <span className={`text-lg font-bold ${
                        service.uptime.current >= 99.9 ? 'text-green-500' : 
                        service.uptime.current >= 99 ? 'text-yellow-500' : 'text-red-500'
                      }`}>
                        {service.uptime.current.toFixed(2)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          service.uptime.current >= 99.9 ? 'bg-green-500' : 
                          service.uptime.current >= 99 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${service.uptime.current}%` }}
                      />
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                    <div>
                      <p className="text-gray-400">Daily</p>
                      <p className="font-bold">{service.uptime.daily}%</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Weekly</p>
                      <p className="font-bold">{service.uptime.weekly}%</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Monthly</p>
                      <p className="font-bold">{service.uptime.monthly}%</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Yearly</p>
                      <p className="font-bold">{service.uptime.yearly}%</p>
                    </div>
                  </div>
                  
                  <div className="pt-3 border-t border-gray-800 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-400">SLA Target</span>
                      <span>{service.sla.target}%</span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-400">SLA Actual</span>
                      <span className={service.sla.compliant ? 'text-green-500' : 'text-red-500'}>
                        {service.sla.actual}% {service.sla.compliant ? '✓' : '✗'}
                      </span>
                    </div>
                    {service.lastIncident && (
                      <div className="text-xs text-gray-500">
                        Last incident: {service.lastIncident.timestamp}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Availability Trend */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
              <h3 className="text-sm font-bold mb-4">Availability Trend</h3>
              <div className="h-32 flex items-end space-x-2">
                {trends.map((point, idx) => (
                  <div key={idx} className="flex-1 flex flex-col items-center">
                    <div 
                      className={`w-full rounded-t ${
                        point.availability >= 99.9 ? 'bg-green-500' : 
                        point.availability >= 99 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ height: `${(point.availability / 100) * 128}px` }}
                    />
                    <span className="text-xs text-gray-500 mt-1">{point.timestamp}</span>
                    {point.incidents > 0 && (
                      <span className="text-xs text-red-500">{point.incidents} inc</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {viewMode === 'detailed' && (
          <div className="space-y-4">
            {filteredServices.map(service => (
              <div key={service.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    {getTypeIcon(service.type)}
                    <div>
                      <h3 className="text-sm font-bold">{service.name}</h3>
                      <p className="text-xs text-gray-400">Service ID: {service.id}</p>
                    </div>
                  </div>
                  <span className={`px-3 py-1 text-sm rounded ${getStatusColor(service.status)}`}>
                    {service.status.replace('_', ' ').toUpperCase()}
                  </span>
                </div>
                
                <div className="grid grid-cols-5 gap-4 mb-4">
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Current</p>
                    <p className="text-lg font-bold">{service.uptime.current.toFixed(2)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Daily</p>
                    <p className="text-lg font-bold">{service.uptime.daily}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Weekly</p>
                    <p className="text-lg font-bold">{service.uptime.weekly}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Monthly</p>
                    <p className="text-lg font-bold">{service.uptime.monthly}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Yearly</p>
                    <p className="text-lg font-bold">{service.uptime.yearly}%</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-4 p-3 bg-gray-800 rounded">
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Health Check</p>
                    <p className="text-sm">Every {service.healthChecks.frequency}s</p>
                    <p className="text-xs text-gray-500">{service.healthChecks.endpoint}</p>
                    <p className="text-xs text-green-500">Success: {service.healthChecks.successRate}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Response Time</p>
                    <p className="text-sm">{service.healthChecks.responseTime}ms</p>
                    <p className="text-xs text-gray-500">Last: {service.healthChecks.lastCheck}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Reliability Metrics</p>
                    <p className="text-sm">MTBF: {service.metrics.mtbf}h</p>
                    <p className="text-sm">MTTR: {service.metrics.mttr}h</p>
                    <p className="text-sm">Availability: {service.metrics.availability}%</p>
                  </div>
                </div>
                
                {service.dependencies.length > 0 && (
                  <div className="mt-3 text-xs">
                    <span className="text-gray-400">Dependencies: </span>
                    {service.dependencies.map((dep, idx) => (
                      <span key={idx} className="ml-2 px-2 py-1 bg-gray-800 rounded">
                        {dep}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {viewMode === 'timeline' && (
          <>
            {/* Active Incidents */}
            {activeIncidents > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-bold mb-3 text-red-500">Active Incidents</h3>
                <div className="space-y-2">
                  {incidents.filter(i => i.status === 'active').map(incident => (
                    <div key={incident.id} className="bg-red-900/20 border border-red-800 rounded-lg p-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <XCircle className="w-4 h-4 text-red-500" />
                          <div>
                            <p className="text-sm font-bold">{incident.serviceName}</p>
                            <p className="text-xs text-gray-400">{incident.impact}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className={`text-xs font-bold ${getSeverityColor(incident.severity)}`}>
                            {incident.severity.toUpperCase()}
                          </p>
                          <p className="text-xs text-gray-500">Started {incident.startTime}</p>
                          <p className="text-xs text-gray-500">{incident.affectedUsers} users affected</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Incident History */}
            <div>
              <h3 className="text-sm font-bold mb-3">Incident History</h3>
              <div className="space-y-2">
                {incidents.filter(i => i.status === 'resolved').map(incident => (
                  <div key={incident.id} className="bg-gray-900 border border-gray-800 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <div>
                          <p className="text-sm font-bold">{incident.serviceName}</p>
                          <p className="text-xs text-gray-400">{incident.impact}</p>
                          {incident.rootCause && (
                            <p className="text-xs text-gray-500">Root cause: {incident.rootCause}</p>
                          )}
                        </div>
                      </div>
                      <div className="text-right">
                        <p className={`text-xs font-bold ${getSeverityColor(incident.severity)}`}>
                          {incident.severity.toUpperCase()}
                        </p>
                        <p className="text-xs text-gray-500">{incident.startTime} - {incident.endTime}</p>
                        <p className="text-xs text-gray-500">Duration: {incident.duration}m</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}