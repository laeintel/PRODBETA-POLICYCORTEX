'use client';

import React, { useState, useEffect } from 'react';
import { 
  Heart, Activity, Server, Database, Globe, Shield, Cloud, Network,
  CheckCircle, XCircle, AlertTriangle, RefreshCw, Clock, TrendingUp,
  TrendingDown, BarChart, PieChart, Zap, Cpu, HardDrive, Wifi,
  Package, GitBranch, Terminal, Monitor, Bell, Info
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface ServiceHealthData {
  id: string;
  name: string;
  category: string;
  status: 'healthy' | 'degraded' | 'outage' | 'maintenance';
  uptime: number;
  responseTime: number;
  errorRate: number;
  throughput: number;
  dependencies: string[];
  lastCheck: string;
  incidents: number;
  region: string;
}

interface HealthIncident {
  id: string;
  service: string;
  type: 'outage' | 'degradation' | 'maintenance';
  severity: 'low' | 'medium' | 'high' | 'critical';
  startTime: string;
  endTime?: string;
  impact: string;
  status: 'active' | 'resolved' | 'investigating';
}

export default function ServiceHealth() {
  const [services, setServices] = useState<ServiceHealthData[]>([]);
  const [incidents, setIncidents] = useState<HealthIncident[]>([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedRegion, setSelectedRegion] = useState('all');
  const [timeRange, setTimeRange] = useState('24h');
  const [refreshing, setRefreshing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Initialize with mock data
  useEffect(() => {
    setServices([
      {
        id: 'api-gateway',
        name: 'API Gateway',
        category: 'Core Services',
        status: 'healthy',
        uptime: 99.99,
        responseTime: 87,
        errorRate: 0.01,
        throughput: 5234,
        dependencies: ['Database Cluster', 'Cache Service'],
        lastCheck: '30 seconds ago',
        incidents: 0,
        region: 'East US'
      },
      {
        id: 'database-cluster',
        name: 'Database Cluster',
        category: 'Data Services',
        status: 'healthy',
        uptime: 99.97,
        responseTime: 12,
        errorRate: 0.02,
        throughput: 8923,
        dependencies: ['Storage Service'],
        lastCheck: '45 seconds ago',
        incidents: 0,
        region: 'East US'
      },
      {
        id: 'cache-service',
        name: 'Cache Service',
        category: 'Core Services',
        status: 'healthy',
        uptime: 100,
        responseTime: 2,
        errorRate: 0,
        throughput: 15234,
        dependencies: [],
        lastCheck: '15 seconds ago',
        incidents: 0,
        region: 'East US'
      },
      {
        id: 'message-queue',
        name: 'Message Queue',
        category: 'Integration',
        status: 'degraded',
        uptime: 99.5,
        responseTime: 145,
        errorRate: 0.8,
        throughput: 3421,
        dependencies: ['Storage Service'],
        lastCheck: '1 minute ago',
        incidents: 1,
        region: 'East US'
      },
      {
        id: 'storage-service',
        name: 'Storage Service',
        category: 'Data Services',
        status: 'healthy',
        uptime: 99.98,
        responseTime: 23,
        errorRate: 0.01,
        throughput: 4567,
        dependencies: [],
        lastCheck: '20 seconds ago',
        incidents: 0,
        region: 'East US'
      },
      {
        id: 'search-engine',
        name: 'Search Engine',
        category: 'Data Services',
        status: 'healthy',
        uptime: 99.95,
        responseTime: 34,
        errorRate: 0.03,
        throughput: 2345,
        dependencies: ['Database Cluster'],
        lastCheck: '50 seconds ago',
        incidents: 0,
        region: 'West Europe'
      },
      {
        id: 'ml-pipeline',
        name: 'ML Pipeline',
        category: 'AI Services',
        status: 'maintenance',
        uptime: 99.9,
        responseTime: 234,
        errorRate: 0.05,
        throughput: 892,
        dependencies: ['GPU Cluster', 'Storage Service'],
        lastCheck: '2 minutes ago',
        incidents: 0,
        region: 'West US'
      },
      {
        id: 'notification-service',
        name: 'Notification Service',
        category: 'Communication',
        status: 'outage',
        uptime: 98.2,
        responseTime: 567,
        errorRate: 12.5,
        throughput: 1234,
        dependencies: ['Message Queue', 'Email Gateway'],
        lastCheck: '5 seconds ago',
        incidents: 3,
        region: 'East US'
      },
      {
        id: 'auth-service',
        name: 'Authentication Service',
        category: 'Security',
        status: 'healthy',
        uptime: 99.99,
        responseTime: 45,
        errorRate: 0.001,
        throughput: 7890,
        dependencies: ['Database Cluster', 'Cache Service'],
        lastCheck: '10 seconds ago',
        incidents: 0,
        region: 'East US'
      },
      {
        id: 'cdn',
        name: 'Content Delivery Network',
        category: 'Network',
        status: 'healthy',
        uptime: 99.99,
        responseTime: 8,
        errorRate: 0.001,
        throughput: 45678,
        dependencies: ['Storage Service'],
        lastCheck: '25 seconds ago',
        incidents: 0,
        region: 'Global'
      }
    ]);

    setIncidents([
      {
        id: 'inc-001',
        service: 'Notification Service',
        type: 'outage',
        severity: 'critical',
        startTime: '10:45 AM',
        impact: 'Email notifications are not being delivered',
        status: 'investigating'
      },
      {
        id: 'inc-002',
        service: 'Message Queue',
        type: 'degradation',
        severity: 'medium',
        startTime: '9:30 AM',
        impact: 'Increased latency in message processing',
        status: 'active'
      },
      {
        id: 'inc-003',
        service: 'ML Pipeline',
        type: 'maintenance',
        severity: 'low',
        startTime: '8:00 AM',
        endTime: '12:00 PM',
        impact: 'Scheduled maintenance - reduced capacity',
        status: 'active'
      }
    ]);
  }, []);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      refreshData();
    }, 30000);
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const refreshData = () => {
    setRefreshing(true);
    // Simulate refresh
    setTimeout(() => {
      setRefreshing(false);
    }, 1000);
  };

  const categories = [
    'all',
    'Core Services',
    'Data Services',
    'AI Services',
    'Security',
    'Network',
    'Integration',
    'Communication'
  ];

  const regions = ['all', 'East US', 'West US', 'West Europe', 'Global'];

  const filteredServices = services.filter(service => {
    if (selectedCategory !== 'all' && service.category !== selectedCategory) return false;
    if (selectedRegion !== 'all' && service.region !== selectedRegion) return false;
    return true;
  });

  const healthStats = {
    healthy: services.filter(s => s.status === 'healthy').length,
    degraded: services.filter(s => s.status === 'degraded').length,
    outage: services.filter(s => s.status === 'outage').length,
    maintenance: services.filter(s => s.status === 'maintenance').length,
    totalUptime: (services.reduce((sum, s) => sum + s.uptime, 0) / services.length).toFixed(2),
    avgResponseTime: Math.round(services.reduce((sum, s) => sum + s.responseTime, 0) / services.length)
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'healthy': return 'text-green-500';
      case 'degraded': return 'text-yellow-500';
      case 'outage': return 'text-red-500';
      case 'maintenance': return 'text-blue-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusBg = (status: string) => {
    switch(status) {
      case 'healthy': return 'bg-green-900/20 border-green-900/30';
      case 'degraded': return 'bg-yellow-900/20 border-yellow-900/30';
      case 'outage': return 'bg-red-900/20 border-red-900/30';
      case 'maintenance': return 'bg-blue-900/20 border-blue-900/30';
      default: return 'bg-gray-900/20 border-gray-900/30';
    }
  };

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'healthy': return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'degraded': return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'outage': return <XCircle className="w-5 h-5 text-red-500" />;
      case 'maintenance': return <Clock className="w-5 h-5 text-blue-500" />;
      default: return null;
    }
  };

  const getCategoryIcon = (category: string) => {
    switch(category) {
      case 'Core Services': return Server;
      case 'Data Services': return Database;
      case 'AI Services': return Cpu;
      case 'Security': return Shield;
      case 'Network': return Network;
      case 'Integration': return GitBranch;
      case 'Communication': return Bell;
      default: return Server;
    }
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Service Health Monitor</h1>
            <p className="text-sm text-gray-400 mt-1">Real-time health status of all services and dependencies</p>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* Time Range */}
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-3 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
            >
              <option value="1h">Last Hour</option>
              <option value="6h">Last 6 Hours</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
            
            {/* Auto Refresh Toggle */}
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-1 rounded text-sm flex items-center space-x-2 ${
                autoRefresh ? 'bg-green-900/20 text-green-500 border border-green-900/30' : 
                'bg-gray-800 text-gray-400 border border-gray-700'
              }`}
            >
              <Activity className="w-3 h-3" />
              <span>Auto Refresh</span>
            </button>
            
            {/* Manual Refresh */}
            <button
              onClick={refreshData}
              className={`p-2 hover:bg-gray-800 rounded ${refreshing ? 'animate-spin' : ''}`}
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Health Overview */}
        <div className="grid grid-cols-6 gap-4 mb-6">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-gray-400">Healthy</p>
              <CheckCircle className="w-4 h-4 text-green-500" />
            </div>
            <p className="text-2xl font-bold text-green-500">{healthStats.healthy}</p>
            <p className="text-xs text-gray-500 mt-1">Services</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-gray-400">Degraded</p>
              <AlertTriangle className="w-4 h-4 text-yellow-500" />
            </div>
            <p className="text-2xl font-bold text-yellow-500">{healthStats.degraded}</p>
            <p className="text-xs text-gray-500 mt-1">Services</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-gray-400">Outage</p>
              <XCircle className="w-4 h-4 text-red-500" />
            </div>
            <p className="text-2xl font-bold text-red-500">{healthStats.outage}</p>
            <p className="text-xs text-gray-500 mt-1">Services</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-gray-400">Maintenance</p>
              <Clock className="w-4 h-4 text-blue-500" />
            </div>
            <p className="text-2xl font-bold text-blue-500">{healthStats.maintenance}</p>
            <p className="text-xs text-gray-500 mt-1">Services</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-gray-400">Avg Uptime</p>
              <TrendingUp className="w-4 h-4 text-cyan-500" />
            </div>
            <p className="text-2xl font-bold">{healthStats.totalUptime}%</p>
            <p className="text-xs text-gray-500 mt-1">Overall</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-gray-400">Avg Response</p>
              <Zap className="w-4 h-4 text-purple-500" />
            </div>
            <p className="text-2xl font-bold">{healthStats.avgResponseTime}ms</p>
            <p className="text-xs text-gray-500 mt-1">Latency</p>
          </div>
        </div>

        {/* Active Incidents */}
        {incidents.length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-bold mb-3">Active Incidents</h3>
            <div className="space-y-2">
              {incidents.map(incident => (
                <div key={incident.id} className={`p-4 rounded-lg border ${
                  incident.severity === 'critical' ? 'bg-red-900/10 border-red-900/30' :
                  incident.severity === 'high' ? 'bg-orange-900/10 border-orange-900/30' :
                  incident.severity === 'medium' ? 'bg-yellow-900/10 border-yellow-900/30' :
                  'bg-blue-900/10 border-blue-900/30'
                }`}>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <span className={`px-2 py-1 text-xs rounded ${
                          incident.type === 'outage' ? 'bg-red-900/30 text-red-500' :
                          incident.type === 'degradation' ? 'bg-yellow-900/30 text-yellow-500' :
                          'bg-blue-900/30 text-blue-500'
                        }`}>
                          {incident.type.toUpperCase()}
                        </span>
                        <span className="text-sm font-medium">{incident.service}</span>
                        <span className={`text-xs ${
                          incident.status === 'investigating' ? 'text-yellow-500' :
                          incident.status === 'active' ? 'text-orange-500' :
                          'text-green-500'
                        }`}>
                          • {incident.status.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-sm text-gray-300">{incident.impact}</p>
                      <div className="flex items-center space-x-4 mt-2 text-xs text-gray-400">
                        <span>Started: {incident.startTime}</span>
                        {incident.endTime && <span>Expected End: {incident.endTime}</span>}
                      </div>
                    </div>
                    <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                      View Details
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="flex items-center space-x-3 mb-6">
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            {categories.map(cat => (
              <option key={cat} value={cat}>
                {cat === 'all' ? 'All Categories' : cat}
              </option>
            ))}
          </select>
          
          <select
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            {regions.map(region => (
              <option key={region} value={region}>
                {region === 'all' ? 'All Regions' : region}
              </option>
            ))}
          </select>
        </div>

        {/* Services Grid */}
        <div className="grid grid-cols-2 gap-4">
          {filteredServices.map(service => {
            const CategoryIcon = getCategoryIcon(service.category);
            return (
              <div key={service.id} className={`bg-gray-900 border rounded-lg p-4 ${getStatusBg(service.status)}`}>
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <CategoryIcon className="w-5 h-5 text-gray-400" />
                    <div>
                      <p className="font-medium">{service.name}</p>
                      <p className="text-xs text-gray-400">{service.category} • {service.region}</p>
                    </div>
                  </div>
                  {getStatusIcon(service.status)}
                </div>
                
                <div className="grid grid-cols-4 gap-3 mb-3">
                  <div>
                    <p className="text-xs text-gray-400">Uptime</p>
                    <p className="text-sm font-bold">{service.uptime}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">Response</p>
                    <p className="text-sm font-bold">{service.responseTime}ms</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">Error Rate</p>
                    <p className="text-sm font-bold">{service.errorRate}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">Throughput</p>
                    <p className="text-sm font-bold">{service.throughput.toLocaleString()}</p>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4 text-xs text-gray-400">
                    <span>Last check: {service.lastCheck}</span>
                    {service.incidents > 0 && (
                      <span className="text-yellow-500">{service.incidents} incidents</span>
                    )}
                  </div>
                  <span className={`px-2 py-1 text-xs rounded ${getStatusBg(service.status)} ${getStatusColor(service.status)}`}>
                    {service.status.toUpperCase()}
                  </span>
                </div>
                
                {service.dependencies.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-800">
                    <p className="text-xs text-gray-400 mb-1">Dependencies:</p>
                    <div className="flex flex-wrap gap-1">
                      {service.dependencies.map(dep => (
                        <span key={dep} className="px-2 py-0.5 bg-gray-800 rounded text-xs">
                          {dep}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}