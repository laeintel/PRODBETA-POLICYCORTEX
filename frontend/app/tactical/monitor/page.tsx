'use client';

import React, { useState, useEffect } from 'react';
import { 
  Activity, Server, Database, Network, Cpu, HardDrive, Wifi, 
  AlertTriangle, CheckCircle, XCircle, RefreshCw, Clock, 
  TrendingUp, TrendingDown, Zap, BarChart, Monitor, Globe,
  Shield, Users, Cloud, Package, GitBranch, Terminal
} from 'lucide-react';
import { api } from '../../../lib/api-client';

export default function RealTimeMonitor() {
  const [refreshing, setRefreshing] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState('all');
  const [timeRange, setTimeRange] = useState('1h');
  
  // Real-time metrics
  const [systemMetrics, setSystemMetrics] = useState({
    cpu: { current: 73, trend: 'up', history: [65, 68, 70, 72, 73] },
    memory: { current: 61, trend: 'stable', history: [60, 61, 61, 60, 61] },
    disk: { current: 84, trend: 'up', history: [82, 82, 83, 83, 84] },
    network: { current: 42, trend: 'down', history: [45, 44, 43, 42, 42] },
    requests: { current: 1247, trend: 'up', history: [1100, 1150, 1200, 1230, 1247] },
    errors: { current: 3, trend: 'down', history: [5, 4, 4, 3, 3] }
  });

  const [services, setServices] = useState([
    { id: 1, name: 'API Gateway', status: 'healthy', uptime: 99.99, latency: 87, requests: 5234 },
    { id: 2, name: 'Database Cluster', status: 'healthy', uptime: 99.97, latency: 12, requests: 8923 },
    { id: 3, name: 'Cache Service', status: 'healthy', uptime: 100, latency: 2, requests: 15234 },
    { id: 4, name: 'Message Queue', status: 'warning', uptime: 99.5, latency: 145, requests: 3421 },
    { id: 5, name: 'Storage Service', status: 'healthy', uptime: 99.98, latency: 23, requests: 4567 },
    { id: 6, name: 'Search Engine', status: 'healthy', uptime: 99.95, latency: 34, requests: 2345 },
    { id: 7, name: 'ML Pipeline', status: 'healthy', uptime: 99.9, latency: 234, requests: 892 },
    { id: 8, name: 'Notification Service', status: 'error', uptime: 98.2, latency: 567, requests: 1234 }
  ]);

  const [events, setEvents] = useState([
    { id: 1, type: 'info', message: 'Deployment completed successfully', service: 'API Gateway', time: '2 min ago' },
    { id: 2, type: 'warning', message: 'High memory usage detected', service: 'Database Cluster', time: '5 min ago' },
    { id: 3, type: 'error', message: 'Connection timeout', service: 'Notification Service', time: '12 min ago' },
    { id: 4, type: 'success', message: 'Auto-scaling triggered', service: 'API Gateway', time: '15 min ago' },
    { id: 5, type: 'info', message: 'Backup completed', service: 'Database Cluster', time: '30 min ago' }
  ]);

  const [regions] = useState([
    { name: 'East US', status: 'operational', latency: 12, load: 67 },
    { name: 'West Europe', status: 'operational', latency: 45, load: 54 },
    { name: 'Southeast Asia', status: 'degraded', latency: 123, load: 82 },
    { name: 'Brazil South', status: 'operational', latency: 67, load: 43 }
  ]);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemMetrics(prev => ({
        cpu: {
          ...prev.cpu,
          current: Math.min(100, Math.max(0, prev.cpu.current + (Math.random() - 0.5) * 10)),
          history: [...prev.cpu.history.slice(1), prev.cpu.current]
        },
        memory: {
          ...prev.memory,
          current: Math.min(100, Math.max(0, prev.memory.current + (Math.random() - 0.5) * 5)),
          history: [...prev.memory.history.slice(1), prev.memory.current]
        },
        disk: {
          ...prev.disk,
          current: Math.min(100, Math.max(0, prev.disk.current + (Math.random() - 0.5) * 2)),
          history: [...prev.disk.history.slice(1), prev.disk.current]
        },
        network: {
          ...prev.network,
          current: Math.min(100, Math.max(0, prev.network.current + (Math.random() - 0.5) * 15)),
          history: [...prev.network.history.slice(1), prev.network.current]
        },
        requests: {
          ...prev.requests,
          current: Math.floor(prev.requests.current + (Math.random() - 0.3) * 100),
          history: [...prev.requests.history.slice(1), prev.requests.current]
        },
        errors: {
          ...prev.errors,
          current: Math.max(0, Math.floor(prev.errors.current + (Math.random() - 0.7) * 2)),
          history: [...prev.errors.history.slice(1), prev.errors.current]
        }
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const refreshData = () => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1000);
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'healthy':
      case 'operational':
        return 'text-green-500';
      case 'warning':
      case 'degraded':
        return 'text-yellow-500';
      case 'error':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  const getStatusBg = (status: string) => {
    switch(status) {
      case 'healthy':
      case 'operational':
        return 'bg-green-900/20';
      case 'warning':
      case 'degraded':
        return 'bg-yellow-900/20';
      case 'error':
        return 'bg-red-900/20';
      default:
        return 'bg-gray-900/20';
    }
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Real-time System Monitor</h1>
            <p className="text-sm text-gray-400 mt-1">Live monitoring of all system components and services</p>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* Time Range Selector */}
            <select 
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-3 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
            >
              <option value="5m">Last 5 minutes</option>
              <option value="15m">Last 15 minutes</option>
              <option value="1h">Last hour</option>
              <option value="6h">Last 6 hours</option>
              <option value="24h">Last 24 hours</option>
            </select>
            
            <button
              onClick={refreshData}
              className={`p-2 hover:bg-gray-800 rounded ${refreshing ? 'animate-spin' : ''}`}
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            
            <div className="flex items-center space-x-2 px-3 py-1 bg-green-900/20 border border-green-900/30 rounded">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-xs text-green-500">LIVE</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* System Metrics Grid */}
        <div className="grid grid-cols-6 gap-4 mb-6">
          <MetricCard
            title="CPU Usage"
            value={`${systemMetrics.cpu.current.toFixed(1)}%`}
            icon={Cpu}
            trend={systemMetrics.cpu.trend}
            history={systemMetrics.cpu.history}
            color="blue"
          />
          <MetricCard
            title="Memory"
            value={`${systemMetrics.memory.current.toFixed(1)}%`}
            icon={Monitor}
            trend={systemMetrics.memory.trend}
            history={systemMetrics.memory.history}
            color="green"
          />
          <MetricCard
            title="Disk Usage"
            value={`${systemMetrics.disk.current.toFixed(1)}%`}
            icon={HardDrive}
            trend={systemMetrics.disk.trend}
            history={systemMetrics.disk.history}
            color="yellow"
          />
          <MetricCard
            title="Network I/O"
            value={`${systemMetrics.network.current.toFixed(1)}%`}
            icon={Network}
            trend={systemMetrics.network.trend}
            history={systemMetrics.network.history}
            color="purple"
          />
          <MetricCard
            title="Requests/min"
            value={systemMetrics.requests.current.toLocaleString()}
            icon={Activity}
            trend={systemMetrics.requests.trend}
            history={systemMetrics.requests.history}
            color="cyan"
          />
          <MetricCard
            title="Errors"
            value={systemMetrics.errors.current.toString()}
            icon={AlertTriangle}
            trend={systemMetrics.errors.trend}
            history={systemMetrics.errors.history}
            color="red"
          />
        </div>

        <div className="grid grid-cols-12 gap-6">
          {/* Service Health Status */}
          <div className="col-span-8 bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold mb-4">Service Health Status</h3>
            <div className="space-y-2">
              {services.map(service => (
                <div key={service.id} className="flex items-center justify-between p-3 bg-gray-800 rounded hover:bg-gray-750">
                  <div className="flex items-center space-x-3">
                    <div className={`w-2 h-2 rounded-full ${
                      service.status === 'healthy' ? 'bg-green-500' :
                      service.status === 'warning' ? 'bg-yellow-500' :
                      'bg-red-500'
                    } ${service.status === 'healthy' ? 'animate-pulse' : ''}`} />
                    <div>
                      <p className="text-sm font-medium">{service.name}</p>
                      <p className="text-xs text-gray-400">Uptime: {service.uptime}%</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-6">
                    <div className="text-right">
                      <p className="text-xs text-gray-400">Latency</p>
                      <p className="text-sm font-mono">{service.latency}ms</p>
                    </div>
                    <div className="text-right">
                      <p className="text-xs text-gray-400">Requests</p>
                      <p className="text-sm font-mono">{service.requests.toLocaleString()}</p>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded ${getStatusBg(service.status)} ${getStatusColor(service.status)}`}>
                      {service.status.toUpperCase()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Regional Status */}
          <div className="col-span-4 bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold mb-4">Regional Status</h3>
            <div className="space-y-3">
              {regions.map(region => (
                <div key={region.name} className="p-3 bg-gray-800 rounded">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <Globe className="w-4 h-4 text-gray-400" />
                      <span className="text-sm font-medium">{region.name}</span>
                    </div>
                    <span className={`text-xs ${getStatusColor(region.status)}`}>
                      {region.status.toUpperCase()}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 mt-2">
                    <div>
                      <p className="text-xs text-gray-400">Latency</p>
                      <p className="text-sm font-mono">{region.latency}ms</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Load</p>
                      <div className="flex items-center space-x-1">
                        <div className="flex-1 bg-gray-700 rounded-full h-1.5">
                          <div 
                            className={`h-1.5 rounded-full ${
                              region.load > 80 ? 'bg-red-500' :
                              region.load > 60 ? 'bg-yellow-500' :
                              'bg-green-500'
                            }`}
                            style={{ width: `${region.load}%` }}
                          />
                        </div>
                        <span className="text-xs">{region.load}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Live Event Stream */}
          <div className="col-span-12 bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold">Live Event Stream</h3>
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-gray-400" />
                <span className="text-xs text-gray-400">Auto-updating</span>
              </div>
            </div>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {events.map(event => (
                <div key={event.id} className="flex items-start space-x-3 p-2 hover:bg-gray-800 rounded">
                  <div className={`w-2 h-2 rounded-full mt-1.5 ${
                    event.type === 'error' ? 'bg-red-500' :
                    event.type === 'warning' ? 'bg-yellow-500' :
                    event.type === 'success' ? 'bg-green-500' :
                    'bg-blue-500'
                  }`} />
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <p className="text-sm">{event.message}</p>
                      <span className="text-xs text-gray-500">{event.time}</span>
                    </div>
                    <p className="text-xs text-gray-400">Service: {event.service}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// Metric Card Component with Sparkline
function MetricCard({ title, value, icon: Icon, trend, history, color }: any) {
  const colorClasses = {
    blue: 'text-blue-500 bg-blue-900/20',
    green: 'text-green-500 bg-green-900/20',
    yellow: 'text-yellow-500 bg-yellow-900/20',
    purple: 'text-purple-500 bg-purple-900/20',
    cyan: 'text-cyan-500 bg-cyan-900/20',
    red: 'text-red-500 bg-red-900/20'
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-gray-400">{title}</p>
        <div className={`p-1.5 rounded ${colorClasses[color]}`}>
          <Icon className="w-4 h-4" />
        </div>
      </div>
      <p className="text-xl font-bold mb-2">{value}</p>
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          {trend === 'up' ? (
            <TrendingUp className="w-3 h-3 text-green-500" />
          ) : trend === 'down' ? (
            <TrendingDown className="w-3 h-3 text-red-500" />
          ) : (
            <Activity className="w-3 h-3 text-gray-500" />
          )}
        </div>
        <div className="flex items-end space-x-0.5 h-8">
          {history.map((val: number, i: number) => (
            <div
              key={i}
              className={`w-1 ${colorClasses[color].split(' ')[0]} opacity-${i === history.length - 1 ? '100' : '60'}`}
              style={{ height: `${(val / Math.max(...history)) * 100}%` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}