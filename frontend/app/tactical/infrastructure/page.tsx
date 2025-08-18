'use client';

import React, { useState, useEffect } from 'react';
import { 
  Server, Database, Cloud, HardDrive, Cpu, Network, Shield, Lock,
  Container, Layers, GitBranch, Box, Globe, Wifi, Battery, Thermometer,
  Activity, AlertTriangle, CheckCircle, XCircle, Info, Settings,
  MoreVertical, ChevronRight, ChevronDown, ExternalLink, Terminal,
  Code, FileCode, Package, Archive, Folder, FolderOpen, Upload,
  Download, RefreshCw, Power, PowerOff, Play, Pause, RotateCw,
  Zap, Clock, Timer, Calendar, TrendingUp, TrendingDown, BarChart3
} from 'lucide-react';

interface InfrastructureResource {
  id: string;
  name: string;
  type: 'vm' | 'container' | 'database' | 'storage' | 'network' | 'loadbalancer' | 'cdn' | 'dns';
  status: 'running' | 'stopped' | 'pending' | 'error' | 'maintenance';
  provider: 'azure' | 'aws' | 'gcp' | 'onprem';
  region: string;
  zone?: string;
  specs: {
    cpu?: number;
    memory?: number;
    storage?: number;
    bandwidth?: number;
  };
  utilization: {
    cpu?: number;
    memory?: number;
    disk?: number;
    network?: number;
  };
  cost: {
    hourly: number;
    monthly: number;
    currency: string;
  };
  tags: string[];
  created: string;
  modified: string;
  uptime: number;
  sla: number;
  dependencies: string[];
  alerts: number;
  backups: {
    enabled: boolean;
    lastBackup?: string;
    nextBackup?: string;
    retention: number;
  };
}

interface ResourceGroup {
  id: string;
  name: string;
  environment: 'production' | 'staging' | 'development' | 'test';
  resources: InfrastructureResource[];
  totalCost: number;
  healthScore: number;
  complianceScore: number;
  securityScore: number;
}

interface InfrastructureMetric {
  id: string;
  title: string;
  value: number | string;
  unit?: string;
  change: number;
  trend: 'up' | 'down' | 'stable';
  category: string;
  severity?: 'normal' | 'warning' | 'critical';
}

export default function InfrastructureManagement() {
  const [resourceGroups, setResourceGroups] = useState<ResourceGroup[]>([]);
  const [selectedGroup, setSelectedGroup] = useState<string | null>(null);
  const [selectedResource, setSelectedResource] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'tree' | 'map'>('grid');
  const [filterType, setFilterType] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterProvider, setFilterProvider] = useState('all');
  const [showMetrics, setShowMetrics] = useState(true);
  const [expandedGroups, setExpandedGroups] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<InfrastructureMetric[]>([]);

  useEffect(() => {
    // Initialize with infrastructure data
    const mockResources: InfrastructureResource[] = [
      {
        id: 'RES-001',
        name: 'prod-api-server-01',
        type: 'vm',
        status: 'running',
        provider: 'azure',
        region: 'East US',
        zone: 'Zone 1',
        specs: { cpu: 8, memory: 32, storage: 500 },
        utilization: { cpu: 65, memory: 72, disk: 45, network: 38 },
        cost: { hourly: 0.85, monthly: 612, currency: 'USD' },
        tags: ['production', 'api', 'critical'],
        created: '2024-01-15',
        modified: '2024-12-18',
        uptime: 99.99,
        sla: 99.95,
        dependencies: ['prod-db-01', 'prod-cache-01'],
        alerts: 0,
        backups: { enabled: true, lastBackup: '2 hours ago', nextBackup: 'in 4 hours', retention: 30 }
      },
      {
        id: 'RES-002',
        name: 'prod-db-01',
        type: 'database',
        status: 'running',
        provider: 'azure',
        region: 'East US',
        zone: 'Zone 1',
        specs: { cpu: 16, memory: 64, storage: 2000 },
        utilization: { cpu: 48, memory: 68, disk: 72, network: 25 },
        cost: { hourly: 2.50, monthly: 1800, currency: 'USD' },
        tags: ['production', 'database', 'critical', 'postgresql'],
        created: '2024-01-10',
        modified: '2024-12-17',
        uptime: 99.95,
        sla: 99.99,
        dependencies: [],
        alerts: 1,
        backups: { enabled: true, lastBackup: '30 minutes ago', nextBackup: 'in 30 minutes', retention: 90 }
      },
      {
        id: 'RES-003',
        name: 'prod-k8s-cluster',
        type: 'container',
        status: 'running',
        provider: 'azure',
        region: 'East US',
        specs: { cpu: 32, memory: 128, storage: 1000 },
        utilization: { cpu: 55, memory: 62, disk: 38, network: 45 },
        cost: { hourly: 3.20, monthly: 2304, currency: 'USD' },
        tags: ['production', 'kubernetes', 'container', 'orchestration'],
        created: '2024-02-01',
        modified: '2024-12-18',
        uptime: 99.98,
        sla: 99.95,
        dependencies: ['prod-registry', 'prod-storage'],
        alerts: 0,
        backups: { enabled: true, lastBackup: '1 hour ago', nextBackup: 'in 5 hours', retention: 14 }
      },
      {
        id: 'RES-004',
        name: 'prod-storage-account',
        type: 'storage',
        status: 'running',
        provider: 'azure',
        region: 'East US',
        specs: { storage: 10000 },
        utilization: { disk: 68 },
        cost: { hourly: 0.45, monthly: 324, currency: 'USD' },
        tags: ['production', 'storage', 'blob'],
        created: '2024-01-05',
        modified: '2024-12-16',
        uptime: 99.999,
        sla: 99.99,
        dependencies: [],
        alerts: 0,
        backups: { enabled: true, lastBackup: '6 hours ago', nextBackup: 'in 18 hours', retention: 365 }
      },
      {
        id: 'RES-005',
        name: 'prod-lb-01',
        type: 'loadbalancer',
        status: 'running',
        provider: 'azure',
        region: 'East US',
        specs: { bandwidth: 1000 },
        utilization: { network: 42 },
        cost: { hourly: 0.25, monthly: 180, currency: 'USD' },
        tags: ['production', 'networking', 'load-balancer'],
        created: '2024-01-12',
        modified: '2024-12-18',
        uptime: 100,
        sla: 99.99,
        dependencies: ['prod-api-server-01', 'prod-api-server-02'],
        alerts: 0,
        backups: { enabled: false, retention: 0 }
      },
      {
        id: 'RES-006',
        name: 'dev-test-server',
        type: 'vm',
        status: 'stopped',
        provider: 'azure',
        region: 'West US',
        zone: 'Zone 2',
        specs: { cpu: 4, memory: 16, storage: 200 },
        utilization: { cpu: 0, memory: 0, disk: 0, network: 0 },
        cost: { hourly: 0, monthly: 0, currency: 'USD' },
        tags: ['development', 'test', 'non-critical'],
        created: '2024-03-15',
        modified: '2024-12-10',
        uptime: 0,
        sla: 0,
        dependencies: [],
        alerts: 0,
        backups: { enabled: false, retention: 0 }
      },
      {
        id: 'RES-007',
        name: 'prod-cdn',
        type: 'cdn',
        status: 'running',
        provider: 'azure',
        region: 'Global',
        specs: { bandwidth: 10000 },
        utilization: { network: 28 },
        cost: { hourly: 0.50, monthly: 360, currency: 'USD' },
        tags: ['production', 'cdn', 'global'],
        created: '2024-01-20',
        modified: '2024-12-18',
        uptime: 99.99,
        sla: 99.95,
        dependencies: ['prod-storage-account'],
        alerts: 0,
        backups: { enabled: false, retention: 0 }
      },
      {
        id: 'RES-008',
        name: 'staging-env',
        type: 'container',
        status: 'maintenance',
        provider: 'azure',
        region: 'Central US',
        specs: { cpu: 16, memory: 64, storage: 500 },
        utilization: { cpu: 0, memory: 0, disk: 0, network: 0 },
        cost: { hourly: 0, monthly: 0, currency: 'USD' },
        tags: ['staging', 'container', 'maintenance'],
        created: '2024-02-10',
        modified: '2024-12-18',
        uptime: 95,
        sla: 95,
        dependencies: ['staging-db', 'staging-cache'],
        alerts: 2,
        backups: { enabled: true, lastBackup: '12 hours ago', nextBackup: 'after maintenance', retention: 7 }
      }
    ];

    // Group resources
    const groups: ResourceGroup[] = [
      {
        id: 'GRP-001',
        name: 'Production Environment',
        environment: 'production',
        resources: mockResources.filter(r => r.tags.includes('production')),
        totalCost: 5580,
        healthScore: 92,
        complianceScore: 98,
        securityScore: 95
      },
      {
        id: 'GRP-002',
        name: 'Staging Environment',
        environment: 'staging',
        resources: mockResources.filter(r => r.tags.includes('staging')),
        totalCost: 800,
        healthScore: 75,
        complianceScore: 85,
        securityScore: 88
      },
      {
        id: 'GRP-003',
        name: 'Development Environment',
        environment: 'development',
        resources: mockResources.filter(r => r.tags.includes('development')),
        totalCost: 200,
        healthScore: 100,
        complianceScore: 70,
        securityScore: 75
      }
    ];

    setResourceGroups(groups);

    // Initialize metrics
    setMetrics([
      { id: 'M1', title: 'Total Resources', value: mockResources.length, change: 5, trend: 'up', category: 'inventory' },
      { id: 'M2', title: 'Running VMs', value: mockResources.filter(r => r.type === 'vm' && r.status === 'running').length, change: 0, trend: 'stable', category: 'compute' },
      { id: 'M3', title: 'Total Cost', value: '$6,580', unit: '/month', change: -8, trend: 'down', category: 'cost' },
      { id: 'M4', title: 'Avg CPU Usage', value: 52, unit: '%', change: 12, trend: 'up', category: 'performance', severity: 'normal' },
      { id: 'M5', title: 'Storage Used', value: '6.8', unit: 'TB', change: 18, trend: 'up', category: 'storage', severity: 'warning' },
      { id: 'M6', title: 'Network Traffic', value: '12.4', unit: 'Gbps', change: 25, trend: 'up', category: 'network' },
      { id: 'M7', title: 'Active Alerts', value: 3, change: 50, trend: 'up', category: 'monitoring', severity: 'warning' },
      { id: 'M8', title: 'Backup Health', value: 98, unit: '%', change: 2, trend: 'up', category: 'backup' }
    ]);
  }, []);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'running': return 'text-green-500 bg-green-900/20';
      case 'stopped': return 'text-gray-500 bg-gray-900/20';
      case 'pending': return 'text-yellow-500 bg-yellow-900/20';
      case 'error': return 'text-red-500 bg-red-900/20';
      case 'maintenance': return 'text-blue-500 bg-blue-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getResourceIcon = (type: string) => {
    switch(type) {
      case 'vm': return <Server className="w-4 h-4" />;
      case 'container': return <Container className="w-4 h-4" />;
      case 'database': return <Database className="w-4 h-4" />;
      case 'storage': return <HardDrive className="w-4 h-4" />;
      case 'network': return <Network className="w-4 h-4" />;
      case 'loadbalancer': return <Layers className="w-4 h-4" />;
      case 'cdn': return <Globe className="w-4 h-4" />;
      case 'dns': return <Wifi className="w-4 h-4" />;
      default: return <Box className="w-4 h-4" />;
    }
  };

  const getProviderColor = (provider: string) => {
    switch(provider) {
      case 'azure': return 'text-blue-500';
      case 'aws': return 'text-orange-500';
      case 'gcp': return 'text-green-500';
      case 'onprem': return 'text-purple-500';
      default: return 'text-gray-500';
    }
  };

  const toggleGroupExpansion = (groupId: string) => {
    setExpandedGroups(prev => 
      prev.includes(groupId) 
        ? prev.filter(id => id !== groupId)
        : [...prev, groupId]
    );
  };

  const filteredGroups = resourceGroups.map(group => ({
    ...group,
    resources: group.resources.filter(resource => {
      if (filterType !== 'all' && resource.type !== filterType) return false;
      if (filterStatus !== 'all' && resource.status !== filterStatus) return false;
      if (filterProvider !== 'all' && resource.provider !== filterProvider) return false;
      return true;
    })
  }));

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Infrastructure Management</h1>
            <p className="text-sm text-gray-400 mt-1">Manage cloud and on-premise infrastructure resources</p>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* View Mode Toggle */}
            <div className="flex items-center bg-gray-800 rounded">
              <button
                onClick={() => setViewMode('grid')}
                className={`px-3 py-2 text-sm ${viewMode === 'grid' ? 'bg-gray-700' : ''} rounded-l`}
              >
                Grid
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`px-3 py-2 text-sm ${viewMode === 'list' ? 'bg-gray-700' : ''}`}
              >
                List
              </button>
              <button
                onClick={() => setViewMode('tree')}
                className={`px-3 py-2 text-sm ${viewMode === 'tree' ? 'bg-gray-700' : ''}`}
              >
                Tree
              </button>
              <button
                onClick={() => setViewMode('map')}
                className={`px-3 py-2 text-sm ${viewMode === 'map' ? 'bg-gray-700' : ''} rounded-r`}
              >
                Map
              </button>
            </div>

            <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>

            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Server className="w-4 h-4" />
              <span>Deploy Resource</span>
            </button>
          </div>
        </div>
      </header>

      {/* Metrics Bar */}
      {showMetrics && (
        <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
          <div className="flex items-center space-x-6 overflow-x-auto">
            {metrics.map(metric => (
              <div key={metric.id} className="flex items-center space-x-2 min-w-fit">
                <div className="text-xs">
                  <span className="text-gray-500">{metric.title}:</span>
                  <span className={`ml-2 font-bold ${
                    metric.severity === 'critical' ? 'text-red-500' :
                    metric.severity === 'warning' ? 'text-yellow-500' :
                    'text-white'
                  }`}>
                    {metric.value}{metric.unit}
                  </span>
                  <span className={`ml-1 text-xs ${
                    metric.trend === 'up' ? 'text-green-500' :
                    metric.trend === 'down' ? 'text-red-500' :
                    'text-gray-500'
                  }`}>
                    {metric.trend === 'up' ? '↑' : metric.trend === 'down' ? '↓' : '→'}
                    {Math.abs(metric.change)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <div className="flex h-full">
          {/* Sidebar Filters */}
          <div className="w-64 bg-gray-900/50 border-r border-gray-800 p-4 overflow-y-auto">
            <h3 className="text-sm font-bold mb-4">Filters</h3>
            
            <div className="space-y-4">
              {/* Type Filter */}
              <div>
                <label className="text-xs text-gray-500 uppercase mb-2 block">Resource Type</label>
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                >
                  <option value="all">All Types</option>
                  <option value="vm">Virtual Machines</option>
                  <option value="container">Containers</option>
                  <option value="database">Databases</option>
                  <option value="storage">Storage</option>
                  <option value="network">Network</option>
                  <option value="loadbalancer">Load Balancers</option>
                  <option value="cdn">CDN</option>
                  <option value="dns">DNS</option>
                </select>
              </div>

              {/* Status Filter */}
              <div>
                <label className="text-xs text-gray-500 uppercase mb-2 block">Status</label>
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                >
                  <option value="all">All Status</option>
                  <option value="running">Running</option>
                  <option value="stopped">Stopped</option>
                  <option value="pending">Pending</option>
                  <option value="error">Error</option>
                  <option value="maintenance">Maintenance</option>
                </select>
              </div>

              {/* Provider Filter */}
              <div>
                <label className="text-xs text-gray-500 uppercase mb-2 block">Provider</label>
                <select
                  value={filterProvider}
                  onChange={(e) => setFilterProvider(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                >
                  <option value="all">All Providers</option>
                  <option value="azure">Azure</option>
                  <option value="aws">AWS</option>
                  <option value="gcp">GCP</option>
                  <option value="onprem">On-Premise</option>
                </select>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="mt-6 pt-6 border-t border-gray-800">
              <h3 className="text-sm font-bold mb-4">Quick Actions</h3>
              <div className="space-y-2">
                <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center space-x-2">
                  <Power className="w-4 h-4" />
                  <span>Start All</span>
                </button>
                <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center space-x-2">
                  <PowerOff className="w-4 h-4" />
                  <span>Stop Non-Critical</span>
                </button>
                <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center space-x-2">
                  <Archive className="w-4 h-4" />
                  <span>Backup All</span>
                </button>
                <button className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left flex items-center space-x-2">
                  <RotateCw className="w-4 h-4" />
                  <span>Sync Resources</span>
                </button>
              </div>
            </div>
          </div>

          {/* Resource Display */}
          <div className="flex-1 p-6 overflow-y-auto">
            {viewMode === 'tree' && (
              <div className="space-y-4">
                {filteredGroups.map(group => (
                  <div key={group.id} className="bg-gray-900 border border-gray-800 rounded-lg">
                    <button
                      onClick={() => toggleGroupExpansion(group.id)}
                      className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-800/50"
                    >
                      <div className="flex items-center space-x-3">
                        {expandedGroups.includes(group.id) ? (
                          <ChevronDown className="w-4 h-4" />
                        ) : (
                          <ChevronRight className="w-4 h-4" />
                        )}
                        <Folder className="w-5 h-5 text-yellow-500" />
                        <span className="font-bold">{group.name}</span>
                        <span className="px-2 py-1 bg-gray-800 rounded text-xs">
                          {group.resources.length} resources
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 text-xs">
                        <div className="flex items-center space-x-1">
                          <Activity className="w-3 h-3 text-green-500" />
                          <span>Health: {group.healthScore}%</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Shield className="w-3 h-3 text-blue-500" />
                          <span>Security: {group.securityScore}%</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <TrendingUp className="w-3 h-3 text-purple-500" />
                          <span>Cost: ${group.totalCost}/mo</span>
                        </div>
                      </div>
                    </button>
                    
                    {expandedGroups.includes(group.id) && (
                      <div className="border-t border-gray-800">
                        {group.resources.map(resource => (
                          <div
                            key={resource.id}
                            className="px-8 py-3 flex items-center justify-between hover:bg-gray-800/30 border-b border-gray-800/50 last:border-b-0"
                          >
                            <div className="flex items-center space-x-3">
                              <div className={`p-2 rounded ${getStatusColor(resource.status)}`}>
                                {getResourceIcon(resource.type)}
                              </div>
                              <div>
                                <div className="flex items-center space-x-2">
                                  <span className="text-sm font-medium">{resource.name}</span>
                                  <span className={`text-xs ${getProviderColor(resource.provider)}`}>
                                    {resource.provider}
                                  </span>
                                </div>
                                <div className="flex items-center space-x-3 text-xs text-gray-500 mt-1">
                                  <span>{resource.region}</span>
                                  {resource.zone && <span>• {resource.zone}</span>}
                                  <span>• ${resource.cost.monthly}/mo</span>
                                  {resource.alerts > 0 && (
                                    <span className="text-yellow-500">• {resource.alerts} alerts</span>
                                  )}
                                </div>
                              </div>
                            </div>
                            
                            <div className="flex items-center space-x-4">
                              {resource.utilization.cpu !== undefined && (
                                <div className="text-xs">
                                  <span className="text-gray-500">CPU:</span>
                                  <span className={`ml-1 ${
                                    resource.utilization.cpu > 80 ? 'text-red-500' :
                                    resource.utilization.cpu > 60 ? 'text-yellow-500' :
                                    'text-green-500'
                                  }`}>
                                    {resource.utilization.cpu}%
                                  </span>
                                </div>
                              )}
                              {resource.utilization.memory !== undefined && (
                                <div className="text-xs">
                                  <span className="text-gray-500">Mem:</span>
                                  <span className={`ml-1 ${
                                    resource.utilization.memory > 80 ? 'text-red-500' :
                                    resource.utilization.memory > 60 ? 'text-yellow-500' :
                                    'text-green-500'
                                  }`}>
                                    {resource.utilization.memory}%
                                  </span>
                                </div>
                              )}
                              <button className="p-1 hover:bg-gray-800 rounded">
                                <MoreVertical className="w-4 h-4 text-gray-500" />
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {viewMode === 'grid' && (
              <div className="space-y-6">
                {filteredGroups.map(group => (
                  <div key={group.id}>
                    <h2 className="text-sm font-bold mb-3 flex items-center space-x-2">
                      <Folder className="w-4 h-4 text-yellow-500" />
                      <span>{group.name}</span>
                      <span className="text-xs text-gray-500">({group.resources.length})</span>
                    </h2>
                    <div className="grid grid-cols-3 gap-4">
                      {group.resources.map(resource => (
                        <div key={resource.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                          <div className="flex items-start justify-between mb-3">
                            <div className="flex items-center space-x-2">
                              <div className={`p-2 rounded ${getStatusColor(resource.status)}`}>
                                {getResourceIcon(resource.type)}
                              </div>
                              <div>
                                <h3 className="text-sm font-bold">{resource.name}</h3>
                                <p className="text-xs text-gray-500">{resource.type}</p>
                              </div>
                            </div>
                            <span className={`text-xs ${getProviderColor(resource.provider)}`}>
                              {resource.provider}
                            </span>
                          </div>
                          
                          <div className="space-y-2 text-xs">
                            <div className="flex justify-between">
                              <span className="text-gray-500">Region</span>
                              <span>{resource.region}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Cost</span>
                              <span>${resource.cost.monthly}/mo</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Uptime</span>
                              <span className="text-green-500">{resource.uptime}%</span>
                            </div>
                            {resource.backups.enabled && (
                              <div className="flex justify-between">
                                <span className="text-gray-500">Last Backup</span>
                                <span>{resource.backups.lastBackup}</span>
                              </div>
                            )}
                          </div>
                          
                          <div className="mt-3 pt-3 border-t border-gray-800 flex justify-between">
                            <button className="text-xs text-blue-500 hover:text-blue-400">
                              Details
                            </button>
                            <button className="text-xs text-gray-500 hover:text-gray-400">
                              Actions
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {viewMode === 'list' && (
              <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
                <table className="w-full">
                  <thead>
                    <tr className="bg-gray-800">
                      <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Resource</th>
                      <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Type</th>
                      <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Status</th>
                      <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Provider</th>
                      <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Region</th>
                      <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Cost</th>
                      <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">CPU</th>
                      <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Memory</th>
                      <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredGroups.flatMap(group => group.resources).map(resource => (
                      <tr key={resource.id} className="border-t border-gray-800 hover:bg-gray-800/30">
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-2">
                            {getResourceIcon(resource.type)}
                            <span className="text-sm">{resource.name}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm">{resource.type}</td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 rounded text-xs ${getStatusColor(resource.status)}`}>
                            {resource.status}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`text-sm ${getProviderColor(resource.provider)}`}>
                            {resource.provider}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm">{resource.region}</td>
                        <td className="px-4 py-3 text-sm">${resource.cost.monthly}/mo</td>
                        <td className="px-4 py-3 text-sm">
                          {resource.utilization.cpu !== undefined ? `${resource.utilization.cpu}%` : '-'}
                        </td>
                        <td className="px-4 py-3 text-sm">
                          {resource.utilization.memory !== undefined ? `${resource.utilization.memory}%` : '-'}
                        </td>
                        <td className="px-4 py-3">
                          <button className="p-1 hover:bg-gray-800 rounded">
                            <MoreVertical className="w-4 h-4 text-gray-500" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {viewMode === 'map' && (
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-8">
                <div className="text-center">
                  <Globe className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-lg font-bold mb-2">Geographic View</h3>
                  <p className="text-sm text-gray-500 mb-6">Interactive map showing resource distribution across regions</p>
                  
                  {/* Region Summary */}
                  <div className="grid grid-cols-3 gap-4 max-w-2xl mx-auto">
                    <div className="bg-gray-800 rounded-lg p-4">
                      <h4 className="text-sm font-bold mb-2">East US</h4>
                      <p className="text-2xl font-bold text-blue-500">5</p>
                      <p className="text-xs text-gray-500">resources</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4">
                      <h4 className="text-sm font-bold mb-2">West US</h4>
                      <p className="text-2xl font-bold text-green-500">1</p>
                      <p className="text-xs text-gray-500">resources</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4">
                      <h4 className="text-sm font-bold mb-2">Central US</h4>
                      <p className="text-2xl font-bold text-purple-500">1</p>
                      <p className="text-xs text-gray-500">resources</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}