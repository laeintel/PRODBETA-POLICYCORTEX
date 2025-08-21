'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { toast } from '@/hooks/useToast';
import {
  Server,
  Database,
  Cloud,
  HardDrive,
  Network,
  Shield,
  Activity,
  DollarSign,
  Settings,
  Play,
  Pause,
  RotateCw,
  Trash2,
  MoreVertical,
  CheckCircle,
  AlertCircle,
  XCircle,
  TrendingUp,
  TrendingDown,
  Tag,
  Layers,
  Cpu,
  MemoryStick,
  Gauge,
  Zap
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
  Treemap
} from 'recharts';

interface Resource {
  id: string;
  name: string;
  type: 'vm' | 'database' | 'storage' | 'network' | 'container' | 'function';
  status: 'running' | 'stopped' | 'warning' | 'error';
  region: string;
  resourceGroup: string;
  subscription: string;
  tags: Record<string, string>;
  metrics: {
    cpu: number;
    memory: number;
    disk: number;
    network: number;
  };
  cost: {
    daily: number;
    monthly: number;
    trend: 'up' | 'down' | 'stable';
  };
  health: {
    score: number;
    issues: number;
    recommendations: string[];
  };
  dependencies: string[];
  lastModified: string;
  createdDate: string;
}

export default function ResourceManagementPage() {
  const router = useRouter();
  const [resources, setResources] = useState<Resource[]>([]);
  const [selectedResource, setSelectedResource] = useState<Resource | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState({
    type: 'all',
    status: 'all',
    resourceGroup: 'all',
    search: ''
  });
  const [utilizationData, setUtilizationData] = useState<any[]>([]);
  const [costTrend, setCostTrend] = useState<any[]>([]);

  useEffect(() => {
    fetchResourceData();
    const interval = setInterval(fetchResourceData, 30000);
    return () => clearInterval(interval);
  }, [filter]);

  const fetchResourceData = async () => {
    try {
      setLoading(true);

      // Mock data - replace with actual Azure API calls
      const mockResources: Resource[] = [
        {
          id: 'res-1',
          name: 'pcx-web-server-01',
          type: 'vm',
          status: 'running',
          region: 'East US',
          resourceGroup: 'rg-production',
          subscription: 'Production',
          tags: { environment: 'prod', owner: 'devops', project: 'policycortex' },
          metrics: { cpu: 45, memory: 62, disk: 38, network: 25 },
          cost: { daily: 12.50, monthly: 375, trend: 'stable' },
          health: { 
            score: 95, 
            issues: 1,
            recommendations: ['Enable backup', 'Update OS patches']
          },
          dependencies: ['res-2', 'res-3'],
          lastModified: new Date(Date.now() - 86400000).toISOString(),
          createdDate: new Date(Date.now() - 2592000000).toISOString()
        },
        {
          id: 'res-2',
          name: 'pcx-sql-database',
          type: 'database',
          status: 'running',
          region: 'East US',
          resourceGroup: 'rg-production',
          subscription: 'Production',
          tags: { environment: 'prod', owner: 'data-team', critical: 'true' },
          metrics: { cpu: 78, memory: 85, disk: 92, network: 45 },
          cost: { daily: 45.00, monthly: 1350, trend: 'up' },
          health: { 
            score: 88, 
            issues: 2,
            recommendations: ['Optimize queries', 'Increase DTU', 'Enable auto-tuning']
          },
          dependencies: [],
          lastModified: new Date(Date.now() - 3600000).toISOString(),
          createdDate: new Date(Date.now() - 5184000000).toISOString()
        },
        {
          id: 'res-3',
          name: 'pcx-storage-account',
          type: 'storage',
          status: 'running',
          region: 'East US',
          resourceGroup: 'rg-production',
          subscription: 'Production',
          tags: { environment: 'prod', owner: 'devops', tier: 'hot' },
          metrics: { cpu: 0, memory: 0, disk: 67, network: 32 },
          cost: { daily: 8.25, monthly: 247.50, trend: 'up' },
          health: { 
            score: 92, 
            issues: 0,
            recommendations: ['Enable lifecycle management']
          },
          dependencies: [],
          lastModified: new Date(Date.now() - 7200000).toISOString(),
          createdDate: new Date(Date.now() - 7776000000).toISOString()
        },
        {
          id: 'res-4',
          name: 'pcx-container-registry',
          type: 'container',
          status: 'running',
          region: 'East US',
          resourceGroup: 'rg-containers',
          subscription: 'Production',
          tags: { environment: 'prod', owner: 'platform-team' },
          metrics: { cpu: 12, memory: 28, disk: 45, network: 18 },
          cost: { daily: 5.00, monthly: 150, trend: 'stable' },
          health: { 
            score: 98, 
            issues: 0,
            recommendations: []
          },
          dependencies: ['res-3'],
          lastModified: new Date(Date.now() - 1800000).toISOString(),
          createdDate: new Date(Date.now() - 10368000000).toISOString()
        },
        {
          id: 'res-5',
          name: 'pcx-app-gateway',
          type: 'network',
          status: 'running',
          region: 'East US',
          resourceGroup: 'rg-network',
          subscription: 'Production',
          tags: { environment: 'prod', owner: 'network-team' },
          metrics: { cpu: 25, memory: 35, disk: 0, network: 88 },
          cost: { daily: 15.00, monthly: 450, trend: 'down' },
          health: { 
            score: 96, 
            issues: 0,
            recommendations: ['Review firewall rules']
          },
          dependencies: ['res-1'],
          lastModified: new Date(Date.now() - 10800000).toISOString(),
          createdDate: new Date(Date.now() - 15552000000).toISOString()
        },
        {
          id: 'res-6',
          name: 'pcx-function-app',
          type: 'function',
          status: 'warning',
          region: 'East US',
          resourceGroup: 'rg-serverless',
          subscription: 'Production',
          tags: { environment: 'prod', owner: 'api-team', runtime: 'nodejs' },
          metrics: { cpu: 65, memory: 72, disk: 12, network: 42 },
          cost: { daily: 3.75, monthly: 112.50, trend: 'up' },
          health: { 
            score: 75, 
            issues: 3,
            recommendations: ['Scale up plan', 'Fix timeout errors', 'Update runtime']
          },
          dependencies: ['res-2', 'res-3'],
          lastModified: new Date(Date.now() - 600000).toISOString(),
          createdDate: new Date(Date.now() - 20736000000).toISOString()
        }
      ];

      const mockUtilization = [
        { time: '00:00', cpu: 32, memory: 45, disk: 62, network: 28 },
        { time: '04:00', cpu: 28, memory: 42, disk: 62, network: 22 },
        { time: '08:00', cpu: 45, memory: 58, disk: 63, network: 45 },
        { time: '12:00', cpu: 72, memory: 78, disk: 65, network: 68 },
        { time: '16:00', cpu: 68, memory: 72, disk: 66, network: 62 },
        { time: '20:00', cpu: 52, memory: 55, disk: 67, network: 48 },
        { time: '23:59', cpu: 38, memory: 48, disk: 67, network: 32 }
      ];

      const mockCostTrend = [
        { month: 'Jan', actual: 12500, forecast: 12000 },
        { month: 'Feb', actual: 13200, forecast: 12500 },
        { month: 'Mar', actual: 12800, forecast: 13000 },
        { month: 'Apr', actual: 13500, forecast: 13200 },
        { month: 'May', actual: 14200, forecast: 13800 },
        { month: 'Jun', actual: 13900, forecast: 14000 }
      ];

      setResources(mockResources);
      setUtilizationData(mockUtilization);
      setCostTrend(mockCostTrend);
    } catch (error) {
      console.error('Error fetching resource data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getResourceIcon = (type: string) => {
    switch (type) {
      case 'vm': return <Server className="w-5 h-5" />;
      case 'database': return <Database className="w-5 h-5" />;
      case 'storage': return <HardDrive className="w-5 h-5" />;
      case 'network': return <Network className="w-5 h-5" />;
      case 'container': return <Layers className="w-5 h-5" />;
      case 'function': return <Zap className="w-5 h-5" />;
      default: return <Cloud className="w-5 h-5" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-green-400';
      case 'stopped': return 'text-gray-400';
      case 'warning': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <CheckCircle className="w-4 h-4" />;
      case 'stopped': return <XCircle className="w-4 h-4" />;
      case 'warning': return <AlertCircle className="w-4 h-4" />;
      case 'error': return <XCircle className="w-4 h-4" />;
      default: return null;
    }
  };

  const handleQuickAction = (action: string, resourceId: string) => {
    console.log(`Executing ${action} on resource ${resourceId}`);
    // Implement actual actions here
  };

  const filteredResources = resources.filter(resource => {
    if (filter.type !== 'all' && resource.type !== filter.type) return false;
    if (filter.status !== 'all' && resource.status !== filter.status) return false;
    if (filter.resourceGroup !== 'all' && resource.resourceGroup !== filter.resourceGroup) return false;
    if (filter.search && !resource.name.toLowerCase().includes(filter.search.toLowerCase())) return false;
    return true;
  });

  const totalCost = resources.reduce((sum, r) => sum + r.cost.monthly, 0);
  const avgHealth = resources.length > 0 ? resources.reduce((sum, r) => sum + r.health.score, 0) / resources.length : 0;
  const totalIssues = resources.reduce((sum, r) => sum + r.health.issues, 0);

  const costByType = [
    { name: 'VMs', value: 375, color: '#3b82f6' },
    { name: 'Databases', value: 1350, color: '#10b981' },
    { name: 'Storage', value: 247.50, color: '#f59e0b' },
    { name: 'Network', value: 450, color: '#8b5cf6' },
    { name: 'Containers', value: 150, color: '#ef4444' },
    { name: 'Functions', value: 112.50, color: '#06b6d4' }
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-400">Loading resource inventory...</p>
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
                <Server className="w-8 h-8 text-blue-500" />
                Operations Resources
              </h1>
              <p className="text-gray-400 mt-2">Azure resource inventory and management</p>
            </div>
            <div className="flex items-center gap-4">
              <button type="button" 
                onClick={fetchResourceData}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors">
                <Activity className="w-4 h-4" />
                Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-between mb-4">
              <Server className="w-8 h-8 text-blue-500" />
              <span className="text-2xl font-bold text-white">{resources.length}</span>
            </div>
            <h3 className="text-gray-400 text-sm">Total Resources</h3>
            <p className="text-xs text-gray-500 mt-1">Across all subscriptions</p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-between mb-4">
              <DollarSign className="w-8 h-8 text-green-500" />
              <span className="text-2xl font-bold text-white">${totalCost.toLocaleString()}</span>
            </div>
            <h3 className="text-gray-400 text-sm">Monthly Cost</h3>
            <p className="text-xs text-gray-500 mt-1">Estimated spend</p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-between mb-4">
              <Shield className="w-8 h-8 text-purple-500" />
              <span className="text-2xl font-bold text-white">{avgHealth.toFixed(0)}%</span>
            </div>
            <h3 className="text-gray-400 text-sm">Health Score</h3>
            <p className="text-xs text-gray-500 mt-1">Average across resources</p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-between mb-4">
              <AlertCircle className="w-8 h-8 text-orange-500" />
              <span className="text-2xl font-bold text-white">{totalIssues}</span>
            </div>
            <h3 className="text-gray-400 text-sm">Active Issues</h3>
            <p className="text-xs text-gray-500 mt-1">Requires attention</p>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-gray-900 rounded-xl p-4 border border-gray-800 mb-6">
          <div className="flex flex-wrap items-center gap-4">
            <input
              type="text"
              placeholder="Search resources..."
              value={filter.search}
              onChange={(e) => setFilter({ ...filter, search: e.target.value })}
              className="bg-gray-800 text-white px-4 py-2 rounded-lg border border-gray-700 focus:border-blue-500 focus:outline-none flex-1 min-w-[200px]"
            />
            <select
              value={filter.type}
              onChange={(e) => setFilter({ ...filter, type: e.target.value })}
              className="bg-gray-800 text-white px-4 py-2 rounded-lg border border-gray-700 focus:border-blue-500 focus:outline-none"
            >
              <option value="all">All Types</option>
              <option value="vm">Virtual Machines</option>
              <option value="database">Databases</option>
              <option value="storage">Storage</option>
              <option value="network">Network</option>
              <option value="container">Containers</option>
              <option value="function">Functions</option>
            </select>
            <select
              value={filter.status}
              onChange={(e) => setFilter({ ...filter, status: e.target.value })}
              className="bg-gray-800 text-white px-4 py-2 rounded-lg border border-gray-700 focus:border-blue-500 focus:outline-none"
            >
              <option value="all">All Status</option>
              <option value="running">Running</option>
              <option value="stopped">Stopped</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
            </select>
          </div>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Resource Utilization */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <Gauge className="w-5 h-5 text-blue-500" />
              Resource Utilization
            </h2>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={utilizationData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  labelStyle={{ color: '#f3f4f6' }}
                />
                <Area type="monotone" dataKey="cpu" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                <Area type="monotone" dataKey="memory" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                <Area type="monotone" dataKey="disk" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.3} />
                <Area type="monotone" dataKey="network" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
                <Legend />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Cost Distribution */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <DollarSign className="w-5 h-5 text-green-500" />
              Cost Distribution
            </h2>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={costByType}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {costByType.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  formatter={(value: any) => `$${value.toLocaleString()}`}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-2 gap-2 mt-4">
              {costByType.map((item, index) => (
                <div key={index} className="flex items-center gap-2 text-xs">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: item.color }}></div>
                  <span className="text-gray-400">{item.name}:</span>
                  <span className="text-white">${item.value.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Cost Trend */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-purple-500" />
              Cost Trend & Forecast
            </h2>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={costTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="month" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  labelStyle={{ color: '#f3f4f6' }}
                  formatter={(value: any) => `$${value.toLocaleString()}`}
                />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6' }} />
                <Line type="monotone" dataKey="forecast" stroke="#10b981" strokeWidth={2} strokeDasharray="5 5" dot={{ fill: '#10b981' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Resource List */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <h2 className="text-xl font-semibold text-white mb-4">Resource Inventory</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gray-800">
                  <th className="pb-3">Resource</th>
                  <th className="pb-3">Type</th>
                  <th className="pb-3">Status</th>
                  <th className="pb-3">Region</th>
                  <th className="pb-3">Utilization</th>
                  <th className="pb-3">Cost/Month</th>
                  <th className="pb-3">Health</th>
                  <th className="pb-3">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredResources.map((resource) => (
                  <tr key={resource.id} className="border-b border-gray-800 hover:bg-gray-800/50">
                    <td className="py-4">
                      <div className="flex items-center gap-3">
                        <div className="text-gray-400">{getResourceIcon(resource.type)}</div>
                        <div>
                          <p className="text-white font-medium">{resource.name}</p>
                          <p className="text-xs text-gray-500">{resource.resourceGroup}</p>
                        </div>
                      </div>
                    </td>
                    <td className="py-4">
                      <span className="text-gray-300 capitalize">{resource.type}</span>
                    </td>
                    <td className="py-4">
                      <div className={`flex items-center gap-2 ${getStatusColor(resource.status)}`}>
                        {getStatusIcon(resource.status)}
                        <span className="capitalize">{resource.status}</span>
                      </div>
                    </td>
                    <td className="py-4 text-gray-300">{resource.region}</td>
                    <td className="py-4">
                      <div className="flex items-center gap-4 text-xs">
                        <div className="flex items-center gap-1">
                          <Cpu className="w-3 h-3 text-gray-400" />
                          <span className="text-gray-300">{resource.metrics.cpu}%</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <MemoryStick className="w-3 h-3 text-gray-400" />
                          <span className="text-gray-300">{resource.metrics.memory}%</span>
                        </div>
                      </div>
                    </td>
                    <td className="py-4">
                      <div className="flex items-center gap-2">
                        <span className="text-white font-medium">${resource.cost.monthly}</span>
                        {resource.cost.trend === 'up' && <TrendingUp className="w-3 h-3 text-red-400" />}
                        {resource.cost.trend === 'down' && <TrendingDown className="w-3 h-3 text-green-400" />}
                      </div>
                    </td>
                    <td className="py-4">
                      <div>
                        <div className={`text-sm font-medium ${
                          resource.health.score >= 90 ? 'text-green-400' :
                          resource.health.score >= 70 ? 'text-yellow-400' :
                          'text-red-400'
                        }`}>
                          {resource.health.score}%
                        </div>
                        {resource.health.issues > 0 && (
                          <p className="text-xs text-orange-400">{resource.health.issues} issues</p>
                        )}
                      </div>
                    </td>
                    <td className="py-4">
                      <div className="flex items-center gap-2">
                        {resource.status === 'running' ? (
                          <button type="button"
                            onClick={() => handleQuickAction('stop', resource.id)}
                            className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white transition-colors"
                            title="Stop"
                          >
                            <Pause className="w-4 h-4" />
                          </button>
                        ) : (
                          <button type="button"
                            onClick={() => handleQuickAction('start', resource.id)}
                            className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white transition-colors"
                            title="Start"
                          >
                            <Play className="w-4 h-4" />
                          </button>
                        )}
                        <button type="button"
                          onClick={() => handleQuickAction('restart', resource.id)}
                          className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white transition-colors"
                          title="Restart"
                        >
                          <RotateCw className="w-4 h-4" />
                        </button>
                        <button type="button"
                          onClick={() => setSelectedResource(resource)}
                          className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white transition-colors"
                          title="Details"
                        >
                          <MoreVertical className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Resource Details Modal */}
        {selectedResource && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-white flex items-center gap-2">
                  {getResourceIcon(selectedResource.type)}
                  {selectedResource.name}
                </h2>
                <button type="button"
                  onClick={() => setSelectedResource(null)}
                  className="text-gray-400 hover:text-white"
                >
                  <XCircle className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-gray-400 text-sm">Resource Group</p>
                    <p className="text-white">{selectedResource.resourceGroup}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">Subscription</p>
                    <p className="text-white">{selectedResource.subscription}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">Region</p>
                    <p className="text-white">{selectedResource.region}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">Created</p>
                    <p className="text-white">{new Date(selectedResource.createdDate).toLocaleDateString()}</p>
                  </div>
                </div>

                <div>
                  <p className="text-gray-400 text-sm mb-2">Tags</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(selectedResource.tags).map(([key, value]) => (
                      <span key={key} className="bg-gray-800 px-2 py-1 rounded text-xs text-gray-300">
                        {key}: {value}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-gray-400 text-sm mb-2">Recommendations</p>
                  <ul className="space-y-1">
                    {selectedResource.health.recommendations.map((rec, index) => (
                      <li key={index} className="text-yellow-400 text-sm flex items-center gap-2">
                        <AlertCircle className="w-3 h-3" />
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="flex gap-3 pt-4 border-t border-gray-800">
                  <button type="button" 
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors" 
                    onClick={() => {
                      if (selectedResource) {
                        router.push(`/operations/resources/${selectedResource.id}/configure`);
                      }
                    }}>
                    <Settings className="w-4 h-4" />
                    Configure
                  </button>
                  <button type="button" 
                    className="bg-gray-800 hover:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors" 
                    onClick={() => {
                      if (selectedResource) {
                        router.push(`/operations/resources/${selectedResource.id}/tags`);
                      }
                    }}>
                    <Tag className="w-4 h-4" />
                    Edit Tags
                  </button>
                  <button type="button" 
                    className="bg-red-600/20 hover:bg-red-600/30 text-red-400 px-4 py-2 rounded-lg flex items-center gap-2 transition-colors" 
                    onClick={() => {
                      if (selectedResource) {
                        if (confirm(`Are you sure you want to delete ${selectedResource.name}?`)) {
                          toast({ 
                            title: 'Resource deletion', 
                            description: `Initiating deletion of ${selectedResource.name}...` 
                          });
                          // In production, this would call the API
                          setTimeout(() => {
                            setResources(prev => prev.filter(r => r.id !== selectedResource.id));
                            setSelectedResource(null);
                            toast({ 
                              title: 'Success', 
                              description: 'Resource deleted successfully' 
                            });
                          }, 1000);
                        }
                      }
                    }}>
                    <Trash2 className="w-4 h-4" />
                    Delete
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}