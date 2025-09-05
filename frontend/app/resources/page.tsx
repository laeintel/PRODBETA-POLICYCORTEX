'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Server,
  Database,
  HardDrive,
  Network,
  Cloud,
  Shield,
  Lock,
  Key,
  Cpu,
  Activity,
  Globe,
  Folder,
  FileText,
  Settings,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Info,
  Search,
  Filter,
  Download,
  RefreshCw,
  Plus,
  Trash2,
  Edit,
  Eye,
  Power,
  PauseCircle,
  PlayCircle,
  Tag,
  DollarSign,
  Clock,
  TrendingUp,
  TrendingDown,
  MoreVertical
} from 'lucide-react';

interface Resource {
  id: string;
  name: string;
  type: 'vm' | 'database' | 'storage' | 'network' | 'security' | 'app-service';
  status: 'running' | 'stopped' | 'warning' | 'error';
  region: string;
  resourceGroup: string;
  subscription: string;
  tags: { [key: string]: string };
  created: Date;
  modified: Date;
  cost: number;
  costTrend: 'up' | 'down' | 'stable';
  compliance: 'compliant' | 'non-compliant' | 'partial';
  performance: {
    cpu?: number;
    memory?: number;
    storage?: number;
    network?: number;
  };
  policies: number;
  alerts: number;
}

interface ResourceGroup {
  id: string;
  name: string;
  subscription: string;
  region: string;
  resourceCount: number;
  totalCost: number;
  complianceStatus: 'compliant' | 'non-compliant' | 'partial';
}

interface ResourceMetrics {
  totalResources: number;
  runningResources: number;
  stoppedResources: number;
  totalCost: number;
  complianceRate: number;
  criticalAlerts: number;
}

export default function ResourcesManagement() {
  const [loading, setLoading] = useState(true);
  const [resources, setResources] = useState<Resource[]>([]);
  const [resourceGroups, setResourceGroups] = useState<ResourceGroup[]>([]);
  const [metrics, setMetrics] = useState<ResourceMetrics | null>(null);
  const [selectedView, setSelectedView] = useState<'list' | 'grid'>('list');
  const [selectedType, setSelectedType] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedResource, setSelectedResource] = useState<Resource | null>(null);

  useEffect(() => {
    // Simulate loading data
    setTimeout(() => {
      // Mock resources
      setResources([
        {
          id: 'res-001',
          name: 'prod-web-server-01',
          type: 'vm',
          status: 'running',
          region: 'East US',
          resourceGroup: 'production-rg',
          subscription: 'Enterprise Production',
          tags: { environment: 'production', owner: 'web-team', project: 'main-app' },
          created: new Date('2023-06-15'),
          modified: new Date('2024-01-08'),
          cost: 450,
          costTrend: 'up',
          compliance: 'compliant',
          performance: { cpu: 65, memory: 72, storage: 45, network: 30 },
          policies: 12,
          alerts: 0
        },
        {
          id: 'res-002',
          name: 'analytics-db-primary',
          type: 'database',
          status: 'running',
          region: 'East US',
          resourceGroup: 'analytics-rg',
          subscription: 'Enterprise Production',
          tags: { environment: 'production', owner: 'data-team', criticality: 'high' },
          created: new Date('2023-03-20'),
          modified: new Date('2024-01-09'),
          cost: 1200,
          costTrend: 'stable',
          compliance: 'compliant',
          performance: { cpu: 45, memory: 60, storage: 78 },
          policies: 15,
          alerts: 1
        },
        {
          id: 'res-003',
          name: 'backup-storage-account',
          type: 'storage',
          status: 'running',
          region: 'West US',
          resourceGroup: 'backup-rg',
          subscription: 'Enterprise Production',
          tags: { environment: 'production', owner: 'ops-team', type: 'backup' },
          created: new Date('2023-01-10'),
          modified: new Date('2024-01-07'),
          cost: 320,
          costTrend: 'up',
          compliance: 'partial',
          performance: { storage: 85 },
          policies: 8,
          alerts: 2
        },
        {
          id: 'res-004',
          name: 'dev-app-service',
          type: 'app-service',
          status: 'stopped',
          region: 'Central US',
          resourceGroup: 'development-rg',
          subscription: 'Development',
          tags: { environment: 'development', owner: 'dev-team', project: 'new-feature' },
          created: new Date('2023-09-05'),
          modified: new Date('2024-01-05'),
          cost: 150,
          costTrend: 'down',
          compliance: 'non-compliant',
          performance: { cpu: 0, memory: 0 },
          policies: 5,
          alerts: 3
        },
        {
          id: 'res-005',
          name: 'vpn-gateway-main',
          type: 'network',
          status: 'running',
          region: 'East US',
          resourceGroup: 'network-rg',
          subscription: 'Enterprise Production',
          tags: { environment: 'production', owner: 'network-team', type: 'vpn' },
          created: new Date('2023-02-28'),
          modified: new Date('2024-01-09'),
          cost: 200,
          costTrend: 'stable',
          compliance: 'compliant',
          performance: { network: 40 },
          policies: 10,
          alerts: 0
        },
        {
          id: 'res-006',
          name: 'keyvault-secrets',
          type: 'security',
          status: 'running',
          region: 'East US',
          resourceGroup: 'security-rg',
          subscription: 'Enterprise Production',
          tags: { environment: 'production', owner: 'security-team', criticality: 'critical' },
          created: new Date('2023-01-01'),
          modified: new Date('2024-01-09'),
          cost: 50,
          costTrend: 'stable',
          compliance: 'compliant',
          performance: {},
          policies: 20,
          alerts: 0
        },
        {
          id: 'res-007',
          name: 'test-vm-02',
          type: 'vm',
          status: 'stopped',
          region: 'West US',
          resourceGroup: 'test-rg',
          subscription: 'Testing',
          tags: { environment: 'test', owner: 'qa-team' },
          created: new Date('2023-11-15'),
          modified: new Date('2023-12-20'),
          cost: 100,
          costTrend: 'down',
          compliance: 'partial',
          performance: { cpu: 0, memory: 0, storage: 20 },
          policies: 6,
          alerts: 1
        },
        {
          id: 'res-008',
          name: 'cdn-endpoint',
          type: 'network',
          status: 'running',
          region: 'Global',
          resourceGroup: 'cdn-rg',
          subscription: 'Enterprise Production',
          tags: { environment: 'production', owner: 'web-team', type: 'cdn' },
          created: new Date('2023-04-10'),
          modified: new Date('2024-01-08'),
          cost: 180,
          costTrend: 'up',
          compliance: 'compliant',
          performance: { network: 75 },
          policies: 7,
          alerts: 0
        }
      ]);

      // Mock resource groups
      setResourceGroups([
        {
          id: 'rg-001',
          name: 'production-rg',
          subscription: 'Enterprise Production',
          region: 'East US',
          resourceCount: 25,
          totalCost: 5200,
          complianceStatus: 'compliant'
        },
        {
          id: 'rg-002',
          name: 'development-rg',
          subscription: 'Development',
          region: 'Central US',
          resourceCount: 15,
          totalCost: 1800,
          complianceStatus: 'partial'
        },
        {
          id: 'rg-003',
          name: 'analytics-rg',
          subscription: 'Enterprise Production',
          region: 'East US',
          resourceCount: 8,
          totalCost: 3200,
          complianceStatus: 'compliant'
        },
        {
          id: 'rg-004',
          name: 'backup-rg',
          subscription: 'Enterprise Production',
          region: 'West US',
          resourceCount: 12,
          totalCost: 1500,
          complianceStatus: 'partial'
        }
      ]);

      // Mock metrics
      setMetrics({
        totalResources: 85,
        runningResources: 62,
        stoppedResources: 23,
        totalCost: 12500,
        complianceRate: 78,
        criticalAlerts: 7
      });

      setLoading(false);
    }, 1000);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30';
      case 'stopped':
        return 'text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-900/30';
      case 'warning':
        return 'text-amber-600 bg-amber-100 dark:text-amber-400 dark:bg-amber-900/30';
      case 'error':
        return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30';
      default:
        return 'text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-900/30';
    }
  };

  const getComplianceColor = (compliance: string) => {
    switch (compliance) {
      case 'compliant':
        return 'text-green-600 dark:text-green-400';
      case 'non-compliant':
        return 'text-red-600 dark:text-red-400';
      case 'partial':
        return 'text-amber-600 dark:text-amber-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getResourceIcon = (type: string) => {
    switch (type) {
      case 'vm':
        return Server;
      case 'database':
        return Database;
      case 'storage':
        return HardDrive;
      case 'network':
        return Network;
      case 'security':
        return Shield;
      case 'app-service':
        return Cloud;
      default:
        return Server;
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0
    }).format(value);
  };

  const filteredResources = resources.filter(resource => {
    const matchesType = selectedType === 'all' || resource.type === selectedType;
    const matchesSearch = resource.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          resource.resourceGroup.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          Object.values(resource.tags).some(tag => 
                            tag.toLowerCase().includes(searchQuery.toLowerCase())
                          );
    return matchesType && matchesSearch;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading resources...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 pb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
              <Cloud className="h-8 w-8 text-blue-600" />
              Resources Management
            </h1>
            <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
              Manage and monitor your cloud resources across all subscriptions
            </p>
          </div>
          <div className="flex gap-2">
            <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
              <Plus className="h-4 w-4" />
              New Resource
            </button>
            <button className="px-4 py-2 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center gap-2">
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Cards */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Total Resources
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metrics.totalResources}</div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Across all subscriptions
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Running
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {metrics.runningResources}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Active resources
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Stopped
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-gray-600 dark:text-gray-400">
                {metrics.stoppedResources}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Inactive resources
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Monthly Cost
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatCurrency(metrics.totalCost)}</div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Current month
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Compliance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metrics.complianceRate}%</div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Policy compliance
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Critical Alerts
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                {metrics.criticalAlerts}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Require attention
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Resource Groups */}
      <Card>
        <CardHeader>
          <CardTitle>Resource Groups</CardTitle>
          <CardDescription>Overview of resource groups across subscriptions</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {resourceGroups.map((group) => (
              <div key={group.id} className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:shadow-md transition-shadow cursor-pointer">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <h4 className="font-semibold">{group.name}</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{group.subscription}</p>
                  </div>
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                    group.complianceStatus === 'compliant'
                      ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                      : group.complianceStatus === 'partial'
                      ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
                      : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                  }`}>
                    {group.complianceStatus}
                  </span>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Resources</span>
                    <span className="font-medium">{group.resourceCount}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Region</span>
                    <span className="font-medium">{group.region}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Cost</span>
                    <span className="font-medium">{formatCurrency(group.totalCost)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Filters and Search */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search resources..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <select
          value={selectedType}
          onChange={(e) => setSelectedType(e.target.value)}
          className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="all">All Types</option>
          <option value="vm">Virtual Machines</option>
          <option value="database">Databases</option>
          <option value="storage">Storage</option>
          <option value="network">Network</option>
          <option value="security">Security</option>
          <option value="app-service">App Services</option>
        </select>
        <div className="flex gap-2">
          <button
            onClick={() => setSelectedView('list')}
            className={`p-2 rounded-lg ${
              selectedView === 'list'
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
            }`}
          >
            List View
          </button>
          <button
            onClick={() => setSelectedView('grid')}
            className={`p-2 rounded-lg ${
              selectedView === 'grid'
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
            }`}
          >
            Grid View
          </button>
        </div>
      </div>

      {/* Resources List/Grid */}
      {selectedView === 'list' ? (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="border-b border-gray-200 dark:border-gray-700">
                  <tr>
                    <th className="text-left py-3 px-4">Resource</th>
                    <th className="text-left py-3 px-4">Type</th>
                    <th className="text-center py-3 px-4">Status</th>
                    <th className="text-left py-3 px-4">Region</th>
                    <th className="text-right py-3 px-4">Cost/Month</th>
                    <th className="text-center py-3 px-4">Compliance</th>
                    <th className="text-center py-3 px-4">Alerts</th>
                    <th className="text-center py-3 px-4">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredResources.map((resource) => {
                    const Icon = getResourceIcon(resource.type);
                    return (
                      <tr key={resource.id} className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900/50">
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-3">
                            <Icon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
                            <div>
                              <div className="font-medium">{resource.name}</div>
                              <div className="text-sm text-gray-600 dark:text-gray-400">
                                {resource.resourceGroup}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="py-3 px-4 capitalize">{resource.type.replace('-', ' ')}</td>
                        <td className="text-center py-3 px-4">
                          <span className={`inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(resource.status)}`}>
                            {resource.status === 'running' && <PlayCircle className="h-3 w-3" />}
                            {resource.status === 'stopped' && <PauseCircle className="h-3 w-3" />}
                            {resource.status === 'warning' && <AlertTriangle className="h-3 w-3" />}
                            {resource.status === 'error' && <XCircle className="h-3 w-3" />}
                            {resource.status}
                          </span>
                        </td>
                        <td className="py-3 px-4">{resource.region}</td>
                        <td className="text-right py-3 px-4">
                          <div className="flex items-center justify-end gap-1">
                            {formatCurrency(resource.cost)}
                            {resource.costTrend === 'up' && <TrendingUp className="h-4 w-4 text-red-500" />}
                            {resource.costTrend === 'down' && <TrendingDown className="h-4 w-4 text-green-500" />}
                          </div>
                        </td>
                        <td className="text-center py-3 px-4">
                          <span className={getComplianceColor(resource.compliance)}>
                            {resource.compliance === 'compliant' && <CheckCircle className="h-4 w-4 inline" />}
                            {resource.compliance === 'non-compliant' && <XCircle className="h-4 w-4 inline" />}
                            {resource.compliance === 'partial' && <AlertTriangle className="h-4 w-4 inline" />}
                          </span>
                        </td>
                        <td className="text-center py-3 px-4">
                          {resource.alerts > 0 ? (
                            <span className="inline-flex items-center gap-1 px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-full text-xs font-medium">
                              <AlertTriangle className="h-3 w-3" />
                              {resource.alerts}
                            </span>
                          ) : (
                            <span className="text-gray-400">-</span>
                          )}
                        </td>
                        <td className="text-center py-3 px-4">
                          <div className="flex items-center justify-center gap-1">
                            <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded">
                              <Eye className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                            </button>
                            <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded">
                              <Edit className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                            </button>
                            <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded">
                              <MoreVertical className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredResources.map((resource) => {
            const Icon = getResourceIcon(resource.type);
            return (
              <Card key={resource.id} className="hover:shadow-lg transition-shadow cursor-pointer">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
                        <Icon className="h-6 w-6 text-gray-600 dark:text-gray-400" />
                      </div>
                      <div>
                        <CardTitle className="text-lg">{resource.name}</CardTitle>
                        <CardDescription>{resource.resourceGroup}</CardDescription>
                      </div>
                    </div>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(resource.status)}`}>
                      {resource.status}
                    </span>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Type</span>
                      <span className="font-medium capitalize">{resource.type.replace('-', ' ')}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Region</span>
                      <span className="font-medium">{resource.region}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Cost</span>
                      <span className="font-medium flex items-center gap-1">
                        {formatCurrency(resource.cost)}/mo
                        {resource.costTrend === 'up' && <TrendingUp className="h-3 w-3 text-red-500" />}
                        {resource.costTrend === 'down' && <TrendingDown className="h-3 w-3 text-green-500" />}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Compliance</span>
                      <span className={`font-medium ${getComplianceColor(resource.compliance)}`}>
                        {resource.compliance}
                      </span>
                    </div>
                    {resource.performance.cpu !== undefined && (
                      <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                        <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">Performance</div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          {resource.performance.cpu !== undefined && (
                            <div>
                              <span className="text-gray-500">CPU:</span>
                              <span className="ml-1 font-medium">{resource.performance.cpu}%</span>
                            </div>
                          )}
                          {resource.performance.memory !== undefined && (
                            <div>
                              <span className="text-gray-500">Memory:</span>
                              <span className="ml-1 font-medium">{resource.performance.memory}%</span>
                            </div>
                          )}
                          {resource.performance.storage !== undefined && (
                            <div>
                              <span className="text-gray-500">Storage:</span>
                              <span className="ml-1 font-medium">{resource.performance.storage}%</span>
                            </div>
                          )}
                          {resource.performance.network !== undefined && (
                            <div>
                              <span className="text-gray-500">Network:</span>
                              <span className="ml-1 font-medium">{resource.performance.network}%</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    <div className="flex gap-2 pt-2">
                      <button className="flex-1 px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700">
                        Manage
                      </button>
                      <button className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-800">
                        <MoreVertical className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}