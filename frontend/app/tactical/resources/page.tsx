'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import AuthGuard from '../../../components/AuthGuard';
import { api } from '../../../lib/api-client';
import toast from 'react-hot-toast';
import { Server, Database, Cloud, Network, HardDrive, Cpu, Activity, AlertCircle, CheckCircle, XCircle } from 'lucide-react';

interface Resource {
  id: string;
  name: string;
  type: string;
  status: 'running' | 'stopped' | 'warning' | 'error';
  region: string;
  resourceGroup: string;
  created: string;
  cost: number;
  health: number;
  tags: string[];
}

interface ResourceMetrics {
  total: number;
  running: number;
  stopped: number;
  warning: number;
  error: number;
  byType: {
    [key: string]: number;
  };
  byRegion: {
    [key: string]: number;
  };
}

export default function ResourceManagementCenter() {
  return (
    <AuthGuard requireAuth={true}>
      <ResourceManagementCenterContent />
    </AuthGuard>
  );
}

function ResourceManagementCenterContent() {
  const [resources, setResources] = useState<Resource[]>([]);
  const [metrics, setMetrics] = useState<ResourceMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedType, setSelectedType] = useState('all');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    fetchResourceData();
    const interval = setInterval(fetchResourceData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchResourceData = async () => {
    try {
      const resp = await api.getResources()
      if (resp.error) {
        const mockData = getMockResourceData();
        setResources(mockData);
        setMetrics(calculateMetrics(mockData));
      } else {
        const items = (resp.data as any)?.resources || (resp.data as any) || []
        setResources(items);
        setMetrics(calculateMetrics(items));
      }
    } catch (error) {
      const mockData = getMockResourceData();
      setResources(mockData);
      setMetrics(calculateMetrics(mockData));
    } finally {
      setLoading(false);
    }
  };

  const triggerAction = async (actionType: string) => {
    try {
      const resp = await api.createAction('global', actionType)
      if (resp.error || resp.status >= 400) {
        toast.error(`Action failed: ${actionType}`)
        return
      }
      toast.success(`${actionType.replace('_',' ')} started`)
      const id = resp.data?.action_id || resp.data?.id
      if (id) {
        const stop = api.streamActionEvents(String(id), (m) => console.log('[resources-action]', id, m))
        setTimeout(stop, 60000)
      }
    } catch (e) {
      toast.error(`Action error: ${actionType}`)
    }
  }

  const getMockResourceData = (): Resource[] => [
    { id: 'vm-01', name: 'VM-PROD-WEB-01', type: 'VirtualMachine', status: 'running', region: 'East US', resourceGroup: 'Production', created: '2024-01-15', cost: 450, health: 98, tags: ['production', 'web'] },
    { id: 'vm-02', name: 'VM-PROD-API-01', type: 'VirtualMachine', status: 'running', region: 'East US', resourceGroup: 'Production', created: '2024-01-20', cost: 380, health: 95, tags: ['production', 'api'] },
    { id: 'sql-01', name: 'SQL-PROD-01', type: 'Database', status: 'running', region: 'East US', resourceGroup: 'Production', created: '2024-01-10', cost: 890, health: 100, tags: ['production', 'database'] },
    { id: 'storage-01', name: 'STORAGE-PROD-01', type: 'StorageAccount', status: 'running', region: 'East US', resourceGroup: 'Production', created: '2024-01-05', cost: 120, health: 100, tags: ['production', 'storage'] },
    { id: 'vm-03', name: 'VM-DEV-01', type: 'VirtualMachine', status: 'stopped', region: 'West US', resourceGroup: 'Development', created: '2024-02-01', cost: 0, health: 0, tags: ['development'] },
    { id: 'aks-01', name: 'AKS-PROD-01', type: 'Kubernetes', status: 'running', region: 'East US', resourceGroup: 'Production', created: '2024-01-25', cost: 1200, health: 97, tags: ['production', 'kubernetes'] },
    { id: 'app-01', name: 'APP-SERVICE-01', type: 'AppService', status: 'warning', region: 'Central US', resourceGroup: 'Production', created: '2024-02-10', cost: 250, health: 75, tags: ['production', 'webapp'] },
    { id: 'lb-01', name: 'LB-PROD-01', type: 'LoadBalancer', status: 'running', region: 'East US', resourceGroup: 'Production', created: '2024-01-18', cost: 180, health: 100, tags: ['production', 'network'] },
    { id: 'nsg-01', name: 'NSG-PROD-01', type: 'NetworkSecurityGroup', status: 'running', region: 'East US', resourceGroup: 'Production', created: '2024-01-12', cost: 0, health: 100, tags: ['production', 'security'] },
    { id: 'vm-04', name: 'VM-TEST-01', type: 'VirtualMachine', status: 'error', region: 'North Europe', resourceGroup: 'Testing', created: '2024-03-01', cost: 0, health: 0, tags: ['testing'] }
  ];

  const calculateMetrics = (resources: Resource[]): ResourceMetrics => {
    const metrics: ResourceMetrics = {
      total: resources.length,
      running: resources.filter(r => r.status === 'running').length,
      stopped: resources.filter(r => r.status === 'stopped').length,
      warning: resources.filter(r => r.status === 'warning').length,
      error: resources.filter(r => r.status === 'error').length,
      byType: {},
      byRegion: {}
    };

    resources.forEach(resource => {
      metrics.byType[resource.type] = (metrics.byType[resource.type] || 0) + 1;
      metrics.byRegion[resource.region] = (metrics.byRegion[resource.region] || 0) + 1;
    });

    return metrics;
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'VirtualMachine': return <Server className="w-4 h-4" />;
      case 'Database': return <Database className="w-4 h-4" />;
      case 'StorageAccount': return <HardDrive className="w-4 h-4" />;
      case 'Kubernetes': return <Network className="w-4 h-4" />;
      case 'AppService': return <Cloud className="w-4 h-4" />;
      default: return <Cpu className="w-4 h-4" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'stopped': return <XCircle className="w-4 h-4 text-gray-500" />;
      case 'warning': return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      case 'error': return <XCircle className="w-4 h-4 text-red-500" />;
      default: return null;
    }
  };

  const filteredResources = resources.filter(resource => {
    if (selectedType !== 'all' && resource.type !== selectedType) return false;
    if (selectedStatus !== 'all' && resource.status !== selectedStatus) return false;
    if (searchTerm && !resource.name.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-gray-400">SCANNING AZURE RESOURCES...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/tactical" className="text-gray-400 hover:text-gray-200">
                ← BACK
              </Link>
              <div className="h-6 w-px bg-gray-700" />
              <h1 className="text-xl font-bold">RESOURCE MANAGEMENT CENTER</h1>
              <div className="px-3 py-1 bg-blue-900/30 text-blue-500 rounded text-xs font-bold">
                {metrics?.total} RESOURCES
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <input
                type="text"
                placeholder="Search resources..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="px-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm w-64"
              />
              <button onClick={() => triggerAction('deploy_resource')} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors">
                DEPLOY RESOURCE
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="p-6">
        {/* Metrics */}
        <div className="grid grid-cols-5 gap-4 mb-6">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Total Resources</p>
            <p className="text-3xl font-bold font-mono">{metrics?.total}</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Running</p>
            <p className="text-3xl font-bold font-mono text-green-500">{metrics?.running}</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Stopped</p>
            <p className="text-3xl font-bold font-mono text-gray-500">{metrics?.stopped}</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Warning</p>
            <p className="text-3xl font-bold font-mono text-yellow-500">{metrics?.warning}</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Error</p>
            <p className="text-3xl font-bold font-mono text-red-500">{metrics?.error}</p>
          </div>
        </div>

        {/* Filters */}
        <div className="flex space-x-4 mb-6">
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            className="px-4 py-2 bg-gray-900 border border-gray-800 rounded text-sm"
          >
            <option value="all">All Types</option>
            {Object.keys(metrics?.byType || {}).map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
          
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="px-4 py-2 bg-gray-900 border border-gray-800 rounded text-sm"
          >
            <option value="all">All Status</option>
            <option value="running">Running</option>
            <option value="stopped">Stopped</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
          </select>
        </div>

        {/* Resources Table */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-800/50 border-b border-gray-800">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Name</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Type</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Region</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Resource Group</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Health</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Cost/Month</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {filteredResources.map((resource) => (
                <tr key={resource.id} className="hover:bg-gray-800/50 transition-colors">
                  <td className="px-4 py-3">
                    {getStatusIcon(resource.status)}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      {getTypeIcon(resource.type)}
                      <span className="font-medium text-sm">{resource.name}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-400">{resource.type}</td>
                  <td className="px-4 py-3 text-sm text-gray-400">{resource.region}</td>
                  <td className="px-4 py-3 text-sm text-gray-400">{resource.resourceGroup}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            resource.health >= 90 ? 'bg-green-500' :
                            resource.health >= 70 ? 'bg-yellow-500' :
                            'bg-red-500'
                          }`}
                          style={{ width: `${resource.health}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500">{resource.health}%</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm font-mono">${resource.cost}</td>
                  <td className="px-4 py-3">
                    <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-xs transition-colors">
                      MANAGE
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}