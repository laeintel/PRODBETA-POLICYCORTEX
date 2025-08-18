'use client';

import React, { useState, useEffect } from 'react';
import { 
  FileCode, Layers, Package, GitBranch, Copy, Download, Upload,
  Play, Pause, CheckCircle, XCircle, AlertTriangle, Clock,
  Shield, Lock, Unlock, Eye, EyeOff, Settings, Filter,
  Search, RefreshCw, MoreVertical, ChevronRight, ChevronDown,
  Zap, Database, Server, Cloud, Globe, Building, Users,
  Calendar, Timer, TrendingUp, BarChart3, Tag, Hash
} from 'lucide-react';

interface Blueprint {
  id: string;
  name: string;
  description: string;
  version: string;
  type: 'infrastructure' | 'application' | 'network' | 'security' | 'governance';
  category: 'compute' | 'storage' | 'database' | 'networking' | 'identity' | 'monitoring';
  status: 'published' | 'draft' | 'deprecated' | 'testing';
  author: string;
  lastModified: string;
  deployments: number;
  resources: number;
  estimatedCost: number;
  complianceLevel: 'high' | 'medium' | 'low';
  tags: string[];
  parameters: Parameter[];
  artifacts: Artifact[];
}

interface Parameter {
  id: string;
  name: string;
  type: 'string' | 'number' | 'boolean' | 'array' | 'object';
  required: boolean;
  defaultValue?: any;
  description: string;
  validation?: string;
}

interface Artifact {
  id: string;
  name: string;
  type: 'template' | 'script' | 'policy' | 'configuration';
  size: string;
  lastModified: string;
}

interface Deployment {
  id: string;
  blueprintId: string;
  blueprintName: string;
  environment: string;
  status: 'running' | 'succeeded' | 'failed' | 'pending';
  startTime: string;
  duration: string;
  resourcesCreated: number;
  cost: number;
  initiatedBy: string;
}

interface BlueprintVersion {
  version: string;
  releaseDate: string;
  changes: string[];
  author: string;
  downloads: number;
}

export default function Blueprints() {
  const [blueprints, setBlueprints] = useState<Blueprint[]>([]);
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [versions, setVersions] = useState<BlueprintVersion[]>([]);
  const [viewMode, setViewMode] = useState<'gallery' | 'deployments' | 'editor' | 'versions'>('gallery');
  const [selectedBlueprint, setSelectedBlueprint] = useState<Blueprint | null>(null);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    // Initialize with blueprint data
    setBlueprints([
      {
        id: 'BP-001',
        name: 'Azure Landing Zone',
        description: 'Enterprise-scale landing zone with hub-spoke network topology',
        version: '2.4.0',
        type: 'infrastructure',
        category: 'networking',
        status: 'published',
        author: 'Platform Team',
        lastModified: '2 days ago',
        deployments: 45,
        resources: 28,
        estimatedCost: 2500,
        complianceLevel: 'high',
        tags: ['enterprise', 'hub-spoke', 'secure'],
        parameters: [
          { id: 'P1', name: 'region', type: 'string', required: true, description: 'Azure region for deployment', defaultValue: 'eastus' },
          { id: 'P2', name: 'addressSpace', type: 'string', required: true, description: 'Virtual network address space', defaultValue: '10.0.0.0/16' },
          { id: 'P3', name: 'enableFirewall', type: 'boolean', required: false, description: 'Deploy Azure Firewall', defaultValue: true }
        ],
        artifacts: [
          { id: 'A1', name: 'main.bicep', type: 'template', size: '124 KB', lastModified: '2 days ago' },
          { id: 'A2', name: 'network.bicep', type: 'template', size: '89 KB', lastModified: '2 days ago' },
          { id: 'A3', name: 'policies.json', type: 'policy', size: '45 KB', lastModified: '1 week ago' }
        ]
      },
      {
        id: 'BP-002',
        name: 'AKS Production Cluster',
        description: 'Production-ready Azure Kubernetes Service cluster with monitoring',
        version: '3.1.2',
        type: 'application',
        category: 'compute',
        status: 'published',
        author: 'DevOps Team',
        lastModified: '1 week ago',
        deployments: 78,
        resources: 35,
        estimatedCost: 3800,
        complianceLevel: 'high',
        tags: ['kubernetes', 'production', 'monitoring'],
        parameters: [
          { id: 'P1', name: 'nodeCount', type: 'number', required: true, description: 'Number of nodes', defaultValue: 3 },
          { id: 'P2', name: 'vmSize', type: 'string', required: true, description: 'Node VM size', defaultValue: 'Standard_D4s_v3' }
        ],
        artifacts: [
          { id: 'A1', name: 'aks.bicep', type: 'template', size: '156 KB', lastModified: '1 week ago' },
          { id: 'A2', name: 'monitoring.yaml', type: 'configuration', size: '23 KB', lastModified: '1 week ago' }
        ]
      },
      {
        id: 'BP-003',
        name: 'SQL Database HA',
        description: 'High-availability SQL Database with geo-replication',
        version: '1.8.0',
        type: 'infrastructure',
        category: 'database',
        status: 'published',
        author: 'Data Team',
        lastModified: '3 days ago',
        deployments: 32,
        resources: 12,
        estimatedCost: 1800,
        complianceLevel: 'high',
        tags: ['sql', 'high-availability', 'geo-replication'],
        parameters: [
          { id: 'P1', name: 'databaseName', type: 'string', required: true, description: 'Database name' },
          { id: 'P2', name: 'skuName', type: 'string', required: true, description: 'SKU tier', defaultValue: 'S3' },
          { id: 'P3', name: 'geoReplication', type: 'boolean', required: false, description: 'Enable geo-replication', defaultValue: true }
        ],
        artifacts: [
          { id: 'A1', name: 'database.bicep', type: 'template', size: '67 KB', lastModified: '3 days ago' }
        ]
      },
      {
        id: 'BP-004',
        name: 'Zero Trust Network',
        description: 'Zero trust network architecture with micro-segmentation',
        version: '2.0.1',
        type: 'security',
        category: 'networking',
        status: 'testing',
        author: 'Security Team',
        lastModified: '5 days ago',
        deployments: 12,
        resources: 42,
        estimatedCost: 4200,
        complianceLevel: 'high',
        tags: ['zero-trust', 'security', 'micro-segmentation'],
        parameters: [
          { id: 'P1', name: 'segments', type: 'array', required: true, description: 'Network segments' },
          { id: 'P2', name: 'enableMFA', type: 'boolean', required: true, description: 'Enforce MFA', defaultValue: true }
        ],
        artifacts: [
          { id: 'A1', name: 'zerotrust.bicep', type: 'template', size: '234 KB', lastModified: '5 days ago' },
          { id: 'A2', name: 'security-policies.json', type: 'policy', size: '89 KB', lastModified: '5 days ago' }
        ]
      },
      {
        id: 'BP-005',
        name: 'Cost Optimization Pack',
        description: 'Automated cost optimization with auto-scaling and scheduling',
        version: '1.5.0',
        type: 'governance',
        category: 'monitoring',
        status: 'published',
        author: 'FinOps Team',
        lastModified: '1 day ago',
        deployments: 156,
        resources: 8,
        estimatedCost: 0,
        complianceLevel: 'medium',
        tags: ['cost', 'optimization', 'auto-scaling'],
        parameters: [
          { id: 'P1', name: 'schedules', type: 'object', required: true, description: 'Shutdown schedules' },
          { id: 'P2', name: 'autoScale', type: 'boolean', required: false, description: 'Enable auto-scaling', defaultValue: true }
        ],
        artifacts: [
          { id: 'A1', name: 'cost-optimization.ps1', type: 'script', size: '12 KB', lastModified: '1 day ago' }
        ]
      },
      {
        id: 'BP-006',
        name: 'Storage Account Secure',
        description: 'Secure storage account with encryption and private endpoints',
        version: '1.2.3',
        type: 'infrastructure',
        category: 'storage',
        status: 'draft',
        author: 'Storage Team',
        lastModified: '6 hours ago',
        deployments: 0,
        resources: 5,
        estimatedCost: 150,
        complianceLevel: 'high',
        tags: ['storage', 'encryption', 'private-endpoint'],
        parameters: [
          { id: 'P1', name: 'storageAccountName', type: 'string', required: true, description: 'Storage account name' },
          { id: 'P2', name: 'enableEncryption', type: 'boolean', required: true, description: 'Enable encryption', defaultValue: true }
        ],
        artifacts: [
          { id: 'A1', name: 'storage.bicep', type: 'template', size: '34 KB', lastModified: '6 hours ago' }
        ]
      }
    ]);

    setDeployments([
      {
        id: 'DEP-001',
        blueprintId: 'BP-001',
        blueprintName: 'Azure Landing Zone',
        environment: 'Production',
        status: 'succeeded',
        startTime: '2 hours ago',
        duration: '45 minutes',
        resourcesCreated: 28,
        cost: 2450,
        initiatedBy: 'john.doe@company.com'
      },
      {
        id: 'DEP-002',
        blueprintId: 'BP-002',
        blueprintName: 'AKS Production Cluster',
        environment: 'Staging',
        status: 'running',
        startTime: '30 minutes ago',
        duration: '30 minutes',
        resourcesCreated: 18,
        cost: 1200,
        initiatedBy: 'devops@company.com'
      },
      {
        id: 'DEP-003',
        blueprintId: 'BP-003',
        blueprintName: 'SQL Database HA',
        environment: 'Development',
        status: 'failed',
        startTime: '4 hours ago',
        duration: '12 minutes',
        resourcesCreated: 3,
        cost: 0,
        initiatedBy: 'data.team@company.com'
      },
      {
        id: 'DEP-004',
        blueprintId: 'BP-005',
        blueprintName: 'Cost Optimization Pack',
        environment: 'Production',
        status: 'succeeded',
        startTime: '1 day ago',
        duration: '5 minutes',
        resourcesCreated: 8,
        cost: 0,
        initiatedBy: 'finops@company.com'
      }
    ]);

    setVersions([
      { version: '2.4.0', releaseDate: '2 days ago', changes: ['Added firewall rules', 'Updated network topology', 'Security improvements'], author: 'Platform Team', downloads: 234 },
      { version: '2.3.2', releaseDate: '2 weeks ago', changes: ['Bug fixes', 'Performance improvements'], author: 'Platform Team', downloads: 456 },
      { version: '2.3.0', releaseDate: '1 month ago', changes: ['New hub-spoke design', 'Added monitoring'], author: 'Platform Team', downloads: 789 },
      { version: '2.2.0', releaseDate: '2 months ago', changes: ['Initial release'], author: 'Platform Team', downloads: 1234 }
    ]);
  }, []);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'published': case 'succeeded': return 'text-green-500 bg-green-900/20';
      case 'draft': case 'pending': return 'text-yellow-500 bg-yellow-900/20';
      case 'deprecated': case 'failed': return 'text-red-500 bg-red-900/20';
      case 'testing': case 'running': return 'text-blue-500 bg-blue-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getTypeIcon = (type: string) => {
    switch(type) {
      case 'infrastructure': return <Layers className="w-4 h-4" />;
      case 'application': return <Package className="w-4 h-4" />;
      case 'network': return <Globe className="w-4 h-4" />;
      case 'security': return <Shield className="w-4 h-4" />;
      case 'governance': return <Settings className="w-4 h-4" />;
      default: return <FileCode className="w-4 h-4" />;
    }
  };

  const totalBlueprints = blueprints.length;
  const publishedBlueprints = blueprints.filter(b => b.status === 'published').length;
  const totalDeployments = blueprints.reduce((acc, b) => acc + b.deployments, 0);
  const activeDeployments = deployments.filter(d => d.status === 'running').length;

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold flex items-center space-x-2">
              <FileCode className="w-6 h-6 text-purple-500" />
              <span>Blueprints</span>
            </h1>
            <p className="text-sm text-gray-400 mt-1">Infrastructure as Code templates and deployment blueprints</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
            
            <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
              <Upload className="w-4 h-4" />
              <span>Import</span>
            </button>
            
            <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm flex items-center space-x-2">
              <FileCode className="w-4 h-4" />
              <span>New Blueprint</span>
            </button>
          </div>
        </div>
      </header>

      {/* Summary Stats */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="text-center">
            <div className="text-xs text-gray-500">Total Blueprints</div>
            <div className="text-2xl font-bold">{totalBlueprints}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Published</div>
            <div className="text-2xl font-bold text-green-500">{publishedBlueprints}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Total Deployments</div>
            <div className="text-2xl font-bold">{totalDeployments}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Active</div>
            <div className="text-2xl font-bold text-blue-500">{activeDeployments}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Success Rate</div>
            <div className="text-2xl font-bold text-green-500">94%</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Avg Deploy Time</div>
            <div className="text-2xl font-bold">28m</div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-gray-900/30 border-b border-gray-800 px-6">
        <div className="flex space-x-6">
          {['gallery', 'deployments', 'editor', 'versions'].map(view => (
            <button
              key={view}
              onClick={() => setViewMode(view as any)}
              className={`py-3 border-b-2 text-sm capitalize ${
                viewMode === view 
                  ? 'border-purple-500 text-purple-500' 
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              {view === 'gallery' ? 'Blueprint Gallery' : view}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {viewMode === 'gallery' && (
          <div className="space-y-4">
            {/* Search and Filter */}
            <div className="flex items-center space-x-3 mb-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
                <input
                  type="text"
                  placeholder="Search blueprints..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                />
              </div>
              
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Categories</option>
                <option value="compute">Compute</option>
                <option value="storage">Storage</option>
                <option value="database">Database</option>
                <option value="networking">Networking</option>
                <option value="identity">Identity</option>
                <option value="monitoring">Monitoring</option>
              </select>
            </div>

            {/* Blueprint Grid */}
            <div className="grid grid-cols-2 gap-4">
              {blueprints.map(blueprint => (
                <div key={blueprint.id} className="bg-gray-900 border border-gray-800 rounded-lg">
                  <div className="p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-start space-x-3">
                        <div className={`p-2 rounded ${
                          blueprint.type === 'infrastructure' ? 'bg-blue-900/20' :
                          blueprint.type === 'application' ? 'bg-green-900/20' :
                          blueprint.type === 'security' ? 'bg-red-900/20' :
                          blueprint.type === 'network' ? 'bg-purple-900/20' :
                          'bg-yellow-900/20'
                        }`}>
                          {getTypeIcon(blueprint.type)}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <h3 className="text-sm font-bold">{blueprint.name}</h3>
                            <span className="text-xs text-gray-500">v{blueprint.version}</span>
                            <span className={`px-2 py-1 rounded text-xs ${getStatusColor(blueprint.status)}`}>
                              {blueprint.status}
                            </span>
                          </div>
                          <p className="text-xs text-gray-400 mb-2">{blueprint.description}</p>
                          <div className="flex items-center space-x-4 text-xs text-gray-500">
                            <span>By {blueprint.author}</span>
                            <span>{blueprint.deployments} deployments</span>
                            <span>{blueprint.lastModified}</span>
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => setSelectedBlueprint(blueprint)}
                        className="p-1 hover:bg-gray-800 rounded"
                      >
                        <MoreVertical className="w-4 h-4 text-gray-500" />
                      </button>
                    </div>

                    <div className="grid grid-cols-4 gap-2 mb-3 text-xs">
                      <div className="bg-gray-800 rounded p-2">
                        <div className="text-gray-500">Resources</div>
                        <div className="font-bold">{blueprint.resources}</div>
                      </div>
                      <div className="bg-gray-800 rounded p-2">
                        <div className="text-gray-500">Est. Cost</div>
                        <div className="font-bold">${blueprint.estimatedCost}</div>
                      </div>
                      <div className="bg-gray-800 rounded p-2">
                        <div className="text-gray-500">Compliance</div>
                        <div className={`font-bold ${
                          blueprint.complianceLevel === 'high' ? 'text-green-500' :
                          blueprint.complianceLevel === 'medium' ? 'text-yellow-500' :
                          'text-red-500'
                        }`}>
                          {blueprint.complianceLevel}
                        </div>
                      </div>
                      <div className="bg-gray-800 rounded p-2">
                        <div className="text-gray-500">Artifacts</div>
                        <div className="font-bold">{blueprint.artifacts.length}</div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex flex-wrap gap-1">
                        {blueprint.tags.slice(0, 3).map(tag => (
                          <span key={tag} className="px-2 py-1 bg-gray-800 rounded text-xs">
                            {tag}
                          </span>
                        ))}
                      </div>
                      <div className="flex space-x-2">
                        <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                          Deploy
                        </button>
                        <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                          Preview
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {viewMode === 'deployments' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-800">
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Blueprint</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Environment</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Status</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Started</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Duration</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Resources</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Cost</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Initiated By</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Actions</th>
                </tr>
              </thead>
              <tbody>
                {deployments.map(deployment => (
                  <tr key={deployment.id} className="border-t border-gray-800 hover:bg-gray-800/30">
                    <td className="px-4 py-3 text-sm">{deployment.blueprintName}</td>
                    <td className="px-4 py-3 text-sm">{deployment.environment}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs ${getStatusColor(deployment.status)}`}>
                        {deployment.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm">{deployment.startTime}</td>
                    <td className="px-4 py-3 text-sm">{deployment.duration}</td>
                    <td className="px-4 py-3 text-sm">{deployment.resourcesCreated}</td>
                    <td className="px-4 py-3 text-sm">${deployment.cost}</td>
                    <td className="px-4 py-3 text-sm text-xs">{deployment.initiatedBy}</td>
                    <td className="px-4 py-3">
                      <button className="text-xs text-blue-500 hover:text-blue-400">
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {viewMode === 'editor' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold">Blueprint Editor</h3>
              <div className="flex space-x-2">
                <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                  Save Draft
                </button>
                <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                  Validate
                </button>
                <button className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs">
                  Publish
                </button>
              </div>
            </div>
            <div className="bg-gray-800 rounded p-4 font-mono text-xs text-gray-300 h-96 overflow-auto">
              <pre>{`{
  "name": "New Blueprint",
  "version": "1.0.0",
  "description": "Blueprint description",
  "type": "infrastructure",
  "resources": [
    {
      "type": "Microsoft.Network/virtualNetworks",
      "apiVersion": "2021-02-01",
      "name": "vnet-main",
      "location": "[parameters('location')]",
      "properties": {
        "addressSpace": {
          "addressPrefixes": ["10.0.0.0/16"]
        }
      }
    }
  ],
  "parameters": {
    "location": {
      "type": "string",
      "defaultValue": "eastus"
    }
  }
}`}</pre>
            </div>
          </div>
        )}

        {viewMode === 'versions' && (
          <div className="space-y-4">
            {versions.map(version => (
              <div key={version.version} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <div className="flex items-center space-x-3">
                      <h3 className="text-sm font-bold">Version {version.version}</h3>
                      <span className="text-xs text-gray-500">Released {version.releaseDate}</span>
                      <span className="text-xs text-gray-500">by {version.author}</span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className="text-sm text-gray-500">{version.downloads} downloads</span>
                    <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                      Download
                    </button>
                  </div>
                </div>
                <div className="space-y-1">
                  <h4 className="text-xs font-bold text-gray-400 mb-2">Changes:</h4>
                  {version.changes.map((change, idx) => (
                    <div key={idx} className="flex items-center space-x-2 text-xs">
                      <CheckCircle className="w-3 h-3 text-green-500" />
                      <span>{change}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}