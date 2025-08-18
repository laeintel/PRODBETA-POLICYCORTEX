'use client';

import React, { useState, useEffect } from 'react';
import { 
  Tag, Search, Filter, Plus, Edit, Trash2, CheckCircle, XCircle,
  AlertTriangle, Info, Download, Upload, RefreshCw, Copy, Shield,
  Settings, BarChart3, PieChart, TrendingUp, Users, Building,
  Calendar, Clock, ChevronRight, ChevronDown, MoreVertical,
  Globe, Server, Database, Cloud, GitBranch, Hash, Key
} from 'lucide-react';

interface ResourceTag {
  id: string;
  key: string;
  value: string;
  description: string;
  required: boolean;
  category: 'cost' | 'compliance' | 'technical' | 'business' | 'security';
  resourceTypes: string[];
  appliedCount: number;
  createdBy: string;
  createdDate: string;
  lastModified: string;
  validationRule?: string;
  defaultValue?: string;
}

interface TagPolicy {
  id: string;
  name: string;
  description: string;
  scope: string[];
  requiredTags: string[];
  enforcement: 'audit' | 'warn' | 'deny';
  status: 'active' | 'draft' | 'disabled';
  violations: number;
  resources: number;
  lastEvaluated: string;
}

interface TagCompliance {
  id: string;
  resourceId: string;
  resourceName: string;
  resourceType: string;
  subscription: string;
  requiredTags: string[];
  appliedTags: string[];
  missingTags: string[];
  complianceScore: number;
  status: 'compliant' | 'partial' | 'non-compliant';
  lastChecked: string;
}

interface TagUsage {
  tagKey: string;
  count: number;
  percentage: number;
  trend: 'up' | 'down' | 'stable';
  topValues: { value: string; count: number }[];
}

export default function ResourceTags() {
  const [tags, setTags] = useState<ResourceTag[]>([]);
  const [policies, setPolicies] = useState<TagPolicy[]>([]);
  const [compliance, setCompliance] = useState<TagCompliance[]>([]);
  const [usage, setUsage] = useState<TagUsage[]>([]);
  const [viewMode, setViewMode] = useState<'tags' | 'policies' | 'compliance' | 'analytics'>('tags');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedTag, setExpandedTag] = useState<string | null>(null);

  useEffect(() => {
    // Initialize with tag data
    setTags([
      {
        id: 'TAG-001',
        key: 'Environment',
        value: 'Production|Staging|Development|Test',
        description: 'Deployment environment classification',
        required: true,
        category: 'technical',
        resourceTypes: ['All'],
        appliedCount: 2450,
        createdBy: 'Platform Team',
        createdDate: '2024-01-15',
        lastModified: '2 weeks ago',
        validationRule: '^(Production|Staging|Development|Test)$',
        defaultValue: 'Development'
      },
      {
        id: 'TAG-002',
        key: 'CostCenter',
        value: 'CC-[0-9]{4}',
        description: 'Financial cost center for billing',
        required: true,
        category: 'cost',
        resourceTypes: ['All'],
        appliedCount: 2380,
        createdBy: 'Finance Team',
        createdDate: '2024-01-10',
        lastModified: '1 month ago',
        validationRule: '^CC-\\d{4}$'
      },
      {
        id: 'TAG-003',
        key: 'Owner',
        value: 'Email address',
        description: 'Resource owner contact',
        required: true,
        category: 'business',
        resourceTypes: ['All'],
        appliedCount: 2420,
        createdBy: 'Governance Team',
        createdDate: '2024-01-05',
        lastModified: '3 weeks ago',
        validationRule: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
      },
      {
        id: 'TAG-004',
        key: 'Project',
        value: 'Project identifier',
        description: 'Associated project or initiative',
        required: false,
        category: 'business',
        resourceTypes: ['All'],
        appliedCount: 1850,
        createdBy: 'PMO',
        createdDate: '2024-02-01',
        lastModified: '1 week ago'
      },
      {
        id: 'TAG-005',
        key: 'DataClassification',
        value: 'Public|Internal|Confidential|Restricted',
        description: 'Data sensitivity classification',
        required: true,
        category: 'security',
        resourceTypes: ['Storage', 'Database'],
        appliedCount: 890,
        createdBy: 'Security Team',
        createdDate: '2024-01-20',
        lastModified: '4 days ago',
        validationRule: '^(Public|Internal|Confidential|Restricted)$',
        defaultValue: 'Internal'
      },
      {
        id: 'TAG-006',
        key: 'Compliance',
        value: 'GDPR|HIPAA|PCI|SOC2|None',
        description: 'Regulatory compliance requirements',
        required: false,
        category: 'compliance',
        resourceTypes: ['All'],
        appliedCount: 1250,
        createdBy: 'Compliance Team',
        createdDate: '2024-01-25',
        lastModified: '2 weeks ago',
        validationRule: '^(GDPR|HIPAA|PCI|SOC2|None)$'
      },
      {
        id: 'TAG-007',
        key: 'Department',
        value: 'Department name',
        description: 'Organizational department',
        required: true,
        category: 'business',
        resourceTypes: ['All'],
        appliedCount: 2400,
        createdBy: 'IT Admin',
        createdDate: '2024-01-08',
        lastModified: '3 weeks ago'
      },
      {
        id: 'TAG-008',
        key: 'BackupPolicy',
        value: 'Daily|Weekly|Monthly|None',
        description: 'Backup frequency requirement',
        required: false,
        category: 'technical',
        resourceTypes: ['VM', 'Database', 'Storage'],
        appliedCount: 750,
        createdBy: 'Operations Team',
        createdDate: '2024-02-10',
        lastModified: '5 days ago',
        validationRule: '^(Daily|Weekly|Monthly|None)$',
        defaultValue: 'Weekly'
      }
    ]);

    setPolicies([
      {
        id: 'POL-001',
        name: 'Mandatory Tagging Policy',
        description: 'Enforce required tags on all resources',
        scope: ['All Subscriptions'],
        requiredTags: ['Environment', 'CostCenter', 'Owner', 'Department'],
        enforcement: 'deny',
        status: 'active',
        violations: 45,
        resources: 2450,
        lastEvaluated: '1 hour ago'
      },
      {
        id: 'POL-002',
        name: 'Production Tagging Standards',
        description: 'Additional tags required for production resources',
        scope: ['Production'],
        requiredTags: ['DataClassification', 'BackupPolicy', 'Compliance'],
        enforcement: 'warn',
        status: 'active',
        violations: 12,
        resources: 580,
        lastEvaluated: '3 hours ago'
      },
      {
        id: 'POL-003',
        name: 'Cost Allocation Tags',
        description: 'Tags required for cost management',
        scope: ['All Subscriptions'],
        requiredTags: ['CostCenter', 'Project', 'Department'],
        enforcement: 'audit',
        status: 'active',
        violations: 120,
        resources: 2450,
        lastEvaluated: '30 minutes ago'
      },
      {
        id: 'POL-004',
        name: 'Security Classification',
        description: 'Data classification tags for sensitive resources',
        scope: ['Storage', 'Database'],
        requiredTags: ['DataClassification', 'Compliance'],
        enforcement: 'deny',
        status: 'draft',
        violations: 0,
        resources: 890,
        lastEvaluated: 'Not evaluated'
      }
    ]);

    setCompliance([
      {
        id: 'COMP-001',
        resourceId: 'vm-prod-web-01',
        resourceName: 'Production Web Server',
        resourceType: 'Virtual Machine',
        subscription: 'Production',
        requiredTags: ['Environment', 'CostCenter', 'Owner', 'Department'],
        appliedTags: ['Environment', 'Owner', 'Department'],
        missingTags: ['CostCenter'],
        complianceScore: 75,
        status: 'partial',
        lastChecked: '10 minutes ago'
      },
      {
        id: 'COMP-002',
        resourceId: 'storage-backup-01',
        resourceName: 'Backup Storage Account',
        resourceType: 'Storage',
        subscription: 'Production',
        requiredTags: ['Environment', 'CostCenter', 'Owner', 'DataClassification'],
        appliedTags: ['Environment', 'CostCenter', 'Owner', 'DataClassification'],
        missingTags: [],
        complianceScore: 100,
        status: 'compliant',
        lastChecked: '15 minutes ago'
      },
      {
        id: 'COMP-003',
        resourceId: 'db-analytics-01',
        resourceName: 'Analytics Database',
        resourceType: 'Database',
        subscription: 'Analytics',
        requiredTags: ['Environment', 'CostCenter', 'Owner', 'DataClassification'],
        appliedTags: ['Environment'],
        missingTags: ['CostCenter', 'Owner', 'DataClassification'],
        complianceScore: 25,
        status: 'non-compliant',
        lastChecked: '5 minutes ago'
      }
    ]);

    setUsage([
      {
        tagKey: 'Environment',
        count: 2450,
        percentage: 98,
        trend: 'stable',
        topValues: [
          { value: 'Production', count: 580 },
          { value: 'Development', count: 1200 },
          { value: 'Staging', count: 420 },
          { value: 'Test', count: 250 }
        ]
      },
      {
        tagKey: 'CostCenter',
        count: 2380,
        percentage: 95,
        trend: 'up',
        topValues: [
          { value: 'CC-1001', count: 650 },
          { value: 'CC-2001', count: 580 },
          { value: 'CC-3001', count: 720 },
          { value: 'CC-4001', count: 430 }
        ]
      },
      {
        tagKey: 'Owner',
        count: 2420,
        percentage: 97,
        trend: 'up',
        topValues: [
          { value: 'devops@company.com', count: 450 },
          { value: 'admin@company.com', count: 380 },
          { value: 'platform@company.com', count: 620 }
        ]
      },
      {
        tagKey: 'Department',
        count: 2400,
        percentage: 96,
        trend: 'stable',
        topValues: [
          { value: 'Engineering', count: 850 },
          { value: 'Operations', count: 620 },
          { value: 'Analytics', count: 480 },
          { value: 'Marketing', count: 450 }
        ]
      }
    ]);
  }, []);

  const getCategoryColor = (category: string) => {
    switch(category) {
      case 'cost': return 'text-green-500 bg-green-900/20';
      case 'compliance': return 'text-blue-500 bg-blue-900/20';
      case 'technical': return 'text-purple-500 bg-purple-900/20';
      case 'business': return 'text-yellow-500 bg-yellow-900/20';
      case 'security': return 'text-red-500 bg-red-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': case 'compliant': return 'text-green-500 bg-green-900/20';
      case 'draft': case 'partial': return 'text-yellow-500 bg-yellow-900/20';
      case 'disabled': case 'non-compliant': return 'text-red-500 bg-red-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const totalTags = tags.length;
  const requiredTags = tags.filter(t => t.required).length;
  const totalResources = 2500;
  const taggedResources = 2450;
  const complianceRate = Math.round((taggedResources / totalResources) * 100);
  const activePolices = policies.filter(p => p.status === 'active').length;

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold flex items-center space-x-2">
              <Tag className="w-6 h-6 text-blue-500" />
              <span>Resource Tags</span>
            </h1>
            <p className="text-sm text-gray-400 mt-1">Tag governance, policies, and compliance management</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
            
            <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export</span>
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Plus className="w-4 h-4" />
              <span>New Tag</span>
            </button>
          </div>
        </div>
      </header>

      {/* Summary Stats */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="text-center">
            <div className="text-xs text-gray-500">Total Tags</div>
            <div className="text-2xl font-bold">{totalTags}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Required</div>
            <div className="text-2xl font-bold text-orange-500">{requiredTags}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Resources</div>
            <div className="text-2xl font-bold">{totalResources}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Tagged</div>
            <div className="text-2xl font-bold text-green-500">{taggedResources}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Compliance</div>
            <div className="text-2xl font-bold text-blue-500">{complianceRate}%</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Policies</div>
            <div className="text-2xl font-bold">{activePolices}</div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-gray-900/30 border-b border-gray-800 px-6">
        <div className="flex space-x-6">
          {['tags', 'policies', 'compliance', 'analytics'].map(view => (
            <button
              key={view}
              onClick={() => setViewMode(view as any)}
              className={`py-3 border-b-2 text-sm capitalize ${
                viewMode === view 
                  ? 'border-blue-500 text-blue-500' 
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              {view === 'tags' ? 'Tag Definitions' : view}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {viewMode === 'tags' && (
          <div className="space-y-4">
            {/* Search and Filter */}
            <div className="flex items-center space-x-3 mb-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
                <input
                  type="text"
                  placeholder="Search tags..."
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
                <option value="cost">Cost</option>
                <option value="compliance">Compliance</option>
                <option value="technical">Technical</option>
                <option value="business">Business</option>
                <option value="security">Security</option>
              </select>
            </div>

            {/* Tags List */}
            {tags.map(tag => (
              <div key={tag.id} className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start space-x-3">
                      <Hash className="w-5 h-5 text-blue-500 mt-0.5" />
                      <div>
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className="text-sm font-bold">{tag.key}</h3>
                          {tag.required && (
                            <span className="px-2 py-1 bg-orange-900/20 text-orange-500 rounded text-xs">
                              Required
                            </span>
                          )}
                          <span className={`px-2 py-1 rounded text-xs ${getCategoryColor(tag.category)}`}>
                            {tag.category}
                          </span>
                        </div>
                        <p className="text-xs text-gray-400 mb-2">{tag.description}</p>
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <span>Applied to {tag.appliedCount} resources</span>
                          <span>Created by {tag.createdBy}</span>
                          <span>Modified {tag.lastModified}</span>
                        </div>
                      </div>
                    </div>
                    <button 
                      onClick={() => setExpandedTag(expandedTag === tag.id ? null : tag.id)}
                      className="p-1 hover:bg-gray-800 rounded"
                    >
                      <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${
                        expandedTag === tag.id ? 'rotate-180' : ''
                      }`} />
                    </button>
                  </div>

                  {expandedTag === tag.id && (
                    <div className="mt-4 pt-4 border-t border-gray-800 space-y-3">
                      <div className="grid grid-cols-2 gap-4 text-xs">
                        <div>
                          <div className="text-gray-500 mb-1">Allowed Values</div>
                          <div className="font-mono bg-gray-800 rounded p-2">{tag.value}</div>
                        </div>
                        {tag.validationRule && (
                          <div>
                            <div className="text-gray-500 mb-1">Validation Rule</div>
                            <div className="font-mono bg-gray-800 rounded p-2">{tag.validationRule}</div>
                          </div>
                        )}
                      </div>
                      
                      {tag.defaultValue && (
                        <div className="text-xs">
                          <div className="text-gray-500 mb-1">Default Value</div>
                          <div className="font-mono bg-gray-800 rounded p-2">{tag.defaultValue}</div>
                        </div>
                      )}
                      
                      <div className="text-xs">
                        <div className="text-gray-500 mb-1">Applicable Resource Types</div>
                        <div className="flex flex-wrap gap-2">
                          {tag.resourceTypes.map(type => (
                            <span key={type} className="px-2 py-1 bg-gray-800 rounded">
                              {type}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="flex space-x-2 mt-3">
                        <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                          Edit Tag
                        </button>
                        <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                          View Usage
                        </button>
                        <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                          Clone
                        </button>
                        <button className="px-3 py-1 bg-red-900/20 hover:bg-red-900/30 text-red-500 rounded text-xs">
                          Delete
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'policies' && (
          <div className="space-y-4">
            {policies.map(policy => (
              <div key={policy.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center space-x-2 mb-1">
                      <h3 className="text-sm font-bold">{policy.name}</h3>
                      <span className={`px-2 py-1 rounded text-xs ${getStatusColor(policy.status)}`}>
                        {policy.status}
                      </span>
                      <span className="px-2 py-1 bg-gray-800 rounded text-xs">
                        {policy.enforcement}
                      </span>
                    </div>
                    <p className="text-xs text-gray-400">{policy.description}</p>
                  </div>
                  <button className="p-1 hover:bg-gray-800 rounded">
                    <MoreVertical className="w-4 h-4 text-gray-500" />
                  </button>
                </div>

                <div className="grid grid-cols-4 gap-4 mb-3">
                  <div>
                    <div className="text-xs text-gray-500">Required Tags</div>
                    <div className="text-lg font-bold">{policy.requiredTags.length}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Resources</div>
                    <div className="text-lg font-bold">{policy.resources}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Violations</div>
                    <div className={`text-lg font-bold ${policy.violations > 0 ? 'text-red-500' : 'text-green-500'}`}>
                      {policy.violations}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Last Evaluated</div>
                    <div className="text-sm">{policy.lastEvaluated}</div>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex flex-wrap gap-2">
                    {policy.requiredTags.map(tag => (
                      <span key={tag} className="px-2 py-1 bg-gray-800 rounded text-xs">
                        {tag}
                      </span>
                    ))}
                  </div>
                  <div className="flex space-x-2">
                    <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                      Edit Policy
                    </button>
                    <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                      Evaluate Now
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'compliance' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-800">
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Resource</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Type</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Subscription</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Compliance</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Missing Tags</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Status</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Actions</th>
                </tr>
              </thead>
              <tbody>
                {compliance.map(item => (
                  <tr key={item.id} className="border-t border-gray-800 hover:bg-gray-800/30">
                    <td className="px-4 py-3 text-sm">{item.resourceName}</td>
                    <td className="px-4 py-3 text-sm">{item.resourceType}</td>
                    <td className="px-4 py-3 text-sm">{item.subscription}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center space-x-2">
                        <div className="w-24 h-2 bg-gray-800 rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${
                              item.complianceScore === 100 ? 'bg-green-500' :
                              item.complianceScore >= 75 ? 'bg-yellow-500' :
                              'bg-red-500'
                            }`}
                            style={{ width: `${item.complianceScore}%` }}
                          />
                        </div>
                        <span className="text-xs">{item.complianceScore}%</span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      {item.missingTags.length > 0 ? (
                        <div className="flex flex-wrap gap-1">
                          {item.missingTags.map(tag => (
                            <span key={tag} className="px-2 py-1 bg-red-900/20 text-red-500 rounded text-xs">
                              {tag}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs ${getStatusColor(item.status)}`}>
                        {item.status}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <button className="text-xs text-blue-500 hover:text-blue-400">
                        Fix Tags
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {viewMode === 'analytics' && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold mb-4">Tag Usage Distribution</h3>
                <div className="space-y-3">
                  {usage.map(item => (
                    <div key={item.tagKey}>
                      <div className="flex justify-between text-xs mb-1">
                        <span>{item.tagKey}</span>
                        <span>{item.percentage}%</span>
                      </div>
                      <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-blue-600 to-blue-400"
                          style={{ width: `${item.percentage}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold mb-4">Top Tag Values</h3>
                <div className="space-y-2">
                  {usage[0].topValues.map(value => (
                    <div key={value.value} className="flex items-center justify-between p-2 bg-gray-800 rounded">
                      <span className="text-sm font-mono">{value.value}</span>
                      <span className="text-sm text-gray-500">{value.count} resources</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Tag Coverage by Category</h3>
              <div className="grid grid-cols-5 gap-4">
                {['cost', 'compliance', 'technical', 'business', 'security'].map(category => {
                  const categoryTags = tags.filter(t => t.category === category);
                  const totalApplied = categoryTags.reduce((acc, t) => acc + t.appliedCount, 0);
                  return (
                    <div key={category} className="bg-gray-800 rounded p-3 text-center">
                      <div className={`text-xs mb-1 capitalize ${getCategoryColor(category).replace('bg-', 'text-').split(' ')[0]}`}>
                        {category}
                      </div>
                      <div className="text-2xl font-bold">{categoryTags.length}</div>
                      <div className="text-xs text-gray-500">tags</div>
                      <div className="text-xs text-gray-400 mt-1">{totalApplied} applications</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}