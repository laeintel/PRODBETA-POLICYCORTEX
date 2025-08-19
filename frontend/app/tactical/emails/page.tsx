'use client';

import React, { useState } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { 
  Mail, 
  Plus, 
  Search, 
  Filter, 
  Download, 
  Settings, 
  Edit3, 
  Trash2, 
  Eye, 
  Copy, 
  Send,
  BarChart3,
  Clock,
  Users,
  CheckCircle,
  FileText,
  Layout,
  Code,
  Smartphone,
  Monitor,
  Globe
} from 'lucide-react';

interface EmailTemplate {
  id: string;
  name: string;
  subject: string;
  category: string;
  status: 'active' | 'draft' | 'archived';
  type: 'alert' | 'notification' | 'report' | 'welcome' | 'reminder';
  lastModified: Date;
  createdBy: string;
  usageCount: number;
  openRate: number;
  clickRate: number;
  isDefault: boolean;
  variables: string[];
  preview: string;
  htmlContent?: string;
  textContent?: string;
}

const mockTemplates: EmailTemplate[] = [
  {
    id: '1',
    name: 'Critical Security Alert',
    subject: 'URGENT: Security Incident Detected - {{incident_id}}',
    category: 'Security',
    status: 'active',
    type: 'alert',
    lastModified: new Date('2024-01-20'),
    createdBy: 'security@company.com',
    usageCount: 1247,
    openRate: 98.5,
    clickRate: 87.2,
    isDefault: true,
    variables: ['incident_id', 'severity', 'resource_name', 'timestamp', 'action_required'],
    preview: 'A critical security incident has been detected in your Azure environment. Immediate action required.'
  },
  {
    id: '2',
    name: 'Policy Compliance Report',
    subject: 'Weekly Compliance Status - {{week_ending}}',
    category: 'Compliance',
    status: 'active',
    type: 'report',
    lastModified: new Date('2024-01-18'),
    createdBy: 'compliance@company.com',
    usageCount: 456,
    openRate: 92.1,
    clickRate: 74.3,
    isDefault: false,
    variables: ['week_ending', 'compliance_score', 'violations_count', 'recommendations'],
    preview: 'Your weekly compliance report is ready. Review your current compliance posture and recommendations.'
  },
  {
    id: '3',
    name: 'Cost Optimization Alert',
    subject: 'Cost Savings Opportunity - {{resource_type}}',
    category: 'Cost Management',
    status: 'active',
    type: 'notification',
    lastModified: new Date('2024-01-15'),
    createdBy: 'finops@company.com',
    usageCount: 234,
    openRate: 86.7,
    clickRate: 68.9,
    isDefault: false,
    variables: ['resource_type', 'current_cost', 'potential_savings', 'recommendation'],
    preview: 'We\'ve identified a cost optimization opportunity for your Azure resources.'
  },
  {
    id: '4',
    name: 'Welcome to PolicyCortex',
    subject: 'Welcome to PolicyCortex - Get Started Guide',
    category: 'Onboarding',
    status: 'active',
    type: 'welcome',
    lastModified: new Date('2024-01-10'),
    createdBy: 'support@company.com',
    usageCount: 89,
    openRate: 95.4,
    clickRate: 82.1,
    isDefault: true,
    variables: ['user_name', 'company_name', 'dashboard_url', 'support_email'],
    preview: 'Welcome to PolicyCortex! Let\'s get you started with your Azure governance journey.'
  },
  {
    id: '5',
    name: 'Resource Provisioning Notification',
    subject: 'New Resource Provisioned - {{resource_name}}',
    category: 'Infrastructure',
    status: 'draft',
    type: 'notification',
    lastModified: new Date('2024-01-08'),
    createdBy: 'devops@company.com',
    usageCount: 0,
    openRate: 0,
    clickRate: 0,
    isDefault: false,
    variables: ['resource_name', 'resource_type', 'location', 'provisioned_by', 'cost_estimate'],
    preview: 'A new Azure resource has been provisioned in your environment.'
  }
];

export default function Page() {
  const [templates, setTemplates] = useState<EmailTemplate[]>(mockTemplates);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [selectedTemplate, setSelectedTemplate] = useState<EmailTemplate | null>(null);
  const [previewMode, setPreviewMode] = useState<'desktop' | 'mobile'>('desktop');

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400 bg-green-400/10';
      case 'draft': return 'text-yellow-400 bg-yellow-400/10';
      case 'archived': return 'text-gray-400 bg-gray-400/10';
      default: return 'text-gray-400 bg-gray-400/10';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'alert': return 'text-red-400 bg-red-400/10';
      case 'notification': return 'text-blue-400 bg-blue-400/10';
      case 'report': return 'text-purple-400 bg-purple-400/10';
      case 'welcome': return 'text-green-400 bg-green-400/10';
      case 'reminder': return 'text-orange-400 bg-orange-400/10';
      default: return 'text-gray-400 bg-gray-400/10';
    }
  };

  const filteredTemplates = templates.filter(template => {
    const matchesSearch = template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         template.subject.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         template.category.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || template.status === statusFilter;
    const matchesCategory = categoryFilter === 'all' || template.category === categoryFilter;
    
    return matchesSearch && matchesStatus && matchesCategory;
  });

  const categories = [...new Set(templates.map(t => t.category))];

  const duplicateTemplate = (templateId: string) => {
    const template = templates.find(t => t.id === templateId);
    if (template) {
      const newTemplate = {
        ...template,
        id: Date.now().toString(),
        name: `${template.name} (Copy)`,
        status: 'draft' as const,
        usageCount: 0,
        isDefault: false,
        lastModified: new Date()
      };
      setTemplates([...templates, newTemplate]);
    }
  };

  const deleteTemplate = (templateId: string) => {
    setTemplates(templates.filter(t => t.id !== templateId));
  };

  return (
    <TacticalPageTemplate title="Email Templates" subtitle="Email Template Management & Analytics" icon={Mail}>
      <div className="space-y-6">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Templates</p>
                <p className="text-2xl font-bold text-white">{templates.length}</p>
              </div>
              <FileText className="w-8 h-8 text-blue-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400 flex items-center">
                <CheckCircle className="w-4 h-4 mr-1" />
                {templates.filter(t => t.status === 'active').length} Active
              </span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Sends</p>
                <p className="text-2xl font-bold text-white">
                  {templates.reduce((sum, t) => sum + t.usageCount, 0).toLocaleString()}
                </p>
              </div>
              <Send className="w-8 h-8 text-green-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400">+15.2% this month</span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Avg Open Rate</p>
                <p className="text-2xl font-bold text-white">
                  {templates.length > 0 
                    ? (templates.reduce((sum, t) => sum + t.openRate, 0) / templates.length).toFixed(1)
                    : '0'}%
                </p>
              </div>
              <Eye className="w-8 h-8 text-purple-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400">Industry leading</span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Avg Click Rate</p>
                <p className="text-2xl font-bold text-white">
                  {templates.length > 0 
                    ? (templates.reduce((sum, t) => sum + t.clickRate, 0) / templates.length).toFixed(1)
                    : '0'}%
                </p>
              </div>
              <BarChart3 className="w-8 h-8 text-orange-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400">Above benchmark</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search templates..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg pl-10 pr-4 py-2 w-full sm:w-80 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              
              <div className="flex gap-2">
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="draft">Draft</option>
                  <option value="archived">Archived</option>
                </select>
                
                <select
                  value={categoryFilter}
                  onChange={(e) => setCategoryFilter(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Categories</option>
                  {categories.map(category => (
                    <option key={category} value={category}>{category}</option>
                  ))}
                </select>
              </div>
            </div>
            
            <div className="flex gap-2">
              <button className="flex items-center space-x-2 bg-gray-800 hover:bg-gray-700 text-gray-300 px-4 py-2 rounded-lg border border-gray-700 transition-colors">
                <Download className="w-4 h-4" />
                <span>Export</span>
              </button>
              <button className="flex items-center space-x-2 bg-gray-800 hover:bg-gray-700 text-gray-300 px-4 py-2 rounded-lg border border-gray-700 transition-colors">
                <Settings className="w-4 h-4" />
                <span>Settings</span>
              </button>
              <button className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors">
                <Plus className="w-4 h-4" />
                <span>Create Template</span>
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Templates List */}
          <div className="lg:col-span-2">
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="px-6 py-4 border-b border-gray-800">
                <h3 className="text-lg font-semibold text-white">Email Templates</h3>
                <p className="text-sm text-gray-400 mt-1">{filteredTemplates.length} templates found</p>
              </div>
              <div className="divide-y divide-gray-800">
                {filteredTemplates.map((template) => (
                  <div 
                    key={template.id} 
                    className={`p-6 hover:bg-gray-800/50 cursor-pointer transition-colors ${selectedTemplate?.id === template.id ? 'bg-gray-800/50 border-l-4 border-blue-500' : ''}`}
                    onClick={() => setSelectedTemplate(template)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-3">
                          <h4 className="text-lg font-medium text-white truncate">{template.name}</h4>
                          {template.isDefault && (
                            <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-blue-900/30 text-blue-300 border border-blue-800">
                              Default
                            </span>
                          )}
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${getStatusColor(template.status)}`}>
                            {template.status}
                          </span>
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${getTypeColor(template.type)}`}>
                            {template.type}
                          </span>
                        </div>
                        <p className="text-sm text-gray-300 mt-1 font-medium">{template.subject}</p>
                        <p className="text-sm text-gray-400 mt-2 truncate">{template.preview}</p>
                        <div className="flex items-center justify-between mt-4">
                          <div className="flex items-center space-x-6 text-sm">
                            <span className="text-gray-400">
                              <Users className="w-4 h-4 inline mr-1" />
                              {template.usageCount} sends
                            </span>
                            <span className="text-gray-400">
                              <Eye className="w-4 h-4 inline mr-1" />
                              {template.openRate}% open
                            </span>
                            <span className="text-gray-400">
                              <BarChart3 className="w-4 h-4 inline mr-1" />
                              {template.clickRate}% click
                            </span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <button 
                              onClick={(e) => { e.stopPropagation(); duplicateTemplate(template.id); }}
                              className="text-gray-400 hover:text-blue-400 transition-colors"
                            >
                              <Copy className="w-4 h-4" />
                            </button>
                            <button className="text-gray-400 hover:text-yellow-400 transition-colors">
                              <Edit3 className="w-4 h-4" />
                            </button>
                            <button 
                              onClick={(e) => { e.stopPropagation(); deleteTemplate(template.id); }}
                              className="text-gray-400 hover:text-red-400 transition-colors"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-1 mt-3">
                      {template.variables.slice(0, 4).map((variable, idx) => (
                        <span key={idx} className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-700 text-gray-300 font-mono">
                          {`{{${variable}}}`}
                        </span>
                      ))}
                      {template.variables.length > 4 && (
                        <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-700 text-gray-300">
                          +{template.variables.length - 4} more
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              
              {filteredTemplates.length === 0 && (
                <div className="text-center py-12">
                  <Mail className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-300 mb-2">No templates found</h3>
                  <p className="text-gray-500">Try adjusting your search terms or filters.</p>
                </div>
              )}
            </div>
          </div>

          {/* Template Preview */}
          <div className="lg:col-span-1">
            <div className="bg-gray-900 border border-gray-800 rounded-lg sticky top-6">
              <div className="px-6 py-4 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">Preview</h3>
                  <div className="flex items-center space-x-2">
                    <button 
                      onClick={() => setPreviewMode('desktop')}
                      className={`p-2 rounded transition-colors ${previewMode === 'desktop' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
                    >
                      <Monitor className="w-4 h-4" />
                    </button>
                    <button 
                      onClick={() => setPreviewMode('mobile')}
                      className={`p-2 rounded transition-colors ${previewMode === 'mobile' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
                    >
                      <Smartphone className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
              
              {selectedTemplate ? (
                <div className="p-6">
                  <div className={`bg-white rounded-lg overflow-hidden ${previewMode === 'mobile' ? 'max-w-sm mx-auto' : ''}`}>
                    <div className="bg-gray-800 p-4 border-b">
                      <div className="text-xs text-gray-300 mb-1">Subject:</div>
                      <div className="text-sm font-medium text-white">{selectedTemplate.subject}</div>
                    </div>
                    <div className="p-4 text-gray-800 min-h-32">
                      <div className="text-sm leading-relaxed">
                        {selectedTemplate.preview}
                      </div>
                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <div className="text-xs text-gray-500">
                          Variables used: {selectedTemplate.variables.join(', ')}
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-6 space-y-4">
                    <div>
                      <div className="text-sm font-medium text-gray-300 mb-2">Template Details</div>
                      <div className="text-sm text-gray-400 space-y-1">
                        <div>Category: <span className="text-white">{selectedTemplate.category}</span></div>
                        <div>Type: <span className="text-white capitalize">{selectedTemplate.type}</span></div>
                        <div>Created by: <span className="text-white">{selectedTemplate.createdBy}</span></div>
                        <div>Last modified: <span className="text-white">{selectedTemplate.lastModified.toLocaleDateString()}</span></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-sm font-medium text-gray-300 mb-2">Performance</div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-400">Usage Count</span>
                          <span className="text-white">{selectedTemplate.usageCount}</span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-400">Open Rate</span>
                          <span className="text-green-400">{selectedTemplate.openRate}%</span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-400">Click Rate</span>
                          <span className="text-blue-400">{selectedTemplate.clickRate}%</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="pt-4 space-y-2">
                      <button className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm transition-colors">
                        Edit Template
                      </button>
                      <button className="w-full bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg text-sm transition-colors">
                        Test Send
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-6 text-center">
                  <Layout className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400">Select a template to preview</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Analytics Section */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg">
          <div className="px-6 py-4 border-b border-gray-800">
            <h3 className="text-lg font-semibold text-white">Template Analytics</h3>
            <p className="text-sm text-gray-400 mt-1">Performance metrics for all email templates</p>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-400 mb-2">
                  {templates.filter(t => t.status === 'active').length}
                </div>
                <div className="text-sm text-gray-400">Active Templates</div>
                <div className="text-xs text-green-400 mt-1">+2 this week</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-400 mb-2">
                  {(templates.reduce((sum, t) => sum + t.openRate, 0) / templates.length).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-400">Average Open Rate</div>
                <div className="text-xs text-green-400 mt-1">+3.2% vs last month</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-400 mb-2">
                  {(templates.reduce((sum, t) => sum + t.clickRate, 0) / templates.length).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-400">Average Click Rate</div>
                <div className="text-xs text-green-400 mt-1">+1.8% vs last month</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </TacticalPageTemplate>
  );
}