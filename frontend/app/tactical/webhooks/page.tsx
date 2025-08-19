'use client';

import React, { useState } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { 
  Link2, 
  Plus, 
  Search, 
  Filter, 
  Download, 
  Settings, 
  Edit3, 
  Trash2, 
  Eye, 
  Send,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  Activity,
  BarChart3,
  Globe,
  Shield,
  Zap,
  Code,
  Server,
  Webhook,
  RefreshCw,
  PlayCircle,
  PauseCircle
} from 'lucide-react';

interface WebhookEndpoint {
  id: string;
  name: string;
  url: string;
  description: string;
  status: 'active' | 'paused' | 'failed' | 'testing';
  method: 'POST' | 'PUT' | 'PATCH';
  events: string[];
  headers: Record<string, string>;
  retryPolicy: {
    maxAttempts: number;
    backoffMultiplier: number;
    initialDelay: number;
  };
  lastTriggered?: Date;
  successRate: number;
  totalRequests: number;
  failedRequests: number;
  avgResponseTime: number;
  createdAt: Date;
  createdBy: string;
}

const mockWebhooks: WebhookEndpoint[] = [
  {
    id: '1',
    name: 'Security Incident Webhook',
    url: 'https://api.security-system.com/webhooks/incidents',
    description: 'Sends critical security incident notifications to external security platform',
    status: 'active',
    method: 'POST',
    events: ['security.incident.created', 'security.breach.detected', 'security.alert.critical'],
    headers: {
      'Authorization': 'Bearer token-***',
      'Content-Type': 'application/json',
      'X-Source': 'PolicyCortex'
    },
    retryPolicy: {
      maxAttempts: 5,
      backoffMultiplier: 2,
      initialDelay: 1000
    },
    lastTriggered: new Date('2024-01-20T10:30:00'),
    successRate: 98.7,
    totalRequests: 1247,
    failedRequests: 16,
    avgResponseTime: 245,
    createdAt: new Date('2024-01-01'),
    createdBy: 'security@company.com'
  },
  {
    id: '2',
    name: 'Compliance Alert Webhook',
    url: 'https://compliance-dashboard.company.com/api/webhooks',
    description: 'Notifications for policy compliance violations and drift detection',
    status: 'active',
    method: 'POST',
    events: ['compliance.violation.detected', 'policy.drift.identified', 'governance.alert'],
    headers: {
      'Authorization': 'API-Key api-key-***',
      'Content-Type': 'application/json'
    },
    retryPolicy: {
      maxAttempts: 3,
      backoffMultiplier: 1.5,
      initialDelay: 500
    },
    lastTriggered: new Date('2024-01-20T09:15:00'),
    successRate: 95.2,
    totalRequests: 634,
    failedRequests: 30,
    avgResponseTime: 180,
    createdAt: new Date('2024-01-05'),
    createdBy: 'compliance@company.com'
  },
  {
    id: '3',
    name: 'Cost Management Webhook',
    url: 'https://finops-platform.company.com/webhooks/cost-alerts',
    description: 'Budget threshold alerts and cost optimization recommendations',
    status: 'paused',
    method: 'POST',
    events: ['cost.threshold.exceeded', 'budget.alert', 'optimization.recommended'],
    headers: {
      'X-API-Key': 'key-***',
      'Content-Type': 'application/json'
    },
    retryPolicy: {
      maxAttempts: 2,
      backoffMultiplier: 2,
      initialDelay: 1000
    },
    lastTriggered: new Date('2024-01-19T14:20:00'),
    successRate: 92.5,
    totalRequests: 189,
    failedRequests: 14,
    avgResponseTime: 320,
    createdAt: new Date('2024-01-10'),
    createdBy: 'finops@company.com'
  }
];

export default function Page() {
  const [webhooks, setWebhooks] = useState<WebhookEndpoint[]>(mockWebhooks);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [selectedWebhook, setSelectedWebhook] = useState<WebhookEndpoint | null>(null);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400 bg-green-400/10';
      case 'paused': return 'text-yellow-400 bg-yellow-400/10';
      case 'failed': return 'text-red-400 bg-red-400/10';
      case 'testing': return 'text-blue-400 bg-blue-400/10';
      default: return 'text-gray-400 bg-gray-400/10';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle className="w-4 h-4" />;
      case 'failed': return <XCircle className="w-4 h-4" />;
      case 'paused': return <PauseCircle className="w-4 h-4" />;
      case 'testing': return <PlayCircle className="w-4 h-4" />;
      default: return <AlertTriangle className="w-4 h-4" />;
    }
  };

  const filteredWebhooks = webhooks.filter(webhook => {
    const matchesSearch = webhook.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         webhook.url.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         webhook.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || webhook.status === statusFilter;
    
    return matchesSearch && matchesStatus;
  });

  const toggleWebhookStatus = (webhookId: string) => {
    setWebhooks(prev => prev.map(webhook => 
      webhook.id === webhookId 
        ? { ...webhook, status: webhook.status === 'active' ? 'paused' : 'active' }
        : webhook
    ));
  };

  const testWebhook = (webhookId: string) => {
    setWebhooks(prev => prev.map(webhook => 
      webhook.id === webhookId 
        ? { ...webhook, status: 'testing' }
        : webhook
    ));
    
    setTimeout(() => {
      setWebhooks(prev => prev.map(webhook => 
        webhook.id === webhookId 
          ? { ...webhook, status: 'active', lastTriggered: new Date() }
          : webhook
      ));
    }, 2000);
  };

  const totalWebhooks = webhooks.length;
  const activeWebhooks = webhooks.filter(w => w.status === 'active').length;
  const avgSuccessRate = webhooks.length > 0 ? webhooks.reduce((sum, w) => sum + w.successRate, 0) / webhooks.length : 0;
  const avgResponseTime = webhooks.length > 0 ? webhooks.reduce((sum, w) => sum + w.avgResponseTime, 0) / webhooks.length : 0;

  return (
    <TacticalPageTemplate title="Webhooks" subtitle="Webhook Endpoint Management & Monitoring" icon={Link2}>
      <div className="space-y-6">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Webhooks</p>
                <p className="text-2xl font-bold text-white">{totalWebhooks}</p>
              </div>
              <Webhook className="w-8 h-8 text-blue-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400 flex items-center">
                <CheckCircle className="w-4 h-4 mr-1" />
                {activeWebhooks} Active
              </span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Success Rate</p>
                <p className="text-2xl font-bold text-white">{avgSuccessRate.toFixed(1)}%</p>
              </div>
              <BarChart3 className="w-8 h-8 text-green-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400">Excellent performance</span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Avg Response Time</p>
                <p className="text-2xl font-bold text-white">{avgResponseTime.toFixed(0)}ms</p>
              </div>
              <Activity className="w-8 h-8 text-purple-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400">-15ms vs last week</span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Requests</p>
                <p className="text-2xl font-bold text-white">{webhooks.reduce((sum, w) => sum + w.totalRequests, 0).toLocaleString()}</p>
              </div>
              <Send className="w-8 h-8 text-orange-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400">+23% this week</span>
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
                  placeholder="Search webhooks..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg pl-10 pr-4 py-2 w-full sm:w-80 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="paused">Paused</option>
                <option value="failed">Failed</option>
                <option value="testing">Testing</option>
              </select>
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
                <span>Create Webhook</span>
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Webhooks List */}
          <div className="lg:col-span-2">
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="px-6 py-4 border-b border-gray-800">
                <h3 className="text-lg font-semibold text-white">Webhook Endpoints</h3>
                <p className="text-sm text-gray-400 mt-1">{filteredWebhooks.length} endpoints configured</p>
              </div>
              <div className="divide-y divide-gray-800">
                {filteredWebhooks.map((webhook) => (
                  <div 
                    key={webhook.id} 
                    className={`p-6 hover:bg-gray-800/50 cursor-pointer transition-colors ${
                      selectedWebhook?.id === webhook.id ? 'bg-gray-800/50 border-l-4 border-blue-500' : ''
                    }`}
                    onClick={() => setSelectedWebhook(webhook)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-3">
                          <h4 className="text-lg font-medium text-white truncate">{webhook.name}</h4>
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${getStatusColor(webhook.status)}`}>
                            {getStatusIcon(webhook.status)}
                            <span className="ml-1">{webhook.status}</span>
                          </span>
                          <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-700 text-gray-300 font-mono">
                            {webhook.method}
                          </span>
                        </div>
                        <p className="text-sm text-gray-400 mt-1 truncate">{webhook.description}</p>
                        <p className="text-sm text-blue-400 mt-1 truncate font-mono">{webhook.url}</p>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                          <div>
                            <div className="text-xs font-medium text-gray-400">Success Rate</div>
                            <div className="text-sm font-bold text-green-400">{webhook.successRate.toFixed(1)}%</div>
                          </div>
                          <div>
                            <div className="text-xs font-medium text-gray-400">Total Requests</div>
                            <div className="text-sm font-bold text-white">{webhook.totalRequests.toLocaleString()}</div>
                          </div>
                          <div>
                            <div className="text-xs font-medium text-gray-400">Avg Response</div>
                            <div className="text-sm font-bold text-purple-400">{webhook.avgResponseTime}ms</div>
                          </div>
                          <div>
                            <div className="text-xs font-medium text-gray-400">Events</div>
                            <div className="text-sm font-bold text-blue-400">{webhook.events.length}</div>
                          </div>
                        </div>
                        
                        <div className="flex flex-wrap gap-1 mt-3">
                          {webhook.events.slice(0, 3).map((event, idx) => (
                            <span key={idx} className="inline-flex items-center px-2 py-1 rounded text-xs bg-blue-900/30 text-blue-300 border border-blue-800">
                              {event}
                            </span>
                          ))}
                          {webhook.events.length > 3 && (
                            <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-700 text-gray-300">
                              +{webhook.events.length - 3} more
                            </span>
                          )}
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2 ml-4">
                        <button 
                          onClick={(e) => { e.stopPropagation(); testWebhook(webhook.id); }}
                          className="text-gray-400 hover:text-green-400 transition-colors"
                          title="Test webhook"
                        >
                          <PlayCircle className="w-4 h-4" />
                        </button>
                        <button 
                          onClick={(e) => { e.stopPropagation(); toggleWebhookStatus(webhook.id); }}
                          className="text-gray-400 hover:text-yellow-400 transition-colors"
                          title="Toggle status"
                        >
                          {webhook.status === 'active' ? <PauseCircle className="w-4 h-4" /> : <PlayCircle className="w-4 h-4" />}
                        </button>
                        <button className="text-gray-400 hover:text-blue-400 transition-colors" title="Edit webhook">
                          <Edit3 className="w-4 h-4" />
                        </button>
                        <button className="text-gray-400 hover:text-red-400 transition-colors" title="Delete webhook">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    
                    {webhook.lastTriggered && (
                      <div className="mt-4 text-xs text-gray-400">
                        Last triggered: {webhook.lastTriggered.toLocaleString()}
                      </div>
                    )}
                  </div>
                ))}
              </div>
              
              {filteredWebhooks.length === 0 && (
                <div className="text-center py-12">
                  <Webhook className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-300 mb-2">No webhooks found</h3>
                  <p className="text-gray-500">Try adjusting your search terms or filters.</p>
                </div>
              )}
            </div>
          </div>

          {/* Webhook Details */}
          <div className="lg:col-span-1">
            <div className="bg-gray-900 border border-gray-800 rounded-lg sticky top-6">
              <div className="px-6 py-4 border-b border-gray-800">
                <h3 className="text-lg font-semibold text-white">Webhook Details</h3>
              </div>
              
              {selectedWebhook ? (
                <div className="p-6 space-y-6">
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Configuration</h4>
                    <div className="text-sm text-gray-400 space-y-2">
                      <div className="flex justify-between">
                        <span>Method:</span>
                        <span className="text-white font-mono">{selectedWebhook.method}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Status:</span>
                        <span className={`capitalize ${selectedWebhook.status === 'active' ? 'text-green-400' : selectedWebhook.status === 'failed' ? 'text-red-400' : 'text-yellow-400'}`}>
                          {selectedWebhook.status}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Created by:</span>
                        <span className="text-white">{selectedWebhook.createdBy}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Created:</span>
                        <span className="text-white">{selectedWebhook.createdAt.toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Performance</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">Success Rate</span>
                        <span className="text-green-400 font-bold">{selectedWebhook.successRate.toFixed(1)}%</span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">Total Requests</span>
                        <span className="text-white">{selectedWebhook.totalRequests.toLocaleString()}</span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">Failed Requests</span>
                        <span className="text-red-400">{selectedWebhook.failedRequests}</span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">Avg Response Time</span>
                        <span className="text-purple-400">{selectedWebhook.avgResponseTime}ms</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Retry Policy</h4>
                    <div className="text-sm text-gray-400 space-y-1">
                      <div>Max attempts: <span className="text-white">{selectedWebhook.retryPolicy.maxAttempts}</span></div>
                      <div>Initial delay: <span className="text-white">{selectedWebhook.retryPolicy.initialDelay}ms</span></div>
                      <div>Backoff multiplier: <span className="text-white">{selectedWebhook.retryPolicy.backoffMultiplier}x</span></div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Events ({selectedWebhook.events.length})</h4>
                    <div className="space-y-1">
                      {selectedWebhook.events.map((event, idx) => (
                        <div key={idx} className="text-xs bg-gray-800 rounded px-2 py-1 text-blue-300 font-mono">
                          {event}
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Headers</h4>
                    <div className="space-y-1">
                      {Object.entries(selectedWebhook.headers).map(([key, value]) => (
                        <div key={key} className="text-xs">
                          <span className="text-gray-400">{key}:</span>
                          <span className="text-white ml-2 font-mono">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="pt-4 space-y-2">
                    <button 
                      onClick={() => testWebhook(selectedWebhook.id)}
                      className="w-full bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg text-sm transition-colors flex items-center justify-center"
                    >
                      <PlayCircle className="w-4 h-4 mr-2" />
                      Test Webhook
                    </button>
                    <button className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm transition-colors">
                      Edit Configuration
                    </button>
                  </div>
                </div>
              ) : (
                <div className="p-6 text-center">
                  <Webhook className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400">Select a webhook to view details</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg">
          <div className="px-6 py-4 border-b border-gray-800">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">Recent Webhook Activity</h3>
              <div className="flex items-center space-x-2 text-sm text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Live</span>
              </div>
            </div>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white">
                    <span className="font-medium">Security Incident Webhook</span> triggered successfully
                  </p>
                  <p className="text-xs text-gray-400 mt-1">2 minutes ago • 200 OK • 245ms</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white">
                    <span className="font-medium">Compliance Alert Webhook</span> delivered
                  </p>
                  <p className="text-xs text-gray-400 mt-1">5 minutes ago • 200 OK • 180ms</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-red-400 rounded-full mt-2 flex-shrink-0"></div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white">
                    <span className="font-medium">Cost Management Webhook</span> failed delivery
                  </p>
                  <p className="text-xs text-gray-400 mt-1">8 minutes ago • 500 Error • Retrying in 2m</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </TacticalPageTemplate>
  );
}