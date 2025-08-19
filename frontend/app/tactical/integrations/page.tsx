'use client';

import React, { useState, useEffect } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { 
  Plug, 
  Plus, 
  Search, 
  Filter,
  MoreVertical,
  Edit,
  Trash2,
  Shield,
  Settings,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  Activity,
  RefreshCw,
  Download,
  Upload,
  Eye,
  EyeOff,
  Link,
  Unlink,
  Zap,
  Database,
  Cloud,
  Server,
  Globe,
  Key,
  Lock,
  Unlock,
  Star,
  Calendar,
  Users,
  BarChart3,
  TrendingUp,
  MessageCircle,
  Mail,
  Bell,
  Webhook,
  Code,
  Terminal,
  Github,
  Slack,
  Chrome,
  Smartphone
} from 'lucide-react';
import { Button } from '../../../components/ui/button';
import { Input } from '../../../components/ui/input';
import { Badge } from '../../../components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../components/ui/tabs';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '../../../components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../../components/ui/select';
import { Switch } from '../../../components/ui/switch';
import { Label } from '../../../components/ui/label';
import { Textarea } from '../../../components/ui/textarea';
import { Progress } from '../../../components/ui/progress';

interface Integration {
  id: string;
  name: string;
  displayName: string;
  description: string;
  provider: string;
  category: 'cloud' | 'communication' | 'security' | 'monitoring' | 'automation' | 'analytics' | 'storage';
  status: 'connected' | 'disconnected' | 'error' | 'configuring' | 'authenticating';
  type: 'oauth' | 'api_key' | 'webhook' | 'direct' | 'saml' | 'ldap';
  version: string;
  lastSync: string;
  lastActivity: string;
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  configuration: {
    endpoint?: string;
    credentials?: {
      clientId?: string;
      apiKey?: string;
      username?: string;
      tokenExpiry?: string;
    };
    settings: Record<string, any>;
  };
  features: string[];
  permissions: string[];
  usage: {
    totalRequests: number;
    successfulRequests: number;
    failedRequests: number;
    averageResponseTime: number;
    dataTransferred: number;
  };
  healthCheck: {
    lastCheck: string;
    status: 'healthy' | 'warning' | 'unhealthy';
    responseTime: number;
    message?: string;
  };
  tags: string[];
}

interface IntegrationTemplate {
  id: string;
  name: string;
  displayName: string;
  description: string;
  provider: string;
  category: 'cloud' | 'communication' | 'security' | 'monitoring' | 'automation' | 'analytics' | 'storage';
  type: 'oauth' | 'api_key' | 'webhook' | 'direct' | 'saml' | 'ldap';
  icon: React.ElementType;
  features: string[];
  setupSteps: string[];
  documentation?: string;
  popular: boolean;
}

const mockIntegrations: Integration[] = [
  {
    id: 'int-1',
    name: 'azure-ad',
    displayName: 'Azure Active Directory',
    description: 'Microsoft Azure Active Directory integration for user authentication and management',
    provider: 'Microsoft',
    category: 'security',
    status: 'connected',
    type: 'oauth',
    version: '2.0',
    lastSync: '2024-01-20T10:25:00Z',
    lastActivity: '2024-01-20T10:30:00Z',
    createdAt: '2023-06-15T08:00:00Z',
    updatedAt: '2024-01-15T14:20:00Z',
    createdBy: 'admin@policycortex.com',
    configuration: {
      endpoint: 'https://login.microsoftonline.com/tenant-id',
      credentials: {
        clientId: 'abc123-def456-ghi789',
        tokenExpiry: '2024-06-15T08:00:00Z'
      },
      settings: {
        tenantId: '9ef5b184-d371-462a-bc75-5024ce8baff7',
        syncInterval: 3600,
        groupMapping: true,
        autoProvisioning: true
      }
    },
    features: ['Single Sign-On', 'User Provisioning', 'Group Sync', 'Multi-Factor Authentication'],
    permissions: ['read:users', 'read:groups', 'manage:authentication'],
    usage: {
      totalRequests: 15847,
      successfulRequests: 15792,
      failedRequests: 55,
      averageResponseTime: 245,
      dataTransferred: 2847392
    },
    healthCheck: {
      lastCheck: '2024-01-20T10:30:00Z',
      status: 'healthy',
      responseTime: 198,
      message: 'All systems operational'
    },
    tags: ['authentication', 'microsoft', 'sso', 'production']
  },
  {
    id: 'int-2',
    name: 'slack',
    displayName: 'Slack Workspace',
    description: 'Slack integration for notifications and team collaboration',
    provider: 'Slack Technologies',
    category: 'communication',
    status: 'connected',
    type: 'oauth',
    version: '1.0',
    lastSync: '2024-01-20T10:20:00Z',
    lastActivity: '2024-01-20T10:28:00Z',
    createdAt: '2023-08-10T14:30:00Z',
    updatedAt: '2024-01-12T09:15:00Z',
    createdBy: 'notifications@policycortex.com',
    configuration: {
      endpoint: 'https://hooks.slack.com/services/webhook-url',
      credentials: {
        apiKey: 'xoxb-****-****-****'
      },
      settings: {
        defaultChannel: '#alerts',
        mentionUsers: true,
        enableThreads: false,
        messageFormat: 'detailed'
      }
    },
    features: ['Real-time Notifications', 'Channel Management', 'User Mentions', 'File Sharing'],
    permissions: ['send:messages', 'read:channels', 'manage:webhooks'],
    usage: {
      totalRequests: 3924,
      successfulRequests: 3891,
      failedRequests: 33,
      averageResponseTime: 145,
      dataTransferred: 489273
    },
    healthCheck: {
      lastCheck: '2024-01-20T10:28:00Z',
      status: 'healthy',
      responseTime: 134,
      message: 'Webhook responding normally'
    },
    tags: ['notifications', 'slack', 'communication', 'alerts']
  },
  {
    id: 'int-3',
    name: 'datadog',
    displayName: 'Datadog Monitoring',
    description: 'Datadog integration for infrastructure monitoring and alerting',
    provider: 'Datadog',
    category: 'monitoring',
    status: 'connected',
    type: 'api_key',
    version: '2.0',
    lastSync: '2024-01-20T10:15:00Z',
    lastActivity: '2024-01-20T10:25:00Z',
    createdAt: '2023-07-22T11:45:00Z',
    updatedAt: '2024-01-08T16:30:00Z',
    createdBy: 'devops@policycortex.com',
    configuration: {
      endpoint: 'https://api.datadoghq.com/api/v1',
      credentials: {
        apiKey: 'dd_api_****'
      },
      settings: {
        site: 'datadoghq.com',
        namespace: 'policycortex',
        enableMetrics: true,
        enableLogs: true,
        enableTraces: false
      }
    },
    features: ['Metrics Collection', 'Log Aggregation', 'Custom Dashboards', 'Alerting'],
    permissions: ['send:metrics', 'send:logs', 'read:dashboards'],
    usage: {
      totalRequests: 48392,
      successfulRequests: 47981,
      failedRequests: 411,
      averageResponseTime: 89,
      dataTransferred: 15847293
    },
    healthCheck: {
      lastCheck: '2024-01-20T10:25:00Z',
      status: 'healthy',
      responseTime: 67,
      message: 'API responding within normal parameters'
    },
    tags: ['monitoring', 'metrics', 'datadog', 'infrastructure']
  },
  {
    id: 'int-4',
    name: 'github',
    displayName: 'GitHub Repository',
    description: 'GitHub integration for source code management and CI/CD workflows',
    provider: 'GitHub',
    category: 'automation',
    status: 'error',
    type: 'oauth',
    version: '4.0',
    lastSync: '2024-01-19T15:20:00Z',
    lastActivity: '2024-01-19T15:22:00Z',
    createdAt: '2023-09-05T13:10:00Z',
    updatedAt: '2024-01-19T15:25:00Z',
    createdBy: 'dev@policycortex.com',
    configuration: {
      endpoint: 'https://api.github.com',
      credentials: {
        clientId: 'github_app_123456'
      },
      settings: {
        organization: 'policycortex',
        repositories: ['policy-engine', 'frontend', 'api'],
        webhookEvents: ['push', 'pull_request', 'release'],
        autoDeployment: false
      }
    },
    features: ['Repository Management', 'Webhook Events', 'CI/CD Integration', 'Release Tracking'],
    permissions: ['read:repos', 'write:webhooks', 'read:org'],
    usage: {
      totalRequests: 2847,
      successfulRequests: 2634,
      failedRequests: 213,
      averageResponseTime: 298,
      dataTransferred: 847392
    },
    healthCheck: {
      lastCheck: '2024-01-19T15:22:00Z',
      status: 'unhealthy',
      responseTime: 5000,
      message: 'Authentication token expired - requires renewal'
    },
    tags: ['github', 'automation', 'cicd', 'source-control', 'error']
  },
  {
    id: 'int-5',
    name: 'aws-s3',
    displayName: 'Amazon S3 Storage',
    description: 'AWS S3 integration for secure document and backup storage',
    provider: 'Amazon Web Services',
    category: 'storage',
    status: 'configuring',
    type: 'api_key',
    version: '2006-03-01',
    lastSync: '2024-01-20T08:45:00Z',
    lastActivity: '2024-01-20T09:12:00Z',
    createdAt: '2024-01-18T10:30:00Z',
    updatedAt: '2024-01-20T09:15:00Z',
    createdBy: 'storage@policycortex.com',
    configuration: {
      endpoint: 'https://s3.amazonaws.com',
      credentials: {
        apiKey: 'AKIA****'
      },
      settings: {
        region: 'us-west-2',
        bucketName: 'policycortex-backups',
        encryption: 'AES256',
        versioning: true,
        lifecycle: '90d'
      }
    },
    features: ['File Storage', 'Backup Management', 'Versioning', 'Lifecycle Policies'],
    permissions: ['read:objects', 'write:objects', 'delete:objects'],
    usage: {
      totalRequests: 445,
      successfulRequests: 423,
      failedRequests: 22,
      averageResponseTime: 156,
      dataTransferred: 2847392847
    },
    healthCheck: {
      lastCheck: '2024-01-20T09:12:00Z',
      status: 'warning',
      responseTime: 234,
      message: 'Configuration in progress - bucket permissions being validated'
    },
    tags: ['aws', 's3', 'storage', 'backup', 'new']
  }
];

const mockTemplates: IntegrationTemplate[] = [
  {
    id: 'template-1',
    name: 'azure-devops',
    displayName: 'Azure DevOps',
    description: 'Integrate with Azure DevOps for project management and CI/CD pipelines',
    provider: 'Microsoft',
    category: 'automation',
    type: 'oauth',
    icon: Shield,
    features: ['Pipeline Integration', 'Work Item Tracking', 'Repository Management', 'Release Management'],
    setupSteps: [
      'Create Azure DevOps application',
      'Configure OAuth permissions',
      'Set up webhook endpoints',
      'Test connection and sync'
    ],
    documentation: 'https://docs.microsoft.com/azure-devops',
    popular: true
  },
  {
    id: 'template-2',
    name: 'jira',
    displayName: 'Atlassian Jira',
    description: 'Connect with Jira for issue tracking and project management',
    provider: 'Atlassian',
    category: 'automation',
    type: 'oauth',
    icon: Globe,
    features: ['Issue Tracking', 'Project Management', 'Custom Fields', 'Workflow Integration'],
    setupSteps: [
      'Install Jira app',
      'Configure authentication',
      'Map issue types and fields',
      'Set up synchronization rules'
    ],
    documentation: 'https://developer.atlassian.com/cloud/jira',
    popular: true
  },
  {
    id: 'template-3',
    name: 'prometheus',
    displayName: 'Prometheus Monitoring',
    description: 'Connect to Prometheus for metrics collection and monitoring',
    provider: 'Prometheus',
    category: 'monitoring',
    type: 'direct',
    icon: BarChart3,
    features: ['Metrics Collection', 'Alerting', 'Query Language', 'Service Discovery'],
    setupSteps: [
      'Configure Prometheus endpoint',
      'Set up authentication if required',
      'Define metric collection rules',
      'Test connectivity and queries'
    ],
    documentation: 'https://prometheus.io/docs',
    popular: false
  },
  {
    id: 'template-4',
    name: 'sendgrid',
    displayName: 'SendGrid Email',
    description: 'Email delivery service integration for notifications and communications',
    provider: 'SendGrid',
    category: 'communication',
    type: 'api_key',
    icon: Mail,
    features: ['Email Delivery', 'Template Management', 'Analytics', 'Suppression Management'],
    setupSteps: [
      'Create SendGrid API key',
      'Configure sender authentication',
      'Set up email templates',
      'Test email delivery'
    ],
    documentation: 'https://docs.sendgrid.com',
    popular: true
  }
];

export default function IntegrationsPage() {
  const [integrations, setIntegrations] = useState<Integration[]>(mockIntegrations);
  const [templates, setTemplates] = useState<IntegrationTemplate[]>(mockTemplates);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [activeTab, setActiveTab] = useState('integrations');
  const [selectedIntegration, setSelectedIntegration] = useState<Integration | null>(null);
  const [isViewingDetails, setIsViewingDetails] = useState(false);
  const [isConfiguringIntegration, setIsConfiguringIntegration] = useState(false);

  const filteredIntegrations = integrations.filter(integration => {
    const matchesSearch = 
      integration.displayName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      integration.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      integration.provider.toLowerCase().includes(searchTerm.toLowerCase()) ||
      integration.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesStatus = statusFilter === 'all' || integration.status === statusFilter;
    const matchesCategory = categoryFilter === 'all' || integration.category === categoryFilter;
    const matchesType = typeFilter === 'all' || integration.type === typeFilter;

    return matchesSearch && matchesStatus && matchesCategory && matchesType;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'bg-green-500/20 text-green-400 border-green-500/20';
      case 'disconnected': return 'bg-gray-500/20 text-gray-400 border-gray-500/20';
      case 'error': return 'bg-red-500/20 text-red-400 border-red-500/20';
      case 'configuring': return 'bg-blue-500/20 text-blue-400 border-blue-500/20';
      case 'authenticating': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/20';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return <CheckCircle className="w-4 h-4" />;
      case 'disconnected': return <XCircle className="w-4 h-4" />;
      case 'error': return <AlertTriangle className="w-4 h-4" />;
      case 'configuring': return <Settings className="w-4 h-4 animate-spin" />;
      case 'authenticating': return <Lock className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'cloud': return <Cloud className="w-5 h-5" />;
      case 'communication': return <MessageCircle className="w-5 h-5" />;
      case 'security': return <Shield className="w-5 h-5" />;
      case 'monitoring': return <BarChart3 className="w-5 h-5" />;
      case 'automation': return <Zap className="w-5 h-5" />;
      case 'analytics': return <TrendingUp className="w-5 h-5" />;
      case 'storage': return <Database className="w-5 h-5" />;
      default: return <Plug className="w-5 h-5" />;
    }
  };

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'unhealthy': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const handleTestConnection = (integrationId: string) => {
    setIntegrations(integrations.map(int => 
      int.id === integrationId 
        ? { 
            ...int, 
            status: 'authenticating' as const,
            healthCheck: {
              ...int.healthCheck,
              lastCheck: new Date().toISOString(),
              status: 'healthy' as const,
              message: 'Connection test successful'
            }
          }
        : int
    ));

    // Simulate test completion
    setTimeout(() => {
      setIntegrations(integrations.map(int => 
        int.id === integrationId 
          ? { ...int, status: 'connected' as const }
          : int
      ));
    }, 2000);
  };

  const handleDisconnect = (integrationId: string) => {
    setIntegrations(integrations.map(int => 
      int.id === integrationId 
        ? { ...int, status: 'disconnected' as const }
        : int
    ));
  };

  const connectedIntegrations = integrations.filter(i => i.status === 'connected').length;
  const errorIntegrations = integrations.filter(i => i.status === 'error').length;
  const totalRequests = integrations.reduce((sum, int) => sum + int.usage.totalRequests, 0);
  const avgResponseTime = integrations.length > 0 
    ? Math.round(integrations.reduce((sum, int) => sum + int.usage.averageResponseTime, 0) / integrations.length)
    : 0;

  const content = (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Connected</p>
                <p className="text-2xl font-bold text-white">{connectedIntegrations}</p>
              </div>
              <Plug className="w-8 h-8 text-green-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-green-500">
              <CheckCircle className="w-3 h-3 mr-1" />
              <span>{connectedIntegrations} of {integrations.length} active</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Requests</p>
                <p className="text-2xl font-bold text-white">{totalRequests.toLocaleString()}</p>
              </div>
              <BarChart3 className="w-8 h-8 text-blue-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-blue-500">
              <TrendingUp className="w-3 h-3 mr-1" />
              <span>Across all integrations</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Avg Response</p>
                <p className="text-2xl font-bold text-white">{avgResponseTime}ms</p>
              </div>
              <Clock className="w-8 h-8 text-blue-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-blue-500">
              <Activity className="w-3 h-3 mr-1" />
              <span>Response time</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Issues</p>
                <p className="text-2xl font-bold text-white">{errorIntegrations}</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-red-500">
              <XCircle className="w-3 h-3 mr-1" />
              <span>Require attention</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-gray-900 border border-gray-800">
          <TabsTrigger value="integrations" className="data-[state=active]:bg-black">Integrations</TabsTrigger>
          <TabsTrigger value="marketplace" className="data-[state=active]:bg-black">Marketplace</TabsTrigger>
          <TabsTrigger value="webhooks" className="data-[state=active]:bg-black">Webhooks</TabsTrigger>
          <TabsTrigger value="analytics" className="data-[state=active]:bg-black">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="integrations" className="space-y-6">
          {/* Filters and Actions */}
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
            <div className="flex flex-wrap items-center gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search integrations..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 w-64 bg-gray-900 border-gray-700 text-white"
                />
              </div>
              
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-32 bg-gray-900 border-gray-700">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="connected">Connected</SelectItem>
                  <SelectItem value="disconnected">Disconnected</SelectItem>
                  <SelectItem value="error">Error</SelectItem>
                  <SelectItem value="configuring">Configuring</SelectItem>
                </SelectContent>
              </Select>

              <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                <SelectTrigger className="w-40 bg-gray-900 border-gray-700">
                  <SelectValue placeholder="Category" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="all">All Categories</SelectItem>
                  <SelectItem value="cloud">Cloud</SelectItem>
                  <SelectItem value="communication">Communication</SelectItem>
                  <SelectItem value="security">Security</SelectItem>
                  <SelectItem value="monitoring">Monitoring</SelectItem>
                  <SelectItem value="automation">Automation</SelectItem>
                  <SelectItem value="analytics">Analytics</SelectItem>
                  <SelectItem value="storage">Storage</SelectItem>
                </SelectContent>
              </Select>

              <Select value={typeFilter} onValueChange={setTypeFilter}>
                <SelectTrigger className="w-32 bg-gray-900 border-gray-700">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="oauth">OAuth</SelectItem>
                  <SelectItem value="api_key">API Key</SelectItem>
                  <SelectItem value="webhook">Webhook</SelectItem>
                  <SelectItem value="direct">Direct</SelectItem>
                  <SelectItem value="saml">SAML</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-3">
              <Button
                onClick={() => setIsConfiguringIntegration(true)}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add Integration
              </Button>
              
              <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh All
              </Button>
            </div>
          </div>

          {/* Integrations List */}
          <div className="space-y-4">
            {filteredIntegrations.map((integration) => (
              <Card key={integration.id} className="bg-black border-gray-800">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-4">
                      {getCategoryIcon(integration.category)}
                      <div>
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className="font-semibold text-white">{integration.displayName}</h3>
                          <Badge variant="secondary" className="bg-gray-700 text-gray-300 text-xs">
                            {integration.provider}
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-400">{integration.description}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge className={getStatusColor(integration.status)} variant="outline">
                        {getStatusIcon(integration.status)}
                        <span className="ml-1 capitalize">{integration.status}</span>
                      </Badge>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-4">
                    <div className="space-y-3">
                      <h4 className="text-sm font-medium text-gray-400">Configuration</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Type:</span>
                          <span className="text-white capitalize">{integration.type.replace('_', ' ')}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Version:</span>
                          <span className="text-white">{integration.version}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Last Sync:</span>
                          <span className="text-white">
                            {new Date(integration.lastSync).toLocaleDateString()}
                          </span>
                        </div>
                        {integration.configuration.endpoint && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Endpoint:</span>
                            <span className="text-white text-xs font-mono truncate">
                              {new URL(integration.configuration.endpoint).hostname}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="space-y-3">
                      <h4 className="text-sm font-medium text-gray-400">Usage Statistics</h4>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="text-center p-2 bg-gray-900 rounded">
                          <p className="text-lg font-bold text-green-400">
                            {integration.usage.successfulRequests.toLocaleString()}
                          </p>
                          <p className="text-xs text-gray-400">Success</p>
                        </div>
                        <div className="text-center p-2 bg-gray-900 rounded">
                          <p className="text-lg font-bold text-red-400">
                            {integration.usage.failedRequests.toLocaleString()}
                          </p>
                          <p className="text-xs text-gray-400">Failed</p>
                        </div>
                      </div>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Response Time:</span>
                          <span className="text-white">{integration.usage.averageResponseTime}ms</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Data Transfer:</span>
                          <span className="text-white">
                            {(integration.usage.dataTransferred / 1024 / 1024).toFixed(1)} MB
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <h4 className="text-sm font-medium text-gray-400">Health Status</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Status:</span>
                          <span className={getHealthColor(integration.healthCheck.status)}>
                            {integration.healthCheck.status}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Last Check:</span>
                          <span className="text-white">
                            {new Date(integration.healthCheck.lastCheck).toLocaleTimeString()}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Response:</span>
                          <span className="text-white">{integration.healthCheck.responseTime}ms</span>
                        </div>
                        {integration.healthCheck.message && (
                          <div className="text-xs text-gray-400 p-2 bg-gray-900 rounded">
                            {integration.healthCheck.message}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {integration.features.length > 0 && (
                    <div className="mb-4">
                      <h5 className="text-sm font-medium text-gray-400 mb-2">Features</h5>
                      <div className="flex flex-wrap gap-2">
                        {integration.features.map((feature) => (
                          <Badge key={feature} variant="secondary" className="bg-blue-500/20 text-blue-400 text-xs">
                            {feature}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {integration.tags.length > 0 && (
                    <div className="mb-4">
                      <div className="flex flex-wrap gap-2">
                        {integration.tags.map((tag) => (
                          <Badge key={tag} variant="secondary" className="bg-gray-800 text-gray-300 text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex items-center justify-between pt-4 border-t border-gray-800">
                    <div className="flex items-center space-x-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setSelectedIntegration(integration);
                          setIsViewingDetails(true);
                        }}
                        className="text-blue-400 hover:text-blue-300"
                      >
                        <Eye className="w-3 h-3 mr-1" />
                        View Details
                      </Button>
                      
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleTestConnection(integration.id)}
                        className="text-green-400 hover:text-green-300"
                        disabled={integration.status === 'authenticating'}
                      >
                        <RefreshCw className={`w-3 h-3 mr-1 ${integration.status === 'authenticating' ? 'animate-spin' : ''}`} />
                        Test
                      </Button>
                      
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-yellow-400 hover:text-yellow-300"
                      >
                        <Settings className="w-3 h-3 mr-1" />
                        Configure
                      </Button>
                      
                      {integration.status === 'connected' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDisconnect(integration.id)}
                          className="text-red-400 hover:text-red-300"
                        >
                          <Unlink className="w-3 h-3 mr-1" />
                          Disconnect
                        </Button>
                      )}
                    </div>
                    
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-gray-400 hover:text-white"
                    >
                      <MoreVertical className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="marketplace" className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-white">Integration Marketplace</h3>
            <div className="flex items-center space-x-2">
              <Input
                placeholder="Search integrations..."
                className="w-64 bg-gray-900 border-gray-700 text-white"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {templates.map((template) => {
              const Icon = template.icon;
              return (
                <Card key={template.id} className="bg-black border-gray-800">
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <Icon className="w-8 h-8 text-blue-500" />
                        <div>
                          <div className="flex items-center space-x-2 mb-1">
                            <h4 className="font-semibold text-white">{template.displayName}</h4>
                            {template.popular && (
                              <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/20" variant="outline">
                                <Star className="w-3 h-3 mr-1" />
                                Popular
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm text-gray-400">{template.provider}</p>
                        </div>
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-400 mb-4">{template.description}</p>
                    
                    <div className="space-y-2 mb-4">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Category:</span>
                        <Badge variant="secondary" className="bg-blue-500/20 text-blue-300 capitalize">
                          {template.category}
                        </Badge>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Auth Type:</span>
                        <span className="text-white capitalize">{template.type.replace('_', ' ')}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Features:</span>
                        <span className="text-white">{template.features.length}</span>
                      </div>
                    </div>

                    <div className="mb-4">
                      <h5 className="text-sm font-medium text-gray-400 mb-2">Key Features</h5>
                      <div className="flex flex-wrap gap-1">
                        {template.features.slice(0, 3).map((feature) => (
                          <Badge key={feature} variant="secondary" className="bg-blue-500/20 text-blue-400 text-xs">
                            {feature}
                          </Badge>
                        ))}
                        {template.features.length > 3 && (
                          <Badge variant="secondary" className="bg-gray-700 text-gray-300 text-xs">
                            +{template.features.length - 3} more
                          </Badge>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-4 border-t border-gray-800">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-blue-400 hover:text-blue-300"
                      >
                        <Eye className="w-3 h-3 mr-1" />
                        Learn More
                      </Button>
                      <Button
                        size="sm"
                        className="bg-green-600 hover:bg-green-700 text-white"
                      >
                        <Plus className="w-3 h-3 mr-1" />
                        Install
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </TabsContent>

        <TabsContent value="webhooks" className="space-y-6">
          <Card className="bg-black border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Webhook Management</CardTitle>
              <CardDescription>Manage incoming and outgoing webhooks</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12">
                <Webhook className="w-16 h-16 text-gray-700 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">Webhook Management</h3>
                <p className="text-gray-400 mb-6">Advanced webhook management and monitoring coming soon</p>
                <div className="flex justify-center space-x-4">
                  <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                    <Plus className="w-4 h-4 mr-2" />
                    Create Webhook
                  </Button>
                  <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                    <Settings className="w-4 h-4 mr-2" />
                    Configure
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <Card className="bg-black border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Integration Analytics</CardTitle>
              <CardDescription>Usage patterns and performance metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12">
                <BarChart3 className="w-16 h-16 text-gray-700 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">Analytics Dashboard</h3>
                <p className="text-gray-400 mb-6">Detailed integration analytics and insights coming soon</p>
                <div className="flex justify-center space-x-4">
                  <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                    <TrendingUp className="w-4 h-4 mr-2" />
                    Usage Trends
                  </Button>
                  <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                    <Activity className="w-4 h-4 mr-2" />
                    Performance
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Integration Details Dialog */}
      <Dialog open={isViewingDetails} onOpenChange={setIsViewingDetails}>
        <DialogContent className="bg-gray-900 border-gray-800 text-white max-w-4xl">
          <DialogHeader>
            <DialogTitle>Integration Details: {selectedIntegration?.displayName}</DialogTitle>
            <DialogDescription>
              Complete integration configuration and usage information
            </DialogDescription>
          </DialogHeader>
          {selectedIntegration && (
            <div className="space-y-6 py-4 max-h-96 overflow-y-auto">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-400 mb-2">Basic Information</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Provider:</span>
                        <span className="text-white">{selectedIntegration.provider}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Category:</span>
                        <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/20 capitalize" variant="outline">
                          {selectedIntegration.category}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Type:</span>
                        <span className="text-white capitalize">{selectedIntegration.type.replace('_', ' ')}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Version:</span>
                        <span className="text-white">{selectedIntegration.version}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Created By:</span>
                        <span className="text-white">{selectedIntegration.createdBy}</span>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-400 mb-2">Usage Statistics</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Total Requests:</span>
                        <span className="text-white">{selectedIntegration.usage.totalRequests.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Success Rate:</span>
                        <span className="text-green-400">
                          {selectedIntegration.usage.totalRequests > 0 
                            ? Math.round((selectedIntegration.usage.successfulRequests / selectedIntegration.usage.totalRequests) * 100)
                            : 0}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Failed Requests:</span>
                        <span className="text-red-400">{selectedIntegration.usage.failedRequests.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Avg Response:</span>
                        <span className="text-white">{selectedIntegration.usage.averageResponseTime}ms</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Data Transfer:</span>
                        <span className="text-white">
                          {(selectedIntegration.usage.dataTransferred / 1024 / 1024).toFixed(2)} MB
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-400 mb-3">Features ({selectedIntegration.features.length})</h4>
                <div className="grid grid-cols-2 gap-2">
                  {selectedIntegration.features.map((feature) => (
                    <div key={feature} className="text-sm text-white bg-gray-800 px-3 py-2 rounded">
                      {feature}
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-400 mb-3">Permissions ({selectedIntegration.permissions.length})</h4>
                <div className="grid grid-cols-2 gap-2">
                  {selectedIntegration.permissions.map((permission) => (
                    <div key={permission} className="text-sm text-white bg-gray-800 px-3 py-2 rounded font-mono">
                      {permission}
                    </div>
                  ))}
                </div>
              </div>

              {selectedIntegration.configuration.endpoint && (
                <div>
                  <h4 className="text-sm font-medium text-gray-400 mb-3">Configuration</h4>
                  <div className="bg-gray-800 rounded p-3">
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Endpoint:</span>
                        <span className="text-white font-mono text-xs break-all">
                          {selectedIntegration.configuration.endpoint}
                        </span>
                      </div>
                      {Object.entries(selectedIntegration.configuration.settings).map(([key, value]) => (
                        <div key={key} className="flex justify-between">
                          <span className="text-gray-400 capitalize">{key.replace(/([A-Z])/g, ' $1')}:</span>
                          <span className="text-white">{String(value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          <div className="flex justify-end space-x-3">
            <Button
              variant="outline"
              onClick={() => setIsViewingDetails(false)}
              className="border-gray-700 hover:bg-gray-800"
            >
              Close
            </Button>
            <Button className="bg-blue-600 hover:bg-blue-700">
              <Settings className="w-4 h-4 mr-2" />
              Configure
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );

  return (
    <TacticalPageTemplate 
      title="Integrations" 
      subtitle="Integration Management Operations Center" 
      icon={Plug}
    >
      {content}
    </TacticalPageTemplate>
  );
}