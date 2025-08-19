'use client';

import React, { useState, useEffect } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { 
  Key, 
  Plus, 
  Search, 
  Filter,
  MoreVertical,
  Edit,
  Trash2,
  Shield,
  Copy,
  Eye,
  EyeOff,
  RefreshCw,
  Download,
  Upload,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  Activity,
  Settings,
  Calendar,
  User,
  Globe,
  Lock,
  Unlock,
  Star,
  Zap,
  BarChart3,
  TrendingUp,
  Server,
  Database,
  Cloud,
  Terminal,
  Smartphone,
  Monitor,
  Code,
  Link
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

interface ApiKey {
  id: string;
  name: string;
  description: string;
  key: string;
  prefix: string;
  suffix: string;
  status: 'active' | 'inactive' | 'expired' | 'revoked';
  type: 'full_access' | 'read_only' | 'limited' | 'service_account';
  scopes: string[];
  permissions: string[];
  rateLimits: {
    requestsPerMinute: number;
    requestsPerHour: number;
    requestsPerDay: number;
  };
  usage: {
    totalRequests: number;
    successfulRequests: number;
    failedRequests: number;
    lastUsed: string;
    currentPeriodRequests: number;
  };
  createdAt: string;
  expiresAt?: string;
  lastRotated?: string;
  createdBy: string;
  ipWhitelist: string[];
  userAgent?: string;
  environment: 'production' | 'staging' | 'development' | 'testing';
  tags: string[];
}

interface ApiKeyTemplate {
  id: string;
  name: string;
  description: string;
  type: 'full_access' | 'read_only' | 'limited' | 'service_account';
  scopes: string[];
  permissions: string[];
  rateLimits: {
    requestsPerMinute: number;
    requestsPerHour: number;
    requestsPerDay: number;
  };
  expiryDays?: number;
  recommended: boolean;
}

const mockApiKeys: ApiKey[] = [
  {
    id: 'api-1',
    name: 'Production Dashboard',
    description: 'Main dashboard API access for production environment',
    key: 'pk_live_1234567890abcdef1234567890abcdef12345678',
    prefix: 'pk_live_',
    suffix: '5678',
    status: 'active',
    type: 'full_access',
    scopes: ['policies:read', 'policies:write', 'users:read', 'metrics:read'],
    permissions: ['dashboard:access', 'api:full'],
    rateLimits: {
      requestsPerMinute: 1000,
      requestsPerHour: 60000,
      requestsPerDay: 1440000
    },
    usage: {
      totalRequests: 2847293,
      successfulRequests: 2832156,
      failedRequests: 15137,
      lastUsed: '2024-01-20T10:25:00Z',
      currentPeriodRequests: 15234
    },
    createdAt: '2023-06-15T08:00:00Z',
    expiresAt: '2024-06-15T08:00:00Z',
    lastRotated: '2024-01-01T00:00:00Z',
    createdBy: 'admin@policycortex.com',
    ipWhitelist: ['10.0.0.0/8', '192.168.1.0/24'],
    userAgent: 'PolicyCortex-Dashboard/2.1.0',
    environment: 'production',
    tags: ['critical', 'dashboard', 'production']
  },
  {
    id: 'api-2',
    name: 'Mobile App Integration',
    description: 'API key for mobile application access',
    key: 'pk_live_abcdef1234567890abcdef1234567890abcdef12',
    prefix: 'pk_live_',
    suffix: 'f12',
    status: 'active',
    type: 'limited',
    scopes: ['policies:read', 'metrics:read'],
    permissions: ['mobile:access', 'api:read'],
    rateLimits: {
      requestsPerMinute: 100,
      requestsPerHour: 6000,
      requestsPerDay: 144000
    },
    usage: {
      totalRequests: 458392,
      successfulRequests: 455821,
      failedRequests: 2571,
      lastUsed: '2024-01-20T10:15:00Z',
      currentPeriodRequests: 2847
    },
    createdAt: '2023-09-10T14:30:00Z',
    expiresAt: '2024-09-10T14:30:00Z',
    createdBy: 'mobile-team@policycortex.com',
    ipWhitelist: [],
    userAgent: 'PolicyCortex-Mobile/1.2.5 (iOS)',
    environment: 'production',
    tags: ['mobile', 'ios', 'limited']
  },
  {
    id: 'api-3',
    name: 'Analytics Service',
    description: 'Service account for automated analytics processing',
    key: 'pk_svc_analytics1234567890abcdef1234567890ab',
    prefix: 'pk_svc_',
    suffix: '90ab',
    status: 'active',
    type: 'service_account',
    scopes: ['metrics:read', 'reports:generate'],
    permissions: ['analytics:process', 'reports:create'],
    rateLimits: {
      requestsPerMinute: 500,
      requestsPerHour: 30000,
      requestsPerDay: 720000
    },
    usage: {
      totalRequests: 1849203,
      successfulRequests: 1847891,
      failedRequests: 1312,
      lastUsed: '2024-01-20T10:30:00Z',
      currentPeriodRequests: 8934
    },
    createdAt: '2023-08-05T10:15:00Z',
    createdBy: 'system@policycortex.com',
    ipWhitelist: ['172.16.0.0/12'],
    userAgent: 'PolicyCortex-Analytics-Service/3.0.1',
    environment: 'production',
    tags: ['service', 'analytics', 'automated']
  },
  {
    id: 'api-4',
    name: 'Development Testing',
    description: 'Development environment testing key',
    key: 'pk_dev_test567890abcdef1234567890abcdef1234',
    prefix: 'pk_dev_',
    suffix: '1234',
    status: 'active',
    type: 'read_only',
    scopes: ['policies:read'],
    permissions: ['dev:access'],
    rateLimits: {
      requestsPerMinute: 50,
      requestsPerHour: 3000,
      requestsPerDay: 72000
    },
    usage: {
      totalRequests: 89234,
      successfulRequests: 87912,
      failedRequests: 1322,
      lastUsed: '2024-01-19T16:45:00Z',
      currentPeriodRequests: 234
    },
    createdAt: '2024-01-10T09:00:00Z',
    expiresAt: '2024-04-10T09:00:00Z',
    createdBy: 'dev@policycortex.com',
    ipWhitelist: [],
    userAgent: 'Postman/10.20.10',
    environment: 'development',
    tags: ['development', 'testing', 'temporary']
  },
  {
    id: 'api-5',
    name: 'Legacy Integration',
    description: 'Legacy system integration - scheduled for deprecation',
    key: 'pk_legacy_old1234567890abcdef1234567890abc',
    prefix: 'pk_legacy_',
    suffix: '0abc',
    status: 'inactive',
    type: 'limited',
    scopes: ['policies:read'],
    permissions: ['legacy:access'],
    rateLimits: {
      requestsPerMinute: 10,
      requestsPerHour: 600,
      requestsPerDay: 14400
    },
    usage: {
      totalRequests: 234891,
      successfulRequests: 229847,
      failedRequests: 5044,
      lastUsed: '2023-12-15T14:20:00Z',
      currentPeriodRequests: 0
    },
    createdAt: '2022-03-20T11:30:00Z',
    expiresAt: '2024-03-20T11:30:00Z',
    createdBy: 'legacy-admin@policycortex.com',
    ipWhitelist: ['203.0.113.0/24'],
    userAgent: 'LegacySystem/1.0',
    environment: 'production',
    tags: ['legacy', 'deprecated', 'scheduled-removal']
  }
];

const mockTemplates: ApiKeyTemplate[] = [
  {
    id: 'template-1',
    name: 'Dashboard Access',
    description: 'Standard dashboard access with read permissions',
    type: 'read_only',
    scopes: ['policies:read', 'metrics:read', 'users:read'],
    permissions: ['dashboard:view'],
    rateLimits: {
      requestsPerMinute: 100,
      requestsPerHour: 6000,
      requestsPerDay: 144000
    },
    expiryDays: 365,
    recommended: true
  },
  {
    id: 'template-2',
    name: 'Service Account',
    description: 'Service account with automated processing permissions',
    type: 'service_account',
    scopes: ['metrics:read', 'reports:generate', 'analytics:process'],
    permissions: ['service:automated'],
    rateLimits: {
      requestsPerMinute: 500,
      requestsPerHour: 30000,
      requestsPerDay: 720000
    },
    recommended: true
  },
  {
    id: 'template-3',
    name: 'Mobile Application',
    description: 'Mobile app integration with limited permissions',
    type: 'limited',
    scopes: ['policies:read', 'metrics:read'],
    permissions: ['mobile:access'],
    rateLimits: {
      requestsPerMinute: 60,
      requestsPerHour: 3600,
      requestsPerDay: 86400
    },
    expiryDays: 180,
    recommended: false
  }
];

export default function ApiKeysPage() {
  const [apiKeys, setApiKeys] = useState<ApiKey[]>(mockApiKeys);
  const [templates, setTemplates] = useState<ApiKeyTemplate[]>(mockTemplates);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [environmentFilter, setEnvironmentFilter] = useState<string>('all');
  const [activeTab, setActiveTab] = useState('keys');
  const [selectedKey, setSelectedKey] = useState<ApiKey | null>(null);
  const [isCreatingKey, setIsCreatingKey] = useState(false);
  const [isViewingKey, setIsViewingKey] = useState(false);
  const [showKeyValue, setShowKeyValue] = useState<Record<string, boolean>>({});

  const [newKey, setNewKey] = useState({
    name: '',
    description: '',
    type: 'read_only' as const,
    scopes: [] as string[],
    permissions: [] as string[],
    expiryDays: 365,
    environment: 'development' as const,
    ipWhitelist: [] as string[],
    tags: [] as string[],
    rateLimits: {
      requestsPerMinute: 100,
      requestsPerHour: 6000,
      requestsPerDay: 144000
    }
  });

  const filteredApiKeys = apiKeys.filter(key => {
    const matchesSearch = 
      key.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      key.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      key.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesStatus = statusFilter === 'all' || key.status === statusFilter;
    const matchesType = typeFilter === 'all' || key.type === typeFilter;
    const matchesEnvironment = environmentFilter === 'all' || key.environment === environmentFilter;

    return matchesSearch && matchesStatus && matchesType && matchesEnvironment;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500/20 text-green-400 border-green-500/20';
      case 'inactive': return 'bg-gray-500/20 text-gray-400 border-gray-500/20';
      case 'expired': return 'bg-red-500/20 text-red-400 border-red-500/20';
      case 'revoked': return 'bg-orange-500/20 text-orange-400 border-orange-500/20';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle className="w-4 h-4" />;
      case 'inactive': return <XCircle className="w-4 h-4" />;
      case 'expired': return <Clock className="w-4 h-4" />;
      case 'revoked': return <AlertTriangle className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'full_access': return <Star className="w-4 h-4" />;
      case 'read_only': return <Eye className="w-4 h-4" />;
      case 'limited': return <Shield className="w-4 h-4" />;
      case 'service_account': return <Server className="w-4 h-4" />;
      default: return <Key className="w-4 h-4" />;
    }
  };

  const getEnvironmentColor = (environment: string) => {
    switch (environment) {
      case 'production': return 'bg-red-500/20 text-red-400 border-red-500/20';
      case 'staging': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/20';
      case 'development': return 'bg-blue-500/20 text-blue-400 border-blue-500/20';
      case 'testing': return 'bg-purple-500/20 text-purple-400 border-purple-500/20';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/20';
    }
  };

  const maskApiKey = (key: string) => {
    if (key.length < 16) return key;
    return `${key.slice(0, 8)}...${key.slice(-8)}`;
  };

  const handleCreateKey = () => {
    const newApiKey: ApiKey = {
      id: `api-${Date.now()}`,
      name: newKey.name,
      description: newKey.description,
      key: `pk_${newKey.environment}_${Math.random().toString(36).substr(2, 32)}`,
      prefix: `pk_${newKey.environment}_`,
      suffix: Math.random().toString(36).substr(-4),
      status: 'active',
      type: newKey.type,
      scopes: newKey.scopes,
      permissions: newKey.permissions,
      rateLimits: newKey.rateLimits,
      usage: {
        totalRequests: 0,
        successfulRequests: 0,
        failedRequests: 0,
        lastUsed: 'never',
        currentPeriodRequests: 0
      },
      createdAt: new Date().toISOString(),
      expiresAt: newKey.expiryDays ? new Date(Date.now() + newKey.expiryDays * 24 * 60 * 60 * 1000).toISOString() : undefined,
      createdBy: 'current-user@policycortex.com',
      ipWhitelist: newKey.ipWhitelist,
      environment: newKey.environment,
      tags: newKey.tags
    };

    setApiKeys([...apiKeys, newApiKey]);
    setNewKey({
      name: '',
      description: '',
      type: 'read_only',
      scopes: [],
      permissions: [],
      expiryDays: 365,
      environment: 'development',
      ipWhitelist: [],
      tags: [],
      rateLimits: {
        requestsPerMinute: 100,
        requestsPerHour: 6000,
        requestsPerDay: 144000
      }
    });
    setIsCreatingKey(false);
  };

  const handleRevokeKey = (keyId: string) => {
    setApiKeys(apiKeys.map(key => 
      key.id === keyId 
        ? { ...key, status: 'revoked' as const }
        : key
    ));
  };

  const handleRotateKey = (keyId: string) => {
    setApiKeys(apiKeys.map(key => 
      key.id === keyId 
        ? { 
            ...key, 
            key: `${key.prefix}${Math.random().toString(36).substr(2, 32)}`,
            suffix: Math.random().toString(36).substr(-4),
            lastRotated: new Date().toISOString()
          }
        : key
    ));
  };

  const toggleKeyVisibility = (keyId: string) => {
    setShowKeyValue(prev => ({
      ...prev,
      [keyId]: !prev[keyId]
    }));
  };

  const activeKeys = apiKeys.filter(k => k.status === 'active').length;
  const totalRequests = apiKeys.reduce((sum, key) => sum + key.usage.totalRequests, 0);
  const expiringKeys = apiKeys.filter(k => 
    k.expiresAt && new Date(k.expiresAt) < new Date(Date.now() + 30 * 24 * 60 * 60 * 1000)
  ).length;
  const successRate = totalRequests > 0 ? 
    Math.round((apiKeys.reduce((sum, key) => sum + key.usage.successfulRequests, 0) / totalRequests) * 100) 
    : 0;

  const content = (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Active Keys</p>
                <p className="text-2xl font-bold text-white">{activeKeys}</p>
              </div>
              <Key className="w-8 h-8 text-green-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-green-500">
              <CheckCircle className="w-3 h-3 mr-1" />
              <span>{activeKeys} of {apiKeys.length} active</span>
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
              <span>{successRate}% success rate</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Expiring Soon</p>
                <p className="text-2xl font-bold text-white">{expiringKeys}</p>
              </div>
              <Clock className="w-8 h-8 text-yellow-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-yellow-500">
              <Calendar className="w-3 h-3 mr-1" />
              <span>Within 30 days</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Environments</p>
                <p className="text-2xl font-bold text-white">{[...new Set(apiKeys.map(k => k.environment))].length}</p>
              </div>
              <Globe className="w-8 h-8 text-purple-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-purple-500">
              <Server className="w-3 h-3 mr-1" />
              <span>Active environments</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-gray-900 border border-gray-800">
          <TabsTrigger value="keys" className="data-[state=active]:bg-black">API Keys</TabsTrigger>
          <TabsTrigger value="usage" className="data-[state=active]:bg-black">Usage Analytics</TabsTrigger>
          <TabsTrigger value="templates" className="data-[state=active]:bg-black">Templates</TabsTrigger>
          <TabsTrigger value="security" className="data-[state=active]:bg-black">Security</TabsTrigger>
        </TabsList>

        <TabsContent value="keys" className="space-y-6">
          {/* Filters and Actions */}
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
            <div className="flex flex-wrap items-center gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search API keys..."
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
                  <SelectItem value="active">Active</SelectItem>
                  <SelectItem value="inactive">Inactive</SelectItem>
                  <SelectItem value="expired">Expired</SelectItem>
                  <SelectItem value="revoked">Revoked</SelectItem>
                </SelectContent>
              </Select>

              <Select value={typeFilter} onValueChange={setTypeFilter}>
                <SelectTrigger className="w-40 bg-gray-900 border-gray-700">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="full_access">Full Access</SelectItem>
                  <SelectItem value="read_only">Read Only</SelectItem>
                  <SelectItem value="limited">Limited</SelectItem>
                  <SelectItem value="service_account">Service Account</SelectItem>
                </SelectContent>
              </Select>

              <Select value={environmentFilter} onValueChange={setEnvironmentFilter}>
                <SelectTrigger className="w-36 bg-gray-900 border-gray-700">
                  <SelectValue placeholder="Environment" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="all">All Environments</SelectItem>
                  <SelectItem value="production">Production</SelectItem>
                  <SelectItem value="staging">Staging</SelectItem>
                  <SelectItem value="development">Development</SelectItem>
                  <SelectItem value="testing">Testing</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-3">
              <Button
                onClick={() => setIsCreatingKey(true)}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                <Plus className="w-4 h-4 mr-2" />
                Create API Key
              </Button>
              
              <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
          </div>

          {/* API Keys List */}
          <div className="space-y-4">
            {filteredApiKeys.map((apiKey) => (
              <Card key={apiKey.id} className="bg-black border-gray-800">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-4">
                      {getTypeIcon(apiKey.type)}
                      <div>
                        <h3 className="font-semibold text-white">{apiKey.name}</h3>
                        <p className="text-sm text-gray-400">{apiKey.description}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge className={getEnvironmentColor(apiKey.environment)} variant="outline">
                        {apiKey.environment}
                      </Badge>
                      <Badge className={getStatusColor(apiKey.status)} variant="outline">
                        {getStatusIcon(apiKey.status)}
                        <span className="ml-1 capitalize">{apiKey.status}</span>
                      </Badge>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-4">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between p-3 bg-gray-900 rounded-lg">
                        <div className="flex items-center space-x-2">
                          <Key className="w-4 h-4 text-gray-400" />
                          <span className="text-sm font-mono text-white">
                            {showKeyValue[apiKey.id] ? apiKey.key : maskApiKey(apiKey.key)}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleKeyVisibility(apiKey.id)}
                            className="h-6 w-6 p-0 text-gray-400 hover:text-white"
                          >
                            {showKeyValue[apiKey.id] ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => navigator.clipboard.writeText(apiKey.key)}
                            className="h-6 w-6 p-0 text-gray-400 hover:text-white"
                          >
                            <Copy className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>

                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Type:</span>
                          <span className="text-white capitalize">{apiKey.type.replace('_', ' ')}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Created:</span>
                          <span className="text-white">{new Date(apiKey.createdAt).toLocaleDateString()}</span>
                        </div>
                        {apiKey.expiresAt && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Expires:</span>
                            <span className="text-white">{new Date(apiKey.expiresAt).toLocaleDateString()}</span>
                          </div>
                        )}
                        <div className="flex justify-between">
                          <span className="text-gray-400">Last Used:</span>
                          <span className="text-white">
                            {apiKey.usage.lastUsed === 'never' ? 'Never' : new Date(apiKey.usage.lastUsed).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="grid grid-cols-3 gap-2">
                        <div className="text-center p-2 bg-gray-900 rounded">
                          <p className="text-lg font-bold text-green-400">{apiKey.usage.successfulRequests.toLocaleString()}</p>
                          <p className="text-xs text-gray-400">Success</p>
                        </div>
                        <div className="text-center p-2 bg-gray-900 rounded">
                          <p className="text-lg font-bold text-red-400">{apiKey.usage.failedRequests.toLocaleString()}</p>
                          <p className="text-xs text-gray-400">Failed</p>
                        </div>
                        <div className="text-center p-2 bg-gray-900 rounded">
                          <p className="text-lg font-bold text-blue-400">{apiKey.usage.currentPeriodRequests.toLocaleString()}</p>
                          <p className="text-xs text-gray-400">Today</p>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Rate Limits:</span>
                        </div>
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Per minute:</span>
                            <span className="text-white">{apiKey.rateLimits.requestsPerMinute.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Per hour:</span>
                            <span className="text-white">{apiKey.rateLimits.requestsPerHour.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Per day:</span>
                            <span className="text-white">{apiKey.rateLimits.requestsPerDay.toLocaleString()}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {apiKey.tags.length > 0 && (
                    <div className="mb-4">
                      <div className="flex flex-wrap gap-2">
                        {apiKey.tags.map((tag) => (
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
                          setSelectedKey(apiKey);
                          setIsViewingKey(true);
                        }}
                        className="text-blue-400 hover:text-blue-300"
                      >
                        <Eye className="w-3 h-3 mr-1" />
                        View Details
                      </Button>
                      
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleRotateKey(apiKey.id)}
                        className="text-yellow-400 hover:text-yellow-300"
                      >
                        <RefreshCw className="w-3 h-3 mr-1" />
                        Rotate
                      </Button>
                      
                      {apiKey.status === 'active' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleRevokeKey(apiKey.id)}
                          className="text-red-400 hover:text-red-300"
                        >
                          <Lock className="w-3 h-3 mr-1" />
                          Revoke
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

        <TabsContent value="usage" className="space-y-6">
          <Card className="bg-black border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Usage Analytics</CardTitle>
              <CardDescription>API key usage patterns and performance metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12">
                <BarChart3 className="w-16 h-16 text-gray-700 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">Usage Analytics</h3>
                <p className="text-gray-400 mb-6">Detailed usage analytics and performance insights coming soon</p>
                <div className="flex justify-center space-x-4">
                  <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                    <TrendingUp className="w-4 h-4 mr-2" />
                    Usage Trends
                  </Button>
                  <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                    <BarChart3 className="w-4 h-4 mr-2" />
                    Performance
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="templates" className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-white">API Key Templates</h3>
            <Button className="bg-purple-600 hover:bg-purple-700 text-white">
              <Plus className="w-4 h-4 mr-2" />
              Create Template
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {templates.map((template) => (
              <Card key={template.id} className="bg-black border-gray-800">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <div className="flex items-center space-x-2 mb-1">
                        <h4 className="font-semibold text-white">{template.name}</h4>
                        {template.recommended && (
                          <Badge className="bg-green-500/20 text-green-400 border-green-500/20" variant="outline">
                            <Star className="w-3 h-3 mr-1" />
                            Recommended
                          </Badge>
                        )}
                      </div>
                      <Badge className={getStatusColor(template.type)} variant="outline">
                        {getTypeIcon(template.type)}
                        <span className="ml-1 capitalize">{template.type.replace('_', ' ')}</span>
                      </Badge>
                    </div>
                  </div>
                  
                  <p className="text-sm text-gray-400 mb-4">{template.description}</p>
                  
                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Permissions:</span>
                      <span className="text-white">{template.permissions.length}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Scopes:</span>
                      <span className="text-white">{template.scopes.length}</span>
                    </div>
                    {template.expiryDays && (
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Default Expiry:</span>
                        <span className="text-white">{template.expiryDays} days</span>
                      </div>
                    )}
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-gray-800">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-purple-400 hover:text-purple-300"
                    >
                      <Eye className="w-3 h-3 mr-1" />
                      Preview
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-green-400 hover:text-green-300"
                    >
                      <Plus className="w-3 h-3 mr-1" />
                      Use Template
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="security" className="space-y-6">
          <Card className="bg-black border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Security Overview</CardTitle>
              <CardDescription>API key security settings and compliance status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h4 className="text-sm font-medium text-gray-400">Security Metrics</h4>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-400">Keys with IP restrictions:</span>
                        <span className="text-white font-semibold">
                          {apiKeys.filter(k => k.ipWhitelist.length > 0).length} / {apiKeys.length}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-400">Keys expiring in 30 days:</span>
                        <span className={expiringKeys > 0 ? 'text-yellow-400 font-semibold' : 'text-green-400 font-semibold'}>
                          {expiringKeys}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-400">Average success rate:</span>
                        <span className="text-green-400 font-semibold">{successRate}%</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <h4 className="text-sm font-medium text-gray-400">Recommendations</h4>
                    <div className="space-y-2">
                      {expiringKeys > 0 && (
                        <div className="flex items-start space-x-2 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded">
                          <AlertTriangle className="w-4 h-4 text-yellow-500 mt-0.5" />
                          <div className="text-xs text-yellow-400">
                            <p className="font-medium">Keys expiring soon</p>
                            <p>Consider rotating keys before expiration</p>
                          </div>
                        </div>
                      )}
                      
                      {apiKeys.filter(k => k.ipWhitelist.length === 0 && k.environment === 'production').length > 0 && (
                        <div className="flex items-start space-x-2 p-2 bg-orange-500/10 border border-orange-500/20 rounded">
                          <Shield className="w-4 h-4 text-orange-500 mt-0.5" />
                          <div className="text-xs text-orange-400">
                            <p className="font-medium">Production keys without IP restrictions</p>
                            <p>Consider adding IP whitelisting for enhanced security</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Create API Key Dialog */}
      <Dialog open={isCreatingKey} onOpenChange={setIsCreatingKey}>
        <DialogContent className="bg-gray-900 border-gray-800 text-white max-w-3xl">
          <DialogHeader>
            <DialogTitle>Create New API Key</DialogTitle>
            <DialogDescription>
              Generate a new API key with specific permissions and access controls
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-6 py-4 max-h-96 overflow-y-auto">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="key-name" className="text-sm font-medium">
                  Key Name
                </Label>
                <Input
                  id="key-name"
                  value={newKey.name}
                  onChange={(e) => setNewKey({...newKey, name: e.target.value})}
                  placeholder="My API Key"
                  className="bg-black border-gray-700 mt-1"
                />
              </div>
              <div>
                <Label htmlFor="key-environment" className="text-sm font-medium">
                  Environment
                </Label>
                <Select value={newKey.environment} onValueChange={(value: any) => setNewKey({...newKey, environment: value})}>
                  <SelectTrigger className="bg-black border-gray-700 mt-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-900 border-gray-700">
                    <SelectItem value="development">Development</SelectItem>
                    <SelectItem value="staging">Staging</SelectItem>
                    <SelectItem value="production">Production</SelectItem>
                    <SelectItem value="testing">Testing</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div>
              <Label htmlFor="key-description" className="text-sm font-medium">
                Description
              </Label>
              <Textarea
                id="key-description"
                value={newKey.description}
                onChange={(e) => setNewKey({...newKey, description: e.target.value})}
                placeholder="Describe the purpose of this API key..."
                rows={3}
                className="bg-black border-gray-700 mt-1"
              />
            </div>

            <div>
              <Label htmlFor="key-type" className="text-sm font-medium">
                Access Type
              </Label>
              <Select value={newKey.type} onValueChange={(value: any) => setNewKey({...newKey, type: value})}>
                <SelectTrigger className="bg-black border-gray-700 mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="read_only">Read Only</SelectItem>
                  <SelectItem value="limited">Limited Access</SelectItem>
                  <SelectItem value="service_account">Service Account</SelectItem>
                  <SelectItem value="full_access">Full Access</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-sm font-medium mb-3 block">
                Rate Limits
              </Label>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <Label htmlFor="requests-per-minute" className="text-xs text-gray-400">
                    Per Minute
                  </Label>
                  <Input
                    id="requests-per-minute"
                    type="number"
                    value={newKey.rateLimits.requestsPerMinute}
                    onChange={(e) => setNewKey({
                      ...newKey,
                      rateLimits: {
                        ...newKey.rateLimits,
                        requestsPerMinute: parseInt(e.target.value) || 0
                      }
                    })}
                    className="bg-black border-gray-700 mt-1"
                  />
                </div>
                <div>
                  <Label htmlFor="requests-per-hour" className="text-xs text-gray-400">
                    Per Hour
                  </Label>
                  <Input
                    id="requests-per-hour"
                    type="number"
                    value={newKey.rateLimits.requestsPerHour}
                    onChange={(e) => setNewKey({
                      ...newKey,
                      rateLimits: {
                        ...newKey.rateLimits,
                        requestsPerHour: parseInt(e.target.value) || 0
                      }
                    })}
                    className="bg-black border-gray-700 mt-1"
                  />
                </div>
                <div>
                  <Label htmlFor="requests-per-day" className="text-xs text-gray-400">
                    Per Day
                  </Label>
                  <Input
                    id="requests-per-day"
                    type="number"
                    value={newKey.rateLimits.requestsPerDay}
                    onChange={(e) => setNewKey({
                      ...newKey,
                      rateLimits: {
                        ...newKey.rateLimits,
                        requestsPerDay: parseInt(e.target.value) || 0
                      }
                    })}
                    className="bg-black border-gray-700 mt-1"
                  />
                </div>
              </div>
            </div>

            <div>
              <Label htmlFor="expiry-days" className="text-sm font-medium">
                Expiry (days)
              </Label>
              <Input
                id="expiry-days"
                type="number"
                value={newKey.expiryDays}
                onChange={(e) => setNewKey({...newKey, expiryDays: parseInt(e.target.value) || 0})}
                placeholder="365"
                className="bg-black border-gray-700 mt-1"
              />
              <p className="text-xs text-gray-400 mt-1">
                Leave empty for no expiration
              </p>
            </div>
          </div>
          <div className="flex justify-end space-x-3">
            <Button
              variant="outline"
              onClick={() => setIsCreatingKey(false)}
              className="border-gray-700 hover:bg-gray-800"
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreateKey}
              disabled={!newKey.name || !newKey.description}
              className="bg-green-600 hover:bg-green-700"
            >
              <Key className="w-4 h-4 mr-2" />
              Create API Key
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* View API Key Details Dialog */}
      <Dialog open={isViewingKey} onOpenChange={setIsViewingKey}>
        <DialogContent className="bg-gray-900 border-gray-800 text-white max-w-4xl">
          <DialogHeader>
            <DialogTitle>API Key Details: {selectedKey?.name}</DialogTitle>
            <DialogDescription>
              Complete API key information and usage statistics
            </DialogDescription>
          </DialogHeader>
          {selectedKey && (
            <div className="space-y-6 py-4 max-h-96 overflow-y-auto">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-400 mb-2">Basic Information</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Name:</span>
                        <span className="text-white">{selectedKey.name}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Type:</span>
                        <Badge className={getStatusColor(selectedKey.type)} variant="outline">
                          {selectedKey.type.replace('_', ' ')}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Environment:</span>
                        <Badge className={getEnvironmentColor(selectedKey.environment)} variant="outline">
                          {selectedKey.environment}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Created By:</span>
                        <span className="text-white">{selectedKey.createdBy}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">User Agent:</span>
                        <span className="text-white text-xs">{selectedKey.userAgent || 'Not specified'}</span>
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
                        <span className="text-white">{selectedKey.usage.totalRequests.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Success Rate:</span>
                        <span className="text-green-400">
                          {selectedKey.usage.totalRequests > 0 
                            ? Math.round((selectedKey.usage.successfulRequests / selectedKey.usage.totalRequests) * 100)
                            : 0}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Failed Requests:</span>
                        <span className="text-red-400">{selectedKey.usage.failedRequests.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Today's Usage:</span>
                        <span className="text-blue-400">{selectedKey.usage.currentPeriodRequests.toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-400 mb-3">API Key Value</h4>
                <div className="flex items-center justify-between p-3 bg-gray-800 rounded border">
                  <span className="text-sm font-mono text-white break-all">
                    {showKeyValue[selectedKey.id] ? selectedKey.key : maskApiKey(selectedKey.key)}
                  </span>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => toggleKeyVisibility(selectedKey.id)}
                      className="h-8 w-8 p-0 text-gray-400 hover:text-white"
                    >
                      {showKeyValue[selectedKey.id] ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => navigator.clipboard.writeText(selectedKey.key)}
                      className="h-8 w-8 p-0 text-gray-400 hover:text-white"
                    >
                      <Copy className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div>
                  <h4 className="text-sm font-medium text-gray-400 mb-3">Scopes ({selectedKey.scopes.length})</h4>
                  <div className="space-y-1 max-h-32 overflow-y-auto">
                    {selectedKey.scopes.map((scope) => (
                      <div key={scope} className="text-sm text-white bg-gray-800 px-2 py-1 rounded">
                        {scope}
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-400 mb-3">IP Whitelist</h4>
                  <div className="space-y-1 max-h-32 overflow-y-auto">
                    {selectedKey.ipWhitelist.length > 0 ? (
                      selectedKey.ipWhitelist.map((ip) => (
                        <div key={ip} className="text-sm text-white bg-gray-800 px-2 py-1 rounded font-mono">
                          {ip}
                        </div>
                      ))
                    ) : (
                      <p className="text-sm text-gray-500 italic">No IP restrictions</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
          <div className="flex justify-end">
            <Button
              variant="outline"
              onClick={() => setIsViewingKey(false)}
              className="border-gray-700 hover:bg-gray-800"
            >
              Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );

  return (
    <TacticalPageTemplate 
      title="API Keys" 
      subtitle="API Key Management Operations Center" 
      icon={Key}
    >
      {content}
    </TacticalPageTemplate>
  );
}