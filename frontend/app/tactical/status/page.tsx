'use client';

import React, { useState, useEffect } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { 
  Activity, 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  Clock, 
  Zap, 
  RefreshCw,
  Server,
  Database,
  Globe,
  Shield,
  Cloud,
  Settings,
  Plus,
  Bell,
  TrendingUp,
  TrendingDown,
  Minus,
  Eye,
  ExternalLink,
  Calendar,
  Users,
  MessageSquare,
  FileText,
  Search,
  Filter,
  Download,
  Upload,
  Edit,
  Trash2,
  MoreVertical,
  Play,
  Pause,
  Square,
  BarChart3,
  PieChart,
  LineChart,
  Info
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

interface ServiceStatus {
  id: string;
  name: string;
  category: 'core' | 'external' | 'database' | 'infrastructure' | 'monitoring';
  status: 'operational' | 'degraded' | 'partial_outage' | 'major_outage' | 'maintenance';
  uptime: number;
  responseTime: number;
  lastChecked: string;
  incidents: number;
  description?: string;
  endpoint?: string;
  dependencies: string[];
  metrics: {
    cpu: number;
    memory: number;
    storage: number;
    requests: number;
  };
}

interface Incident {
  id: string;
  title: string;
  description: string;
  status: 'investigating' | 'identified' | 'monitoring' | 'resolved';
  severity: 'low' | 'medium' | 'high' | 'critical';
  affectedServices: string[];
  startTime: string;
  endTime?: string;
  updates: IncidentUpdate[];
  assignee?: string;
  rootCause?: string;
}

interface IncidentUpdate {
  id: string;
  timestamp: string;
  status: string;
  message: string;
  author: string;
}

interface MaintenanceWindow {
  id: string;
  title: string;
  description: string;
  services: string[];
  startTime: string;
  endTime: string;
  status: 'scheduled' | 'in_progress' | 'completed';
  impact: 'no_impact' | 'minimal' | 'moderate' | 'significant';
  type: 'emergency' | 'planned' | 'security';
}

interface Metric {
  name: string;
  value: number;
  unit: string;
  status: 'good' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  change: number;
}

const mockServices: ServiceStatus[] = [
  {
    id: 'api-gateway',
    name: 'API Gateway',
    category: 'core',
    status: 'operational',
    uptime: 99.97,
    responseTime: 45,
    lastChecked: '2024-01-20T10:30:00Z',
    incidents: 0,
    description: 'Main API gateway handling all requests',
    endpoint: 'https://api.policycortex.com',
    dependencies: ['auth-service', 'database'],
    metrics: { cpu: 35, memory: 62, storage: 45, requests: 2847 }
  },
  {
    id: 'auth-service',
    name: 'Authentication Service',
    category: 'core',
    status: 'operational',
    uptime: 99.99,
    responseTime: 28,
    lastChecked: '2024-01-20T10:29:45Z',
    incidents: 0,
    dependencies: ['database', 'redis'],
    metrics: { cpu: 28, memory: 45, storage: 23, requests: 1547 }
  },
  {
    id: 'policy-engine',
    name: 'Policy Engine',
    category: 'core',
    status: 'degraded',
    uptime: 98.45,
    responseTime: 156,
    lastChecked: '2024-01-20T10:30:00Z',
    incidents: 1,
    dependencies: ['database', 'ai-service'],
    metrics: { cpu: 78, memory: 84, storage: 67, requests: 945 }
  },
  {
    id: 'database',
    name: 'Primary Database',
    category: 'database',
    status: 'operational',
    uptime: 99.95,
    responseTime: 12,
    lastChecked: '2024-01-20T10:30:00Z',
    incidents: 0,
    dependencies: [],
    metrics: { cpu: 42, memory: 68, storage: 73, requests: 0 }
  },
  {
    id: 'redis',
    name: 'Redis Cache',
    category: 'database',
    status: 'operational',
    uptime: 99.89,
    responseTime: 3,
    lastChecked: '2024-01-20T10:30:00Z',
    incidents: 0,
    dependencies: [],
    metrics: { cpu: 15, memory: 34, storage: 28, requests: 0 }
  },
  {
    id: 'azure-integration',
    name: 'Azure Integration',
    category: 'external',
    status: 'operational',
    uptime: 99.2,
    responseTime: 234,
    lastChecked: '2024-01-20T10:29:30Z',
    incidents: 2,
    dependencies: [],
    metrics: { cpu: 0, memory: 0, storage: 0, requests: 234 }
  }
];

const mockIncidents: Incident[] = [
  {
    id: 'inc-1',
    title: 'Policy Engine Performance Degradation',
    description: 'Policy evaluation response times increased by 300% affecting compliance checks',
    status: 'monitoring',
    severity: 'medium',
    affectedServices: ['policy-engine'],
    startTime: '2024-01-20T09:15:00Z',
    assignee: 'Sarah Chen',
    updates: [
      {
        id: 'up-1',
        timestamp: '2024-01-20T10:15:00Z',
        status: 'monitoring',
        message: 'Performance has improved after scaling policy engine instances. Continuing to monitor.',
        author: 'Sarah Chen'
      },
      {
        id: 'up-2',
        timestamp: '2024-01-20T09:45:00Z',
        status: 'identified',
        message: 'Root cause identified as increased memory usage. Scaling instances.',
        author: 'Sarah Chen'
      },
      {
        id: 'up-3',
        timestamp: '2024-01-20T09:15:00Z',
        status: 'investigating',
        message: 'Investigating policy engine performance issues.',
        author: 'Sarah Chen'
      }
    ]
  },
  {
    id: 'inc-2',
    title: 'Azure API Rate Limiting',
    description: 'Intermittent Azure API rate limiting affecting resource discovery',
    status: 'resolved',
    severity: 'low',
    affectedServices: ['azure-integration'],
    startTime: '2024-01-20T08:30:00Z',
    endTime: '2024-01-20T09:45:00Z',
    assignee: 'Mike Rodriguez',
    rootCause: 'Azure API quota exceeded due to increased scanning frequency',
    updates: [
      {
        id: 'up-4',
        timestamp: '2024-01-20T09:45:00Z',
        status: 'resolved',
        message: 'Issue resolved by implementing exponential backoff and request batching.',
        author: 'Mike Rodriguez'
      }
    ]
  }
];

const mockMaintenance: MaintenanceWindow[] = [
  {
    id: 'maint-1',
    title: 'Database Maintenance',
    description: 'Scheduled database maintenance to apply security patches and optimize performance',
    services: ['database', 'api-gateway'],
    startTime: '2024-01-21T02:00:00Z',
    endTime: '2024-01-21T04:00:00Z',
    status: 'scheduled',
    impact: 'minimal',
    type: 'planned'
  },
  {
    id: 'maint-2',
    title: 'Security Updates',
    description: 'Critical security updates for authentication service',
    services: ['auth-service'],
    startTime: '2024-01-22T01:00:00Z',
    endTime: '2024-01-22T01:30:00Z',
    status: 'scheduled',
    impact: 'moderate',
    type: 'security'
  }
];

const mockMetrics: Metric[] = [
  { name: 'Overall Uptime', value: 99.8, unit: '%', status: 'good', trend: 'stable', change: 0.1 },
  { name: 'Avg Response Time', value: 78, unit: 'ms', status: 'good', trend: 'down', change: -12 },
  { name: 'Error Rate', value: 0.02, unit: '%', status: 'good', trend: 'down', change: -0.01 },
  { name: 'Active Incidents', value: 1, unit: '', status: 'warning', trend: 'up', change: 1 },
  { name: 'API Requests/min', value: 2847, unit: '', status: 'good', trend: 'up', change: 234 },
  { name: 'Service Health Score', value: 94, unit: '%', status: 'good', trend: 'stable', change: 0 }
];

export default function StatusPage() {
  const [services, setServices] = useState<ServiceStatus[]>(mockServices);
  const [incidents, setIncidents] = useState<Incident[]>(mockIncidents);
  const [maintenance, setMaintenance] = useState<MaintenanceWindow[]>(mockMaintenance);
  const [metrics, setMetrics] = useState<Metric[]>(mockMetrics);
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [isCreatingIncident, setIsCreatingIncident] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const [newIncident, setNewIncident] = useState({
    title: '',
    description: '',
    severity: 'medium' as const,
    affectedServices: [] as string[]
  });

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        // Simulate real-time updates
        setServices(services.map(service => ({
          ...service,
          lastChecked: new Date().toISOString(),
          responseTime: service.responseTime + (Math.random() - 0.5) * 10,
          metrics: {
            ...service.metrics,
            cpu: Math.max(0, Math.min(100, service.metrics.cpu + (Math.random() - 0.5) * 5)),
            memory: Math.max(0, Math.min(100, service.metrics.memory + (Math.random() - 0.5) * 3)),
            requests: service.metrics.requests + Math.floor(Math.random() * 10)
          }
        })));
      }, 30000); // Update every 30 seconds

      return () => clearInterval(interval);
    }
  }, [autoRefresh, services]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'operational': return 'bg-green-500/20 text-green-400 border-green-500/20';
      case 'degraded': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/20';
      case 'partial_outage': return 'bg-orange-500/20 text-orange-400 border-orange-500/20';
      case 'major_outage': return 'bg-red-500/20 text-red-400 border-red-500/20';
      case 'maintenance': return 'bg-blue-500/20 text-blue-400 border-blue-500/20';
      case 'investigating': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/20';
      case 'identified': return 'bg-orange-500/20 text-orange-400 border-orange-500/20';
      case 'monitoring': return 'bg-blue-500/20 text-blue-400 border-blue-500/20';
      case 'resolved': return 'bg-green-500/20 text-green-400 border-green-500/20';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'operational': return <CheckCircle className="w-4 h-4" />;
      case 'degraded': return <AlertTriangle className="w-4 h-4" />;
      case 'partial_outage': return <XCircle className="w-4 h-4" />;
      case 'major_outage': return <XCircle className="w-4 h-4" />;
      case 'maintenance': return <Settings className="w-4 h-4" />;
      case 'investigating': return <Search className="w-4 h-4" />;
      case 'identified': return <Eye className="w-4 h-4" />;
      case 'monitoring': return <Activity className="w-4 h-4" />;
      case 'resolved': return <CheckCircle className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'core': return <Server className="w-5 h-5" />;
      case 'database': return <Database className="w-5 h-5" />;
      case 'external': return <Globe className="w-5 h-5" />;
      case 'infrastructure': return <Cloud className="w-5 h-5" />;
      case 'monitoring': return <Activity className="w-5 h-5" />;
      default: return <Server className="w-5 h-5" />;
    }
  };

  const getMetricStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'critical': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="w-3 h-3" />;
      case 'down': return <TrendingDown className="w-3 h-3" />;
      case 'stable': return <Minus className="w-3 h-3" />;
      default: return <Minus className="w-3 h-3" />;
    }
  };

  const filteredServices = services.filter(service => {
    const matchesCategory = selectedCategory === 'all' || service.category === selectedCategory;
    const matchesSearch = service.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         service.description?.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const overallStatus = services.some(s => s.status === 'major_outage') ? 'major_outage' :
                       services.some(s => s.status === 'partial_outage') ? 'partial_outage' :
                       services.some(s => s.status === 'degraded') ? 'degraded' : 'operational';

  const operationalServices = services.filter(s => s.status === 'operational').length;
  const activeIncidents = incidents.filter(i => i.status !== 'resolved').length;

  const handleCreateIncident = () => {
    const incident: Incident = {
      id: `inc-${Date.now()}`,
      title: newIncident.title,
      description: newIncident.description,
      status: 'investigating',
      severity: newIncident.severity,
      affectedServices: newIncident.affectedServices,
      startTime: new Date().toISOString(),
      assignee: 'Current User',
      updates: [{
        id: `up-${Date.now()}`,
        timestamp: new Date().toISOString(),
        status: 'investigating',
        message: 'Incident created and investigation started.',
        author: 'Current User'
      }]
    };

    setIncidents([incident, ...incidents]);
    setNewIncident({
      title: '',
      description: '',
      severity: 'medium',
      affectedServices: []
    });
    setIsCreatingIncident(false);
  };

  const content = (
    <div className="space-y-6">
      {/* Overall Status */}
      <Card className="bg-black border-gray-800">
        <CardContent className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                {getStatusIcon(overallStatus)}
                <h2 className="text-2xl font-bold text-white">System Status</h2>
              </div>
              <Badge className={getStatusColor(overallStatus)} variant="outline">
                {overallStatus.replace('_', ' ').toUpperCase()}
              </Badge>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={autoRefresh}
                  onCheckedChange={setAutoRefresh}
                  className="data-[state=checked]:bg-green-600"
                />
                <Label className="text-sm text-gray-400">Auto-refresh</Label>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="border-gray-700 hover:bg-gray-800"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <p className="text-3xl font-bold text-green-400">{operationalServices}</p>
              <p className="text-sm text-gray-400">Services Online</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold text-blue-400">{services.length}</p>
              <p className="text-sm text-gray-400">Total Services</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold text-yellow-400">{activeIncidents}</p>
              <p className="text-sm text-gray-400">Active Incidents</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold text-purple-400">{maintenance.filter(m => m.status === 'scheduled').length}</p>
              <p className="text-sm text-gray-400">Scheduled Maintenance</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metrics.map((metric) => (
          <Card key={metric.name} className="bg-black border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-400">{metric.name}</p>
                  <div className="flex items-baseline space-x-2">
                    <p className={`text-2xl font-bold ${getMetricStatusColor(metric.status)}`}>
                      {metric.value}
                    </p>
                    <span className="text-sm text-gray-400">{metric.unit}</span>
                  </div>
                </div>
                <div className="flex items-center space-x-1 text-xs text-gray-500">
                  {getTrendIcon(metric.trend)}
                  <span className={metric.change > 0 ? 'text-green-400' : metric.change < 0 ? 'text-red-400' : 'text-gray-400'}>
                    {metric.change > 0 ? '+' : ''}{metric.change}{metric.unit}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-gray-900 border border-gray-800">
          <TabsTrigger value="overview" className="data-[state=active]:bg-black">Services</TabsTrigger>
          <TabsTrigger value="incidents" className="data-[state=active]:bg-black">Incidents</TabsTrigger>
          <TabsTrigger value="maintenance" className="data-[state=active]:bg-black">Maintenance</TabsTrigger>
          <TabsTrigger value="history" className="data-[state=active]:bg-black">History</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search services..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 bg-gray-900 border-gray-700 text-white"
                />
              </div>
              <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                <SelectTrigger className="w-40 bg-gray-900 border-gray-700">
                  <SelectValue placeholder="Category" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="all">All Categories</SelectItem>
                  <SelectItem value="core">Core Services</SelectItem>
                  <SelectItem value="database">Databases</SelectItem>
                  <SelectItem value="external">External APIs</SelectItem>
                  <SelectItem value="infrastructure">Infrastructure</SelectItem>
                  <SelectItem value="monitoring">Monitoring</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {filteredServices.map((service) => (
              <Card key={service.id} className="bg-black border-gray-800">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      {getCategoryIcon(service.category)}
                      <div>
                        <h3 className="font-semibold text-white">{service.name}</h3>
                        <p className="text-sm text-gray-400 capitalize">{service.category}</p>
                      </div>
                    </div>
                    <Badge className={getStatusColor(service.status)} variant="outline">
                      {getStatusIcon(service.status)}
                      <span className="ml-1 capitalize">{service.status.replace('_', ' ')}</span>
                    </Badge>
                  </div>

                  {service.description && (
                    <p className="text-sm text-gray-400 mb-4">{service.description}</p>
                  )}

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Uptime:</span>
                        <span className="text-white font-semibold">{service.uptime.toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Response:</span>
                        <span className="text-white font-semibold">{Math.round(service.responseTime)}ms</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Incidents:</span>
                        <span className={service.incidents > 0 ? 'text-yellow-400' : 'text-green-400'}>
                          {service.incidents}
                        </span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-400">CPU</span>
                          <span className="text-white">{Math.round(service.metrics.cpu)}%</span>
                        </div>
                        <Progress value={service.metrics.cpu} className="h-2" />
                      </div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-400">Memory</span>
                          <span className="text-white">{Math.round(service.metrics.memory)}%</span>
                        </div>
                        <Progress value={service.metrics.memory} className="h-2" />
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-gray-800 text-xs text-gray-500">
                    <span>Updated: {new Date(service.lastChecked).toLocaleTimeString()}</span>
                    <div className="flex items-center space-x-2">
                      {service.endpoint && (
                        <Button variant="ghost" size="sm" className="h-6 px-2">
                          <ExternalLink className="w-3 h-3" />
                        </Button>
                      )}
                      <Button variant="ghost" size="sm" className="h-6 px-2">
                        <BarChart3 className="w-3 h-3" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="incidents" className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-white">Incident Management</h3>
            <Button
              onClick={() => setIsCreatingIncident(true)}
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              <Plus className="w-4 h-4 mr-2" />
              Report Incident
            </Button>
          </div>

          <div className="space-y-4">
            {incidents.map((incident) => (
              <Card key={incident.id} className="bg-black border-gray-800">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <h4 className="font-semibold text-white">{incident.title}</h4>
                        <Badge className={getStatusColor(incident.severity)}>
                          {incident.severity.toUpperCase()}
                        </Badge>
                        <Badge className={getStatusColor(incident.status)}>
                          {getStatusIcon(incident.status)}
                          <span className="ml-1 capitalize">{incident.status.replace('_', ' ')}</span>
                        </Badge>
                      </div>
                      <p className="text-sm text-gray-400 mb-3">{incident.description}</p>
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>Started: {new Date(incident.startTime).toLocaleString()}</span>
                        {incident.endTime && (
                          <span>Ended: {new Date(incident.endTime).toLocaleString()}</span>
                        )}
                        {incident.assignee && (
                          <span>Assignee: {incident.assignee}</span>
                        )}
                        <span>Services: {incident.affectedServices.length}</span>
                      </div>
                    </div>
                  </div>

                  {incident.updates && incident.updates.length > 0 && (
                    <div className="border-t border-gray-800 pt-4">
                      <h5 className="text-sm font-medium text-white mb-3">Latest Update</h5>
                      <div className="bg-gray-900 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <Badge className={getStatusColor(incident.updates[0].status)} variant="outline">
                            {incident.updates[0].status.toUpperCase()}
                          </Badge>
                          <span className="text-xs text-gray-500">
                            {new Date(incident.updates[0].timestamp).toLocaleString()}
                          </span>
                        </div>
                        <p className="text-sm text-gray-300">{incident.updates[0].message}</p>
                        <p className="text-xs text-gray-500 mt-2">By {incident.updates[0].author}</p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="maintenance" className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-white">Scheduled Maintenance</h3>
            <Button className="bg-blue-600 hover:bg-blue-700 text-white">
              <Plus className="w-4 h-4 mr-2" />
              Schedule Maintenance
            </Button>
          </div>

          <div className="space-y-4">
            {maintenance.map((window) => (
              <Card key={window.id} className="bg-black border-gray-800">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <div className="flex items-center space-x-2 mb-2">
                        <h4 className="font-semibold text-white">{window.title}</h4>
                        <Badge className={getStatusColor(window.status)}>
                          {window.status.replace('_', ' ').toUpperCase()}
                        </Badge>
                        <Badge variant="outline" className={`
                          ${window.impact === 'no_impact' ? 'border-green-500/20 text-green-400' : ''}
                          ${window.impact === 'minimal' ? 'border-blue-500/20 text-blue-400' : ''}
                          ${window.impact === 'moderate' ? 'border-yellow-500/20 text-yellow-400' : ''}
                          ${window.impact === 'significant' ? 'border-red-500/20 text-red-400' : ''}
                        `}>
                          {window.impact.replace('_', ' ').toUpperCase()}
                        </Badge>
                      </div>
                      <p className="text-sm text-gray-400 mb-3">{window.description}</p>
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>Start: {new Date(window.startTime).toLocaleString()}</span>
                        <span>End: {new Date(window.endTime).toLocaleString()}</span>
                        <span>Duration: {Math.round((new Date(window.endTime).getTime() - new Date(window.startTime).getTime()) / (1000 * 60 * 60))}h</span>
                        <span>Services: {window.services.length}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-gray-800">
                    <div className="flex items-center space-x-2">
                      {window.services.map(service => (
                        <Badge key={service} variant="secondary" className="bg-gray-800 text-gray-300">
                          {service}
                        </Badge>
                      ))}
                    </div>
                    <div className="flex items-center space-x-2">
                      <Button variant="ghost" size="sm" className="text-gray-400 hover:text-white">
                        <Edit className="w-3 h-3" />
                      </Button>
                      <Button variant="ghost" size="sm" className="text-gray-400 hover:text-white">
                        <Calendar className="w-3 h-3" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          <Card className="bg-black border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">System History</CardTitle>
              <CardDescription>Historical system status and incident data</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12">
                <LineChart className="w-16 h-16 text-gray-700 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">Historical Data</h3>
                <p className="text-gray-400 mb-6">Detailed historical analysis and trending data coming soon</p>
                <div className="flex justify-center space-x-4">
                  <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                    <Download className="w-4 h-4 mr-2" />
                    Export Data
                  </Button>
                  <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                    <BarChart3 className="w-4 h-4 mr-2" />
                    View Analytics
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Create Incident Dialog */}
      <Dialog open={isCreatingIncident} onOpenChange={setIsCreatingIncident}>
        <DialogContent className="bg-gray-900 border-gray-800 text-white max-w-2xl">
          <DialogHeader>
            <DialogTitle>Report New Incident</DialogTitle>
            <DialogDescription>
              Create a new incident report to track and manage system issues
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-6 py-4">
            <div>
              <Label htmlFor="incident-title" className="text-sm font-medium">
                Incident Title
              </Label>
              <Input
                id="incident-title"
                value={newIncident.title}
                onChange={(e) => setNewIncident({...newIncident, title: e.target.value})}
                placeholder="Brief description of the incident"
                className="bg-black border-gray-700 mt-1"
              />
            </div>

            <div>
              <Label htmlFor="incident-description" className="text-sm font-medium">
                Description
              </Label>
              <Textarea
                id="incident-description"
                value={newIncident.description}
                onChange={(e) => setNewIncident({...newIncident, description: e.target.value})}
                placeholder="Detailed description of the incident and its impact"
                rows={4}
                className="bg-black border-gray-700 mt-1"
              />
            </div>

            <div>
              <Label htmlFor="incident-severity" className="text-sm font-medium">
                Severity Level
              </Label>
              <Select value={newIncident.severity} onValueChange={(value: any) => setNewIncident({...newIncident, severity: value})}>
                <SelectTrigger className="bg-black border-gray-700 mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="low">Low - Minor impact</SelectItem>
                  <SelectItem value="medium">Medium - Moderate impact</SelectItem>
                  <SelectItem value="high">High - Significant impact</SelectItem>
                  <SelectItem value="critical">Critical - Major system impact</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-sm font-medium mb-3 block">
                Affected Services
              </Label>
              <div className="grid grid-cols-2 gap-2 max-h-40 overflow-y-auto">
                {services.map((service) => (
                  <div key={service.id} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id={`service-${service.id}`}
                      checked={newIncident.affectedServices.includes(service.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setNewIncident({
                            ...newIncident,
                            affectedServices: [...newIncident.affectedServices, service.id]
                          });
                        } else {
                          setNewIncident({
                            ...newIncident,
                            affectedServices: newIncident.affectedServices.filter(id => id !== service.id)
                          });
                        }
                      }}
                      className="rounded border-gray-700 bg-black"
                    />
                    <Label htmlFor={`service-${service.id}`} className="text-sm">
                      {service.name}
                    </Label>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="flex justify-end space-x-3">
            <Button
              variant="outline"
              onClick={() => setIsCreatingIncident(false)}
              className="border-gray-700 hover:bg-gray-800"
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreateIncident}
              disabled={!newIncident.title || !newIncident.description}
              className="bg-red-600 hover:bg-red-700"
            >
              Create Incident
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );

  return (
    <TacticalPageTemplate 
      title="Status Page" 
      subtitle="System Health & Incident Management" 
      icon={Activity}
    >
      {content}
    </TacticalPageTemplate>
  );
}