'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Package,
  Server,
  Database,
  Cloud,
  Shield,
  Users,
  Settings,
  Activity,
  Lock,
  Key,
  Globe,
  Cpu,
  HardDrive,
  Network,
  Monitor,
  Smartphone,
  Wifi,
  Mail,
  Calendar,
  FileText,
  Search,
  Filter,
  Grid,
  List,
  Star,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
  ChevronRight,
  ExternalLink,
  ShoppingCart,
  DollarSign,
  Zap
} from 'lucide-react';

interface ServiceItem {
  id: string;
  name: string;
  category: string;
  description: string;
  shortDescription: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  price: number;
  priceUnit: 'per-month' | 'per-user' | 'per-gb' | 'one-time';
  sla: string;
  deliveryTime: string;
  popularity: number;
  status: 'available' | 'coming-soon' | 'deprecated';
  tags: string[];
  features: string[];
  prerequisites: string[];
}

interface ServiceCategory {
  id: string;
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  itemCount: number;
  color: string;
}

interface ServiceRequest {
  id: string;
  serviceId: string;
  serviceName: string;
  requestedBy: string;
  requestedDate: Date;
  status: 'pending' | 'approved' | 'in-progress' | 'completed' | 'rejected';
  priority: 'low' | 'medium' | 'high' | 'critical';
  estimatedCompletion: Date;
}

export default function ITSMCatalog() {
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [services, setServices] = useState<ServiceItem[]>([]);
  const [categories, setCategories] = useState<ServiceCategory[]>([]);
  const [recentRequests, setRecentRequests] = useState<ServiceRequest[]>([]);
  const [selectedService, setSelectedService] = useState<ServiceItem | null>(null);

  useEffect(() => {
    // Simulate loading data
    setTimeout(() => {
      // Mock categories
      setCategories([
        {
          id: 'infrastructure',
          name: 'Infrastructure',
          description: 'Virtual machines, storage, and networking',
          icon: Server,
          itemCount: 12,
          color: '#3B82F6'
        },
        {
          id: 'database',
          name: 'Database Services',
          description: 'SQL, NoSQL, and data warehousing',
          icon: Database,
          itemCount: 8,
          color: '#10B981'
        },
        {
          id: 'security',
          name: 'Security & Compliance',
          description: 'Identity, access management, and protection',
          icon: Shield,
          itemCount: 15,
          color: '#EF4444'
        },
        {
          id: 'collaboration',
          name: 'Collaboration Tools',
          description: 'Email, chat, and productivity apps',
          icon: Users,
          itemCount: 10,
          color: '#8B5CF6'
        },
        {
          id: 'monitoring',
          name: 'Monitoring & Analytics',
          description: 'Logging, metrics, and insights',
          icon: Activity,
          itemCount: 7,
          color: '#F59E0B'
        },
        {
          id: 'development',
          name: 'Development Tools',
          description: 'CI/CD, repositories, and testing',
          icon: Settings,
          itemCount: 9,
          color: '#06B6D4'
        }
      ]);

      // Mock services
      setServices([
        {
          id: 'vm-standard',
          name: 'Virtual Machine - Standard',
          category: 'infrastructure',
          description: 'Standard compute instances for general-purpose workloads with flexible configurations',
          shortDescription: 'General-purpose VMs',
          icon: Server,
          color: '#3B82F6',
          price: 150,
          priceUnit: 'per-month',
          sla: '99.9%',
          deliveryTime: '15 minutes',
          popularity: 95,
          status: 'available',
          tags: ['compute', 'scalable', 'windows', 'linux'],
          features: ['Auto-scaling', 'Load balancing', 'Backup included', 'SSD storage'],
          prerequisites: ['Valid subscription', 'Network configuration']
        },
        {
          id: 'sql-database',
          name: 'Azure SQL Database',
          category: 'database',
          description: 'Fully managed relational database with built-in intelligence and security',
          shortDescription: 'Managed SQL database',
          icon: Database,
          color: '#10B981',
          price: 250,
          priceUnit: 'per-month',
          sla: '99.99%',
          deliveryTime: '5 minutes',
          popularity: 88,
          status: 'available',
          tags: ['database', 'sql', 'managed', 'backup'],
          features: ['Automated backups', 'Geo-replication', 'AI-powered optimization', 'Advanced security'],
          prerequisites: ['Database design', 'Capacity planning']
        },
        {
          id: 'identity-management',
          name: 'Identity & Access Management',
          category: 'security',
          description: 'Enterprise identity service for single sign-on and multi-factor authentication',
          shortDescription: 'SSO and MFA service',
          icon: Key,
          color: '#EF4444',
          price: 6,
          priceUnit: 'per-user',
          sla: '99.99%',
          deliveryTime: '1 hour',
          popularity: 92,
          status: 'available',
          tags: ['security', 'identity', 'sso', 'mfa'],
          features: ['Single sign-on', 'MFA support', 'Conditional access', 'Identity protection'],
          prerequisites: ['User directory', 'Security policies']
        },
        {
          id: 'blob-storage',
          name: 'Blob Storage - Hot Tier',
          category: 'infrastructure',
          description: 'Object storage solution for unstructured data with high availability',
          shortDescription: 'Object storage',
          icon: HardDrive,
          color: '#3B82F6',
          price: 0.02,
          priceUnit: 'per-gb',
          sla: '99.9%',
          deliveryTime: 'Instant',
          popularity: 85,
          status: 'available',
          tags: ['storage', 'blob', 'scalable', 'backup'],
          features: ['Unlimited capacity', 'Geo-redundancy', 'Lifecycle management', 'CDN integration'],
          prerequisites: ['Storage account', 'Access policies']
        },
        {
          id: 'teams-enterprise',
          name: 'Microsoft Teams Enterprise',
          category: 'collaboration',
          description: 'Complete collaboration platform with chat, video, and file sharing',
          shortDescription: 'Team collaboration',
          icon: Users,
          color: '#8B5CF6',
          price: 12,
          priceUnit: 'per-user',
          sla: '99.9%',
          deliveryTime: '30 minutes',
          popularity: 90,
          status: 'available',
          tags: ['collaboration', 'chat', 'video', 'teams'],
          features: ['Unlimited chat', 'Video conferencing', 'File sharing', 'App integration'],
          prerequisites: ['User licenses', 'O365 subscription']
        },
        {
          id: 'app-insights',
          name: 'Application Insights',
          category: 'monitoring',
          description: 'Application performance management and analytics service',
          shortDescription: 'APM and analytics',
          icon: Activity,
          color: '#F59E0B',
          price: 50,
          priceUnit: 'per-month',
          sla: '99.9%',
          deliveryTime: '10 minutes',
          popularity: 78,
          status: 'available',
          tags: ['monitoring', 'apm', 'analytics', 'alerts'],
          features: ['Real-time monitoring', 'Smart detection', 'Custom dashboards', 'Alert rules'],
          prerequisites: ['Application code', 'Instrumentation key']
        },
        {
          id: 'vpn-gateway',
          name: 'VPN Gateway',
          category: 'infrastructure',
          description: 'Secure cross-premises connectivity for hybrid cloud scenarios',
          shortDescription: 'Site-to-site VPN',
          icon: Network,
          color: '#3B82F6',
          price: 200,
          priceUnit: 'per-month',
          sla: '99.95%',
          deliveryTime: '45 minutes',
          popularity: 72,
          status: 'available',
          tags: ['network', 'vpn', 'security', 'hybrid'],
          features: ['Site-to-site VPN', 'Point-to-site VPN', 'High availability', 'BGP support'],
          prerequisites: ['Network design', 'On-premises gateway']
        },
        {
          id: 'devops-pipeline',
          name: 'Azure DevOps Pipeline',
          category: 'development',
          description: 'CI/CD pipeline for automated build and deployment',
          shortDescription: 'CI/CD automation',
          icon: Settings,
          color: '#06B6D4',
          price: 30,
          priceUnit: 'per-user',
          sla: '99.9%',
          deliveryTime: '15 minutes',
          popularity: 83,
          status: 'available',
          tags: ['devops', 'ci-cd', 'automation', 'deployment'],
          features: ['Unlimited builds', 'Parallel jobs', 'Container support', 'Release management'],
          prerequisites: ['Source code', 'Build configuration']
        },
        {
          id: 'cosmos-db',
          name: 'Cosmos DB',
          category: 'database',
          description: 'Globally distributed, multi-model database service',
          shortDescription: 'NoSQL database',
          icon: Globe,
          color: '#10B981',
          price: 300,
          priceUnit: 'per-month',
          sla: '99.999%',
          deliveryTime: '5 minutes',
          popularity: 75,
          status: 'available',
          tags: ['database', 'nosql', 'global', 'scalable'],
          features: ['Global distribution', 'Multiple APIs', 'Automatic indexing', 'Guaranteed latency'],
          prerequisites: ['Data model', 'Partition strategy']
        },
        {
          id: 'backup-service',
          name: 'Azure Backup Service',
          category: 'infrastructure',
          description: 'Centralized backup solution for cloud and on-premises workloads',
          shortDescription: 'Backup solution',
          icon: Shield,
          color: '#3B82F6',
          price: 100,
          priceUnit: 'per-month',
          sla: '99.9%',
          deliveryTime: '30 minutes',
          popularity: 87,
          status: 'available',
          tags: ['backup', 'recovery', 'protection', 'compliance'],
          features: ['Automated backups', 'Long-term retention', 'Instant restore', 'Encryption'],
          prerequisites: ['Backup policy', 'Recovery vault']
        }
      ]);

      // Mock recent requests
      setRecentRequests([
        {
          id: 'REQ001',
          serviceId: 'vm-standard',
          serviceName: 'Virtual Machine - Standard',
          requestedBy: 'John Smith',
          requestedDate: new Date('2024-01-09T10:00:00'),
          status: 'in-progress',
          priority: 'high',
          estimatedCompletion: new Date('2024-01-09T10:15:00')
        },
        {
          id: 'REQ002',
          serviceId: 'sql-database',
          serviceName: 'Azure SQL Database',
          requestedBy: 'Sarah Johnson',
          requestedDate: new Date('2024-01-09T09:30:00'),
          status: 'completed',
          priority: 'medium',
          estimatedCompletion: new Date('2024-01-09T09:35:00')
        },
        {
          id: 'REQ003',
          serviceId: 'identity-management',
          serviceName: 'Identity & Access Management',
          requestedBy: 'Mike Davis',
          requestedDate: new Date('2024-01-09T08:45:00'),
          status: 'pending',
          priority: 'critical',
          estimatedCompletion: new Date('2024-01-09T11:00:00')
        }
      ]);

      setLoading(false);
    }, 1000);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'available':
        return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'coming-soon':
        return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'deprecated':
        return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
      default:
        return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
    }
  };

  const getRequestStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'in-progress':
        return <Clock className="h-4 w-4 text-blue-500" />;
      case 'pending':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case 'rejected':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const formatPrice = (price: number, unit: string) => {
    const formatted = price.toLocaleString('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: unit === 'per-gb' ? 2 : 0
    });
    return `${formatted} ${unit.replace('per-', '/')}`;
  };

  const filteredServices = services.filter(service => {
    const matchesCategory = selectedCategory === 'all' || service.category === selectedCategory;
    const matchesSearch = service.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          service.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          service.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesCategory && matchesSearch;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading service catalog...</p>
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
              <Package className="h-8 w-8 text-blue-600" />
              IT Service Catalog
            </h1>
            <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
              Browse and request IT services for your organization
            </p>
          </div>
          <div className="flex gap-2">
            <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
              <ShoppingCart className="h-4 w-4" />
              My Requests ({recentRequests.length})
            </button>
          </div>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search services..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('grid')}
            className={`p-2 rounded-lg ${
              viewMode === 'grid'
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
            }`}
          >
            <Grid className="h-5 w-5" />
          </button>
          <button
            onClick={() => setViewMode('list')}
            className={`p-2 rounded-lg ${
              viewMode === 'list'
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
            }`}
          >
            <List className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Categories */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <button
          onClick={() => setSelectedCategory('all')}
          className={`p-4 rounded-lg border transition-all ${
            selectedCategory === 'all'
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
              : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
          }`}
        >
          <Package className="h-8 w-8 mx-auto mb-2 text-gray-600 dark:text-gray-400" />
          <div className="font-medium">All Services</div>
          <div className="text-sm text-gray-500 dark:text-gray-400">
            {services.length} items
          </div>
        </button>
        {categories.map((category) => {
          const Icon = category.icon;
          return (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(category.id)}
              className={`p-4 rounded-lg border transition-all ${
                selectedCategory === category.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              <span style={{ color: category.color }}>
                <Icon className="h-8 w-8 mx-auto mb-2" />
              </span>
              <div className="font-medium text-sm">{category.name}</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {category.itemCount} items
              </div>
            </button>
          );
        })}
      </div>

      {/* Services Grid/List */}
      {viewMode === 'grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredServices.map((service) => {
            const Icon = service.icon;
            return (
              <Card key={service.id} className="hover:shadow-lg transition-shadow cursor-pointer"
                    onClick={() => setSelectedService(service)}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
                      <span style={{ color: service.color }}>
                        <Icon className="h-6 w-6" />
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-1">
                        <Star className="h-4 w-4 text-yellow-500 fill-yellow-500" />
                        <span className="text-sm">{(service.popularity / 20).toFixed(1)}</span>
                      </div>
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(service.status)}`}>
                        {service.status}
                      </span>
                    </div>
                  </div>
                  <CardTitle className="text-lg mt-3">{service.name}</CardTitle>
                  <CardDescription>{service.shortDescription}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Price</span>
                      <span className="font-medium">{formatPrice(service.price, service.priceUnit)}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">SLA</span>
                      <span className="font-medium">{service.sla}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Delivery</span>
                      <span className="font-medium">{service.deliveryTime}</span>
                    </div>
                    <div className="flex flex-wrap gap-1 mt-3">
                      {service.tags.slice(0, 3).map((tag) => (
                        <span key={tag} className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-800 rounded-full">
                          {tag}
                        </span>
                      ))}
                      {service.tags.length > 3 && (
                        <span className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-800 rounded-full">
                          +{service.tags.length - 3}
                        </span>
                      )}
                    </div>
                  </div>
                  <button className="w-full mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center justify-center gap-2">
                    Request Service
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </CardContent>
              </Card>
            );
          })}
        </div>
      ) : (
        <Card>
          <CardContent className="p-0">
            <table className="w-full">
              <thead className="border-b border-gray-200 dark:border-gray-700">
                <tr>
                  <th className="text-left py-3 px-4">Service</th>
                  <th className="text-left py-3 px-4">Category</th>
                  <th className="text-right py-3 px-4">Price</th>
                  <th className="text-center py-3 px-4">SLA</th>
                  <th className="text-center py-3 px-4">Delivery</th>
                  <th className="text-center py-3 px-4">Status</th>
                  <th className="text-center py-3 px-4">Action</th>
                </tr>
              </thead>
              <tbody>
                {filteredServices.map((service) => {
                  const Icon = service.icon;
                  return (
                    <tr key={service.id} className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900/50">
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-3">
                          <span style={{ color: service.color }}>
                            <Icon className="h-5 w-5" />
                          </span>
                          <div>
                            <div className="font-medium">{service.name}</div>
                            <div className="text-sm text-gray-600 dark:text-gray-400">
                              {service.shortDescription}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        {categories.find(c => c.id === service.category)?.name}
                      </td>
                      <td className="text-right py-3 px-4">
                        {formatPrice(service.price, service.priceUnit)}
                      </td>
                      <td className="text-center py-3 px-4">{service.sla}</td>
                      <td className="text-center py-3 px-4">{service.deliveryTime}</td>
                      <td className="text-center py-3 px-4">
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(service.status)}`}>
                          {service.status}
                        </span>
                      </td>
                      <td className="text-center py-3 px-4">
                        <button className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700">
                          Request
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {/* Recent Requests */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Service Requests</CardTitle>
          <CardDescription>Track your submitted service requests</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {recentRequests.map((request) => (
              <div key={request.id} className="flex items-center justify-between p-3 border border-gray-200 dark:border-gray-700 rounded-lg">
                <div className="flex items-center gap-3">
                  {getRequestStatusIcon(request.status)}
                  <div>
                    <div className="font-medium">{request.serviceName}</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {request.id} • Requested by {request.requestedBy}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium">
                    {request.requestedDate.toLocaleDateString()}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    Est. completion: {request.estimatedCompletion.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}