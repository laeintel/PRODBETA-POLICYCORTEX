'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Upload,
  Play,
  Pause,
  RotateCcw,
  AlertTriangle,
  CheckCircle,
  Clock,
  XCircle,
  GitBranch,
  Calendar,
  Users,
  Activity,
  Zap,
  Shield,
  Target,
  TrendingUp,
  Gauge,
  Server,
  GitCommit,
  Eye,
  Download,
  Filter,
  Search,
  MoreVertical,
  RefreshCw,
  ArrowRight,
  Database,
  Cloud,
  Code,
  FileText,
  Settings,
  AlertCircle,
  Loader,
  ChevronDown,
  ChevronRight
} from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area
} from 'recharts';

interface Deployment {
  id: string;
  name: string;
  application: string;
  version: string;
  environment: string;
  status: 'deploying' | 'success' | 'failed' | 'pending' | 'cancelled' | 'rollback';
  startTime: string;
  endTime?: string;
  duration?: number;
  progress: number;
  author: string;
  branch: string;
  commit: string;
  rollbackAvailable: boolean;
  logs: string[];
  artifacts: number;
  size: string;
  strategy: string;
  approvals: number;
  requiredApprovals: number;
  targetInstances: number;
  successfulInstances: number;
  failedInstances: number;
}

interface Environment {
  id: string;
  name: string;
  type: 'production' | 'staging' | 'testing' | 'development';
  status: 'healthy' | 'degraded' | 'down';
  deployments: number;
  lastDeployment: string;
  uptime: number;
  instances: number;
  cpu: number;
  memory: number;
  requests: number;
}

interface DeploymentMetrics {
  totalDeployments: number;
  successRate: number;
  averageDuration: number;
  failureRate: number;
  rolledBack: number;
  pendingApprovals: number;
}

export default function DeploymentsPage() {
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [environments, setEnvironments] = useState<Environment[]>([]);
  const [metrics, setMetrics] = useState<DeploymentMetrics>({
    totalDeployments: 0,
    successRate: 0,
    averageDuration: 0,
    failureRate: 0,
    rolledBack: 0,
    pendingApprovals: 0
  });
  const [selectedTab, setSelectedTab] = useState('overview');
  const [selectedEnvironment, setSelectedEnvironment] = useState('all');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDeployment, setSelectedDeployment] = useState<Deployment | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const [expandedDeployments, setExpandedDeployments] = useState<Set<string>>(new Set());
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Mock data generation
  useEffect(() => {
    const generateMockDeployments = (): Deployment[] => {
      const applications = ['PolicyCortex-Web', 'PolicyCortex-API', 'ML-Engine', 'Event-Processor', 'Analytics-Service', 'Notification-Service'];
      const environments = ['production', 'staging', 'testing', 'development'];
      const statuses: Deployment['status'][] = ['success', 'failed', 'deploying', 'pending', 'cancelled', 'rollback'];
      const strategies = ['Blue-Green', 'Rolling', 'Canary', 'Recreate', 'A/B Testing'];
      const authors = ['John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Wilson', 'David Brown', 'Emily Davis'];
      
      return Array.from({ length: 25 }, (_, i) => {
        const startTime = new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000);
        const duration = Math.floor(Math.random() * 1800) + 60; // 1-30 minutes
        const endTime = new Date(startTime.getTime() + duration * 1000);
        const status = statuses[Math.floor(Math.random() * statuses.length)];
        const targetInstances = Math.floor(Math.random() * 10) + 3;
        const successfulInstances = status === 'success' ? targetInstances : 
                                  status === 'failed' ? Math.floor(targetInstances * 0.3) :
                                  status === 'deploying' ? Math.floor(targetInstances * 0.7) : 0;
        
        return {
          id: `deploy-${i + 1}`,
          name: `${applications[Math.floor(Math.random() * applications.length)]}-${(Math.random() * 1000).toFixed(0)}`,
          application: applications[Math.floor(Math.random() * applications.length)],
          version: `v${Math.floor(Math.random() * 5) + 1}.${Math.floor(Math.random() * 10)}.${Math.floor(Math.random() * 20)}`,
          environment: environments[Math.floor(Math.random() * environments.length)],
          status,
          startTime: startTime.toISOString(),
          endTime: status !== 'deploying' && status !== 'pending' ? endTime.toISOString() : undefined,
          duration: status !== 'deploying' && status !== 'pending' ? duration : undefined,
          progress: status === 'deploying' ? Math.floor(Math.random() * 80) + 20 :
                   status === 'success' ? 100 :
                   status === 'failed' ? Math.floor(Math.random() * 60) + 10 : 0,
          author: authors[Math.floor(Math.random() * authors.length)],
          branch: Math.random() > 0.7 ? 'main' : Math.random() > 0.5 ? 'develop' : 'feature/new-feature',
          commit: Math.random().toString(36).substring(2, 9).toUpperCase(),
          rollbackAvailable: status === 'success' || status === 'failed',
          logs: [
            'Starting deployment process...',
            'Pulling latest image from registry...',
            'Validating deployment configuration...',
            'Creating deployment resources...',
            status === 'failed' ? 'ERROR: Health check failed' : 'Health checks passed',
            status === 'success' ? 'Deployment completed successfully!' : 'Deployment process interrupted'
          ],
          artifacts: Math.floor(Math.random() * 5) + 1,
          size: `${(Math.random() * 500 + 50).toFixed(1)} MB`,
          strategy: strategies[Math.floor(Math.random() * strategies.length)],
          approvals: Math.floor(Math.random() * 3),
          requiredApprovals: Math.floor(Math.random() * 2) + 1,
          targetInstances,
          successfulInstances,
          failedInstances: targetInstances - successfulInstances
        };
      });
    };

    const generateMockEnvironments = (): Environment[] => {
      return [
        {
          id: 'prod',
          name: 'Production',
          type: 'production',
          status: 'healthy',
          deployments: 156,
          lastDeployment: '2 hours ago',
          uptime: 99.9,
          instances: 12,
          cpu: 65,
          memory: 78,
          requests: 15420
        },
        {
          id: 'staging',
          name: 'Staging',
          type: 'staging',
          status: 'healthy',
          deployments: 89,
          lastDeployment: '30 minutes ago',
          uptime: 98.5,
          instances: 6,
          cpu: 45,
          memory: 62,
          requests: 2840
        },
        {
          id: 'testing',
          name: 'Testing',
          type: 'testing',
          status: 'degraded',
          deployments: 234,
          lastDeployment: '1 hour ago',
          uptime: 97.2,
          instances: 4,
          cpu: 82,
          memory: 91,
          requests: 1250
        },
        {
          id: 'dev',
          name: 'Development',
          type: 'development',
          status: 'healthy',
          deployments: 412,
          lastDeployment: '15 minutes ago',
          uptime: 95.8,
          instances: 8,
          cpu: 38,
          memory: 55,
          requests: 890
        }
      ];
    };

    const mockDeployments = generateMockDeployments();
    const mockEnvironments = generateMockEnvironments();
    
    setDeployments(mockDeployments);
    setEnvironments(mockEnvironments);

    // Calculate metrics
    const successCount = mockDeployments.filter(d => d.status === 'success').length;
    const failureCount = mockDeployments.filter(d => d.status === 'failed').length;
    const rollbackCount = mockDeployments.filter(d => d.status === 'rollback').length;
    const pendingApprovals = mockDeployments.filter(d => d.approvals < d.requiredApprovals).length;
    const completedDeployments = mockDeployments.filter(d => d.duration);
    const avgDuration = completedDeployments.length > 0 
      ? completedDeployments.reduce((acc, d) => acc + (d.duration || 0), 0) / completedDeployments.length 
      : 0;

    setMetrics({
      totalDeployments: mockDeployments.length,
      successRate: mockDeployments.length > 0 ? (successCount / mockDeployments.length) * 100 : 0,
      averageDuration: avgDuration,
      failureRate: mockDeployments.length > 0 ? (failureCount / mockDeployments.length) * 100 : 0,
      rolledBack: rollbackCount,
      pendingApprovals
    });
  }, []);

  // Auto refresh effect
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      setDeployments(prev => 
        prev.map(deployment => {
          if (deployment.status === 'deploying') {
            const newProgress = Math.min(deployment.progress + Math.random() * 5, 100);
            if (newProgress >= 100) {
              return {
                ...deployment,
                status: Math.random() > 0.8 ? 'failed' : 'success',
                progress: 100,
                endTime: new Date().toISOString(),
                duration: Math.floor((Date.now() - new Date(deployment.startTime).getTime()) / 1000)
              };
            }
            return { ...deployment, progress: newProgress };
          }
          return deployment;
        })
      );
    }, 2000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const getStatusColor = (status: Deployment['status']) => {
    switch (status) {
      case 'success': return 'text-green-400 bg-green-500/10';
      case 'failed': return 'text-red-400 bg-red-500/10';
      case 'deploying': return 'text-blue-400 bg-blue-500/10';
      case 'pending': return 'text-yellow-400 bg-yellow-500/10';
      case 'cancelled': return 'text-gray-400 bg-gray-500/10';
      case 'rollback': return 'text-orange-400 bg-orange-500/10';
      default: return 'text-gray-400 bg-gray-500/10';
    }
  };

  const getEnvironmentStatusColor = (status: Environment['status']) => {
    switch (status) {
      case 'healthy': return 'text-green-400 bg-green-500/10';
      case 'degraded': return 'text-yellow-400 bg-yellow-500/10';
      case 'down': return 'text-red-400 bg-red-500/10';
      default: return 'text-gray-400 bg-gray-500/10';
    }
  };

  const filteredDeployments = deployments.filter(deployment => {
    const matchesEnvironment = selectedEnvironment === 'all' || deployment.environment === selectedEnvironment;
    const matchesStatus = selectedStatus === 'all' || deployment.status === selectedStatus;
    const matchesSearch = deployment.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         deployment.application.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         deployment.author.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesEnvironment && matchesStatus && matchesSearch;
  });

  const handleRollback = (deploymentId: string) => {
    setDeployments(prev => 
      prev.map(d => 
        d.id === deploymentId 
          ? { ...d, status: 'deploying' as const, progress: 0, startTime: new Date().toISOString() }
          : d
      )
    );
  };

  const handleRetry = (deploymentId: string) => {
    setDeployments(prev => 
      prev.map(d => 
        d.id === deploymentId 
          ? { ...d, status: 'deploying' as const, progress: 0, startTime: new Date().toISOString() }
          : d
      )
    );
  };

  const toggleDeploymentExpansion = (deploymentId: string) => {
    setExpandedDeployments(prev => {
      const newSet = new Set(prev);
      if (newSet.has(deploymentId)) {
        newSet.delete(deploymentId);
      } else {
        newSet.add(deploymentId);
      }
      return newSet;
    });
  };

  // Chart data
  const deploymentTrendData = Array.from({ length: 7 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (6 - i));
    return {
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      successful: Math.floor(Math.random() * 15) + 5,
      failed: Math.floor(Math.random() * 5) + 1,
      total: Math.floor(Math.random() * 20) + 10
    };
  });

  const environmentDistributionData = environments.map(env => ({
    name: env.name,
    value: env.deployments,
    color: env.type === 'production' ? '#10b981' : 
           env.type === 'staging' ? '#3b82f6' :
           env.type === 'testing' ? '#f59e0b' : '#8b5cf6'
  }));

  const durationAnalysisData = [
    { range: '0-5min', count: 45, color: '#10b981' },
    { range: '5-15min', count: 32, color: '#3b82f6' },
    { range: '15-30min', count: 18, color: '#f59e0b' },
    { range: '30min+', count: 7, color: '#ef4444' }
  ];

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <div className="bg-gray-900/50 border-b border-gray-800 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-blue-500/10 rounded-xl">
              <Upload className="h-8 w-8 text-blue-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Deployments</h1>
              <p className="text-gray-400">Continuous deployment operations center</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm text-gray-400">Auto Refresh</label>
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  autoRefresh ? 'bg-blue-500' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    autoRefresh ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
            <button className="flex items-center space-x-2 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 px-4 py-2 rounded-lg transition-colors">
              <RefreshCw className="h-4 w-4" />
              <span>Refresh</span>
            </button>
            <button className="flex items-center space-x-2 bg-green-500/10 hover:bg-green-500/20 text-green-400 px-4 py-2 rounded-lg transition-colors">
              <Upload className="h-4 w-4" />
              <span>New Deployment</span>
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Overview */}
      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Total Deployments</p>
                <p className="text-2xl font-bold text-white">{metrics.totalDeployments}</p>
              </div>
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <Activity className="h-6 w-6 text-blue-400" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Success Rate</p>
                <p className="text-2xl font-bold text-green-400">{metrics.successRate.toFixed(1)}%</p>
              </div>
              <div className="p-2 bg-green-500/10 rounded-lg">
                <CheckCircle className="h-6 w-6 text-green-400" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Avg Duration</p>
                <p className="text-2xl font-bold text-blue-400">{Math.floor(metrics.averageDuration / 60)}m {Math.floor(metrics.averageDuration % 60)}s</p>
              </div>
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <Clock className="h-6 w-6 text-blue-400" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Failure Rate</p>
                <p className="text-2xl font-bold text-red-400">{metrics.failureRate.toFixed(1)}%</p>
              </div>
              <div className="p-2 bg-red-500/10 rounded-lg">
                <XCircle className="h-6 w-6 text-red-400" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Rollbacks</p>
                <p className="text-2xl font-bold text-orange-400">{metrics.rolledBack}</p>
              </div>
              <div className="p-2 bg-orange-500/10 rounded-lg">
                <RotateCcw className="h-6 w-6 text-orange-400" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Pending Approvals</p>
                <p className="text-2xl font-bold text-yellow-400">{metrics.pendingApprovals}</p>
              </div>
              <div className="p-2 bg-yellow-500/10 rounded-lg">
                <Shield className="h-6 w-6 text-yellow-400" />
              </div>
            </div>
          </motion.div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 mb-8 bg-gray-900/30 p-1 rounded-lg">
          {[
            { id: 'overview', label: 'Overview', icon: Activity },
            { id: 'deployments', label: 'Deployments', icon: Upload },
            { id: 'environments', label: 'Environments', icon: Server },
            { id: 'analytics', label: 'Analytics', icon: TrendingUp }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
                selectedTab === tab.id
                  ? 'bg-blue-500/20 text-blue-400'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <tab.icon className="h-4 w-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {selectedTab === 'overview' && (
          <div className="space-y-8">
            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Deployment Trends */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4">Deployment Trends</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={deploymentTrendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="date" stroke="#9CA3AF" />
                    <YAxis stroke="#9CA3AF" />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '8px'
                      }}
                    />
                    <Area type="monotone" dataKey="successful" stackId="1" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                    <Area type="monotone" dataKey="failed" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              </motion.div>

              {/* Environment Distribution */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4">Environment Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={environmentDistributionData}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                    >
                      {environmentDistributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '8px'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </motion.div>
            </div>

            {/* Recent Activity */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
            >
              <h3 className="text-lg font-semibold text-white mb-4">Recent Deployments</h3>
              <div className="space-y-3">
                {deployments.slice(0, 5).map((deployment, index) => (
                  <div key={deployment.id} className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className={`p-2 rounded-lg ${getStatusColor(deployment.status)}`}>
                        {deployment.status === 'success' && <CheckCircle className="h-4 w-4" />}
                        {deployment.status === 'failed' && <XCircle className="h-4 w-4" />}
                        {deployment.status === 'deploying' && <Loader className="h-4 w-4 animate-spin" />}
                        {deployment.status === 'pending' && <Clock className="h-4 w-4" />}
                        {deployment.status === 'cancelled' && <XCircle className="h-4 w-4" />}
                        {deployment.status === 'rollback' && <RotateCcw className="h-4 w-4" />}
                      </div>
                      <div>
                        <p className="font-medium text-white">{deployment.name}</p>
                        <p className="text-sm text-gray-400">{deployment.application} • {deployment.environment}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium text-white">{deployment.version}</p>
                      <p className="text-xs text-gray-400">{new Date(deployment.startTime).toLocaleTimeString()}</p>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        )}

        {/* Deployments Tab */}
        {selectedTab === 'deployments' && (
          <div className="space-y-6">
            {/* Filters */}
            <div className="flex flex-wrap items-center gap-4 bg-gray-900/30 p-4 rounded-lg">
              <div className="flex items-center space-x-2">
                <Search className="h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search deployments..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              <select
                value={selectedEnvironment}
                onChange={(e) => setSelectedEnvironment(e.target.value)}
                className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Environments</option>
                <option value="production">Production</option>
                <option value="staging">Staging</option>
                <option value="testing">Testing</option>
                <option value="development">Development</option>
              </select>

              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
                className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Status</option>
                <option value="success">Success</option>
                <option value="failed">Failed</option>
                <option value="deploying">Deploying</option>
                <option value="pending">Pending</option>
                <option value="cancelled">Cancelled</option>
                <option value="rollback">Rollback</option>
              </select>
            </div>

            {/* Deployments List */}
            <div className="space-y-4">
              {filteredDeployments.map((deployment, index) => (
                <motion.div
                  key={deployment.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-4">
                      <button
                        onClick={() => toggleDeploymentExpansion(deployment.id)}
                        className="p-1 hover:bg-gray-800 rounded transition-colors"
                      >
                        {expandedDeployments.has(deployment.id) ? 
                          <ChevronDown className="h-4 w-4 text-gray-400" /> :
                          <ChevronRight className="h-4 w-4 text-gray-400" />
                        }
                      </button>
                      <div className={`p-2 rounded-lg ${getStatusColor(deployment.status)}`}>
                        {deployment.status === 'success' && <CheckCircle className="h-5 w-5" />}
                        {deployment.status === 'failed' && <XCircle className="h-5 w-5" />}
                        {deployment.status === 'deploying' && <Loader className="h-5 w-5 animate-spin" />}
                        {deployment.status === 'pending' && <Clock className="h-5 w-5" />}
                        {deployment.status === 'cancelled' && <XCircle className="h-5 w-5" />}
                        {deployment.status === 'rollback' && <RotateCcw className="h-5 w-5" />}
                      </div>
                      <div>
                        <h4 className="font-semibold text-white text-lg">{deployment.name}</h4>
                        <div className="flex items-center space-x-4 text-sm text-gray-400">
                          <span>{deployment.application}</span>
                          <span>•</span>
                          <span>{deployment.version}</span>
                          <span>•</span>
                          <span className="capitalize">{deployment.environment}</span>
                          <span>•</span>
                          <span>{deployment.strategy}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <p className="text-sm font-medium text-white">
                          {deployment.duration ? 
                            `${Math.floor(deployment.duration / 60)}m ${deployment.duration % 60}s` :
                            'In Progress'
                          }
                        </p>
                        <p className="text-xs text-gray-400">
                          {new Date(deployment.startTime).toLocaleString()}
                        </p>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        {deployment.status === 'failed' && (
                          <button
                            onClick={() => handleRetry(deployment.id)}
                            className="p-2 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 rounded-lg transition-colors"
                            title="Retry Deployment"
                          >
                            <RefreshCw className="h-4 w-4" />
                          </button>
                        )}
                        {deployment.rollbackAvailable && (
                          <button
                            onClick={() => handleRollback(deployment.id)}
                            className="p-2 bg-orange-500/10 hover:bg-orange-500/20 text-orange-400 rounded-lg transition-colors"
                            title="Rollback"
                          >
                            <RotateCcw className="h-4 w-4" />
                          </button>
                        )}
                        <button
                          onClick={() => setSelectedDeployment(deployment)}
                          className="p-2 bg-gray-500/10 hover:bg-gray-500/20 text-gray-400 rounded-lg transition-colors"
                          title="View Details"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                        <button className="p-2 bg-gray-500/10 hover:bg-gray-500/20 text-gray-400 rounded-lg transition-colors">
                          <MoreVertical className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  {deployment.status === 'deploying' && (
                    <div className="mb-4">
                      <div className="flex items-center justify-between text-sm mb-2">
                        <span className="text-gray-400">Deployment Progress</span>
                        <span className="text-blue-400">{deployment.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <motion.div
                          className="bg-blue-500 h-2 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${deployment.progress}%` }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div className="flex items-center space-x-2">
                      <GitBranch className="h-4 w-4 text-gray-400" />
                      <span className="text-sm text-gray-300">{deployment.branch}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <GitCommit className="h-4 w-4 text-gray-400" />
                      <span className="text-sm font-mono text-gray-300">{deployment.commit}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Users className="h-4 w-4 text-gray-400" />
                      <span className="text-sm text-gray-300">{deployment.author}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Database className="h-4 w-4 text-gray-400" />
                      <span className="text-sm text-gray-300">{deployment.size}</span>
                    </div>
                  </div>

                  {/* Instance Status */}
                  <div className="flex items-center space-x-6 text-sm">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span className="text-gray-400">Successful: {deployment.successfulInstances}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                      <span className="text-gray-400">Failed: {deployment.failedInstances}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                      <span className="text-gray-400">Total: {deployment.targetInstances}</span>
                    </div>
                    {deployment.approvals < deployment.requiredApprovals && (
                      <div className="flex items-center space-x-2">
                        <Shield className="h-4 w-4 text-yellow-400" />
                        <span className="text-yellow-400">Approvals: {deployment.approvals}/{deployment.requiredApprovals}</span>
                      </div>
                    )}
                  </div>

                  {/* Expanded Details */}
                  {expandedDeployments.has(deployment.id) && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="mt-4 pt-4 border-t border-gray-700"
                    >
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div>
                          <h5 className="font-medium text-white mb-2">Deployment Logs</h5>
                          <div className="bg-gray-800/50 rounded-lg p-3 space-y-1 max-h-40 overflow-y-auto">
                            {deployment.logs.map((log, logIndex) => (
                              <div key={logIndex} className="text-xs font-mono text-gray-300">
                                <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span> {log}
                              </div>
                            ))}
                          </div>
                        </div>
                        <div>
                          <h5 className="font-medium text-white mb-2">Artifacts & Resources</h5>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between p-2 bg-gray-800/30 rounded">
                              <span className="text-sm text-gray-300">Build Artifacts</span>
                              <span className="text-sm text-blue-400">{deployment.artifacts} files</span>
                            </div>
                            <div className="flex items-center justify-between p-2 bg-gray-800/30 rounded">
                              <span className="text-sm text-gray-300">Package Size</span>
                              <span className="text-sm text-blue-400">{deployment.size}</span>
                            </div>
                            <div className="flex items-center justify-between p-2 bg-gray-800/30 rounded">
                              <span className="text-sm text-gray-300">Strategy</span>
                              <span className="text-sm text-blue-400">{deployment.strategy}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* Environments Tab */}
        {selectedTab === 'environments' && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
            {environments.map((environment, index) => (
              <motion.div
                key={environment.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-semibold text-white text-lg">{environment.name}</h3>
                    <p className="text-sm text-gray-400 capitalize">{environment.type}</p>
                  </div>
                  <div className={`p-2 rounded-lg ${getEnvironmentStatusColor(environment.status)}`}>
                    <Server className="h-5 w-5" />
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Status</span>
                    <span className={`text-sm font-medium capitalize ${
                      environment.status === 'healthy' ? 'text-green-400' :
                      environment.status === 'degraded' ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {environment.status}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Deployments</span>
                    <span className="text-sm font-medium text-white">{environment.deployments}</span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Last Deployment</span>
                    <span className="text-sm font-medium text-white">{environment.lastDeployment}</span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Uptime</span>
                    <span className="text-sm font-medium text-green-400">{environment.uptime}%</span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Instances</span>
                    <span className="text-sm font-medium text-white">{environment.instances}</span>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">CPU Usage</span>
                      <span className="text-blue-400">{environment.cpu}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-1.5">
                      <div 
                        className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${environment.cpu}%` }}
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Memory Usage</span>
                      <span className="text-purple-400">{environment.memory}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-1.5">
                      <div 
                        className="bg-purple-500 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${environment.memory}%` }}
                      />
                    </div>
                  </div>

                  <div className="pt-2 border-t border-gray-700">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Requests/min</span>
                      <span className="text-sm font-medium text-white">{environment.requests.toLocaleString()}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-4 flex space-x-2">
                  <button className="flex-1 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 py-2 px-3 rounded-lg text-sm transition-colors">
                    Deploy
                  </button>
                  <button className="flex-1 bg-gray-500/10 hover:bg-gray-500/20 text-gray-400 py-2 px-3 rounded-lg text-sm transition-colors">
                    Monitor
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        )}

        {/* Analytics Tab */}
        {selectedTab === 'analytics' && (
          <div className="space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Duration Analysis */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4">Deployment Duration Analysis</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={durationAnalysisData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="range" stroke="#9CA3AF" />
                    <YAxis stroke="#9CA3AF" />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '8px'
                      }}
                    />
                    <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </motion.div>

              {/* Success Rate Trend */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4">Success Rate Trend</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={deploymentTrendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="date" stroke="#9CA3AF" />
                    <YAxis stroke="#9CA3AF" />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '8px'
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="successful" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      dot={{ fill: '#10b981', strokeWidth: 2 }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="failed" 
                      stroke="#ef4444" 
                      strokeWidth={2}
                      dot={{ fill: '#ef4444', strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </motion.div>
            </div>

            {/* Performance Metrics */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6"
            >
              <h3 className="text-lg font-semibold text-white mb-4">Performance Insights</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gray-800/30 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <Zap className="h-5 w-5 text-yellow-400" />
                    <h4 className="font-medium text-white">Fastest Deployment</h4>
                  </div>
                  <p className="text-2xl font-bold text-yellow-400">2m 15s</p>
                  <p className="text-sm text-gray-400">PolicyCortex-API v1.2.3</p>
                </div>
                
                <div className="bg-gray-800/30 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <Target className="h-5 w-5 text-green-400" />
                    <h4 className="font-medium text-white">Best Success Rate</h4>
                  </div>
                  <p className="text-2xl font-bold text-green-400">98.7%</p>
                  <p className="text-sm text-gray-400">Production environment</p>
                </div>
                
                <div className="bg-gray-800/30 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <TrendingUp className="h-5 w-5 text-blue-400" />
                    <h4 className="font-medium text-white">Improvement</h4>
                  </div>
                  <p className="text-2xl font-bold text-blue-400">+23%</p>
                  <p className="text-sm text-gray-400">Deployment speed this month</p>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </div>

      {/* Deployment Details Modal */}
      {selectedDeployment && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-4xl w-full max-h-[80vh] overflow-y-auto"
          >
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-xl font-bold text-white">{selectedDeployment.name}</h2>
                <p className="text-gray-400">{selectedDeployment.application} • {selectedDeployment.version}</p>
              </div>
              <button
                onClick={() => setSelectedDeployment(null)}
                className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
              >
                <XCircle className="h-6 w-6 text-gray-400" />
              </button>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-white mb-3">Deployment Details</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Status</span>
                    <span className={`capitalize ${
                      selectedDeployment.status === 'success' ? 'text-green-400' :
                      selectedDeployment.status === 'failed' ? 'text-red-400' :
                      selectedDeployment.status === 'deploying' ? 'text-blue-400' :
                      'text-yellow-400'
                    }`}>
                      {selectedDeployment.status}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Environment</span>
                    <span className="text-white capitalize">{selectedDeployment.environment}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Author</span>
                    <span className="text-white">{selectedDeployment.author}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Branch</span>
                    <span className="text-white">{selectedDeployment.branch}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Commit</span>
                    <span className="text-white font-mono">{selectedDeployment.commit}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Strategy</span>
                    <span className="text-white">{selectedDeployment.strategy}</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="font-semibold text-white mb-3">Deployment Logs</h3>
                <div className="bg-gray-800/50 rounded-lg p-3 space-y-1 h-64 overflow-y-auto">
                  {selectedDeployment.logs.map((log, index) => (
                    <div key={index} className="text-xs font-mono text-gray-300">
                      <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span> {log}
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="mt-6 flex justify-end space-x-3">
              {selectedDeployment.rollbackAvailable && (
                <button className="flex items-center space-x-2 bg-orange-500/10 hover:bg-orange-500/20 text-orange-400 px-4 py-2 rounded-lg transition-colors">
                  <RotateCcw className="h-4 w-4" />
                  <span>Rollback</span>
                </button>
              )}
              <button className="flex items-center space-x-2 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 px-4 py-2 rounded-lg transition-colors">
                <Download className="h-4 w-4" />
                <span>Download Logs</span>
              </button>
              <button
                onClick={() => setSelectedDeployment(null)}
                className="flex items-center space-x-2 bg-gray-500/10 hover:bg-gray-500/20 text-gray-400 px-4 py-2 rounded-lg transition-colors"
              >
                <span>Close</span>
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}