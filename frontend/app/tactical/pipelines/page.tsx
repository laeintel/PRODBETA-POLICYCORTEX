'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { 
  GitBranch, Play, Pause, Square, RefreshCw, Clock, CheckCircle, XCircle, 
  AlertTriangle, Download, Upload, Settings, Filter, Search, Calendar,
  Activity, TrendingUp, TrendingDown, Eye, Edit, Trash2, Plus, MoreHorizontal,
  Timer, Code, FileText, Package, Deploy, Terminal, GitMerge, Users,
  Server, Database, Shield, Zap, Target, Award, BarChart3, PieChart
} from 'lucide-react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';

interface Pipeline {
  id: string;
  name: string;
  repository: string;
  branch: string;
  status: 'running' | 'success' | 'failed' | 'pending' | 'cancelled' | 'queued';
  stage: string;
  progress: number;
  duration: number;
  startedAt: string;
  triggeredBy: string;
  buildNumber: number;
  environment: string;
  lastRun: string;
  successRate: number;
  avgDuration: number;
  deployments: number;
  commits: number;
  coverage: number;
  quality: number;
  security: number;
  performance: number;
  tags: string[];
  artifacts: number;
  tests: {
    total: number;
    passed: number;
    failed: number;
    skipped: number;
  };
}

interface PipelineTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  stages: string[];
  estimatedDuration: number;
  complexity: 'low' | 'medium' | 'high';
  popularity: number;
}

interface BuildMetrics {
  totalBuilds: number;
  successfulBuilds: number;
  failedBuilds: number;
  avgBuildTime: number;
  activeBuilds: number;
  queuedBuilds: number;
  deploymentsToday: number;
  successRate: number;
  throughput: number;
  mttr: number; // Mean Time To Recovery
  leadTime: number;
  deploymentFreq: number;
}

const mockPipelines: Pipeline[] = [
  {
    id: '1',
    name: 'PolicyCortex Core API',
    repository: 'policycortex/core-api',
    branch: 'main',
    status: 'running',
    stage: 'Test Execution',
    progress: 65,
    duration: 12,
    startedAt: '2024-01-20T10:30:00Z',
    triggeredBy: 'Sarah Chen',
    buildNumber: 1247,
    environment: 'production',
    lastRun: '2024-01-20T08:15:00Z',
    successRate: 94.5,
    avgDuration: 15,
    deployments: 142,
    commits: 8,
    coverage: 87.5,
    quality: 92.3,
    security: 96.1,
    performance: 88.7,
    tags: ['critical', 'api', 'rust'],
    artifacts: 5,
    tests: { total: 1247, passed: 1198, failed: 12, skipped: 37 }
  },
  {
    id: '2',
    name: 'Frontend Web App',
    repository: 'policycortex/frontend',
    branch: 'develop',
    status: 'success',
    stage: 'Completed',
    progress: 100,
    duration: 8,
    startedAt: '2024-01-20T09:45:00Z',
    triggeredBy: 'Auto trigger',
    buildNumber: 892,
    environment: 'staging',
    lastRun: '2024-01-20T09:53:00Z',
    successRate: 96.8,
    avgDuration: 9,
    deployments: 78,
    commits: 12,
    coverage: 82.1,
    quality: 89.6,
    security: 93.4,
    performance: 91.2,
    tags: ['frontend', 'nextjs', 'typescript'],
    artifacts: 3,
    tests: { total: 687, passed: 679, failed: 0, skipped: 8 }
  },
  {
    id: '3',
    name: 'AI Engine Services',
    repository: 'policycortex/ai-engine',
    branch: 'feature/ml-optimization',
    status: 'failed',
    stage: 'Security Scan',
    progress: 45,
    duration: 22,
    startedAt: '2024-01-20T07:20:00Z',
    triggeredBy: 'Mike Rodriguez',
    buildNumber: 456,
    environment: 'development',
    lastRun: '2024-01-20T07:42:00Z',
    successRate: 87.2,
    avgDuration: 18,
    deployments: 23,
    commits: 5,
    coverage: 79.3,
    quality: 85.1,
    security: 72.8,
    performance: 86.9,
    tags: ['ai', 'python', 'ml'],
    artifacts: 8,
    tests: { total: 423, passed: 387, failed: 28, skipped: 8 }
  },
  {
    id: '4',
    name: 'GraphQL Gateway',
    repository: 'policycortex/graphql-gateway',
    branch: 'main',
    status: 'pending',
    stage: 'Queue',
    progress: 0,
    duration: 0,
    startedAt: '2024-01-20T10:35:00Z',
    triggeredBy: 'Lisa Wang',
    buildNumber: 234,
    environment: 'staging',
    lastRun: '2024-01-19T16:22:00Z',
    successRate: 91.7,
    avgDuration: 7,
    deployments: 56,
    commits: 3,
    coverage: 85.7,
    quality: 93.2,
    security: 94.8,
    performance: 89.3,
    tags: ['graphql', 'nodejs', 'gateway'],
    artifacts: 2,
    tests: { total: 298, passed: 291, failed: 2, skipped: 5 }
  },
  {
    id: '5',
    name: 'Mobile App (iOS)',
    repository: 'policycortex/mobile-ios',
    branch: 'release/v2.1',
    status: 'queued',
    stage: 'Waiting',
    progress: 0,
    duration: 0,
    startedAt: '2024-01-20T10:40:00Z',
    triggeredBy: 'Auto trigger',
    buildNumber: 167,
    environment: 'production',
    lastRun: '2024-01-19T14:30:00Z',
    successRate: 89.4,
    avgDuration: 25,
    deployments: 12,
    commits: 7,
    coverage: 76.5,
    quality: 87.9,
    security: 91.2,
    performance: 84.6,
    tags: ['mobile', 'ios', 'swift'],
    artifacts: 4,
    tests: { total: 534, passed: 489, failed: 31, skipped: 14 }
  },
  {
    id: '6',
    name: 'Infrastructure as Code',
    repository: 'policycortex/infrastructure',
    branch: 'main',
    status: 'cancelled',
    stage: 'Infrastructure Deploy',
    progress: 30,
    duration: 5,
    startedAt: '2024-01-20T09:15:00Z',
    triggeredBy: 'DevOps Team',
    buildNumber: 89,
    environment: 'production',
    lastRun: '2024-01-20T09:20:00Z',
    successRate: 93.6,
    avgDuration: 12,
    deployments: 34,
    commits: 2,
    coverage: 0,
    quality: 95.4,
    security: 97.2,
    performance: 0,
    tags: ['infrastructure', 'terraform', 'azure'],
    artifacts: 1,
    tests: { total: 0, passed: 0, failed: 0, skipped: 0 }
  }
];

const mockTemplates: PipelineTemplate[] = [
  {
    id: '1',
    name: 'Node.js Web Application',
    description: 'Complete CI/CD pipeline for Node.js applications with testing, security scanning, and deployment',
    category: 'Web Applications',
    stages: ['Build', 'Test', 'Security Scan', 'Deploy to Staging', 'Integration Tests', 'Deploy to Production'],
    estimatedDuration: 12,
    complexity: 'medium',
    popularity: 85
  },
  {
    id: '2',
    name: 'Rust Microservice',
    description: 'High-performance Rust service pipeline with cargo build, clippy, and container deployment',
    category: 'Microservices',
    stages: ['Cargo Build', 'Unit Tests', 'Clippy Check', 'Security Audit', 'Container Build', 'Deploy'],
    estimatedDuration: 18,
    complexity: 'high',
    popularity: 72
  },
  {
    id: '3',
    name: 'Python ML Pipeline',
    description: 'Machine learning pipeline with model training, validation, and deployment to inference endpoints',
    category: 'AI/ML',
    stages: ['Environment Setup', 'Data Validation', 'Model Training', 'Model Testing', 'Deploy Model', 'Monitor'],
    estimatedDuration: 45,
    complexity: 'high',
    popularity: 68
  },
  {
    id: '4',
    name: 'Infrastructure Deployment',
    description: 'Terraform-based infrastructure provisioning with validation and drift detection',
    category: 'Infrastructure',
    stages: ['Plan', 'Validate', 'Security Check', 'Apply', 'Test Infrastructure', 'Notify'],
    estimatedDuration: 15,
    complexity: 'medium',
    popularity: 91
  }
];

const mockMetrics: BuildMetrics = {
  totalBuilds: 2847,
  successfulBuilds: 2634,
  failedBuilds: 213,
  avgBuildTime: 14.5,
  activeBuilds: 3,
  queuedBuilds: 7,
  deploymentsToday: 12,
  successRate: 92.5,
  throughput: 156, // builds per day
  mttr: 23, // minutes
  leadTime: 4.2, // hours
  deploymentFreq: 8.5 // per day
};

export default function PipelinesPage() {
  const [pipelines, setPipelines] = useState<Pipeline[]>(mockPipelines);
  const [selectedPipeline, setSelectedPipeline] = useState<Pipeline | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [environmentFilter, setEnvironmentFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'name' | 'lastRun' | 'successRate' | 'duration'>('lastRun');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [showTemplates, setShowTemplates] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedEnvironments, setSelectedEnvironments] = useState<string[]>([]);
  const [dateRange, setDateRange] = useState<'today' | 'week' | 'month'>('today');

  // Auto-refresh simulation
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      setPipelines(prevPipelines => 
        prevPipelines.map(pipeline => {
          if (pipeline.status === 'running') {
            const newProgress = Math.min(100, pipeline.progress + Math.random() * 10);
            if (newProgress >= 100) {
              return {
                ...pipeline,
                status: Math.random() > 0.15 ? 'success' : 'failed',
                progress: 100,
                duration: pipeline.duration + Math.random() * 5
              };
            }
            return {
              ...pipeline,
              progress: newProgress,
              duration: pipeline.duration + 0.5
            };
          }
          return pipeline;
        })
      );
    }, 2000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const filteredAndSortedPipelines = useMemo(() => {
    let filtered = pipelines.filter(pipeline => {
      const matchesSearch = pipeline.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           pipeline.repository.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           pipeline.branch.toLowerCase().includes(searchTerm.toLowerCase());
      
      const matchesStatus = statusFilter === 'all' || pipeline.status === statusFilter;
      const matchesEnvironment = environmentFilter === 'all' || pipeline.environment === environmentFilter;
      
      return matchesSearch && matchesStatus && matchesEnvironment;
    });

    filtered.sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'lastRun':
          comparison = new Date(a.lastRun).getTime() - new Date(b.lastRun).getTime();
          break;
        case 'successRate':
          comparison = a.successRate - b.successRate;
          break;
        case 'duration':
          comparison = a.avgDuration - b.avgDuration;
          break;
      }
      
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [pipelines, searchTerm, statusFilter, environmentFilter, sortBy, sortOrder]);

  const getStatusIcon = (status: Pipeline['status']) => {
    switch (status) {
      case 'running': return <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />;
      case 'success': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-400" />;
      case 'pending': return <Clock className="w-4 h-4 text-yellow-400" />;
      case 'cancelled': return <Square className="w-4 h-4 text-gray-400" />;
      case 'queued': return <Timer className="w-4 h-4 text-purple-400" />;
      default: return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: Pipeline['status']) => {
    switch (status) {
      case 'running': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'success': return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'failed': return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'pending': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'cancelled': return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
      case 'queued': return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const formatDuration = (minutes: number) => {
    if (minutes < 60) return `${Math.round(minutes)}m`;
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `${hours}h ${mins}m`;
  };

  const formatTimeAgo = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / (1000 * 60));
    
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  const executeAction = (pipeline: Pipeline, action: 'run' | 'stop' | 'restart') => {
    setPipelines(prev => prev.map(p => {
      if (p.id === pipeline.id) {
        switch (action) {
          case 'run':
            return { ...p, status: 'running', progress: 0, duration: 0, startedAt: new Date().toISOString() };
          case 'stop':
            return { ...p, status: 'cancelled', progress: p.progress };
          case 'restart':
            return { ...p, status: 'running', progress: 0, duration: 0, startedAt: new Date().toISOString() };
        }
      }
      return p;
    }));
  };

  return (
    <TacticalPageTemplate
      title="CI/CD Pipelines"
      subtitle="Continuous Integration & Deployment Management"
      icon={GitBranch}
    >
      <div className="space-y-6">
        {/* Metrics Dashboard */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-green-400" />
                <span className="text-sm font-medium text-gray-300">Active Builds</span>
              </div>
              <span className="text-xs text-green-400 bg-green-400/10 px-2 py-1 rounded-full">
                Live
              </span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-white">{mockMetrics.activeBuilds}</span>
              <span className="text-sm text-gray-400">/ {mockMetrics.queuedBuilds} queued</span>
            </div>
          </div>

          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-medium text-gray-300">Success Rate</span>
              </div>
              <TrendingUp className="w-4 h-4 text-green-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-white">{mockMetrics.successRate}%</span>
              <span className="text-sm text-green-400">+2.3%</span>
            </div>
          </div>

          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-yellow-400" />
                <span className="text-sm font-medium text-gray-300">Avg Build Time</span>
              </div>
              <TrendingDown className="w-4 h-4 text-green-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-white">{formatDuration(mockMetrics.avgBuildTime)}</span>
              <span className="text-sm text-green-400">-1.2m</span>
            </div>
          </div>

          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Deploy className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium text-gray-300">Deployments Today</span>
              </div>
              <Target className="w-4 h-4 text-blue-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-white">{mockMetrics.deploymentsToday}</span>
              <span className="text-sm text-blue-400">+4 since 9 AM</span>
            </div>
          </div>
        </div>

        {/* Advanced Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-white">DORA Metrics</h3>
              <BarChart3 className="w-4 h-4 text-gray-400" />
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Deployment Frequency</span>
                <span className="text-sm font-medium text-white">{mockMetrics.deploymentFreq}/day</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Lead Time</span>
                <span className="text-sm font-medium text-white">{mockMetrics.leadTime}h</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">MTTR</span>
                <span className="text-sm font-medium text-white">{mockMetrics.mttr}m</span>
              </div>
            </div>
          </div>

          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-white">Build Trends</h3>
              <TrendingUp className="w-4 h-4 text-green-400" />
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Throughput</span>
                <span className="text-sm font-medium text-white">{mockMetrics.throughput} builds/day</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Total Builds</span>
                <span className="text-sm font-medium text-white">{mockMetrics.totalBuilds.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Failed Builds</span>
                <span className="text-sm font-medium text-red-400">{mockMetrics.failedBuilds}</span>
              </div>
            </div>
          </div>

          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-white">Resource Usage</h3>
              <Server className="w-4 h-4 text-blue-400" />
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">CPU Usage</span>
                <span className="text-sm font-medium text-white">67%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Memory Usage</span>
                <span className="text-sm font-medium text-white">84%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Build Agents</span>
                <span className="text-sm font-medium text-green-400">12/15 active</span>
              </div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-black border border-gray-800 rounded-xl p-4">
          <div className="flex flex-col lg:flex-row gap-4">
            {/* Search and Filters */}
            <div className="flex-1 flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search pipelines..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                />
              </div>
              
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="all">All Statuses</option>
                <option value="running">Running</option>
                <option value="success">Success</option>
                <option value="failed">Failed</option>
                <option value="pending">Pending</option>
                <option value="queued">Queued</option>
                <option value="cancelled">Cancelled</option>
              </select>

              <select
                value={environmentFilter}
                onChange={(e) => setEnvironmentFilter(e.target.value)}
                className="px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="all">All Environments</option>
                <option value="production">Production</option>
                <option value="staging">Staging</option>
                <option value="development">Development</option>
              </select>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowTemplates(!showTemplates)}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                <Plus className="w-4 h-4" />
                New Pipeline
              </button>
              
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`flex items-center gap-2 px-4 py-2 border rounded-lg transition-colors ${
                  autoRefresh 
                    ? 'bg-green-600/20 border-green-500/30 text-green-400' 
                    : 'bg-gray-700 border-gray-600 text-gray-300 hover:bg-gray-600'
                }`}
              >
                <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
                Auto Refresh
              </button>

              <div className="flex border border-gray-700 rounded-lg overflow-hidden">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 transition-colors ${
                    viewMode === 'grid' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <BarChart3 className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 transition-colors ${
                    viewMode === 'list' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <FileText className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Pipeline Templates Modal */}
        {showTemplates && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-black border border-gray-800 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-white">Pipeline Templates</h2>
                  <button
                    onClick={() => setShowTemplates(false)}
                    className="text-gray-400 hover:text-white"
                  >
                    <XCircle className="w-6 h-6" />
                  </button>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {mockTemplates.map((template) => (
                    <div key={template.id} className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h3 className="font-semibold text-white">{template.name}</h3>
                          <p className="text-sm text-gray-400">{template.category}</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-1 rounded-full text-xs ${
                            template.complexity === 'low' ? 'bg-green-500/20 text-green-400' :
                            template.complexity === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                            'bg-red-500/20 text-red-400'
                          }`}>
                            {template.complexity}
                          </span>
                          <span className="text-xs text-gray-400">{template.popularity}% popular</span>
                        </div>
                      </div>
                      
                      <p className="text-sm text-gray-300 mb-4">{template.description}</p>
                      
                      <div className="space-y-2 mb-4">
                        <h4 className="text-xs font-medium text-gray-400">Pipeline Stages:</h4>
                        <div className="flex flex-wrap gap-1">
                          {template.stages.map((stage, index) => (
                            <span key={index} className="px-2 py-1 bg-gray-800 text-xs text-gray-300 rounded">
                              {stage}
                            </span>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-400">
                          Est. Duration: {formatDuration(template.estimatedDuration)}
                        </span>
                        <button className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors">
                          Use Template
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Pipelines Grid/List */}
        {viewMode === 'grid' ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {filteredAndSortedPipelines.map((pipeline) => (
              <div key={pipeline.id} className="bg-black border border-gray-800 rounded-xl p-4 hover:border-gray-700 transition-colors">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-white truncate">{pipeline.name}</h3>
                    <p className="text-sm text-gray-400 truncate">{pipeline.repository}</p>
                  </div>
                  <div className="flex items-center gap-2 ml-2">
                    <span className={`px-2 py-1 rounded-full text-xs border ${getStatusColor(pipeline.status)}`}>
                      {getStatusIcon(pipeline.status)}
                      <span className="ml-1">{pipeline.status}</span>
                    </span>
                  </div>
                </div>

                <div className="space-y-2 mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Branch:</span>
                    <span className="text-gray-300">{pipeline.branch}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Environment:</span>
                    <span className="text-gray-300">{pipeline.environment}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Build #:</span>
                    <span className="text-gray-300">{pipeline.buildNumber}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Last Run:</span>
                    <span className="text-gray-300">{formatTimeAgo(pipeline.lastRun)}</span>
                  </div>
                </div>

                {/* Progress Bar */}
                {pipeline.status === 'running' && (
                  <div className="mb-4">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">{pipeline.stage}</span>
                      <span className="text-gray-300">{Math.round(pipeline.progress)}%</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-1000 ease-out"
                        style={{ width: `${pipeline.progress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Quality Metrics */}
                <div className="grid grid-cols-2 gap-2 mb-4">
                  <div className="text-center">
                    <div className="text-lg font-bold text-white">{pipeline.successRate}%</div>
                    <div className="text-xs text-gray-400">Success Rate</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold text-white">{formatDuration(pipeline.avgDuration)}</div>
                    <div className="text-xs text-gray-400">Avg Duration</div>
                  </div>
                </div>

                {/* Tags */}
                <div className="flex flex-wrap gap-1 mb-4">
                  {pipeline.tags.map((tag) => (
                    <span key={tag} className="px-2 py-1 bg-gray-800 text-xs text-gray-300 rounded">
                      {tag}
                    </span>
                  ))}
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  {pipeline.status === 'running' ? (
                    <button
                      onClick={() => executeAction(pipeline, 'stop')}
                      className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                    >
                      <Square className="w-4 h-4" />
                      Stop
                    </button>
                  ) : (
                    <button
                      onClick={() => executeAction(pipeline, 'run')}
                      className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                    >
                      <Play className="w-4 h-4" />
                      Run
                    </button>
                  )}
                  
                  <button
                    onClick={() => setSelectedPipeline(pipeline)}
                    className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors"
                  >
                    <Eye className="w-4 h-4" />
                  </button>
                  
                  <button className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors">
                    <MoreHorizontal className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-black border border-gray-800 rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Pipeline</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Status</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Branch</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Environment</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Last Run</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Success Rate</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Duration</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredAndSortedPipelines.map((pipeline) => (
                    <tr key={pipeline.id} className="border-b border-gray-800 hover:bg-gray-900/50">
                      <td className="p-4">
                        <div>
                          <div className="font-medium text-white">{pipeline.name}</div>
                          <div className="text-sm text-gray-400">{pipeline.repository}</div>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(pipeline.status)}
                          <span className="text-sm text-gray-300 capitalize">{pipeline.status}</span>
                          {pipeline.status === 'running' && (
                            <span className="text-xs text-gray-400">({Math.round(pipeline.progress)}%)</span>
                          )}
                        </div>
                      </td>
                      <td className="p-4 text-sm text-gray-300">{pipeline.branch}</td>
                      <td className="p-4">
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          pipeline.environment === 'production' ? 'bg-red-500/20 text-red-400' :
                          pipeline.environment === 'staging' ? 'bg-yellow-500/20 text-yellow-400' :
                          'bg-blue-500/20 text-blue-400'
                        }`}>
                          {pipeline.environment}
                        </span>
                      </td>
                      <td className="p-4 text-sm text-gray-300">{formatTimeAgo(pipeline.lastRun)}</td>
                      <td className="p-4 text-sm text-gray-300">{pipeline.successRate}%</td>
                      <td className="p-4 text-sm text-gray-300">{formatDuration(pipeline.avgDuration)}</td>
                      <td className="p-4">
                        <div className="flex gap-2">
                          {pipeline.status === 'running' ? (
                            <button
                              onClick={() => executeAction(pipeline, 'stop')}
                              className="p-1 text-red-400 hover:text-red-300"
                            >
                              <Square className="w-4 h-4" />
                            </button>
                          ) : (
                            <button
                              onClick={() => executeAction(pipeline, 'run')}
                              className="p-1 text-green-400 hover:text-green-300"
                            >
                              <Play className="w-4 h-4" />
                            </button>
                          )}
                          <button
                            onClick={() => setSelectedPipeline(pipeline)}
                            className="p-1 text-blue-400 hover:text-blue-300"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button className="p-1 text-gray-400 hover:text-gray-300">
                            <Settings className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Pipeline Details Modal */}
        {selectedPipeline && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-black border border-gray-800 rounded-xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(selectedPipeline.status)}
                    <h2 className="text-xl font-bold text-white">{selectedPipeline.name}</h2>
                    <span className={`px-2 py-1 rounded-full text-xs border ${getStatusColor(selectedPipeline.status)}`}>
                      {selectedPipeline.status}
                    </span>
                  </div>
                  <button
                    onClick={() => setSelectedPipeline(null)}
                    className="text-gray-400 hover:text-white"
                  >
                    <XCircle className="w-6 h-6" />
                  </button>
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2 space-y-6">
                    {/* Pipeline Progress */}
                    {selectedPipeline.status === 'running' && (
                      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                        <h3 className="font-semibold text-white mb-3">Current Progress</h3>
                        <div className="space-y-3">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-300">{selectedPipeline.stage}</span>
                            <span className="text-blue-400">{Math.round(selectedPipeline.progress)}%</span>
                          </div>
                          <div className="w-full bg-gray-800 rounded-full h-3">
                            <div
                              className="bg-blue-600 h-3 rounded-full transition-all duration-1000 ease-out"
                              style={{ width: `${selectedPipeline.progress}%` }}
                            />
                          </div>
                          <div className="text-sm text-gray-400">
                            Running for {formatDuration(selectedPipeline.duration)}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Test Results */}
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-white mb-3">Test Results</h3>
                      <div className="grid grid-cols-4 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-white">{selectedPipeline.tests.total}</div>
                          <div className="text-xs text-gray-400">Total</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-400">{selectedPipeline.tests.passed}</div>
                          <div className="text-xs text-gray-400">Passed</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-red-400">{selectedPipeline.tests.failed}</div>
                          <div className="text-xs text-gray-400">Failed</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-yellow-400">{selectedPipeline.tests.skipped}</div>
                          <div className="text-xs text-gray-400">Skipped</div>
                        </div>
                      </div>
                    </div>

                    {/* Quality Metrics */}
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-white mb-3">Quality Metrics</h3>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <div className="flex justify-between mb-2">
                            <span className="text-sm text-gray-400">Code Coverage</span>
                            <span className="text-sm text-white">{selectedPipeline.coverage}%</span>
                          </div>
                          <div className="w-full bg-gray-800 rounded-full h-2">
                            <div
                              className="bg-green-600 h-2 rounded-full"
                              style={{ width: `${selectedPipeline.coverage}%` }}
                            />
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between mb-2">
                            <span className="text-sm text-gray-400">Code Quality</span>
                            <span className="text-sm text-white">{selectedPipeline.quality}%</span>
                          </div>
                          <div className="w-full bg-gray-800 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full"
                              style={{ width: `${selectedPipeline.quality}%` }}
                            />
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between mb-2">
                            <span className="text-sm text-gray-400">Security Score</span>
                            <span className="text-sm text-white">{selectedPipeline.security}%</span>
                          </div>
                          <div className="w-full bg-gray-800 rounded-full h-2">
                            <div
                              className="bg-purple-600 h-2 rounded-full"
                              style={{ width: `${selectedPipeline.security}%` }}
                            />
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between mb-2">
                            <span className="text-sm text-gray-400">Performance</span>
                            <span className="text-sm text-white">{selectedPipeline.performance}%</span>
                          </div>
                          <div className="w-full bg-gray-800 rounded-full h-2">
                            <div
                              className="bg-yellow-600 h-2 rounded-full"
                              style={{ width: `${selectedPipeline.performance}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-6">
                    {/* Pipeline Info */}
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-white mb-3">Pipeline Info</h3>
                      <div className="space-y-3">
                        <div>
                          <div className="text-xs text-gray-400">Repository</div>
                          <div className="text-sm text-white">{selectedPipeline.repository}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Branch</div>
                          <div className="text-sm text-white">{selectedPipeline.branch}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Environment</div>
                          <div className="text-sm text-white">{selectedPipeline.environment}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Build Number</div>
                          <div className="text-sm text-white">#{selectedPipeline.buildNumber}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Triggered By</div>
                          <div className="text-sm text-white">{selectedPipeline.triggeredBy}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Artifacts</div>
                          <div className="text-sm text-white">{selectedPipeline.artifacts} files</div>
                        </div>
                      </div>
                    </div>

                    {/* Historical Stats */}
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-white mb-3">Historical Stats</h3>
                      <div className="space-y-3">
                        <div>
                          <div className="text-xs text-gray-400">Success Rate</div>
                          <div className="text-sm text-white">{selectedPipeline.successRate}%</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Average Duration</div>
                          <div className="text-sm text-white">{formatDuration(selectedPipeline.avgDuration)}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Total Deployments</div>
                          <div className="text-sm text-white">{selectedPipeline.deployments}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Recent Commits</div>
                          <div className="text-sm text-white">{selectedPipeline.commits}</div>
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="space-y-2">
                      {selectedPipeline.status === 'running' ? (
                        <button
                          onClick={() => {
                            executeAction(selectedPipeline, 'stop');
                            setSelectedPipeline(null);
                          }}
                          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                        >
                          <Square className="w-4 h-4" />
                          Stop Pipeline
                        </button>
                      ) : (
                        <button
                          onClick={() => {
                            executeAction(selectedPipeline, 'run');
                            setSelectedPipeline(null);
                          }}
                          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                        >
                          <Play className="w-4 h-4" />
                          Run Pipeline
                        </button>
                      )}
                      
                      <button className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors">
                        <Edit className="w-4 h-4" />
                        Edit Pipeline
                      </button>
                      
                      <button className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors">
                        <Download className="w-4 h-4" />
                        Download Logs
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </TacticalPageTemplate>
  );
}