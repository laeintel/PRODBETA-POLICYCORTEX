'use client';

import React, { useState, useEffect } from 'react';
import { 
  GitBranch, GitCommit, GitMerge, GitPullRequest, Package, Archive,
  Play, Pause, StopCircle, RotateCw, CheckCircle, XCircle, Clock,
  Activity, Zap, Server, Container, Layers, Cloud, Database,
  Terminal, Code, FileCode, Settings, Wrench, Hammer,
  TrendingUp, TrendingDown, BarChart3, LineChart, PieChart,
  AlertTriangle, Info, Bell, Shield, Lock, Key, UserCheck,
  Calendar, Timer, RefreshCw, Download, Upload, Search, Filter,
  ChevronRight, ChevronDown, MoreVertical, ExternalLink, Copy,
  Folder, File, FileText, Hash, Tag, Bookmark, Flag, Target
} from 'lucide-react';

interface Pipeline {
  id: string;
  name: string;
  branch: string;
  status: 'running' | 'succeeded' | 'failed' | 'cancelled' | 'pending' | 'queued';
  stage: string;
  startTime: string;
  duration?: number;
  trigger: 'manual' | 'commit' | 'schedule' | 'api' | 'merge';
  author: string;
  commit: {
    hash: string;
    message: string;
    timestamp: string;
  };
  stages: {
    name: string;
    status: 'running' | 'succeeded' | 'failed' | 'skipped' | 'pending';
    duration?: number;
    jobs: number;
  }[];
  artifacts?: {
    name: string;
    size: string;
    type: string;
  }[];
  tests?: {
    passed: number;
    failed: number;
    skipped: number;
    coverage: number;
  };
}

interface Deployment {
  id: string;
  pipeline: string;
  environment: 'development' | 'staging' | 'production' | 'testing';
  status: 'deploying' | 'deployed' | 'failed' | 'rolled_back';
  version: string;
  deployedBy: string;
  deployedAt: string;
  services: {
    name: string;
    status: 'healthy' | 'unhealthy' | 'unknown';
    replicas: number;
    cpu: number;
    memory: number;
  }[];
  rollbackAvailable: boolean;
}

interface Repository {
  id: string;
  name: string;
  language: string;
  branches: number;
  pullRequests: number;
  lastCommit: string;
  cicdEnabled: boolean;
  coverage: number;
  vulnerabilities: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
  activity: {
    commits: number;
    contributors: number;
    issues: number;
  };
}

interface BuildMetric {
  id: string;
  title: string;
  value: number | string;
  unit?: string;
  trend: 'up' | 'down' | 'stable';
  change: number;
  category: string;
}

export default function DevOpsPipelineManagement() {
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [repositories, setRepositories] = useState<Repository[]>([]);
  const [selectedView, setSelectedView] = useState<'pipelines' | 'deployments' | 'repositories' | 'analytics'>('pipelines');
  const [selectedPipeline, setSelectedPipeline] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterEnvironment, setFilterEnvironment] = useState('all');
  const [metrics, setMetrics] = useState<BuildMetric[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    // Initialize with pipeline data
    setPipelines([
      {
        id: 'PIPE-001',
        name: 'Main Build Pipeline',
        branch: 'main',
        status: 'running',
        stage: 'Deploy to Production',
        startTime: '5 minutes ago',
        duration: 312,
        trigger: 'commit',
        author: 'john.doe@company.com',
        commit: {
          hash: 'a1b2c3d4',
          message: 'feat: Add new authentication module',
          timestamp: '10 minutes ago'
        },
        stages: [
          { name: 'Build', status: 'succeeded', duration: 120, jobs: 3 },
          { name: 'Test', status: 'succeeded', duration: 85, jobs: 5 },
          { name: 'Security Scan', status: 'succeeded', duration: 45, jobs: 2 },
          { name: 'Deploy to Staging', status: 'succeeded', duration: 60, jobs: 4 },
          { name: 'Deploy to Production', status: 'running', jobs: 4 }
        ],
        artifacts: [
          { name: 'app-bundle.zip', size: '45.2 MB', type: 'application' },
          { name: 'test-results.xml', size: '128 KB', type: 'test' }
        ],
        tests: {
          passed: 342,
          failed: 0,
          skipped: 5,
          coverage: 87.5
        }
      },
      {
        id: 'PIPE-002',
        name: 'Feature Branch Build',
        branch: 'feature/user-dashboard',
        status: 'succeeded',
        stage: 'Completed',
        startTime: '2 hours ago',
        duration: 485,
        trigger: 'merge',
        author: 'jane.smith@company.com',
        commit: {
          hash: 'e5f6g7h8',
          message: 'Merge pull request #142 from feature/user-dashboard',
          timestamp: '2 hours ago'
        },
        stages: [
          { name: 'Build', status: 'succeeded', duration: 150, jobs: 3 },
          { name: 'Test', status: 'succeeded', duration: 120, jobs: 5 },
          { name: 'Security Scan', status: 'succeeded', duration: 55, jobs: 2 },
          { name: 'Deploy to Dev', status: 'succeeded', duration: 80, jobs: 4 },
          { name: 'Integration Tests', status: 'succeeded', duration: 80, jobs: 3 }
        ],
        tests: {
          passed: 298,
          failed: 2,
          skipped: 8,
          coverage: 82.3
        }
      },
      {
        id: 'PIPE-003',
        name: 'Hotfix Pipeline',
        branch: 'hotfix/security-patch',
        status: 'failed',
        stage: 'Test',
        startTime: '30 minutes ago',
        duration: 180,
        trigger: 'manual',
        author: 'admin@company.com',
        commit: {
          hash: 'i9j0k1l2',
          message: 'fix: Critical security vulnerability patch',
          timestamp: '35 minutes ago'
        },
        stages: [
          { name: 'Build', status: 'succeeded', duration: 100, jobs: 3 },
          { name: 'Test', status: 'failed', duration: 80, jobs: 5 },
          { name: 'Security Scan', status: 'skipped', jobs: 2 },
          { name: 'Deploy', status: 'skipped', jobs: 4 }
        ],
        tests: {
          passed: 285,
          failed: 15,
          skipped: 8,
          coverage: 81.2
        }
      },
      {
        id: 'PIPE-004',
        name: 'Nightly Build',
        branch: 'develop',
        status: 'queued',
        stage: 'Waiting',
        startTime: 'Scheduled',
        trigger: 'schedule',
        author: 'system',
        commit: {
          hash: 'm3n4o5p6',
          message: 'chore: Daily dependency updates',
          timestamp: '8 hours ago'
        },
        stages: [
          { name: 'Build', status: 'pending', jobs: 3 },
          { name: 'Test', status: 'pending', jobs: 5 },
          { name: 'Security Scan', status: 'pending', jobs: 2 },
          { name: 'Deploy to Test', status: 'pending', jobs: 4 }
        ]
      },
      {
        id: 'PIPE-005',
        name: 'Release Pipeline',
        branch: 'release/v2.5.0',
        status: 'succeeded',
        stage: 'Completed',
        startTime: '1 day ago',
        duration: 720,
        trigger: 'api',
        author: 'release-bot',
        commit: {
          hash: 'q7r8s9t0',
          message: 'chore: Release version 2.5.0',
          timestamp: '1 day ago'
        },
        stages: [
          { name: 'Build', status: 'succeeded', duration: 180, jobs: 4 },
          { name: 'Test', status: 'succeeded', duration: 200, jobs: 8 },
          { name: 'Security Scan', status: 'succeeded', duration: 80, jobs: 3 },
          { name: 'Package', status: 'succeeded', duration: 60, jobs: 2 },
          { name: 'Deploy', status: 'succeeded', duration: 200, jobs: 6 }
        ],
        artifacts: [
          { name: 'release-2.5.0.tar.gz', size: '125 MB', type: 'release' },
          { name: 'release-notes.md', size: '12 KB', type: 'documentation' },
          { name: 'checksums.txt', size: '1 KB', type: 'verification' }
        ],
        tests: {
          passed: 1245,
          failed: 0,
          skipped: 12,
          coverage: 91.3
        }
      }
    ]);

    setDeployments([
      {
        id: 'DEP-001',
        pipeline: 'Main Build Pipeline',
        environment: 'production',
        status: 'deploying',
        version: 'v2.4.8',
        deployedBy: 'john.doe@company.com',
        deployedAt: '5 minutes ago',
        services: [
          { name: 'API Gateway', status: 'healthy', replicas: 3, cpu: 45, memory: 62 },
          { name: 'Auth Service', status: 'healthy', replicas: 2, cpu: 38, memory: 55 },
          { name: 'User Service', status: 'unknown', replicas: 3, cpu: 0, memory: 0 }
        ],
        rollbackAvailable: true
      },
      {
        id: 'DEP-002',
        pipeline: 'Feature Branch Build',
        environment: 'staging',
        status: 'deployed',
        version: 'v2.5.0-beta.1',
        deployedBy: 'jane.smith@company.com',
        deployedAt: '2 hours ago',
        services: [
          { name: 'API Gateway', status: 'healthy', replicas: 2, cpu: 32, memory: 48 },
          { name: 'Auth Service', status: 'healthy', replicas: 1, cpu: 25, memory: 40 },
          { name: 'User Service', status: 'healthy', replicas: 2, cpu: 35, memory: 52 },
          { name: 'Notification Service', status: 'healthy', replicas: 1, cpu: 20, memory: 35 }
        ],
        rollbackAvailable: true
      },
      {
        id: 'DEP-003',
        pipeline: 'Hotfix Pipeline',
        environment: 'development',
        status: 'failed',
        version: 'v2.4.7-hotfix.1',
        deployedBy: 'admin@company.com',
        deployedAt: '30 minutes ago',
        services: [
          { name: 'API Gateway', status: 'unhealthy', replicas: 1, cpu: 0, memory: 0 }
        ],
        rollbackAvailable: false
      },
      {
        id: 'DEP-004',
        pipeline: 'Release Pipeline',
        environment: 'production',
        status: 'deployed',
        version: 'v2.4.7',
        deployedBy: 'release-bot',
        deployedAt: '1 day ago',
        services: [
          { name: 'API Gateway', status: 'healthy', replicas: 3, cpu: 40, memory: 58 },
          { name: 'Auth Service', status: 'healthy', replicas: 2, cpu: 35, memory: 50 },
          { name: 'User Service', status: 'healthy', replicas: 3, cpu: 42, memory: 55 },
          { name: 'Notification Service', status: 'healthy', replicas: 2, cpu: 28, memory: 42 },
          { name: 'Analytics Service', status: 'healthy', replicas: 2, cpu: 38, memory: 60 }
        ],
        rollbackAvailable: true
      }
    ]);

    setRepositories([
      {
        id: 'REPO-001',
        name: 'backend-api',
        language: 'TypeScript',
        branches: 24,
        pullRequests: 8,
        lastCommit: '2 hours ago',
        cicdEnabled: true,
        coverage: 87.5,
        vulnerabilities: { critical: 0, high: 2, medium: 5, low: 12 },
        activity: { commits: 142, contributors: 8, issues: 23 }
      },
      {
        id: 'REPO-002',
        name: 'frontend-app',
        language: 'React',
        branches: 18,
        pullRequests: 5,
        lastCommit: '5 hours ago',
        cicdEnabled: true,
        coverage: 78.3,
        vulnerabilities: { critical: 1, high: 3, medium: 8, low: 15 },
        activity: { commits: 98, contributors: 6, issues: 15 }
      },
      {
        id: 'REPO-003',
        name: 'mobile-app',
        language: 'React Native',
        branches: 12,
        pullRequests: 3,
        lastCommit: '1 day ago',
        cicdEnabled: true,
        coverage: 72.1,
        vulnerabilities: { critical: 0, high: 1, medium: 4, low: 8 },
        activity: { commits: 65, contributors: 4, issues: 10 }
      },
      {
        id: 'REPO-004',
        name: 'infrastructure',
        language: 'Terraform',
        branches: 8,
        pullRequests: 2,
        lastCommit: '3 days ago',
        cicdEnabled: true,
        coverage: 0,
        vulnerabilities: { critical: 0, high: 0, medium: 2, low: 5 },
        activity: { commits: 34, contributors: 3, issues: 5 }
      }
    ]);

    setMetrics([
      { id: 'M1', title: 'Build Success Rate', value: 85.2, unit: '%', trend: 'up', change: 5.3, category: 'reliability' },
      { id: 'M2', title: 'Avg Build Time', value: 6.5, unit: 'min', trend: 'down', change: -12, category: 'performance' },
      { id: 'M3', title: 'Deployments Today', value: 24, trend: 'up', change: 20, category: 'activity' },
      { id: 'M4', title: 'Failed Builds', value: 3, trend: 'down', change: -25, category: 'reliability' },
      { id: 'M5', title: 'Code Coverage', value: 82.4, unit: '%', trend: 'up', change: 2.1, category: 'quality' },
      { id: 'M6', title: 'Active Pipelines', value: 12, trend: 'stable', change: 0, category: 'activity' },
      { id: 'M7', title: 'Rollback Rate', value: 2.3, unit: '%', trend: 'down', change: -15, category: 'reliability' },
      { id: 'M8', title: 'Lead Time', value: 18, unit: 'hours', trend: 'down', change: -8, category: 'performance' }
    ]);
  }, []);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'running': case 'deploying': return 'text-blue-500 bg-blue-900/20';
      case 'succeeded': case 'deployed': return 'text-green-500 bg-green-900/20';
      case 'failed': return 'text-red-500 bg-red-900/20';
      case 'cancelled': case 'rolled_back': return 'text-gray-500 bg-gray-900/20';
      case 'pending': case 'queued': return 'text-yellow-500 bg-yellow-900/20';
      case 'skipped': return 'text-gray-400 bg-gray-800/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'running': case 'deploying': return <Activity className="w-4 h-4 animate-pulse" />;
      case 'succeeded': case 'deployed': return <CheckCircle className="w-4 h-4" />;
      case 'failed': return <XCircle className="w-4 h-4" />;
      case 'cancelled': case 'rolled_back': return <StopCircle className="w-4 h-4" />;
      case 'pending': case 'queued': return <Clock className="w-4 h-4" />;
      case 'skipped': return <ChevronRight className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  const getTriggerIcon = (trigger: string) => {
    switch(trigger) {
      case 'manual': return <Play className="w-3 h-3" />;
      case 'commit': return <GitCommit className="w-3 h-3" />;
      case 'schedule': return <Calendar className="w-3 h-3" />;
      case 'api': return <Terminal className="w-3 h-3" />;
      case 'merge': return <GitMerge className="w-3 h-3" />;
      default: return <Zap className="w-3 h-3" />;
    }
  };

  const getEnvironmentColor = (env: string) => {
    switch(env) {
      case 'production': return 'text-red-500 bg-red-900/20';
      case 'staging': return 'text-yellow-500 bg-yellow-900/20';
      case 'development': return 'text-green-500 bg-green-900/20';
      case 'testing': return 'text-blue-500 bg-blue-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold flex items-center space-x-2">
              <GitBranch className="w-6 h-6 text-purple-500" />
              <span>DevOps Pipeline Management</span>
            </h1>
            <p className="text-sm text-gray-400 mt-1">CI/CD pipelines, deployments, and repository management</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                autoRefresh ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              <span>{autoRefresh ? 'Auto-refresh' : 'Paused'}</span>
            </button>

            <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm flex items-center space-x-2">
              <Play className="w-4 h-4" />
              <span>Run Pipeline</span>
            </button>
          </div>
        </div>
      </header>

      {/* Metrics Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="flex items-center space-x-6 overflow-x-auto">
          {metrics.map(metric => (
            <div key={metric.id} className="flex items-center space-x-2 min-w-fit">
              <div className="text-xs">
                <span className="text-gray-500">{metric.title}:</span>
                <span className="ml-2 font-bold">
                  {metric.value}{metric.unit}
                </span>
                <span className={`ml-1 text-xs ${
                  metric.trend === 'up' ? 'text-green-500' :
                  metric.trend === 'down' ? 'text-red-500' :
                  'text-gray-500'
                }`}>
                  {metric.trend === 'up' ? '↑' : metric.trend === 'down' ? '↓' : '→'}
                  {Math.abs(metric.change)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-gray-900/30 border-b border-gray-800 px-6">
        <div className="flex space-x-6">
          {['pipelines', 'deployments', 'repositories', 'analytics'].map(view => (
            <button
              key={view}
              onClick={() => setSelectedView(view as any)}
              className={`py-3 border-b-2 text-sm capitalize ${
                selectedView === view 
                  ? 'border-purple-500 text-purple-500' 
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              {view}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {selectedView === 'pipelines' && (
          <div className="space-y-4">
            {/* Pipeline Filters */}
            <div className="flex items-center space-x-3 mb-4">
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Status</option>
                <option value="running">Running</option>
                <option value="succeeded">Succeeded</option>
                <option value="failed">Failed</option>
                <option value="cancelled">Cancelled</option>
                <option value="queued">Queued</option>
              </select>
              
              <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
                <Filter className="w-4 h-4" />
                <span>More Filters</span>
              </button>
            </div>

            {/* Pipelines List */}
            <div className="space-y-3">
              {pipelines.map(pipeline => (
                <div key={pipeline.id} className="bg-gray-900 border border-gray-800 rounded-lg">
                  <div className="p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-start space-x-3">
                        <div className={`p-2 rounded ${getStatusColor(pipeline.status)}`}>
                          {getStatusIcon(pipeline.status)}
                        </div>
                        <div>
                          <div className="flex items-center space-x-2 mb-1">
                            <h3 className="text-sm font-bold">{pipeline.name}</h3>
                            <span className="text-xs text-gray-500">#{pipeline.id}</span>
                          </div>
                          <div className="flex items-center space-x-4 text-xs text-gray-500">
                            <span className="flex items-center space-x-1">
                              <GitBranch className="w-3 h-3" />
                              <span>{pipeline.branch}</span>
                            </span>
                            <span className="flex items-center space-x-1">
                              {getTriggerIcon(pipeline.trigger)}
                              <span>{pipeline.trigger}</span>
                            </span>
                            <span>{pipeline.startTime}</span>
                            {pipeline.duration && <span>{Math.floor(pipeline.duration / 60)}m {pipeline.duration % 60}s</span>}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        {pipeline.status === 'running' && (
                          <button className="p-2 bg-red-600 hover:bg-red-700 rounded">
                            <StopCircle className="w-4 h-4" />
                          </button>
                        )}
                        {(pipeline.status === 'failed' || pipeline.status === 'cancelled') && (
                          <button className="p-2 bg-blue-600 hover:bg-blue-700 rounded">
                            <RotateCw className="w-4 h-4" />
                          </button>
                        )}
                        <button 
                          onClick={() => setSelectedPipeline(selectedPipeline === pipeline.id ? null : pipeline.id)}
                          className="p-2 hover:bg-gray-800 rounded"
                        >
                          <ChevronDown className={`w-4 h-4 transition-transform ${
                            selectedPipeline === pipeline.id ? 'rotate-180' : ''
                          }`} />
                        </button>
                      </div>
                    </div>

                    {/* Commit Info */}
                    <div className="flex items-center space-x-4 mb-3 p-2 bg-gray-800 rounded text-xs">
                      <GitCommit className="w-3 h-3 text-gray-500" />
                      <span className="font-mono text-blue-500">{pipeline.commit.hash}</span>
                      <span className="flex-1">{pipeline.commit.message}</span>
                      <span className="text-gray-500">by {pipeline.author}</span>
                    </div>

                    {/* Pipeline Stages */}
                    <div className="flex items-center space-x-2 overflow-x-auto">
                      {pipeline.stages.map((stage, idx) => (
                        <React.Fragment key={stage.name}>
                          {idx > 0 && <ChevronRight className="w-4 h-4 text-gray-600" />}
                          <div className={`px-3 py-2 rounded text-xs flex items-center space-x-2 ${
                            getStatusColor(stage.status)
                          }`}>
                            {getStatusIcon(stage.status)}
                            <span>{stage.name}</span>
                            {stage.duration && <span className="text-gray-500">({stage.duration}s)</span>}
                          </div>
                        </React.Fragment>
                      ))}
                    </div>

                    {/* Expanded Details */}
                    {selectedPipeline === pipeline.id && (
                      <div className="mt-4 pt-4 border-t border-gray-800">
                        <div className="grid grid-cols-3 gap-4">
                          {/* Test Results */}
                          {pipeline.tests && (
                            <div className="bg-gray-800 rounded p-3">
                              <h4 className="text-xs font-bold mb-2">Test Results</h4>
                              <div className="space-y-1 text-xs">
                                <div className="flex justify-between">
                                  <span className="text-green-500">Passed</span>
                                  <span>{pipeline.tests.passed}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-red-500">Failed</span>
                                  <span>{pipeline.tests.failed}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-500">Skipped</span>
                                  <span>{pipeline.tests.skipped}</span>
                                </div>
                                <div className="flex justify-between pt-2 border-t border-gray-700">
                                  <span>Coverage</span>
                                  <span className="font-bold">{pipeline.tests.coverage}%</span>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Artifacts */}
                          {pipeline.artifacts && (
                            <div className="bg-gray-800 rounded p-3">
                              <h4 className="text-xs font-bold mb-2">Artifacts</h4>
                              <div className="space-y-1">
                                {pipeline.artifacts.map(artifact => (
                                  <div key={artifact.name} className="flex items-center justify-between text-xs">
                                    <span className="flex items-center space-x-1">
                                      <Archive className="w-3 h-3" />
                                      <span className="truncate">{artifact.name}</span>
                                    </span>
                                    <span className="text-gray-500">{artifact.size}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Actions */}
                          <div className="bg-gray-800 rounded p-3">
                            <h4 className="text-xs font-bold mb-2">Actions</h4>
                            <div className="space-y-2">
                              <button className="w-full px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs">
                                View Logs
                              </button>
                              <button className="w-full px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs">
                                Download Artifacts
                              </button>
                              <button className="w-full px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs">
                                View in Repository
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedView === 'deployments' && (
          <div className="space-y-4">
            {/* Deployment Filters */}
            <div className="flex items-center space-x-3 mb-4">
              <select
                value={filterEnvironment}
                onChange={(e) => setFilterEnvironment(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Environments</option>
                <option value="production">Production</option>
                <option value="staging">Staging</option>
                <option value="development">Development</option>
                <option value="testing">Testing</option>
              </select>
            </div>

            {/* Deployments Grid */}
            <div className="grid grid-cols-2 gap-4">
              {deployments.map(deployment => (
                <div key={deployment.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center space-x-2 mb-1">
                        <h3 className="text-sm font-bold">{deployment.pipeline}</h3>
                        <span className={`px-2 py-1 rounded text-xs ${getEnvironmentColor(deployment.environment)}`}>
                          {deployment.environment}
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>Version: {deployment.version}</span>
                        <span>By: {deployment.deployedBy}</span>
                        <span>{deployment.deployedAt}</span>
                      </div>
                    </div>
                    <div className={`px-2 py-1 rounded text-xs ${getStatusColor(deployment.status)}`}>
                      {deployment.status}
                    </div>
                  </div>

                  {/* Services Status */}
                  <div className="space-y-2 mb-3">
                    {deployment.services.map(service => (
                      <div key={service.name} className="flex items-center justify-between p-2 bg-gray-800 rounded text-xs">
                        <div className="flex items-center space-x-2">
                          <Container className={`w-3 h-3 ${
                            service.status === 'healthy' ? 'text-green-500' :
                            service.status === 'unhealthy' ? 'text-red-500' :
                            'text-gray-500'
                          }`} />
                          <span>{service.name}</span>
                          <span className="text-gray-500">({service.replicas} replicas)</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <span>CPU: {service.cpu}%</span>
                          <span>Mem: {service.memory}%</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Actions */}
                  <div className="flex space-x-2">
                    {deployment.rollbackAvailable && (
                      <button className="flex-1 px-3 py-1 bg-orange-600 hover:bg-orange-700 rounded text-xs">
                        Rollback
                      </button>
                    )}
                    <button className="flex-1 px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                      View Logs
                    </button>
                    <button className="flex-1 px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                      Scale
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedView === 'repositories' && (
          <div className="grid grid-cols-2 gap-4">
            {repositories.map(repo => (
              <div key={repo.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="text-sm font-bold flex items-center space-x-2">
                      <Folder className="w-4 h-4" />
                      <span>{repo.name}</span>
                    </h3>
                    <p className="text-xs text-gray-500 mt-1">{repo.language}</p>
                  </div>
                  {repo.cicdEnabled && (
                    <span className="px-2 py-1 bg-green-900/20 text-green-500 rounded text-xs">
                      CI/CD Enabled
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-3 gap-2 mb-3 text-xs">
                  <div className="text-center">
                    <div className="text-2xl font-bold">{repo.branches}</div>
                    <div className="text-gray-500">Branches</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold">{repo.pullRequests}</div>
                    <div className="text-gray-500">PRs</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold">{repo.coverage}%</div>
                    <div className="text-gray-500">Coverage</div>
                  </div>
                </div>

                {/* Vulnerabilities */}
                <div className="flex items-center justify-between mb-3 p-2 bg-gray-800 rounded text-xs">
                  <span className="text-gray-500">Vulnerabilities:</span>
                  <div className="flex items-center space-x-2">
                    {repo.vulnerabilities.critical > 0 && (
                      <span className="text-red-500">{repo.vulnerabilities.critical} critical</span>
                    )}
                    {repo.vulnerabilities.high > 0 && (
                      <span className="text-orange-500">{repo.vulnerabilities.high} high</span>
                    )}
                    <span className="text-yellow-500">{repo.vulnerabilities.medium} medium</span>
                    <span className="text-blue-500">{repo.vulnerabilities.low} low</span>
                  </div>
                </div>

                {/* Activity */}
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>{repo.activity.commits} commits</span>
                  <span>{repo.activity.contributors} contributors</span>
                  <span>{repo.activity.issues} issues</span>
                </div>

                <div className="mt-3 pt-3 border-t border-gray-800 flex space-x-2">
                  <button className="flex-1 px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    View Repo
                  </button>
                  <button className="flex-1 px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    Run Pipeline
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedView === 'analytics' && (
          <div className="grid grid-cols-2 gap-6">
            {/* Build Trends */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Build Trends (Last 7 Days)</h3>
              <div className="h-48 flex items-end justify-between space-x-2">
                {[85, 78, 92, 88, 75, 83, 90].map((value, idx) => (
                  <div key={idx} className="flex-1 flex flex-col items-center">
                    <div className="w-full bg-blue-500 rounded-t" style={{ height: `${value}%` }} />
                    <span className="text-xs text-gray-500 mt-2">Day {idx + 1}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Deployment Frequency */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Deployment Frequency</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Production</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div className="h-full bg-red-500" style={{ width: '65%' }} />
                    </div>
                    <span className="text-xs">8/day</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Staging</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div className="h-full bg-yellow-500" style={{ width: '85%' }} />
                    </div>
                    <span className="text-xs">12/day</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Development</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div className="h-full bg-green-500" style={{ width: '100%' }} />
                    </div>
                    <span className="text-xs">24/day</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Pipeline Performance */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Pipeline Performance</h3>
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Fastest Build</span>
                  <span className="text-green-500">3m 12s</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Slowest Build</span>
                  <span className="text-red-500">18m 45s</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Average Build</span>
                  <span>6m 30s</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Queue Time</span>
                  <span>45s</span>
                </div>
              </div>
            </div>

            {/* Top Contributors */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Top Contributors (This Week)</h3>
              <div className="space-y-2">
                {[
                  { name: 'john.doe', commits: 42, lines: '+1,234 -567' },
                  { name: 'jane.smith', commits: 38, lines: '+987 -321' },
                  { name: 'bob.wilson', commits: 25, lines: '+654 -210' },
                  { name: 'alice.johnson', commits: 18, lines: '+432 -123' }
                ].map(contributor => (
                  <div key={contributor.name} className="flex items-center justify-between text-xs">
                    <span>{contributor.name}</span>
                    <div className="flex items-center space-x-3">
                      <span>{contributor.commits} commits</span>
                      <span className="text-gray-500">{contributor.lines}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}