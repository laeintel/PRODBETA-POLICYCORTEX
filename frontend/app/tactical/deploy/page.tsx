'use client';

import React, { useState, useEffect } from 'react';
import { 
  Upload, Package, GitBranch, Play, Pause, CheckCircle, XCircle,
  AlertTriangle, Clock, Rocket, Server, Cloud, Database, Activity,
  BarChart, TrendingUp, Calendar, User, Users, Settings, Terminal,
  FileCode, Box, Layers, Shield, Zap, RefreshCw, Eye, Download,
  ChevronRight, MoreVertical, Info, ArrowRight, Timer
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface Deployment {
  id: string;
  name: string;
  version: string;
  environment: 'development' | 'staging' | 'production' | 'testing';
  status: 'pending' | 'in_progress' | 'success' | 'failed' | 'rolled_back' | 'cancelled';
  type: 'application' | 'infrastructure' | 'database' | 'configuration' | 'hotfix';
  strategy: 'blue_green' | 'canary' | 'rolling' | 'recreate';
  initiatedBy: string;
  approvedBy: string[];
  startTime: string;
  endTime?: string;
  duration?: number;
  pipeline: {
    id: string;
    name: string;
    stages: PipelineStage[];
  };
  artifacts: {
    name: string;
    size: string;
    checksum: string;
  }[];
  affectedServices: string[];
  rollbackEnabled: boolean;
  healthChecks: {
    name: string;
    status: 'passing' | 'failing' | 'pending';
  }[];
  metrics: {
    successRate: number;
    deploymentFrequency: number;
    leadTime: number;
    mttr: number;
  };
}

interface PipelineStage {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'success' | 'failed' | 'skipped';
  startTime?: string;
  duration?: number;
  steps: {
    name: string;
    status: 'pending' | 'running' | 'success' | 'failed';
    logs?: string[];
  }[];
}

export default function DeploymentCenter() {
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [selectedDeployment, setSelectedDeployment] = useState<Deployment | null>(null);
  const [filterEnvironment, setFilterEnvironment] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [viewMode, setViewMode] = useState<'list' | 'pipeline'>('list');
  const [showNewDeployment, setShowNewDeployment] = useState(false);

  useEffect(() => {
    // Initialize with mock deployments
    setDeployments([
      {
        id: 'DEP-2024-001',
        name: 'API Service v2.5.0',
        version: '2.5.0',
        environment: 'production',
        status: 'in_progress',
        type: 'application',
        strategy: 'blue_green',
        initiatedBy: 'CI/CD Pipeline',
        approvedBy: ['John Smith', 'Sarah Johnson'],
        startTime: '10 minutes ago',
        pipeline: {
          id: 'pipe-001',
          name: 'Production Deploy',
          stages: [
            {
              id: 's1',
              name: 'Build',
              status: 'success',
              duration: 120,
              steps: [
                { name: 'Checkout Code', status: 'success' },
                { name: 'Install Dependencies', status: 'success' },
                { name: 'Run Tests', status: 'success' },
                { name: 'Build Artifacts', status: 'success' }
              ]
            },
            {
              id: 's2',
              name: 'Security Scan',
              status: 'success',
              duration: 45,
              steps: [
                { name: 'Vulnerability Scan', status: 'success' },
                { name: 'License Check', status: 'success' }
              ]
            },
            {
              id: 's3',
              name: 'Deploy',
              status: 'running',
              startTime: '2 minutes ago',
              steps: [
                { name: 'Blue Environment', status: 'success' },
                { name: 'Health Check', status: 'running' },
                { name: 'Switch Traffic', status: 'pending' },
                { name: 'Cleanup', status: 'pending' }
              ]
            },
            {
              id: 's4',
              name: 'Verify',
              status: 'pending',
              steps: [
                { name: 'Smoke Tests', status: 'pending' },
                { name: 'Performance Tests', status: 'pending' }
              ]
            }
          ]
        },
        artifacts: [
          { name: 'api-service.tar.gz', size: '45.2 MB', checksum: 'sha256:abc123...' },
          { name: 'config.yaml', size: '2.1 KB', checksum: 'sha256:def456...' }
        ],
        affectedServices: ['API Gateway', 'User Service', 'Auth Service'],
        rollbackEnabled: true,
        healthChecks: [
          { name: 'API Health', status: 'passing' },
          { name: 'Database Connection', status: 'passing' },
          { name: 'Cache Service', status: 'pending' }
        ],
        metrics: {
          successRate: 98.5,
          deploymentFrequency: 4.2,
          leadTime: 45,
          mttr: 12
        }
      },
      {
        id: 'DEP-2024-002',
        name: 'Database Migration',
        version: '1.0.0',
        environment: 'staging',
        status: 'success',
        type: 'database',
        strategy: 'recreate',
        initiatedBy: 'Mike Chen',
        approvedBy: ['Database Lead'],
        startTime: '2 hours ago',
        endTime: '1 hour ago',
        duration: 3600,
        pipeline: {
          id: 'pipe-002',
          name: 'Database Update',
          stages: [
            {
              id: 's1',
              name: 'Backup',
              status: 'success',
              duration: 300,
              steps: [
                { name: 'Create Snapshot', status: 'success' },
                { name: 'Verify Backup', status: 'success' }
              ]
            },
            {
              id: 's2',
              name: 'Migrate',
              status: 'success',
              duration: 1800,
              steps: [
                { name: 'Schema Update', status: 'success' },
                { name: 'Data Migration', status: 'success' }
              ]
            },
            {
              id: 's3',
              name: 'Validate',
              status: 'success',
              duration: 600,
              steps: [
                { name: 'Data Integrity Check', status: 'success' },
                { name: 'Performance Test', status: 'success' }
              ]
            }
          ]
        },
        artifacts: [
          { name: 'migration-scripts.sql', size: '125 KB', checksum: 'sha256:ghi789...' }
        ],
        affectedServices: ['Database Cluster', 'Analytics Service'],
        rollbackEnabled: true,
        healthChecks: [
          { name: 'Database Health', status: 'passing' },
          { name: 'Replication Status', status: 'passing' }
        ],
        metrics: {
          successRate: 100,
          deploymentFrequency: 0.5,
          leadTime: 120,
          mttr: 30
        }
      },
      {
        id: 'DEP-2024-003',
        name: 'Frontend Hotfix',
        version: '1.2.1',
        environment: 'production',
        status: 'failed',
        type: 'hotfix',
        strategy: 'rolling',
        initiatedBy: 'Emergency Response',
        approvedBy: ['CTO'],
        startTime: 'Yesterday',
        endTime: 'Yesterday',
        duration: 900,
        pipeline: {
          id: 'pipe-003',
          name: 'Hotfix Deploy',
          stages: [
            {
              id: 's1',
              name: 'Build',
              status: 'success',
              duration: 60,
              steps: [
                { name: 'Quick Build', status: 'success' }
              ]
            },
            {
              id: 's2',
              name: 'Deploy',
              status: 'failed',
              duration: 180,
              steps: [
                { name: 'Rolling Update', status: 'failed' }
              ]
            }
          ]
        },
        artifacts: [
          { name: 'frontend-hotfix.zip', size: '12.3 MB', checksum: 'sha256:jkl012...' }
        ],
        affectedServices: ['Web Frontend'],
        rollbackEnabled: true,
        healthChecks: [
          { name: 'Frontend Health', status: 'failing' }
        ],
        metrics: {
          successRate: 85,
          deploymentFrequency: 8,
          leadTime: 15,
          mttr: 5
        }
      },
      {
        id: 'DEP-2024-004',
        name: 'Infrastructure Update',
        version: '3.0.0',
        environment: 'development',
        status: 'pending',
        type: 'infrastructure',
        strategy: 'canary',
        initiatedBy: 'DevOps Team',
        approvedBy: [],
        startTime: 'Scheduled for tomorrow',
        pipeline: {
          id: 'pipe-004',
          name: 'Infra Deploy',
          stages: [
            {
              id: 's1',
              name: 'Plan',
              status: 'pending',
              steps: [
                { name: 'Terraform Plan', status: 'pending' },
                { name: 'Cost Analysis', status: 'pending' }
              ]
            },
            {
              id: 's2',
              name: 'Apply',
              status: 'pending',
              steps: [
                { name: 'Terraform Apply', status: 'pending' }
              ]
            }
          ]
        },
        artifacts: [
          { name: 'terraform.tfstate', size: '45 KB', checksum: 'sha256:mno345...' }
        ],
        affectedServices: ['All Services'],
        rollbackEnabled: true,
        healthChecks: [],
        metrics: {
          successRate: 95,
          deploymentFrequency: 2,
          leadTime: 180,
          mttr: 45
        }
      }
    ]);
  }, []);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'success': return 'text-green-500 bg-green-900/20';
      case 'in_progress':
      case 'running': return 'text-blue-500 bg-blue-900/20';
      case 'failed': return 'text-red-500 bg-red-900/20';
      case 'pending': return 'text-yellow-500 bg-yellow-900/20';
      case 'rolled_back': return 'text-orange-500 bg-orange-900/20';
      case 'cancelled': return 'text-gray-500 bg-gray-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
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

  const filteredDeployments = deployments.filter(deployment => {
    if (filterEnvironment !== 'all' && deployment.environment !== filterEnvironment) return false;
    if (filterStatus !== 'all' && deployment.status !== filterStatus) return false;
    return true;
  });

  const stats = {
    total: deployments.length,
    inProgress: deployments.filter(d => d.status === 'in_progress').length,
    success: deployments.filter(d => d.status === 'success').length,
    failed: deployments.filter(d => d.status === 'failed').length,
    avgSuccessRate: Math.round(deployments.reduce((sum, d) => sum + d.metrics.successRate, 0) / deployments.length),
    avgLeadTime: Math.round(deployments.reduce((sum, d) => sum + d.metrics.leadTime, 0) / deployments.length)
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Deployment Center</h1>
            <p className="text-sm text-gray-400 mt-1">Manage and track all deployments across environments</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setViewMode(viewMode === 'list' ? 'pipeline' : 'list')}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2"
            >
              {viewMode === 'list' ? <GitBranch className="w-4 h-4" /> : <BarChart className="w-4 h-4" />}
              <span>{viewMode === 'list' ? 'Pipeline View' : 'List View'}</span>
            </button>
            
            <button
              onClick={() => setShowNewDeployment(true)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2"
            >
              <Rocket className="w-4 h-4" />
              <span>New Deployment</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <Package className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Total</p>
              <p className="text-xl font-bold">{stats.total}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Activity className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">In Progress</p>
              <p className="text-xl font-bold">{stats.inProgress}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Success</p>
              <p className="text-xl font-bold">{stats.success}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XCircle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Failed</p>
              <p className="text-xl font-bold">{stats.failed}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <TrendingUp className="w-5 h-5 text-cyan-500" />
            <div>
              <p className="text-xs text-gray-400">Success Rate</p>
              <p className="text-xl font-bold">{stats.avgSuccessRate}%</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Timer className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Lead Time</p>
              <p className="text-xl font-bold">{stats.avgLeadTime}m</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Filters */}
        <div className="flex items-center space-x-3 mb-6">
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
          
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="in_progress">In Progress</option>
            <option value="success">Success</option>
            <option value="failed">Failed</option>
            <option value="rolled_back">Rolled Back</option>
          </select>
        </div>

        {/* Deployments List */}
        <div className="space-y-4">
          {filteredDeployments.map(deployment => (
            <div
              key={deployment.id}
              className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              onClick={() => setSelectedDeployment(deployment)}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <span className="text-xs text-gray-500 font-mono">{deployment.id}</span>
                    <span className={`px-2 py-1 text-xs rounded ${getEnvironmentColor(deployment.environment)}`}>
                      {deployment.environment.toUpperCase()}
                    </span>
                    <span className={`px-2 py-1 text-xs rounded ${getStatusColor(deployment.status)}`}>
                      {deployment.status.toUpperCase().replace('_', ' ')}
                    </span>
                    <span className="text-xs text-gray-400">
                      {deployment.strategy.replace('_', '-')}
                    </span>
                  </div>
                  <h3 className="text-sm font-bold mb-1">{deployment.name}</h3>
                  <div className="flex items-center space-x-4 text-xs text-gray-500">
                    <span className="flex items-center space-x-1">
                      <User className="w-3 h-3" />
                      <span>{deployment.initiatedBy}</span>
                    </span>
                    <span className="flex items-center space-x-1">
                      <Clock className="w-3 h-3" />
                      <span>{deployment.startTime}</span>
                    </span>
                    {deployment.duration && (
                      <span>{Math.round(deployment.duration / 60)}m duration</span>
                    )}
                  </div>
                </div>
                <ChevronRight className="w-5 h-5 text-gray-500" />
              </div>
              
              {/* Pipeline Progress */}
              <div className="mb-3">
                <div className="flex items-center space-x-2">
                  {deployment.pipeline.stages.map((stage, idx) => (
                    <React.Fragment key={stage.id}>
                      <div className="flex items-center space-x-1">
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${
                          stage.status === 'success' ? 'bg-green-900/30 text-green-500' :
                          stage.status === 'running' ? 'bg-blue-900/30 text-blue-500' :
                          stage.status === 'failed' ? 'bg-red-900/30 text-red-500' :
                          'bg-gray-900/30 text-gray-500'
                        }`}>
                          {stage.status === 'success' ? '✓' :
                           stage.status === 'failed' ? '✗' :
                           stage.status === 'running' ? '•' : ''}
                        </div>
                        <span className="text-xs text-gray-400">{stage.name}</span>
                      </div>
                      {idx < deployment.pipeline.stages.length - 1 && (
                        <div className={`flex-1 h-0.5 ${
                          deployment.pipeline.stages[idx + 1].status !== 'pending' ? 'bg-gray-600' : 'bg-gray-800'
                        }`} />
                      )}
                    </React.Fragment>
                  ))}
                </div>
              </div>
              
              <div className="flex items-center justify-between pt-3 border-t border-gray-800">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-400">Affected:</span>
                    <div className="flex flex-wrap gap-1">
                      {deployment.affectedServices.slice(0, 2).map(service => (
                        <span key={service} className="px-2 py-0.5 bg-gray-800 rounded text-xs">
                          {service}
                        </span>
                      ))}
                      {deployment.affectedServices.length > 2 && (
                        <span className="text-xs text-gray-500">
                          +{deployment.affectedServices.length - 2}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  {deployment.rollbackEnabled && deployment.status === 'failed' && (
                    <button className="px-3 py-1 bg-orange-600 hover:bg-orange-700 rounded text-xs">
                      Rollback
                    </button>
                  )}
                  {deployment.status === 'in_progress' && (
                    <button className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-xs">
                      Cancel
                    </button>
                  )}
                  <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    View Logs
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}