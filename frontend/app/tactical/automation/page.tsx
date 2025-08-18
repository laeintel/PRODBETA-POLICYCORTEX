'use client';

import React, { useState, useEffect } from 'react';
import { 
  Zap, Play, Pause, Square, RotateCcw, Calendar, Clock, CheckCircle,
  XCircle, AlertTriangle, Activity, Settings, Terminal, FileCode,
  GitBranch, Database, Server, Cloud, Shield, User, Users, Timer,
  TrendingUp, BarChart, Eye, Edit, Trash2, Copy, Download, Upload,
  ChevronRight, MoreVertical, Info, ArrowRight, Cpu, Box
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface AutomationTask {
  id: string;
  name: string;
  description: string;
  type: 'scheduled' | 'event_driven' | 'manual' | 'webhook' | 'conditional';
  status: 'active' | 'inactive' | 'running' | 'failed' | 'completed' | 'paused';
  category: 'deployment' | 'maintenance' | 'monitoring' | 'security' | 'data_processing' | 'backup';
  trigger: {
    type: string;
    value: string;
    lastTriggered?: string;
    nextRun?: string;
  };
  workflow: {
    steps: WorkflowStep[];
    variables: { [key: string]: any };
    conditions: string[];
  };
  execution: {
    totalRuns: number;
    successfulRuns: number;
    failedRuns: number;
    averageDuration: number;
    lastRun?: string;
    lastStatus?: string;
  };
  resources: {
    cpu: number;
    memory: number;
    estimatedCost: number;
  };
  owner: string;
  created: string;
  modified: string;
  tags: string[];
}

interface WorkflowStep {
  id: string;
  name: string;
  action: string;
  parameters: { [key: string]: any };
  status?: 'pending' | 'running' | 'success' | 'failed' | 'skipped';
  output?: string;
  duration?: number;
}

interface ExecutionHistory {
  id: string;
  taskId: string;
  taskName: string;
  startTime: string;
  endTime?: string;
  duration?: number;
  status: 'running' | 'success' | 'failed' | 'cancelled';
  triggeredBy: string;
  logs?: string[];
  metrics?: {
    itemsProcessed?: number;
    dataTransferred?: string;
    errors?: number;
  };
}

export default function AutomationHub() {
  const [tasks, setTasks] = useState<AutomationTask[]>([]);
  const [executionHistory, setExecutionHistory] = useState<ExecutionHistory[]>([]);
  const [selectedTask, setSelectedTask] = useState<AutomationTask | null>(null);
  const [filterCategory, setFilterCategory] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [viewMode, setViewMode] = useState<'tasks' | 'history' | 'insights'>('tasks');
  const [showCreateTask, setShowCreateTask] = useState(false);

  useEffect(() => {
    // Initialize with mock automation tasks
    setTasks([
      {
        id: 'AUTO-001',
        name: 'Database Backup Automation',
        description: 'Automated daily backup of production databases with verification',
        type: 'scheduled',
        status: 'active',
        category: 'backup',
        trigger: {
          type: 'cron',
          value: '0 2 * * *',
          lastTriggered: 'Today 2:00 AM',
          nextRun: 'Tomorrow 2:00 AM'
        },
        workflow: {
          steps: [
            {
              id: 's1',
              name: 'Create Snapshot',
              action: 'database.snapshot',
              parameters: { database: 'production', type: 'full' },
              status: 'success',
              duration: 180
            },
            {
              id: 's2',
              name: 'Compress Data',
              action: 'file.compress',
              parameters: { algorithm: 'gzip', level: 9 },
              status: 'success',
              duration: 120
            },
            {
              id: 's3',
              name: 'Upload to Storage',
              action: 'storage.upload',
              parameters: { bucket: 's3://backups', encryption: true },
              status: 'success',
              duration: 240
            },
            {
              id: 's4',
              name: 'Verify Backup',
              action: 'backup.verify',
              parameters: { checksum: true, testRestore: false },
              status: 'success',
              duration: 60
            },
            {
              id: 's5',
              name: 'Send Notification',
              action: 'notification.send',
              parameters: { channel: 'slack', recipients: ['#devops'] },
              status: 'success',
              duration: 5
            }
          ],
          variables: {
            retention_days: 30,
            compression_enabled: true,
            encryption_key: '***'
          },
          conditions: ['database.isHealthy', 'storage.hasSpace']
        },
        execution: {
          totalRuns: 365,
          successfulRuns: 362,
          failedRuns: 3,
          averageDuration: 605,
          lastRun: 'Today 2:00 AM',
          lastStatus: 'success'
        },
        resources: {
          cpu: 2,
          memory: 4096,
          estimatedCost: 15.50
        },
        owner: 'DevOps Team',
        created: '1 year ago',
        modified: '1 week ago',
        tags: ['critical', 'production', 'backup']
      },
      {
        id: 'AUTO-002',
        name: 'Auto-Scaling Policy',
        description: 'Automatically scale infrastructure based on load metrics',
        type: 'event_driven',
        status: 'running',
        category: 'deployment',
        trigger: {
          type: 'metric',
          value: 'cpu > 80% for 5min',
          lastTriggered: '30 minutes ago'
        },
        workflow: {
          steps: [
            {
              id: 's1',
              name: 'Check Current Load',
              action: 'metrics.query',
              parameters: { metric: 'cpu_usage', period: '5m' },
              status: 'running',
              output: 'CPU: 85%'
            },
            {
              id: 's2',
              name: 'Calculate Required Instances',
              action: 'scaling.calculate',
              parameters: { algorithm: 'predictive', buffer: 20 },
              status: 'pending'
            },
            {
              id: 's3',
              name: 'Provision New Instances',
              action: 'infrastructure.provision',
              parameters: { type: 't3.medium', count: 2 },
              status: 'pending'
            },
            {
              id: 's4',
              name: 'Update Load Balancer',
              action: 'loadbalancer.update',
              parameters: { strategy: 'round-robin' },
              status: 'pending'
            }
          ],
          variables: {
            min_instances: 2,
            max_instances: 20,
            cooldown_period: 300
          },
          conditions: ['budget.withinLimit', 'region.hasCapacity']
        },
        execution: {
          totalRuns: 142,
          successfulRuns: 138,
          failedRuns: 4,
          averageDuration: 180,
          lastRun: '30 minutes ago',
          lastStatus: 'running'
        },
        resources: {
          cpu: 1,
          memory: 2048,
          estimatedCost: 8.75
        },
        owner: 'Infrastructure Team',
        created: '6 months ago',
        modified: '2 days ago',
        tags: ['auto-scaling', 'cost-optimization']
      },
      {
        id: 'AUTO-003',
        name: 'Security Patch Deployment',
        description: 'Automated security patch deployment with rollback capability',
        type: 'webhook',
        status: 'active',
        category: 'security',
        trigger: {
          type: 'webhook',
          value: 'https://api.company.com/webhooks/security',
          lastTriggered: 'Yesterday 11:00 PM'
        },
        workflow: {
          steps: [
            {
              id: 's1',
              name: 'Scan for Vulnerabilities',
              action: 'security.scan',
              parameters: { depth: 'full', severity: 'critical' }
            },
            {
              id: 's2',
              name: 'Download Patches',
              action: 'patch.download',
              parameters: { source: 'vendor', verify: true }
            },
            {
              id: 's3',
              name: 'Test in Staging',
              action: 'deploy.staging',
              parameters: { environment: 'staging', timeout: 300 }
            },
            {
              id: 's4',
              name: 'Deploy to Production',
              action: 'deploy.production',
              parameters: { strategy: 'rolling', batchSize: 5 }
            },
            {
              id: 's5',
              name: 'Verify Deployment',
              action: 'health.check',
              parameters: { endpoints: ['api', 'web'], retries: 3 }
            }
          ],
          variables: {
            approval_required: true,
            rollback_enabled: true,
            maintenance_window: '23:00-02:00'
          },
          conditions: ['environment.isStable', 'backup.isRecent']
        },
        execution: {
          totalRuns: 52,
          successfulRuns: 51,
          failedRuns: 1,
          averageDuration: 1200,
          lastRun: 'Yesterday 11:00 PM',
          lastStatus: 'success'
        },
        resources: {
          cpu: 4,
          memory: 8192,
          estimatedCost: 22.30
        },
        owner: 'Security Team',
        created: '1 year ago',
        modified: '1 month ago',
        tags: ['security', 'compliance', 'patches']
      },
      {
        id: 'AUTO-004',
        name: 'Log Aggregation Pipeline',
        description: 'Collect, process, and analyze logs from all services',
        type: 'scheduled',
        status: 'active',
        category: 'monitoring',
        trigger: {
          type: 'interval',
          value: 'every 5 minutes',
          lastTriggered: '2 minutes ago',
          nextRun: 'In 3 minutes'
        },
        workflow: {
          steps: [
            {
              id: 's1',
              name: 'Collect Logs',
              action: 'logs.collect',
              parameters: { sources: ['app', 'system', 'security'] }
            },
            {
              id: 's2',
              name: 'Parse & Transform',
              action: 'logs.transform',
              parameters: { format: 'json', enrichment: true }
            },
            {
              id: 's3',
              name: 'Detect Anomalies',
              action: 'ml.detectAnomalies',
              parameters: { model: 'isolation_forest', threshold: 0.95 }
            },
            {
              id: 's4',
              name: 'Store in Database',
              action: 'database.insert',
              parameters: { table: 'logs', index: 'timestamp' }
            }
          ],
          variables: {
            batch_size: 10000,
            retention_period: 90,
            compression: true
          },
          conditions: ['storage.available', 'pipeline.healthy']
        },
        execution: {
          totalRuns: 8640,
          successfulRuns: 8598,
          failedRuns: 42,
          averageDuration: 45,
          lastRun: '2 minutes ago',
          lastStatus: 'success'
        },
        resources: {
          cpu: 2,
          memory: 4096,
          estimatedCost: 18.90
        },
        owner: 'Platform Team',
        created: '2 months ago',
        modified: '3 days ago',
        tags: ['logging', 'monitoring', 'analytics']
      },
      {
        id: 'AUTO-005',
        name: 'Cost Optimization Bot',
        description: 'Identify and implement cost-saving opportunities',
        type: 'scheduled',
        status: 'failed',
        category: 'maintenance',
        trigger: {
          type: 'cron',
          value: '0 0 * * MON',
          lastTriggered: 'Last Monday',
          nextRun: 'Next Monday'
        },
        workflow: {
          steps: [
            {
              id: 's1',
              name: 'Analyze Resource Usage',
              action: 'cost.analyze',
              parameters: { period: '7d', services: 'all' },
              status: 'success'
            },
            {
              id: 's2',
              name: 'Identify Idle Resources',
              action: 'resources.findIdle',
              parameters: { threshold: 5, window: '24h' },
              status: 'success'
            },
            {
              id: 's3',
              name: 'Generate Recommendations',
              action: 'ai.recommend',
              parameters: { model: 'cost_optimizer_v2' },
              status: 'failed',
              output: 'Error: Model unavailable'
            }
          ],
          variables: {
            savings_target: 20,
            auto_apply: false,
            notification_threshold: 100
          },
          conditions: ['approval.granted', 'savings.significant']
        },
        execution: {
          totalRuns: 12,
          successfulRuns: 10,
          failedRuns: 2,
          averageDuration: 900,
          lastRun: 'Last Monday',
          lastStatus: 'failed'
        },
        resources: {
          cpu: 1,
          memory: 2048,
          estimatedCost: 5.25
        },
        owner: 'FinOps Team',
        created: '3 months ago',
        modified: '1 week ago',
        tags: ['cost-optimization', 'finops']
      }
    ]);

    setExecutionHistory([
      {
        id: 'EXEC-001',
        taskId: 'AUTO-001',
        taskName: 'Database Backup Automation',
        startTime: 'Today 2:00 AM',
        endTime: 'Today 2:10 AM',
        duration: 605,
        status: 'success',
        triggeredBy: 'Schedule',
        metrics: {
          itemsProcessed: 5,
          dataTransferred: '45.2 GB',
          errors: 0
        }
      },
      {
        id: 'EXEC-002',
        taskId: 'AUTO-002',
        taskName: 'Auto-Scaling Policy',
        startTime: '30 minutes ago',
        status: 'running',
        triggeredBy: 'Metric Alert',
        metrics: {
          itemsProcessed: 2
        }
      },
      {
        id: 'EXEC-003',
        taskId: 'AUTO-004',
        taskName: 'Log Aggregation Pipeline',
        startTime: '2 minutes ago',
        endTime: '1 minute ago',
        duration: 42,
        status: 'success',
        triggeredBy: 'Schedule',
        metrics: {
          itemsProcessed: 8432,
          dataTransferred: '124 MB',
          errors: 0
        }
      },
      {
        id: 'EXEC-004',
        taskId: 'AUTO-005',
        taskName: 'Cost Optimization Bot',
        startTime: 'Last Monday 12:00 AM',
        endTime: 'Last Monday 12:08 AM',
        duration: 480,
        status: 'failed',
        triggeredBy: 'Schedule',
        metrics: {
          itemsProcessed: 245,
          errors: 1
        }
      }
    ]);
  }, []);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active':
      case 'success':
      case 'completed': return 'text-green-500 bg-green-900/20';
      case 'running': return 'text-blue-500 bg-blue-900/20';
      case 'failed': return 'text-red-500 bg-red-900/20';
      case 'inactive':
      case 'paused': return 'text-yellow-500 bg-yellow-900/20';
      case 'cancelled': return 'text-gray-500 bg-gray-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch(category) {
      case 'deployment': return <Cpu className="w-4 h-4 text-blue-500" />;
      case 'maintenance': return <Settings className="w-4 h-4 text-yellow-500" />;
      case 'monitoring': return <Activity className="w-4 h-4 text-green-500" />;
      case 'security': return <Shield className="w-4 h-4 text-red-500" />;
      case 'data_processing': return <Database className="w-4 h-4 text-purple-500" />;
      case 'backup': return <Cloud className="w-4 h-4 text-cyan-500" />;
      default: return <Box className="w-4 h-4 text-gray-500" />;
    }
  };

  const filteredTasks = tasks.filter(task => {
    if (filterCategory !== 'all' && task.category !== filterCategory) return false;
    if (filterStatus !== 'all' && task.status !== filterStatus) return false;
    return true;
  });

  const stats = {
    totalTasks: tasks.length,
    activeTasks: tasks.filter(t => t.status === 'active').length,
    runningTasks: tasks.filter(t => t.status === 'running').length,
    failedTasks: tasks.filter(t => t.status === 'failed').length,
    totalExecutions: tasks.reduce((sum, t) => sum + t.execution.totalRuns, 0),
    successRate: Math.round(
      (tasks.reduce((sum, t) => sum + t.execution.successfulRuns, 0) /
       tasks.reduce((sum, t) => sum + t.execution.totalRuns, 0)) * 100
    )
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Automation Hub</h1>
            <p className="text-sm text-gray-400 mt-1">Manage and monitor automated workflows</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setViewMode(
                viewMode === 'tasks' ? 'history' : 
                viewMode === 'history' ? 'insights' : 'tasks'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2"
            >
              {viewMode === 'tasks' ? <Clock className="w-4 h-4" /> : 
               viewMode === 'history' ? <BarChart className="w-4 h-4" /> :
               <Zap className="w-4 h-4" />}
              <span>
                {viewMode === 'tasks' ? 'History' : 
                 viewMode === 'history' ? 'Insights' : 'Tasks'}
              </span>
            </button>
            
            <button
              onClick={() => setShowCreateTask(true)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2"
            >
              <Zap className="w-4 h-4" />
              <span>New Automation</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <Box className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Total Tasks</p>
              <p className="text-xl font-bold">{stats.totalTasks}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Active</p>
              <p className="text-xl font-bold">{stats.activeTasks}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Activity className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Running</p>
              <p className="text-xl font-bold">{stats.runningTasks}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XCircle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Failed</p>
              <p className="text-xl font-bold text-red-500">{stats.failedTasks}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Timer className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Executions</p>
              <p className="text-xl font-bold">{stats.totalExecutions}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <TrendingUp className="w-5 h-5 text-cyan-500" />
            <div>
              <p className="text-xs text-gray-400">Success Rate</p>
              <p className="text-xl font-bold">{stats.successRate}%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {viewMode === 'tasks' && (
          <>
            {/* Filters */}
            <div className="flex items-center space-x-3 mb-6">
              <select
                value={filterCategory}
                onChange={(e) => setFilterCategory(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Categories</option>
                <option value="deployment">Deployment</option>
                <option value="maintenance">Maintenance</option>
                <option value="monitoring">Monitoring</option>
                <option value="security">Security</option>
                <option value="data_processing">Data Processing</option>
                <option value="backup">Backup</option>
              </select>
              
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
                <option value="running">Running</option>
                <option value="failed">Failed</option>
                <option value="paused">Paused</option>
              </select>
            </div>

            {/* Tasks Grid */}
            <div className="grid grid-cols-2 gap-4">
              {filteredTasks.map(task => (
                <div key={task.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        {getCategoryIcon(task.category)}
                        <span className="text-xs text-gray-500 font-mono">{task.id}</span>
                        <span className={`px-2 py-1 text-xs rounded ${getStatusColor(task.status)}`}>
                          {task.status.toUpperCase()}
                        </span>
                      </div>
                      <h3 className="text-sm font-bold mb-1">{task.name}</h3>
                      <p className="text-xs text-gray-400 mb-2">{task.description}</p>
                    </div>
                    <MoreVertical className="w-4 h-4 text-gray-500 cursor-pointer" />
                  </div>
                  
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-400">Trigger</span>
                      <span>{task.trigger.type}: {task.trigger.value}</span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-400">Last Run</span>
                      <span className={task.execution.lastStatus === 'success' ? 'text-green-500' : 'text-red-500'}>
                        {task.execution.lastRun || 'Never'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-400">Success Rate</span>
                      <span>
                        {Math.round((task.execution.successfulRuns / task.execution.totalRuns) * 100)}%
                        ({task.execution.successfulRuns}/{task.execution.totalRuns})
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-400">Avg Duration</span>
                      <span>{Math.round(task.execution.averageDuration / 60)}m</span>
                    </div>
                  </div>

                  {/* Workflow Steps Preview */}
                  <div className="mb-3 p-2 bg-gray-800 rounded">
                    <p className="text-xs text-gray-400 mb-1">Workflow Steps</p>
                    <div className="flex items-center space-x-1">
                      {task.workflow.steps.slice(0, 4).map((step, idx) => (
                        <React.Fragment key={step.id}>
                          <div className="text-xs text-gray-300" title={step.name}>
                            {idx + 1}
                          </div>
                          {idx < Math.min(3, task.workflow.steps.length - 1) && (
                            <ArrowRight className="w-3 h-3 text-gray-600" />
                          )}
                        </React.Fragment>
                      ))}
                      {task.workflow.steps.length > 4 && (
                        <span className="text-xs text-gray-500">+{task.workflow.steps.length - 4}</span>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between pt-3 border-t border-gray-800">
                    <div className="flex items-center space-x-2 text-xs">
                      <User className="w-3 h-3 text-gray-500" />
                      <span className="text-gray-500">{task.owner}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      {task.status === 'active' && (
                        <button className="px-2 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-xs">
                          <Pause className="w-3 h-3" />
                        </button>
                      )}
                      {task.status === 'inactive' && (
                        <button className="px-2 py-1 bg-green-600 hover:bg-green-700 rounded text-xs">
                          <Play className="w-3 h-3" />
                        </button>
                      )}
                      <button className="px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                        <Terminal className="w-3 h-3" />
                      </button>
                      <button className="px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                        <Edit className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'history' && (
          /* Execution History */
          <div className="space-y-3">
            {executionHistory.map(execution => (
              <div key={execution.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className={`w-2 h-2 rounded-full ${
                      execution.status === 'success' ? 'bg-green-500' :
                      execution.status === 'running' ? 'bg-blue-500 animate-pulse' :
                      execution.status === 'failed' ? 'bg-red-500' :
                      'bg-gray-500'
                    }`} />
                    <div>
                      <p className="text-sm font-bold">{execution.taskName}</p>
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>ID: {execution.id}</span>
                        <span>Started: {execution.startTime}</span>
                        {execution.duration && <span>Duration: {Math.round(execution.duration / 60)}m</span>}
                        <span>Triggered by: {execution.triggeredBy}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    {execution.metrics && (
                      <div className="flex items-center space-x-3 text-xs">
                        {execution.metrics.itemsProcessed !== undefined && (
                          <span>Items: {execution.metrics.itemsProcessed}</span>
                        )}
                        {execution.metrics.dataTransferred && (
                          <span>Data: {execution.metrics.dataTransferred}</span>
                        )}
                        {execution.metrics.errors !== undefined && (
                          <span className={execution.metrics.errors > 0 ? 'text-red-500' : 'text-green-500'}>
                            Errors: {execution.metrics.errors}
                          </span>
                        )}
                      </div>
                    )}
                    <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                      View Logs
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'insights' && (
          /* Automation Insights */
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Execution Trends</h3>
              <div className="h-48 flex items-center justify-center text-gray-600">
                [Execution trend chart would go here]
              </div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Resource Usage</h3>
              <div className="h-48 flex items-center justify-center text-gray-600">
                [Resource usage chart would go here]
              </div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Top Failures</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span>Cost Optimization Bot</span>
                  <span className="text-red-500">2 failures</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span>Auto-Scaling Policy</span>
                  <span className="text-yellow-500">4 failures</span>
                </div>
              </div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Cost Analysis</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span>Total Monthly Cost</span>
                  <span className="text-green-500">$89.70</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span>Cost per Execution</span>
                  <span>$0.012</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}