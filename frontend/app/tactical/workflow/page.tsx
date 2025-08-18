'use client';

import React, { useState, useEffect } from 'react';
import { 
  GitBranch, GitMerge, GitPullRequest, GitCommit, PlayCircle, PauseCircle,
  StopCircle, RotateCcw, CheckCircle, XCircle, AlertTriangle, Clock,
  Activity, Settings, Code, FileCode, Database, Server, Cloud, Shield,
  User, Users, Timer, TrendingUp, BarChart, Eye, Edit, Trash2, Copy,
  Download, Upload, ChevronRight, ChevronDown, MoreVertical, Info,
  ArrowRight, ArrowDown, Plus, Minus, Move, Layers, Box, Zap
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface WorkflowNode {
  id: string;
  type: 'start' | 'end' | 'action' | 'condition' | 'parallel' | 'loop';
  name: string;
  description?: string;
  action?: string;
  parameters?: { [key: string]: any };
  conditions?: string[];
  connections: string[];
  position: { x: number; y: number };
  status?: 'idle' | 'running' | 'success' | 'failed' | 'skipped';
  executionTime?: number;
  error?: string;
}

interface Workflow {
  id: string;
  name: string;
  description: string;
  version: string;
  status: 'draft' | 'active' | 'inactive' | 'deprecated';
  category: 'cicd' | 'data' | 'infrastructure' | 'security' | 'business' | 'custom';
  nodes: WorkflowNode[];
  variables: { [key: string]: any };
  triggers: {
    type: 'manual' | 'schedule' | 'event' | 'api';
    config: any;
  }[];
  settings: {
    timeout: number;
    retryPolicy: {
      maxRetries: number;
      backoffMultiplier: number;
    };
    notifications: {
      onSuccess: boolean;
      onFailure: boolean;
      channels: string[];
    };
  };
  metrics: {
    totalRuns: number;
    successRate: number;
    averageDuration: number;
    lastRun?: string;
  };
  owner: string;
  created: string;
  modified: string;
  tags: string[];
}

interface WorkflowRun {
  id: string;
  workflowId: string;
  workflowName: string;
  status: 'queued' | 'running' | 'success' | 'failed' | 'cancelled';
  startTime: string;
  endTime?: string;
  duration?: number;
  triggeredBy: string;
  currentNode?: string;
  progress: number;
  logs: {
    timestamp: string;
    level: 'info' | 'warning' | 'error';
    message: string;
    nodeId?: string;
  }[];
  artifacts?: {
    name: string;
    size: string;
    url: string;
  }[];
}

export default function WorkflowEngine() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [workflowRuns, setWorkflowRuns] = useState<WorkflowRun[]>([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null);
  const [filterCategory, setFilterCategory] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [viewMode, setViewMode] = useState<'workflows' | 'runs' | 'builder'>('workflows');
  const [showCreateWorkflow, setShowCreateWorkflow] = useState(false);

  useEffect(() => {
    // Initialize with mock workflows
    setWorkflows([
      {
        id: 'WF-001',
        name: 'CI/CD Pipeline',
        description: 'Complete CI/CD pipeline with testing and deployment',
        version: '2.1.0',
        status: 'active',
        category: 'cicd',
        nodes: [
          {
            id: 'start',
            type: 'start',
            name: 'Start',
            connections: ['checkout'],
            position: { x: 100, y: 200 }
          },
          {
            id: 'checkout',
            type: 'action',
            name: 'Checkout Code',
            action: 'git.checkout',
            parameters: { branch: 'main' },
            connections: ['build'],
            position: { x: 250, y: 200 }
          },
          {
            id: 'build',
            type: 'action',
            name: 'Build Application',
            action: 'build.run',
            parameters: { target: 'production' },
            connections: ['test-condition'],
            position: { x: 400, y: 200 }
          },
          {
            id: 'test-condition',
            type: 'condition',
            name: 'Run Tests?',
            conditions: ['config.runTests == true'],
            connections: ['parallel-tests', 'deploy'],
            position: { x: 550, y: 200 }
          },
          {
            id: 'parallel-tests',
            type: 'parallel',
            name: 'Run Tests in Parallel',
            connections: ['unit-tests', 'integration-tests', 'e2e-tests'],
            position: { x: 700, y: 100 }
          },
          {
            id: 'unit-tests',
            type: 'action',
            name: 'Unit Tests',
            action: 'test.unit',
            connections: ['deploy'],
            position: { x: 850, y: 50 }
          },
          {
            id: 'integration-tests',
            type: 'action',
            name: 'Integration Tests',
            action: 'test.integration',
            connections: ['deploy'],
            position: { x: 850, y: 150 }
          },
          {
            id: 'e2e-tests',
            type: 'action',
            name: 'E2E Tests',
            action: 'test.e2e',
            connections: ['deploy'],
            position: { x: 850, y: 250 }
          },
          {
            id: 'deploy',
            type: 'action',
            name: 'Deploy to Production',
            action: 'deploy.production',
            parameters: { strategy: 'blue-green' },
            connections: ['end'],
            position: { x: 1000, y: 200 }
          },
          {
            id: 'end',
            type: 'end',
            name: 'End',
            connections: [],
            position: { x: 1150, y: 200 }
          }
        ],
        variables: {
          environment: 'production',
          runTests: true,
          deploymentTarget: 'kubernetes'
        },
        triggers: [
          {
            type: 'event',
            config: { event: 'push', branch: 'main' }
          },
          {
            type: 'manual',
            config: { requireApproval: true }
          }
        ],
        settings: {
          timeout: 3600,
          retryPolicy: {
            maxRetries: 2,
            backoffMultiplier: 2
          },
          notifications: {
            onSuccess: true,
            onFailure: true,
            channels: ['slack', 'email']
          }
        },
        metrics: {
          totalRuns: 542,
          successRate: 94.5,
          averageDuration: 720,
          lastRun: '2 hours ago'
        },
        owner: 'DevOps Team',
        created: '6 months ago',
        modified: '1 week ago',
        tags: ['production', 'cicd', 'critical']
      },
      {
        id: 'WF-002',
        name: 'Data Processing Pipeline',
        description: 'ETL pipeline for processing customer data',
        version: '1.5.0',
        status: 'active',
        category: 'data',
        nodes: [
          {
            id: 'start',
            type: 'start',
            name: 'Start',
            connections: ['extract'],
            position: { x: 100, y: 200 }
          },
          {
            id: 'extract',
            type: 'action',
            name: 'Extract Data',
            action: 'data.extract',
            parameters: { source: 'database', table: 'customers' },
            connections: ['transform'],
            position: { x: 250, y: 200 }
          },
          {
            id: 'transform',
            type: 'action',
            name: 'Transform Data',
            action: 'data.transform',
            parameters: { rules: 'customer_transform_v2' },
            connections: ['validate'],
            position: { x: 400, y: 200 }
          },
          {
            id: 'validate',
            type: 'action',
            name: 'Validate Data',
            action: 'data.validate',
            parameters: { schema: 'customer_schema' },
            connections: ['load'],
            position: { x: 550, y: 200 }
          },
          {
            id: 'load',
            type: 'action',
            name: 'Load to Warehouse',
            action: 'data.load',
            parameters: { destination: 'data_warehouse' },
            connections: ['end'],
            position: { x: 700, y: 200 }
          },
          {
            id: 'end',
            type: 'end',
            name: 'End',
            connections: [],
            position: { x: 850, y: 200 }
          }
        ],
        variables: {
          batchSize: 10000,
          parallel: true,
          errorThreshold: 0.01
        },
        triggers: [
          {
            type: 'schedule',
            config: { cron: '0 2 * * *' }
          }
        ],
        settings: {
          timeout: 7200,
          retryPolicy: {
            maxRetries: 3,
            backoffMultiplier: 1.5
          },
          notifications: {
            onSuccess: false,
            onFailure: true,
            channels: ['email']
          }
        },
        metrics: {
          totalRuns: 365,
          successRate: 98.2,
          averageDuration: 1800,
          lastRun: 'Today 2:00 AM'
        },
        owner: 'Data Team',
        created: '1 year ago',
        modified: '2 weeks ago',
        tags: ['etl', 'data', 'scheduled']
      },
      {
        id: 'WF-003',
        name: 'Incident Response Workflow',
        description: 'Automated incident response and escalation',
        version: '3.0.0',
        status: 'active',
        category: 'security',
        nodes: [
          {
            id: 'start',
            type: 'start',
            name: 'Incident Detected',
            connections: ['assess'],
            position: { x: 100, y: 200 }
          },
          {
            id: 'assess',
            type: 'action',
            name: 'Assess Severity',
            action: 'incident.assess',
            connections: ['severity-condition'],
            position: { x: 250, y: 200 }
          },
          {
            id: 'severity-condition',
            type: 'condition',
            name: 'Check Severity',
            conditions: ['severity == "critical"', 'severity == "high"', 'severity == "medium"'],
            connections: ['critical-response', 'high-response', 'medium-response'],
            position: { x: 400, y: 200 }
          },
          {
            id: 'critical-response',
            type: 'parallel',
            name: 'Critical Response',
            connections: ['page-oncall', 'isolate-system', 'create-war-room'],
            position: { x: 600, y: 100 }
          },
          {
            id: 'page-oncall',
            type: 'action',
            name: 'Page On-Call',
            action: 'notification.page',
            parameters: { team: 'security', priority: 'critical' },
            connections: ['remediate'],
            position: { x: 750, y: 50 }
          },
          {
            id: 'isolate-system',
            type: 'action',
            name: 'Isolate System',
            action: 'security.isolate',
            connections: ['remediate'],
            position: { x: 750, y: 150 }
          },
          {
            id: 'create-war-room',
            type: 'action',
            name: 'Create War Room',
            action: 'incident.createWarRoom',
            connections: ['remediate'],
            position: { x: 750, y: 250 }
          },
          {
            id: 'high-response',
            type: 'action',
            name: 'High Priority Response',
            action: 'incident.highPriority',
            connections: ['remediate'],
            position: { x: 600, y: 200 }
          },
          {
            id: 'medium-response',
            type: 'action',
            name: 'Medium Priority Response',
            action: 'incident.mediumPriority',
            connections: ['remediate'],
            position: { x: 600, y: 300 }
          },
          {
            id: 'remediate',
            type: 'action',
            name: 'Remediate',
            action: 'incident.remediate',
            connections: ['document'],
            position: { x: 900, y: 200 }
          },
          {
            id: 'document',
            type: 'action',
            name: 'Document Incident',
            action: 'incident.document',
            connections: ['end'],
            position: { x: 1050, y: 200 }
          },
          {
            id: 'end',
            type: 'end',
            name: 'Incident Resolved',
            connections: [],
            position: { x: 1200, y: 200 }
          }
        ],
        variables: {
          escalationTimeout: 300,
          autoRemediate: true,
          notificationChannels: ['slack', 'pagerduty', 'email']
        },
        triggers: [
          {
            type: 'event',
            config: { source: 'monitoring', event: 'alert' }
          },
          {
            type: 'api',
            config: { endpoint: '/api/incidents' }
          }
        ],
        settings: {
          timeout: 1800,
          retryPolicy: {
            maxRetries: 1,
            backoffMultiplier: 1
          },
          notifications: {
            onSuccess: true,
            onFailure: true,
            channels: ['slack', 'email', 'sms']
          }
        },
        metrics: {
          totalRuns: 89,
          successRate: 96.7,
          averageDuration: 420,
          lastRun: 'Yesterday'
        },
        owner: 'Security Team',
        created: '3 months ago',
        modified: '5 days ago',
        tags: ['incident', 'security', 'critical']
      },
      {
        id: 'WF-004',
        name: 'Infrastructure Provisioning',
        description: 'Provision and configure cloud infrastructure',
        version: '1.0.0',
        status: 'draft',
        category: 'infrastructure',
        nodes: [
          {
            id: 'start',
            type: 'start',
            name: 'Start',
            connections: ['validate-request'],
            position: { x: 100, y: 200 }
          },
          {
            id: 'validate-request',
            type: 'action',
            name: 'Validate Request',
            action: 'request.validate',
            connections: ['provision-loop'],
            position: { x: 250, y: 200 }
          },
          {
            id: 'provision-loop',
            type: 'loop',
            name: 'Provision Resources',
            connections: ['provision-vm'],
            position: { x: 400, y: 200 }
          },
          {
            id: 'provision-vm',
            type: 'action',
            name: 'Provision VM',
            action: 'cloud.provisionVM',
            connections: ['configure-vm'],
            position: { x: 550, y: 200 }
          },
          {
            id: 'configure-vm',
            type: 'action',
            name: 'Configure VM',
            action: 'ansible.configure',
            connections: ['verify'],
            position: { x: 700, y: 200 }
          },
          {
            id: 'verify',
            type: 'action',
            name: 'Verify Setup',
            action: 'test.infrastructure',
            connections: ['end'],
            position: { x: 850, y: 200 }
          },
          {
            id: 'end',
            type: 'end',
            name: 'End',
            connections: [],
            position: { x: 1000, y: 200 }
          }
        ],
        variables: {
          resourceCount: 5,
          vmSize: 'Standard_D2s_v3',
          region: 'eastus'
        },
        triggers: [
          {
            type: 'manual',
            config: { requireApproval: true }
          }
        ],
        settings: {
          timeout: 3600,
          retryPolicy: {
            maxRetries: 2,
            backoffMultiplier: 2
          },
          notifications: {
            onSuccess: true,
            onFailure: true,
            channels: ['email']
          }
        },
        metrics: {
          totalRuns: 0,
          successRate: 0,
          averageDuration: 0
        },
        owner: 'Infrastructure Team',
        created: 'Today',
        modified: 'Today',
        tags: ['infrastructure', 'provisioning', 'draft']
      }
    ]);

    setWorkflowRuns([
      {
        id: 'RUN-001',
        workflowId: 'WF-001',
        workflowName: 'CI/CD Pipeline',
        status: 'running',
        startTime: '10 minutes ago',
        triggeredBy: 'Git Push (main)',
        currentNode: 'integration-tests',
        progress: 65,
        logs: [
          { timestamp: '10 minutes ago', level: 'info', message: 'Workflow started', nodeId: 'start' },
          { timestamp: '9 minutes ago', level: 'info', message: 'Checking out code from main branch', nodeId: 'checkout' },
          { timestamp: '8 minutes ago', level: 'info', message: 'Build completed successfully', nodeId: 'build' },
          { timestamp: '5 minutes ago', level: 'info', message: 'Running tests in parallel', nodeId: 'parallel-tests' },
          { timestamp: '2 minutes ago', level: 'info', message: 'Unit tests passed', nodeId: 'unit-tests' },
          { timestamp: '1 minute ago', level: 'warning', message: 'Integration test retry 1/2', nodeId: 'integration-tests' }
        ]
      },
      {
        id: 'RUN-002',
        workflowId: 'WF-002',
        workflowName: 'Data Processing Pipeline',
        status: 'success',
        startTime: 'Today 2:00 AM',
        endTime: 'Today 2:32 AM',
        duration: 1920,
        triggeredBy: 'Schedule',
        progress: 100,
        logs: [
          { timestamp: 'Today 2:00 AM', level: 'info', message: 'Processing started for 45,231 records' },
          { timestamp: 'Today 2:32 AM', level: 'info', message: 'Processing completed successfully' }
        ],
        artifacts: [
          { name: 'processed_data.csv', size: '124 MB', url: '/downloads/processed_data.csv' },
          { name: 'validation_report.pdf', size: '2.1 MB', url: '/downloads/validation_report.pdf' }
        ]
      },
      {
        id: 'RUN-003',
        workflowId: 'WF-003',
        workflowName: 'Incident Response Workflow',
        status: 'failed',
        startTime: 'Yesterday 3:45 PM',
        endTime: 'Yesterday 3:52 PM',
        duration: 420,
        triggeredBy: 'Alert: High CPU Usage',
        progress: 75,
        logs: [
          { timestamp: 'Yesterday 3:45 PM', level: 'info', message: 'Incident detected: High CPU usage on prod-api-01' },
          { timestamp: 'Yesterday 3:46 PM', level: 'info', message: 'Severity assessed as HIGH' },
          { timestamp: 'Yesterday 3:48 PM', level: 'info', message: 'Notification sent to on-call team' },
          { timestamp: 'Yesterday 3:52 PM', level: 'error', message: 'Failed to isolate system: Permission denied' }
        ]
      },
      {
        id: 'RUN-004',
        workflowId: 'WF-001',
        workflowName: 'CI/CD Pipeline',
        status: 'queued',
        startTime: 'Scheduled',
        triggeredBy: 'Manual (John Smith)',
        progress: 0,
        logs: []
      }
    ]);
  }, []);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active':
      case 'success': return 'text-green-500 bg-green-900/20';
      case 'running': return 'text-blue-500 bg-blue-900/20';
      case 'failed': return 'text-red-500 bg-red-900/20';
      case 'draft':
      case 'queued': return 'text-yellow-500 bg-yellow-900/20';
      case 'inactive':
      case 'cancelled': return 'text-gray-500 bg-gray-900/20';
      case 'deprecated': return 'text-orange-500 bg-orange-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch(category) {
      case 'cicd': return <GitBranch className="w-4 h-4 text-blue-500" />;
      case 'data': return <Database className="w-4 h-4 text-purple-500" />;
      case 'infrastructure': return <Server className="w-4 h-4 text-green-500" />;
      case 'security': return <Shield className="w-4 h-4 text-red-500" />;
      case 'business': return <BarChart className="w-4 h-4 text-yellow-500" />;
      case 'custom': return <Code className="w-4 h-4 text-cyan-500" />;
      default: return <Box className="w-4 h-4 text-gray-500" />;
    }
  };

  const getNodeTypeIcon = (type: string) => {
    switch(type) {
      case 'start': return <PlayCircle className="w-4 h-4 text-green-500" />;
      case 'end': return <StopCircle className="w-4 h-4 text-red-500" />;
      case 'action': return <Zap className="w-4 h-4 text-blue-500" />;
      case 'condition': return <GitBranch className="w-4 h-4 text-yellow-500" />;
      case 'parallel': return <GitMerge className="w-4 h-4 text-purple-500" />;
      case 'loop': return <RotateCcw className="w-4 h-4 text-cyan-500" />;
      default: return <Box className="w-4 h-4 text-gray-500" />;
    }
  };

  const filteredWorkflows = workflows.filter(workflow => {
    if (filterCategory !== 'all' && workflow.category !== filterCategory) return false;
    if (filterStatus !== 'all' && workflow.status !== filterStatus) return false;
    return true;
  });

  const stats = {
    totalWorkflows: workflows.length,
    activeWorkflows: workflows.filter(w => w.status === 'active').length,
    totalRuns: workflowRuns.length,
    runningRuns: workflowRuns.filter(r => r.status === 'running').length,
    avgSuccessRate: Math.round(workflows.reduce((sum, w) => sum + w.metrics.successRate, 0) / workflows.length),
    totalExecutions: workflows.reduce((sum, w) => sum + w.metrics.totalRuns, 0)
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Workflow Engine</h1>
            <p className="text-sm text-gray-400 mt-1">Design, execute, and monitor complex workflows</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setViewMode(
                viewMode === 'workflows' ? 'runs' : 
                viewMode === 'runs' ? 'builder' : 'workflows'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2"
            >
              {viewMode === 'workflows' ? <Activity className="w-4 h-4" /> : 
               viewMode === 'runs' ? <Code className="w-4 h-4" /> :
               <GitBranch className="w-4 h-4" />}
              <span>
                {viewMode === 'workflows' ? 'Runs' : 
                 viewMode === 'runs' ? 'Builder' : 'Workflows'}
              </span>
            </button>
            
            <button
              onClick={() => setShowCreateWorkflow(true)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2"
            >
              <Plus className="w-4 h-4" />
              <span>New Workflow</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <GitBranch className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Workflows</p>
              <p className="text-xl font-bold">{stats.totalWorkflows}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Active</p>
              <p className="text-xl font-bold">{stats.activeWorkflows}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Activity className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Total Runs</p>
              <p className="text-xl font-bold">{stats.totalRuns}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <PlayCircle className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Running</p>
              <p className="text-xl font-bold">{stats.runningRuns}</p>
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
            <Timer className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Executions</p>
              <p className="text-xl font-bold">{stats.totalExecutions}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {viewMode === 'workflows' && (
          <>
            {/* Filters */}
            <div className="flex items-center space-x-3 mb-6">
              <select
                value={filterCategory}
                onChange={(e) => setFilterCategory(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Categories</option>
                <option value="cicd">CI/CD</option>
                <option value="data">Data</option>
                <option value="infrastructure">Infrastructure</option>
                <option value="security">Security</option>
                <option value="business">Business</option>
                <option value="custom">Custom</option>
              </select>
              
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Status</option>
                <option value="draft">Draft</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
                <option value="deprecated">Deprecated</option>
              </select>
            </div>

            {/* Workflows List */}
            <div className="space-y-4">
              {filteredWorkflows.map(workflow => (
                <div key={workflow.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        {getCategoryIcon(workflow.category)}
                        <span className="text-xs text-gray-500 font-mono">{workflow.id}</span>
                        <span className={`px-2 py-1 text-xs rounded ${getStatusColor(workflow.status)}`}>
                          {workflow.status.toUpperCase()}
                        </span>
                        <span className="text-xs text-gray-400">v{workflow.version}</span>
                      </div>
                      <h3 className="text-sm font-bold mb-1">{workflow.name}</h3>
                      <p className="text-xs text-gray-400 mb-2">{workflow.description}</p>
                    </div>
                    <ChevronRight className="w-5 h-5 text-gray-500" />
                  </div>
                  
                  {/* Workflow Visualization */}
                  <div className="mb-3 p-3 bg-gray-800 rounded">
                    <div className="flex items-center space-x-2 overflow-x-auto">
                      {workflow.nodes.slice(0, 6).map((node, idx) => (
                        <React.Fragment key={node.id}>
                          <div className="flex items-center space-x-1">
                            {getNodeTypeIcon(node.type)}
                            <span className="text-xs text-gray-400 whitespace-nowrap">{node.name}</span>
                          </div>
                          {idx < Math.min(5, workflow.nodes.length - 1) && (
                            <ArrowRight className="w-3 h-3 text-gray-600 flex-shrink-0" />
                          )}
                        </React.Fragment>
                      ))}
                      {workflow.nodes.length > 6 && (
                        <span className="text-xs text-gray-500">+{workflow.nodes.length - 6} nodes</span>
                      )}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-4 gap-4 mb-3">
                    <div>
                      <p className="text-xs text-gray-400">Total Runs</p>
                      <p className="text-sm font-bold">{workflow.metrics.totalRuns}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Success Rate</p>
                      <p className="text-sm font-bold">{workflow.metrics.successRate}%</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Avg Duration</p>
                      <p className="text-sm font-bold">{Math.round(workflow.metrics.averageDuration / 60)}m</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Last Run</p>
                      <p className="text-sm">{workflow.metrics.lastRun || 'Never'}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between pt-3 border-t border-gray-800">
                    <div className="flex items-center space-x-3 text-xs">
                      <span className="flex items-center space-x-1">
                        <User className="w-3 h-3" />
                        <span>{workflow.owner}</span>
                      </span>
                      <span className="text-gray-500">Modified {workflow.modified}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      {workflow.status === 'active' && (
                        <button className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs">
                          Run
                        </button>
                      )}
                      <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                        Edit
                      </button>
                      <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                        Clone
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'runs' && (
          /* Workflow Runs */
          <div className="space-y-3">
            {workflowRuns.map(run => (
              <div key={run.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-4">
                    <div className={`w-2 h-2 rounded-full ${
                      run.status === 'success' ? 'bg-green-500' :
                      run.status === 'running' ? 'bg-blue-500 animate-pulse' :
                      run.status === 'failed' ? 'bg-red-500' :
                      run.status === 'queued' ? 'bg-yellow-500' :
                      'bg-gray-500'
                    }`} />
                    <div>
                      <div className="flex items-center space-x-3">
                        <span className="text-sm font-bold">{run.workflowName}</span>
                        <span className="text-xs text-gray-500 font-mono">{run.id}</span>
                        <span className={`px-2 py-1 text-xs rounded ${getStatusColor(run.status)}`}>
                          {run.status.toUpperCase()}
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 text-xs text-gray-500 mt-1">
                        <span>Started: {run.startTime}</span>
                        {run.duration && <span>Duration: {Math.round(run.duration / 60)}m</span>}
                        <span>Triggered by: {run.triggeredBy}</span>
                      </div>
                    </div>
                  </div>
                  <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    View Details
                  </button>
                </div>
                
                {run.status === 'running' && (
                  <div className="mb-3">
                    <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                      <span>Progress: {run.progress}%</span>
                      {run.currentNode && <span>Current: {run.currentNode}</span>}
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${run.progress}%` }}
                      />
                    </div>
                  </div>
                )}
                
                {run.logs && run.logs.length > 0 && (
                  <div className="p-2 bg-gray-800 rounded text-xs font-mono space-y-1 max-h-32 overflow-y-auto">
                    {run.logs.slice(-3).map((log, idx) => (
                      <div key={idx} className="flex items-start space-x-2">
                        <span className="text-gray-500">{log.timestamp}</span>
                        <span className={`${
                          log.level === 'error' ? 'text-red-500' :
                          log.level === 'warning' ? 'text-yellow-500' :
                          'text-gray-300'
                        }`}>
                          {log.message}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
                
                {run.artifacts && run.artifacts.length > 0 && (
                  <div className="flex items-center space-x-2 mt-3">
                    <span className="text-xs text-gray-400">Artifacts:</span>
                    {run.artifacts.map(artifact => (
                      <button key={artifact.name} className="px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs flex items-center space-x-1">
                        <Download className="w-3 h-3" />
                        <span>{artifact.name} ({artifact.size})</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {viewMode === 'builder' && (
          /* Workflow Builder */
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-center h-96 text-gray-600">
              <div className="text-center">
                <GitMerge className="w-16 h-16 mx-auto mb-4" />
                <p className="text-lg font-bold mb-2">Visual Workflow Builder</p>
                <p className="text-sm mb-4">Drag and drop nodes to create complex workflows</p>
                <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                  Open Builder
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}