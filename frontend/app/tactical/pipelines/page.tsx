'use client';

import { useState, useMemo } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { GitBranch, Play, Pause, Square, CheckCircle, XCircle, Clock, AlertTriangle, Activity, Settings, RefreshCw, Filter, Calendar, Users } from 'lucide-react';

export default function Page() {
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [selectedTimeframe, setSelectedTimeframe] = useState('24h');
  const [selectedRepo, setSelectedRepo] = useState('all');

  // Mock pipeline data
  const pipelines = [
    {
      id: '1',
      name: 'PolicyCortex Backend',
      repository: 'policycortex/backend',
      branch: 'main',
      status: 'success',
      duration: '2m 34s',
      started: '2024-08-19T10:30:00Z',
      completed: '2024-08-19T10:32:34Z',
      triggeredBy: 'John Doe',
      commit: 'feat: add cost analytics endpoint',
      commitHash: 'a1b2c3d',
      stages: [
        { name: 'Build', status: 'success', duration: '1m 12s' },
        { name: 'Test', status: 'success', duration: '45s' },
        { name: 'Security Scan', status: 'success', duration: '23s' },
        { name: 'Deploy', status: 'success', duration: '14s' }
      ]
    },
    {
      id: '2',
      name: 'PolicyCortex Frontend',
      repository: 'policycortex/frontend',
      branch: 'feature/dashboard-improvements',
      status: 'running',
      duration: '1m 45s',
      started: '2024-08-19T10:28:00Z',
      triggeredBy: 'Jane Smith',
      commit: 'refactor: improve dashboard performance',
      commitHash: 'x7y8z9w',
      stages: [
        { name: 'Build', status: 'success', duration: '1m 2s' },
        { name: 'Test', status: 'running', duration: '43s' },
        { name: 'Security Scan', status: 'pending', duration: null },
        { name: 'Deploy', status: 'pending', duration: null }
      ]
    },
    {
      id: '3',
      name: 'PolicyCortex API Gateway',
      repository: 'policycortex/api-gateway',
      branch: 'hotfix/security-patch',
      status: 'failed',
      duration: '3m 12s',
      started: '2024-08-19T10:25:00Z',
      completed: '2024-08-19T10:28:12Z',
      triggeredBy: 'Mike Johnson',
      commit: 'fix: resolve authentication vulnerability',
      commitHash: 'p4q5r6s',
      stages: [
        { name: 'Build', status: 'success', duration: '1m 20s' },
        { name: 'Test', status: 'failed', duration: '1m 45s' },
        { name: 'Security Scan', status: 'cancelled', duration: null },
        { name: 'Deploy', status: 'cancelled', duration: null }
      ]
    },
    {
      id: '4',
      name: 'PolicyCortex Infrastructure',
      repository: 'policycortex/terraform',
      branch: 'main',
      status: 'queued',
      triggeredBy: 'Sarah Wilson',
      commit: 'feat: add new Azure resources',
      commitHash: 't7u8v9w',
      stages: [
        { name: 'Plan', status: 'pending', duration: null },
        { name: 'Validate', status: 'pending', duration: null },
        { name: 'Apply', status: 'pending', duration: null }
      ]
    }
  ];

  const pipelineStats = useMemo(() => {
    const total = pipelines.length;
    const success = pipelines.filter(p => p.status === 'success').length;
    const failed = pipelines.filter(p => p.status === 'failed').length;
    const running = pipelines.filter(p => p.status === 'running').length;
    const queued = pipelines.filter(p => p.status === 'queued').length;
    const successRate = total > 0 ? (success / (success + failed)) * 100 : 0;

    return { total, success, failed, running, queued, successRate };
  }, [pipelines]);

  const recentActivity = [
    { type: 'deployment', message: 'PolicyCortex Backend deployed to production', time: '5 minutes ago', status: 'success' },
    { type: 'failure', message: 'API Gateway pipeline failed - test stage', time: '8 minutes ago', status: 'failed' },
    { type: 'trigger', message: 'New pipeline triggered for Frontend', time: '12 minutes ago', status: 'info' },
    { type: 'approval', message: 'Infrastructure changes approved for deployment', time: '15 minutes ago', status: 'success' }
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-500" />;
      case 'running': return <Activity className="w-4 h-4 text-blue-500 animate-pulse" />;
      case 'queued': return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'pending': return <Clock className="w-4 h-4 text-gray-500" />;
      case 'cancelled': return <Square className="w-4 h-4 text-gray-500" />;
      default: return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'bg-green-900/50 text-green-300 border-green-500/30';
      case 'failed': return 'bg-red-900/50 text-red-300 border-red-500/30';
      case 'running': return 'bg-blue-900/50 text-blue-300 border-blue-500/30';
      case 'queued': return 'bg-yellow-900/50 text-yellow-300 border-yellow-500/30';
      default: return 'bg-gray-900/50 text-gray-300 border-gray-500/30';
    }
  };

  const content = (
    <div className="space-y-8">
      {/* Header Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <select 
            value={selectedTimeframe} 
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          <select 
            value={selectedRepo} 
            onChange={(e) => setSelectedRepo(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Repositories</option>
            <option value="backend">Backend</option>
            <option value="frontend">Frontend</option>
            <option value="infrastructure">Infrastructure</option>
          </select>
          <select 
            value={selectedFilter} 
            onChange={(e) => setSelectedFilter(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Status</option>
            <option value="running">Running</option>
            <option value="failed">Failed</option>
            <option value="success">Success</option>
          </select>
        </div>
        <div className="flex items-center space-x-3">
          <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white flex items-center space-x-2 transition-colors">
            <Play className="w-4 h-4" />
            <span>Trigger Pipeline</span>
          </button>
          <button className="bg-gray-800 hover:bg-gray-700 px-4 py-2 rounded-lg text-white flex items-center space-x-2 transition-colors">
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Pipeline Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 backdrop-blur-md rounded-xl border border-blue-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <GitBranch className="w-8 h-8 text-blue-400" />
            <div className="text-sm text-blue-300">Total</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">{pipelineStats.total}</p>
          <p className="text-blue-300 text-sm">Pipelines</p>
        </div>

        <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 backdrop-blur-md rounded-xl border border-green-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <CheckCircle className="w-8 h-8 text-green-400" />
            <div className="text-sm text-green-300">{pipelineStats.successRate.toFixed(1)}%</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">{pipelineStats.success}</p>
          <p className="text-green-300 text-sm">Successful</p>
        </div>

        <div className="bg-gradient-to-br from-red-900/50 to-red-800/30 backdrop-blur-md rounded-xl border border-red-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <XCircle className="w-8 h-8 text-red-400" />
            <div className="text-sm text-red-300">Failed</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">{pipelineStats.failed}</p>
          <p className="text-red-300 text-sm">Failures</p>
        </div>

        <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 backdrop-blur-md rounded-xl border border-blue-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <Activity className="w-8 h-8 text-blue-400 animate-pulse" />
            <div className="text-sm text-blue-300">Active</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">{pipelineStats.running}</p>
          <p className="text-blue-300 text-sm">Running</p>
        </div>

        <div className="bg-gradient-to-br from-yellow-900/50 to-yellow-800/30 backdrop-blur-md rounded-xl border border-yellow-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <Clock className="w-8 h-8 text-yellow-400" />
            <div className="text-sm text-yellow-300">Queue</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">{pipelineStats.queued}</p>
          <p className="text-yellow-300 text-sm">Queued</p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        {/* Pipeline List */}
        <div className="xl:col-span-2">
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
              <h3 className="text-xl font-bold text-white">Active Pipelines</h3>
              <div className="flex items-center space-x-2">
                <Filter className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-400">Filter applied</span>
              </div>
            </div>
            <div className="divide-y divide-gray-800">
              {pipelines.map((pipeline) => (
                <div key={pipeline.id} className="p-6 hover:bg-gray-800/30 transition-colors">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(pipeline.status)}
                      <div>
                        <h4 className="text-white font-semibold">{pipeline.name}</h4>
                        <p className="text-gray-400 text-sm">{pipeline.repository} • {pipeline.branch}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(pipeline.status)}`}>
                        {pipeline.status}
                      </span>
                      <button className="text-gray-400 hover:text-white">
                        <Settings className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  <div className="mb-4">
                    <p className="text-gray-300 text-sm mb-1">{pipeline.commit}</p>
                    <div className="flex items-center space-x-4 text-xs text-gray-400">
                      <span>#{pipeline.commitHash}</span>
                      <span className="flex items-center">
                        <Users className="w-3 h-3 mr-1" />
                        {pipeline.triggeredBy}
                      </span>
                      {pipeline.duration && <span>Duration: {pipeline.duration}</span>}
                      <span>
                        {pipeline.completed 
                          ? `Completed ${new Date(pipeline.completed as string).toLocaleTimeString()}`
                          : `Started ${new Date(pipeline.started as string).toLocaleTimeString()}`
                        }
                      </span>
                    </div>
                  </div>

                  {/* Pipeline Stages */}
                  <div className="flex items-center space-x-2">
                    {pipeline.stages.map((stage, index) => (
                      <div key={index} className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-gray-400">{stage.name}</span>
                          {stage.duration && <span className="text-xs text-gray-500">{stage.duration}</span>}
                        </div>
                        <div className={`h-2 rounded-full ${
                          stage.status === 'success' ? 'bg-green-500' :
                          stage.status === 'failed' ? 'bg-red-500' :
                          stage.status === 'running' ? 'bg-blue-500' :
                          stage.status === 'cancelled' ? 'bg-gray-600' :
                          'bg-gray-700'
                        }`} />
                      </div>
                    ))}
                  </div>

                  {pipeline.status === 'failed' && (
                    <div className="mt-4 p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <AlertTriangle className="w-4 h-4 text-red-400" />
                        <span className="text-red-300 text-sm font-medium">Pipeline Failed</span>
                      </div>
                      <p className="text-red-200 text-sm mt-1">
                        Test stage failed: 3 tests failed in authentication module
                      </p>
                      <div className="mt-2 flex items-center space-x-3">
                        <button className="text-red-400 hover:text-red-300 text-sm">View Logs</button>
                        <button className="text-red-400 hover:text-red-300 text-sm">Retry</button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Recent Activity Sidebar */}
        <div className="space-y-6">
          {/* Recent Activity */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center">
              <Activity className="w-5 h-5 text-blue-500 mr-2" />
              Recent Activity
            </h3>
            <div className="space-y-4">
              {recentActivity.map((activity, index) => (
                <div key={index} className="flex items-start space-x-3">
                  <div className={`w-2 h-2 rounded-full mt-2 ${
                    activity.status === 'success' ? 'bg-green-500' :
                    activity.status === 'failed' ? 'bg-red-500' :
                    'bg-blue-500'
                  }`} />
                  <div className="flex-1 min-w-0">
                    <p className="text-white text-sm">{activity.message}</p>
                    <p className="text-gray-400 text-xs">{activity.time}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Pipeline Health */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-white mb-4">Pipeline Health</h3>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">Success Rate</span>
                  <span className="text-sm font-semibold text-green-400">{pipelineStats.successRate.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${pipelineStats.successRate}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">Average Duration</span>
                  <span className="text-sm font-semibold text-blue-400">2m 48s</span>
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">Queue Wait Time</span>
                  <span className="text-sm font-semibold text-yellow-400">1m 12s</span>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-white mb-4">Quick Actions</h3>
            <div className="space-y-3">
              <button className="w-full bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white text-sm transition-colors">
                Run All Tests
              </button>
              <button className="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg text-white text-sm transition-colors">
                Deploy to Staging
              </button>
              <button className="w-full bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg text-white text-sm transition-colors">
                View Pipeline Metrics
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <TacticalPageTemplate 
      title="Pipeline Dashboard" 
      subtitle="CI/CD Pipeline Management & Monitoring" 
      icon={GitBranch}
    >
      {content}
    </TacticalPageTemplate>
  );
}