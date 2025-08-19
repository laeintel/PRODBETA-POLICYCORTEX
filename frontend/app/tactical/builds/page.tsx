'use client';

import { useState, useMemo } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { Package, CheckCircle, XCircle, Clock, AlertTriangle, TrendingUp, TrendingDown, Play, RefreshCw, Download, Filter, GitBranch, Timer } from 'lucide-react';

export default function Page() {
  const [selectedTimeframe, setSelectedTimeframe] = useState('24h');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [selectedProject, setSelectedProject] = useState('all');

  // Mock build data
  const builds = [
    {
      id: '1',
      project: 'PolicyCortex Backend',
      branch: 'main',
      buildNumber: '1234',
      commit: 'feat: add cost analytics endpoint',
      commitHash: 'a1b2c3d4',
      status: 'success',
      duration: '2m 14s',
      started: '2024-08-19T10:30:00Z',
      completed: '2024-08-19T10:32:14Z',
      triggeredBy: 'push',
      author: 'John Doe',
      tests: { passed: 142, failed: 0, total: 142 },
      coverage: 87.5,
      artifacts: ['backend-v1.2.3.jar', 'coverage-report.html'],
      size: '15.2 MB'
    },
    {
      id: '2',
      project: 'PolicyCortex Frontend',
      branch: 'feature/dashboard-improvements',
      buildNumber: '856',
      commit: 'refactor: improve dashboard performance',
      commitHash: 'x7y8z9w1',
      status: 'running',
      duration: '1m 45s',
      started: '2024-08-19T10:28:00Z',
      triggeredBy: 'pull_request',
      author: 'Jane Smith',
      tests: { passed: 89, failed: 2, total: 95 },
      coverage: 92.1,
      currentStep: 'Running tests',
      progress: 65
    },
    {
      id: '3',
      project: 'PolicyCortex API Gateway',
      branch: 'hotfix/security-patch',
      buildNumber: '623',
      commit: 'fix: resolve authentication vulnerability',
      commitHash: 'p4q5r6s7',
      status: 'failed',
      duration: '1m 52s',
      started: '2024-08-19T10:25:00Z',
      completed: '2024-08-19T10:26:52Z',
      triggeredBy: 'push',
      author: 'Mike Johnson',
      tests: { passed: 78, failed: 5, total: 83 },
      coverage: 82.3,
      failureReason: 'Test failures in authentication module',
      artifacts: ['build-logs.txt', 'test-results.xml']
    },
    {
      id: '4',
      project: 'PolicyCortex Mobile App',
      branch: 'main',
      buildNumber: '445',
      commit: 'feat: add biometric authentication',
      commitHash: 't7u8v9w2',
      status: 'queued',
      triggeredBy: 'schedule',
      author: 'Sarah Wilson',
      estimatedDuration: '3m 30s'
    }
  ];

  const buildStats = useMemo(() => {
    const total = builds.length;
    const success = builds.filter(b => b.status === 'success').length;
    const failed = builds.filter(b => b.status === 'failed').length;
    const running = builds.filter(b => b.status === 'running').length;
    const queued = builds.filter(b => b.status === 'queued').length;
    const successRate = total > 0 ? (success / (success + failed)) * 100 : 0;
    const avgDuration = '2m 35s';
    const totalTests = builds.reduce((sum, b) => sum + (b.tests?.total || 0), 0);
    const passedTests = builds.reduce((sum, b) => sum + (b.tests?.passed || 0), 0);
    const testPassRate = totalTests > 0 ? (passedTests / totalTests) * 100 : 0;

    return { total, success, failed, running, queued, successRate, avgDuration, testPassRate };
  }, [builds]);

  const trendData = [
    { date: '19-08', builds: 24, success: 22, failed: 2 },
    { date: '18-08', builds: 28, success: 25, failed: 3 },
    { date: '17-08', builds: 19, success: 18, failed: 1 },
    { date: '16-08', builds: 31, success: 28, failed: 3 },
    { date: '15-08', builds: 26, success: 24, failed: 2 }
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-500" />;
      case 'running': return <Timer className="w-4 h-4 text-blue-500 animate-pulse" />;
      case 'queued': return <Clock className="w-4 h-4 text-yellow-500" />;
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
            value={selectedProject} 
            onChange={(e) => setSelectedProject(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Projects</option>
            <option value="backend">Backend</option>
            <option value="frontend">Frontend</option>
            <option value="api-gateway">API Gateway</option>
            <option value="mobile">Mobile App</option>
          </select>
          <select 
            value={selectedStatus} 
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Status</option>
            <option value="success">Success</option>
            <option value="failed">Failed</option>
            <option value="running">Running</option>
          </select>
        </div>
        <div className="flex items-center space-x-3">
          <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white flex items-center space-x-2 transition-colors">
            <Play className="w-4 h-4" />
            <span>Trigger Build</span>
          </button>
          <button className="bg-gray-800 hover:bg-gray-700 px-4 py-2 rounded-lg text-white flex items-center space-x-2 transition-colors">
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Build Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 backdrop-blur-md rounded-xl border border-blue-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <Package className="w-8 h-8 text-blue-400" />
            <div className="text-sm text-blue-300">Total</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">{buildStats.total}</p>
          <p className="text-blue-300 text-sm">Builds Today</p>
        </div>

        <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 backdrop-blur-md rounded-xl border border-green-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <CheckCircle className="w-8 h-8 text-green-400" />
            <div className="text-sm text-green-300">{buildStats.successRate.toFixed(1)}%</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">{buildStats.success}</p>
          <p className="text-green-300 text-sm">Successful</p>
        </div>

        <div className="bg-gradient-to-br from-red-900/50 to-red-800/30 backdrop-blur-md rounded-xl border border-red-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <XCircle className="w-8 h-8 text-red-400" />
            <div className="text-sm text-red-300">{buildStats.testPassRate.toFixed(1)}%</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">{buildStats.failed}</p>
          <p className="text-red-300 text-sm">Failed Builds</p>
        </div>

        <div className="bg-gradient-to-br from-purple-900/50 to-purple-800/30 backdrop-blur-md rounded-xl border border-purple-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <Timer className="w-8 h-8 text-purple-400" />
            <div className="text-sm text-purple-300">Avg</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">{buildStats.avgDuration}</p>
          <p className="text-purple-300 text-sm">Build Time</p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        {/* Build List */}
        <div className="xl:col-span-2">
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
              <h3 className="text-xl font-bold text-white">Recent Builds</h3>
              <div className="flex items-center space-x-2">
                <Filter className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-400">Filtered</span>
              </div>
            </div>
            <div className="divide-y divide-gray-800">
              {builds.map((build) => (
                <div key={build.id} className="p-6 hover:bg-gray-800/30 transition-colors">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(build.status)}
                      <div>
                        <h4 className="text-white font-semibold">{build.project}</h4>
                        <p className="text-gray-400 text-sm">#{build.buildNumber} • {build.branch}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(build.status)}`}>
                        {build.status}
                      </span>
                      <button className="text-blue-400 hover:text-blue-300 text-sm">
                        View Logs
                      </button>
                    </div>
                  </div>

                  <div className="mb-4">
                    <p className="text-gray-300 text-sm mb-1">{build.commit}</p>
                    <div className="flex items-center space-x-4 text-xs text-gray-400">
                      <span>#{build.commitHash}</span>
                      <span>by {build.author}</span>
                      <span>triggered by {build.triggeredBy}</span>
                      {build.duration && <span>Duration: {build.duration}</span>}
                      {build.size && <span>Size: {build.size}</span>}
                    </div>
                  </div>

                  {build.status === 'running' && build.progress && (
                    <div className="mb-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-blue-300">{build.currentStep}</span>
                        <span className="text-sm text-gray-400">{build.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all duration-1000"
                          style={{ width: `${build.progress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {build.tests && (
                    <div className="grid grid-cols-3 gap-4 mb-4">
                      <div className="text-center">
                        <p className="text-2xl font-bold text-green-400">{build.tests.passed}</p>
                        <p className="text-xs text-gray-400">Passed</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-red-400">{build.tests.failed}</p>
                        <p className="text-xs text-gray-400">Failed</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-blue-400">{build.coverage?.toFixed(1)}%</p>
                        <p className="text-xs text-gray-400">Coverage</p>
                      </div>
                    </div>
                  )}

                  {build.status === 'failed' && build.failureReason && (
                    <div className="mb-4 p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
                      <div className="flex items-center space-x-2 mb-2">
                        <AlertTriangle className="w-4 h-4 text-red-400" />
                        <span className="text-red-300 text-sm font-medium">Build Failed</span>
                      </div>
                      <p className="text-red-200 text-sm">{build.failureReason}</p>
                      <div className="mt-2 flex items-center space-x-3">
                        <button className="text-red-400 hover:text-red-300 text-sm">View Details</button>
                        <button className="text-red-400 hover:text-red-300 text-sm">Rebuild</button>
                      </div>
                    </div>
                  )}

                  {build.artifacts && build.artifacts.length > 0 && (
                    <div className="flex items-center space-x-4">
                      <span className="text-sm text-gray-400">Artifacts:</span>
                      {build.artifacts.map((artifact, index) => (
                        <button 
                          key={index}
                          className="text-blue-400 hover:text-blue-300 text-sm flex items-center space-x-1"
                        >
                          <Download className="w-3 h-3" />
                          <span>{artifact}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Build Trends */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center">
              <TrendingUp className="w-5 h-5 text-green-500 mr-2" />
              Build Trends
            </h3>
            <div className="space-y-3">
              {trendData.map((day, index) => (
                <div key={index} className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">{day.date}</span>
                  <div className="flex items-center space-x-2">
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      <span className="text-xs text-green-400">{day.success}</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-red-500 rounded-full" />
                      <span className="text-xs text-red-400">{day.failed}</span>
                    </div>
                    <span className="text-sm text-gray-400">/{day.builds}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Build Performance */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-white mb-4">Performance Metrics</h3>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">Success Rate</span>
                  <span className="text-sm font-semibold text-green-400">{buildStats.successRate.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${buildStats.successRate}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">Test Pass Rate</span>
                  <span className="text-sm font-semibold text-blue-400">{buildStats.testPassRate.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-blue-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${buildStats.testPassRate}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">Avg Build Time</span>
                  <span className="text-sm font-semibold text-purple-400">{buildStats.avgDuration}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-white mb-4">Quick Actions</h3>
            <div className="space-y-3">
              <button className="w-full bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white text-sm transition-colors">
                Rebuild Failed
              </button>
              <button className="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg text-white text-sm transition-colors">
                Build All Projects
              </button>
              <button className="w-full bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg text-white text-sm transition-colors">
                View Build History
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <TacticalPageTemplate 
      title="Build Status" 
      subtitle="Continuous Integration Build Monitoring" 
      icon={Package}
    >
      {content}
    </TacticalPageTemplate>
  );
}