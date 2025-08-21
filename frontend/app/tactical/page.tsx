'use client';

import React, { useState, useEffect } from 'react';
import {
  AlertTriangle, Shield, Activity, Zap, Terminal, Users,
  AlertCircle, CheckCircle, Clock, TrendingUp, Server,
  Database, Cloud, Lock, Bell, Play, Pause, RefreshCw,
  ChevronRight, Cpu, HardDrive, Wifi, Target, Radio
} from 'lucide-react';
import ResponsiveGrid, { ResponsiveContainer, ResponsiveText } from '@/components/ResponsiveGrid';
import { toast } from '@/hooks/useToast'

interface Alert {
  id: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  time: string;
  source: string;
  status: 'active' | 'acknowledged' | 'resolved';
}

interface SystemMetric {
  name: string;
  value: number;
  unit: string;
  status: 'healthy' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
}

export default function TacticalOperationsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [commandInput, setCommandInput] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [isWarRoomActive, setIsWarRoomActive] = useState(false);

  useEffect(() => {
    // Simulate real-time alerts
    const mockAlerts: Alert[] = [
      {
        id: 'ALT-001',
        severity: 'critical',
        title: 'Database Connection Failure',
        description: 'Primary database cluster experiencing connectivity issues',
        time: '2 min ago',
        source: 'Azure SQL',
        status: 'active'
      },
      {
        id: 'ALT-002',
        severity: 'high',
        title: 'Unusual Login Activity Detected',
        description: 'Multiple failed login attempts from unknown IP addresses',
        time: '5 min ago',
        source: 'Azure AD',
        status: 'acknowledged'
      },
      {
        id: 'ALT-003',
        severity: 'medium',
        title: 'Cost Threshold Exceeded',
        description: 'Monthly spending has exceeded 80% of budget',
        time: '15 min ago',
        source: 'Cost Management',
        status: 'active'
      },
      {
        id: 'ALT-004',
        severity: 'low',
        title: 'Certificate Expiry Warning',
        description: 'SSL certificate expires in 30 days',
        time: '1 hour ago',
        source: 'Key Vault',
        status: 'resolved'
      }
    ];
    setAlerts(mockAlerts);
  }, []);

  const systemMetrics: SystemMetric[] = [
    { name: 'CPU Usage', value: 78, unit: '%', status: 'warning', trend: 'up' },
    { name: 'Memory', value: 62, unit: '%', status: 'healthy', trend: 'stable' },
    { name: 'Network', value: 245, unit: 'Mbps', status: 'healthy', trend: 'up' },
    { name: 'Storage', value: 89, unit: '%', status: 'critical', trend: 'up' },
    { name: 'API Latency', value: 142, unit: 'ms', status: 'healthy', trend: 'down' },
    { name: 'Error Rate', value: 0.3, unit: '%', status: 'healthy', trend: 'down' }
  ];

  const quickActions = [
    { icon: Shield, label: 'Initiate Security Scan', color: 'blue' },
    { icon: RefreshCw, label: 'Restart Services', color: 'green' },
    { icon: Lock, label: 'Lock Down Resources', color: 'red' },
    { icon: Database, label: 'Backup Database', color: 'purple' },
    { icon: Users, label: 'Alert Team', color: 'yellow' },
    { icon: Terminal, label: 'Execute Playbook', color: 'pink' }
  ];

  const handleCommand = (e: React.FormEvent) => {
    e.preventDefault();
    if (commandInput.trim()) {
      setCommandHistory([...commandHistory, `> ${commandInput}`, 'Command executed successfully']);
      setCommandInput('');
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-500 bg-red-500/10 border-red-500/20';
      case 'high': return 'text-orange-500 bg-orange-500/10 border-orange-500/20';
      case 'medium': return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/20';
      case 'low': return 'text-blue-500 bg-blue-500/10 border-blue-500/20';
      default: return 'text-gray-500 bg-gray-500/10 border-gray-500/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-500';
      case 'warning': return 'text-yellow-500';
      case 'critical': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  return (
    <ResponsiveContainer className="min-h-screen bg-gray-900 text-white py-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center space-x-3">
              <Activity className="w-8 h-8 text-blue-500" />
              <span>Tactical Operations Center</span>
            </h1>
            <p className="text-gray-400">Real-time monitoring and incident response</p>
          </div>
          <div className="flex items-center space-x-4">
            <button type="button"
              onClick={() => setIsWarRoomActive(!isWarRoomActive)}
              className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center space-x-2 ${
                isWarRoomActive 
                  ? 'bg-red-600 hover:bg-red-700 animate-pulse' 
                  : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              <Radio className="w-5 h-5" />
              <span>{isWarRoomActive ? 'War Room Active' : 'Activate War Room'}</span>
            </button>
            <button
              type="button"
              className="px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
              onClick={() => toast({ title: 'Alert settings', description: 'Opening alert settings (coming soon)' })}
            >
              <Bell className="w-4 h-4" />
              <span>Alert Settings</span>
            </button>
          </div>
        </div>

        {/* Critical Alert Banner */}
        {alerts.filter(a => a.severity === 'critical' && a.status === 'active').length > 0 && (
          <div className="bg-red-900/20 border border-red-500 rounded-lg p-4 mb-4 animate-pulse">
            <div className="flex items-center space-x-3">
              <AlertTriangle className="w-6 h-6 text-red-500" />
              <div>
                <p className="font-semibold text-red-500">CRITICAL ALERTS ACTIVE</p>
                <p className="text-sm text-gray-300">Immediate action required - {alerts.filter(a => a.severity === 'critical').length} critical issues detected</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* System Metrics Grid */}
      <ResponsiveGrid variant="metrics" className="mb-6">
        {systemMetrics.map((metric, index) => (
          <div key={index} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-gray-400">{metric.name}</span>
              {metric.trend === 'up' && <TrendingUp className="w-3 h-3 text-green-500" />}
              {metric.trend === 'down' && <TrendingUp className="w-3 h-3 text-red-500 rotate-180" />}
              {metric.trend === 'stable' && <div className="w-3 h-3 bg-gray-500 rounded-full" />}
            </div>
            <div className={`text-2xl font-bold ${getStatusColor(metric.status)}`}>
              {metric.value}{metric.unit}
            </div>
            <div className="mt-2 h-1 bg-gray-700 rounded-full overflow-hidden">
              <div 
                className={`h-full ${
                  metric.status === 'critical' ? 'bg-red-500' :
                  metric.status === 'warning' ? 'bg-yellow-500' :
                  'bg-green-500'
                }`}
                style={{ width: `${metric.value}%` }}
              />
            </div>
          </div>
        ))}
      </ResponsiveGrid>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Alert Feed */}
        <div className="lg:col-span-2">
          <div className="bg-gray-800 rounded-lg border border-gray-700">
            <div className="p-4 border-b border-gray-700">
              <h2 className="text-lg font-semibold flex items-center space-x-2">
                <AlertCircle className="w-5 h-5 text-red-500" />
                <span>Active Alerts</span>
                <span className="ml-auto text-sm bg-red-500 text-white px-2 py-1 rounded">
                  {alerts.filter(a => a.status === 'active').length}
                </span>
              </h2>
            </div>
            <div className="divide-y divide-gray-700 max-h-96 overflow-y-auto">
              {alerts.map((alert) => (
                <div
                  key={alert.id}
                  className="p-4 hover:bg-gray-700/50 cursor-pointer transition-colors"
                  onClick={() => setSelectedAlert(alert)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <span className={`px-2 py-1 text-xs rounded border ${getSeverityColor(alert.severity)}`}>
                          {alert.severity.toUpperCase()}
                        </span>
                        <span className="text-xs text-gray-400">{alert.source}</span>
                        <span className="text-xs text-gray-400">{alert.time}</span>
                      </div>
                      <h3 className="font-medium mb-1">{alert.title}</h3>
                      <p className="text-sm text-gray-400">{alert.description}</p>
                    </div>
                    <div className="ml-4">
                      {alert.status === 'active' && <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />}
                      {alert.status === 'acknowledged' && <div className="w-2 h-2 bg-yellow-500 rounded-full" />}
                      {alert.status === 'resolved' && <CheckCircle className="w-4 h-4 text-green-500" />}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Command Center */}
          <div className="mt-6 bg-gray-800 rounded-lg border border-gray-700">
            <div className="p-4 border-b border-gray-700">
              <h2 className="text-lg font-semibold flex items-center space-x-2">
                <Terminal className="w-5 h-5 text-green-500" />
                <span>Command Center</span>
              </h2>
            </div>
            <div className="p-4">
              <div className="bg-black rounded-lg p-4 font-mono text-sm mb-4 h-32 overflow-y-auto">
                {commandHistory.length === 0 ? (
                  <div className="text-gray-500">Ready for commands...</div>
                ) : (
                  commandHistory.map((line, index) => (
                    <div key={index} className={line.startsWith('>') ? 'text-green-400' : 'text-gray-300'}>
                      {line}
                    </div>
                  ))
                )}
              </div>
              <form onSubmit={handleCommand} className="flex space-x-2">
                <input
                  type="text"
                  value={commandInput}
                  onChange={(e) => setCommandInput(e.target.value)}
                  placeholder="Enter command..."
                  className="flex-1 bg-gray-700 rounded px-4 py-2 text-sm"
                />
                <button
                  type="submit"
                  className="px-4 py-2 bg-green-600 rounded hover:bg-green-700 transition-colors"
                >
                  Execute
                </button>
              </form>
            </div>
          </div>
        </div>

        {/* Right Panel */}
        <div className="space-y-6">
          {/* Quick Actions */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
            <div className="grid grid-cols-2 gap-3">
              {quickActions.map((action, index) => {
                const Icon = action.icon;
                return (
                  <button type="button"
                    key={index}
                    className="p-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors flex flex-col items-center space-y-2"
                  >
                    <Icon className={`w-6 h-6 text-${action.color}-500`} />
                    <span className="text-xs text-center">{action.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Incident Response */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h2 className="text-lg font-semibold mb-4">Incident Response</h2>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
                <span className="text-sm">Auto-remediation</span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" className="sr-only peer" defaultChecked />
                  <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
                <span className="text-sm">Alert escalation</span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" className="sr-only peer" defaultChecked />
                  <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>
              <button
                type="button"
                className="w/full p-3 bg-orange-600 hover:bg-orange-700 rounded transition-colors"
                onClick={() => toast({ title: 'Playbook', description: 'Executing emergency playbook...' })}
              >
                Execute Emergency Playbook
              </button>
            </div>
          </div>

          {/* Communication Channels */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h2 className="text-lg font-semibold mb-4">Communication Channels</h2>
            <div className="space-y-2">
              <div className="flex items-center justify-between p-2">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm">Slack - #incidents</span>
                </div>
                <span className="text-xs text-green-500">Connected</span>
              </div>
              <div className="flex items-center justify-between p-2">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm">Teams - Operations</span>
                </div>
                <span className="text-xs text-green-500">Connected</span>
              </div>
              <div className="flex items-center justify-between p-2">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                  <span className="text-sm">PagerDuty</span>
                </div>
                <span className="text-xs text-yellow-500">Stand-by</span>
              </div>
            </div>
            <button
              type="button"
              className="w-full mt-3 p-2 bg-blue-600 hover:bg-blue-700 rounded transition-colors text-sm"
              onClick={() => toast({ title: 'Broadcast', description: 'Broadcasting alert to channels...' })}
            >
              Broadcast Alert
            </button>
          </div>

          {/* Active Playbooks */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h2 className="text-lg font-semibold mb-4">Active Playbooks</h2>
            <div className="space-y-3">
              <div className="p-3 bg-gray-700 rounded">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Database Recovery</span>
                  <Play className="w-4 h-4 text-green-500" />
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div className="bg-blue-500 h-2 rounded-full" style={{ width: '60%' }}></div>
                </div>
                <p className="text-xs text-gray-400 mt-1">Step 3 of 5</p>
              </div>
              <div className="p-3 bg-gray-700 rounded">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Security Lockdown</span>
                  <Pause className="w-4 h-4 text-yellow-500" />
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '30%' }}></div>
                </div>
                <p className="text-xs text-gray-400 mt-1">Paused - Awaiting approval</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Alert Detail Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setSelectedAlert(null)}>
          <div className="bg-gray-800 rounded-lg p-6 max-w-2xl w-full mx-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 className="text-xl font-bold mb-2">{selectedAlert.title}</h2>
                <div className="flex items-center space-x-3">
                  <span className={`px-2 py-1 text-xs rounded border ${getSeverityColor(selectedAlert.severity)}`}>
                    {selectedAlert.severity.toUpperCase()}
                  </span>
                  <span className="text-sm text-gray-400">{selectedAlert.source}</span>
                  <span className="text-sm text-gray-400">{selectedAlert.time}</span>
                </div>
              </div>
              <button type="button"
                onClick={() => setSelectedAlert(null)}
                className="text-gray-400 hover:text-white text-2xl"
              >
                ×
              </button>
            </div>
            
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-400 mb-2">Description</h3>
              <p className="text-gray-300">{selectedAlert.description}</p>
            </div>

            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-400 mb-2">Affected Resources</h3>
              <div className="bg-gray-900 rounded p-3">
                <code className="text-sm text-green-400">
                  /subscriptions/205b477d/resourceGroups/prod-rg/providers/Microsoft.Sql/servers/prod-sql-01
                </code>
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-400 mb-2">Recommended Actions</h3>
              <ol className="list-decimal list-inside space-y-2 text-sm text-gray-300">
                <li>Check database connection strings and network configuration</li>
                <li>Verify firewall rules and security groups</li>
                <li>Review recent changes to the database configuration</li>
                <li>Check Azure service health for ongoing issues</li>
              </ol>
            </div>

            <div className="flex space-x-3">
              <button
                type="button"
                className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors"
                onClick={() => toast({ title: 'Acknowledged', description: `${selectedAlert.title}` })}
              >
                Acknowledge
              </button>
              <button
                type="button"
                className="px-4 py-2 bg-green-600 rounded hover:bg-green-700 transition-colors"
                onClick={() => toast({ title: 'Resolved', description: `${selectedAlert.title}` })}
              >
                Mark Resolved
              </button>
              <button
                type="button"
                className="px-4 py-2 bg-orange-600 rounded hover:bg-orange-700 transition-colors"
                onClick={() => toast({ title: 'Escalated', description: `${selectedAlert.title}` })}
              >
                Escalate
              </button>
              <button
                type="button"
                className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
                onClick={() => toast({ title: 'Note added', description: 'Added note to alert' })}
              >
                Add Note
              </button>
            </div>
          </div>
        </div>
      )}
    </ResponsiveContainer>
  );
}