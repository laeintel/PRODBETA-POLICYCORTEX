'use client';

import React, { useState, useEffect } from 'react';
import { 
  Bell, BellOff, AlertTriangle, AlertCircle, XCircle, CheckCircle,
  Clock, Filter, Search, Settings, Volume2, VolumeX, Mail, MessageSquare,
  Phone, Zap, TrendingUp, BarChart, Calendar, User, Users, Timer,
  ChevronRight, MoreVertical, ExternalLink, Copy, Archive, Trash2
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface Alert {
  id: string;
  title: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  status: 'active' | 'acknowledged' | 'resolved' | 'suppressed';
  source: string;
  service: string;
  metric?: string;
  threshold?: {
    condition: string;
    value: number;
    actualValue: number;
  };
  timestamp: string;
  acknowledgedAt?: string;
  resolvedAt?: string;
  assignee?: string;
  tags: string[];
  affectedResources: string[];
  notifications: {
    channel: 'email' | 'slack' | 'sms' | 'webhook';
    sentAt: string;
    recipient: string;
    status: 'sent' | 'failed' | 'pending';
  }[];
  actions: {
    type: string;
    status: 'pending' | 'completed' | 'failed';
    timestamp: string;
  }[];
}

interface AlertRule {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  condition: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  frequency: string;
  notification: {
    channels: string[];
    recipients: string[];
    delay: number;
  };
  suppressionRules?: {
    timeWindow?: string;
    conditions?: string[];
  };
  lastTriggered?: string;
  triggerCount: number;
}

interface NotificationChannel {
  id: string;
  name: string;
  type: 'email' | 'slack' | 'sms' | 'webhook' | 'teams' | 'pagerduty';
  enabled: boolean;
  config: any;
  testStatus?: 'success' | 'failed' | 'pending';
  lastUsed?: string;
}

export default function AlertManager() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [notificationChannels, setNotificationChannels] = useState<NotificationChannel[]>([]);
  const [selectedSeverity, setSelectedSeverity] = useState('all');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'alerts' | 'rules' | 'channels'>('alerts');
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);

  useEffect(() => {
    // Initialize with mock alert data
    setAlerts([
      {
        id: 'ALT-001',
        title: 'High CPU Usage on Production Server',
        description: 'CPU utilization has exceeded 90% for more than 5 minutes',
        severity: 'critical',
        status: 'active',
        source: 'CloudWatch',
        service: 'API Gateway',
        metric: 'CPUUtilization',
        threshold: {
          condition: '> 90%',
          value: 90,
          actualValue: 94.5
        },
        timestamp: '5 minutes ago',
        tags: ['production', 'infrastructure', 'auto-scaling'],
        affectedResources: ['prod-api-01', 'prod-api-02'],
        notifications: [
          {
            channel: 'slack',
            sentAt: '4 minutes ago',
            recipient: '#alerts-critical',
            status: 'sent'
          },
          {
            channel: 'email',
            sentAt: '4 minutes ago',
            recipient: 'oncall@company.com',
            status: 'sent'
          }
        ],
        actions: [
          {
            type: 'Auto-scale triggered',
            status: 'completed',
            timestamp: '3 minutes ago'
          }
        ]
      },
      {
        id: 'ALT-002',
        title: 'Database Connection Pool Exhausted',
        description: 'Available database connections dropped below threshold',
        severity: 'high',
        status: 'acknowledged',
        source: 'Application Insights',
        service: 'User Service',
        metric: 'ConnectionPoolAvailable',
        threshold: {
          condition: '< 10',
          value: 10,
          actualValue: 5
        },
        timestamp: '15 minutes ago',
        acknowledgedAt: '10 minutes ago',
        assignee: 'john.doe@company.com',
        tags: ['database', 'performance'],
        affectedResources: ['db-primary', 'user-service'],
        notifications: [
          {
            channel: 'slack',
            sentAt: '14 minutes ago',
            recipient: '#database-team',
            status: 'sent'
          }
        ],
        actions: [
          {
            type: 'Connection pool increased',
            status: 'pending',
            timestamp: '8 minutes ago'
          }
        ]
      },
      {
        id: 'ALT-003',
        title: 'SSL Certificate Expiring Soon',
        description: 'SSL certificate for api.company.com expires in 7 days',
        severity: 'medium',
        status: 'active',
        source: 'Certificate Monitor',
        service: 'API Gateway',
        timestamp: '1 hour ago',
        tags: ['security', 'certificate', 'compliance'],
        affectedResources: ['api.company.com'],
        notifications: [
          {
            channel: 'email',
            sentAt: '1 hour ago',
            recipient: 'security@company.com',
            status: 'sent'
          }
        ],
        actions: []
      },
      {
        id: 'ALT-004',
        title: 'Disk Space Warning',
        description: 'Disk usage on log server exceeded 80%',
        severity: 'medium',
        status: 'resolved',
        source: 'Infrastructure Monitor',
        service: 'Log Server',
        metric: 'DiskUsagePercent',
        threshold: {
          condition: '> 80%',
          value: 80,
          actualValue: 82
        },
        timestamp: '2 hours ago',
        acknowledgedAt: '1 hour 45 minutes ago',
        resolvedAt: '30 minutes ago',
        assignee: 'ops-team@company.com',
        tags: ['infrastructure', 'storage'],
        affectedResources: ['log-server-01'],
        notifications: [
          {
            channel: 'slack',
            sentAt: '2 hours ago',
            recipient: '#ops-team',
            status: 'sent'
          }
        ],
        actions: [
          {
            type: 'Log rotation executed',
            status: 'completed',
            timestamp: '35 minutes ago'
          }
        ]
      },
      {
        id: 'ALT-005',
        title: 'Unusual Login Activity Detected',
        description: 'Multiple failed login attempts from unknown IP',
        severity: 'high',
        status: 'active',
        source: 'Security Monitor',
        service: 'Auth Service',
        timestamp: '10 minutes ago',
        tags: ['security', 'authentication', 'threat'],
        affectedResources: ['auth-service'],
        notifications: [
          {
            channel: 'sms',
            sentAt: '9 minutes ago',
            recipient: '+1234567890',
            status: 'sent'
          },
          {
            channel: 'email',
            sentAt: '9 minutes ago',
            recipient: 'security@company.com',
            status: 'sent'
          }
        ],
        actions: [
          {
            type: 'IP blocked',
            status: 'completed',
            timestamp: '8 minutes ago'
          }
        ]
      },
      {
        id: 'ALT-006',
        title: 'API Rate Limit Approaching',
        description: 'API rate limit usage at 85% of threshold',
        severity: 'low',
        status: 'suppressed',
        source: 'API Gateway',
        service: 'Public API',
        metric: 'RateLimitUsage',
        threshold: {
          condition: '> 80%',
          value: 80,
          actualValue: 85
        },
        timestamp: '20 minutes ago',
        tags: ['api', 'rate-limit', 'performance'],
        affectedResources: ['api-gateway'],
        notifications: [],
        actions: []
      }
    ]);

    setAlertRules([
      {
        id: 'RULE-001',
        name: 'High CPU Alert',
        description: 'Alert when CPU usage exceeds threshold',
        enabled: true,
        condition: 'cpu_usage > 90 for 5 minutes',
        severity: 'critical',
        frequency: '1 minute',
        notification: {
          channels: ['slack', 'email'],
          recipients: ['oncall@company.com'],
          delay: 0
        },
        lastTriggered: '5 minutes ago',
        triggerCount: 45
      },
      {
        id: 'RULE-002',
        name: 'Database Connection Alert',
        description: 'Alert on low database connections',
        enabled: true,
        condition: 'db_connections < 10',
        severity: 'high',
        frequency: '30 seconds',
        notification: {
          channels: ['slack'],
          recipients: ['#database-team'],
          delay: 60
        },
        lastTriggered: '15 minutes ago',
        triggerCount: 12
      },
      {
        id: 'RULE-003',
        name: 'Certificate Expiry',
        description: 'Alert before SSL certificates expire',
        enabled: true,
        condition: 'days_until_expiry < 30',
        severity: 'medium',
        frequency: 'daily',
        notification: {
          channels: ['email'],
          recipients: ['security@company.com'],
          delay: 0
        },
        lastTriggered: '1 hour ago',
        triggerCount: 3
      },
      {
        id: 'RULE-004',
        name: 'Security Threat Detection',
        description: 'Alert on suspicious security activity',
        enabled: true,
        condition: 'failed_logins > 5 in 1 minute',
        severity: 'high',
        frequency: 'real-time',
        notification: {
          channels: ['sms', 'email', 'slack'],
          recipients: ['security@company.com'],
          delay: 0
        },
        lastTriggered: '10 minutes ago',
        triggerCount: 8
      }
    ]);

    setNotificationChannels([
      {
        id: 'CH-001',
        name: 'Slack - Critical Alerts',
        type: 'slack',
        enabled: true,
        config: {
          webhook: 'https://hooks.slack.com/...',
          channel: '#alerts-critical',
          username: 'Alert Bot'
        },
        testStatus: 'success',
        lastUsed: '4 minutes ago'
      },
      {
        id: 'CH-002',
        name: 'Email - On-Call Team',
        type: 'email',
        enabled: true,
        config: {
          smtp: 'smtp.company.com',
          from: 'alerts@company.com',
          to: ['oncall@company.com']
        },
        testStatus: 'success',
        lastUsed: '4 minutes ago'
      },
      {
        id: 'CH-003',
        name: 'SMS - Emergency',
        type: 'sms',
        enabled: true,
        config: {
          provider: 'Twilio',
          from: '+1234567890',
          to: ['+0987654321']
        },
        testStatus: 'success',
        lastUsed: '9 minutes ago'
      },
      {
        id: 'CH-004',
        name: 'PagerDuty Integration',
        type: 'pagerduty',
        enabled: false,
        config: {
          apiKey: '***',
          serviceKey: 'service-123'
        },
        testStatus: 'failed'
      },
      {
        id: 'CH-005',
        name: 'Webhook - Custom',
        type: 'webhook',
        enabled: true,
        config: {
          url: 'https://api.company.com/alerts',
          method: 'POST',
          headers: { 'Authorization': 'Bearer ***' }
        },
        testStatus: 'pending'
      }
    ]);
  }, []);

  const getSeverityColor = (severity: string) => {
    switch(severity) {
      case 'critical': return 'text-red-600 bg-red-900/20';
      case 'high': return 'text-orange-500 bg-orange-900/20';
      case 'medium': return 'text-yellow-500 bg-yellow-900/20';
      case 'low': return 'text-blue-500 bg-blue-900/20';
      case 'info': return 'text-gray-500 bg-gray-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': return 'text-red-500 bg-red-900/20';
      case 'acknowledged': return 'text-yellow-500 bg-yellow-900/20';
      case 'resolved': return 'text-green-500 bg-green-900/20';
      case 'suppressed': return 'text-gray-500 bg-gray-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch(severity) {
      case 'critical': return <XCircle className="w-4 h-4" />;
      case 'high': return <AlertTriangle className="w-4 h-4" />;
      case 'medium': return <AlertCircle className="w-4 h-4" />;
      case 'low': return <Bell className="w-4 h-4" />;
      case 'info': return <Bell className="w-4 h-4" />;
      default: return <Bell className="w-4 h-4" />;
    }
  };

  const getChannelIcon = (type: string) => {
    switch(type) {
      case 'email': return <Mail className="w-4 h-4" />;
      case 'slack': return <MessageSquare className="w-4 h-4" />;
      case 'sms': return <Phone className="w-4 h-4" />;
      case 'webhook': return <Zap className="w-4 h-4" />;
      case 'teams': return <Users className="w-4 h-4" />;
      case 'pagerduty': return <Bell className="w-4 h-4" />;
      default: return <Bell className="w-4 h-4" />;
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    if (selectedSeverity !== 'all' && alert.severity !== selectedSeverity) return false;
    if (selectedStatus !== 'all' && alert.status !== selectedStatus) return false;
    if (searchQuery && !alert.title.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const stats = {
    total: alerts.length,
    active: alerts.filter(a => a.status === 'active').length,
    acknowledged: alerts.filter(a => a.status === 'acknowledged').length,
    resolved: alerts.filter(a => a.status === 'resolved').length,
    critical: alerts.filter(a => a.severity === 'critical').length,
    enabledRules: alertRules.filter(r => r.enabled).length
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Alert Manager</h1>
            <p className="text-sm text-gray-400 mt-1">Monitor and manage system alerts</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setViewMode(
                viewMode === 'alerts' ? 'rules' : 
                viewMode === 'rules' ? 'channels' : 'alerts'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2"
            >
              {viewMode === 'alerts' ? <Settings className="w-4 h-4" /> : 
               viewMode === 'rules' ? <Bell className="w-4 h-4" /> :
               <AlertTriangle className="w-4 h-4" />}
              <span>
                {viewMode === 'alerts' ? 'Rules' : 
                 viewMode === 'rules' ? 'Channels' : 'Alerts'}
              </span>
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Bell className="w-4 h-4" />
              <span>New Alert Rule</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <Bell className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Total Alerts</p>
              <p className="text-xl font-bold">{stats.total}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Active</p>
              <p className="text-xl font-bold text-red-500">{stats.active}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Clock className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Acknowledged</p>
              <p className="text-xl font-bold text-yellow-500">{stats.acknowledged}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Resolved</p>
              <p className="text-xl font-bold">{stats.resolved}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XCircle className="w-5 h-5 text-red-600" />
            <div>
              <p className="text-xs text-gray-400">Critical</p>
              <p className="text-xl font-bold text-red-600">{stats.critical}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Settings className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Active Rules</p>
              <p className="text-xl font-bold">{stats.enabledRules}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {viewMode === 'alerts' && (
          <>
            {/* Search and Filters */}
            <div className="flex items-center space-x-3 mb-6">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search alerts..."
                  className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                />
              </div>
              
              <select
                value={selectedSeverity}
                onChange={(e) => setSelectedSeverity(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Severities</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
                <option value="info">Info</option>
              </select>
              
              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="acknowledged">Acknowledged</option>
                <option value="resolved">Resolved</option>
                <option value="suppressed">Suppressed</option>
              </select>
            </div>

            {/* Alerts List */}
            <div className="space-y-3">
              {filteredAlerts.map(alert => (
                <div key={alert.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start space-x-3">
                      <div className={`p-2 rounded ${getSeverityColor(alert.severity)}`}>
                        {getSeverityIcon(alert.severity)}
                      </div>
                      <div className="flex-1">
                        <h3 className="text-sm font-bold mb-1">{alert.title}</h3>
                        <p className="text-xs text-gray-400 mb-2">{alert.description}</p>
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <span>{alert.timestamp}</span>
                          <span>Source: {alert.source}</span>
                          <span>Service: {alert.service}</span>
                          {alert.metric && <span>Metric: {alert.metric}</span>}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 text-xs rounded ${getStatusColor(alert.status)}`}>
                        {alert.status.toUpperCase()}
                      </span>
                      <button className="p-1 hover:bg-gray-800 rounded">
                        <MoreVertical className="w-4 h-4 text-gray-500" />
                      </button>
                    </div>
                  </div>
                  
                  {alert.threshold && (
                    <div className="mb-3 p-2 bg-gray-800 rounded text-xs">
                      <span className="text-gray-400">Threshold: </span>
                      <span>{alert.threshold.condition}</span>
                      <span className="text-gray-400 mx-2">|</span>
                      <span className="text-gray-400">Actual: </span>
                      <span className="text-red-500">{alert.threshold.actualValue}</span>
                    </div>
                  )}
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {alert.tags.map(tag => (
                        <span key={tag} className="px-2 py-1 bg-gray-800 rounded text-xs">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <div className="flex items-center space-x-2">
                      {alert.status === 'active' && (
                        <button className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-xs">
                          Acknowledge
                        </button>
                      )}
                      {(alert.status === 'active' || alert.status === 'acknowledged') && (
                        <button className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs">
                          Resolve
                        </button>
                      )}
                      <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                        Details
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'rules' && (
          <div className="grid grid-cols-2 gap-4">
            {alertRules.map(rule => (
              <div key={rule.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="text-sm font-bold mb-1">{rule.name}</h3>
                    <p className="text-xs text-gray-400">{rule.description}</p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 text-xs rounded ${getSeverityColor(rule.severity)}`}>
                      {rule.severity.toUpperCase()}
                    </span>
                    <button className={`p-1 rounded ${
                      rule.enabled ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-600 hover:bg-gray-700'
                    }`}>
                      {rule.enabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
                
                <div className="space-y-2 text-xs mb-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Condition</span>
                    <span className="font-mono">{rule.condition}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Frequency</span>
                    <span>{rule.frequency}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Channels</span>
                    <span>{rule.notification.channels.join(', ')}</span>
                  </div>
                  {rule.lastTriggered && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Last Triggered</span>
                      <span>{rule.lastTriggered}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-gray-400">Trigger Count</span>
                    <span>{rule.triggerCount}</span>
                  </div>
                </div>
                
                <button className="w-full px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                  Edit Rule
                </button>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'channels' && (
          <div className="grid grid-cols-3 gap-4">
            {notificationChannels.map(channel => (
              <div key={channel.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    {getChannelIcon(channel.type)}
                    <h3 className="text-sm font-bold">{channel.name}</h3>
                  </div>
                  <button className={`p-1 rounded ${
                    channel.enabled ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-600 hover:bg-gray-700'
                  }`}>
                    {channel.enabled ? <Bell className="w-4 h-4" /> : <BellOff className="w-4 h-4" />}
                  </button>
                </div>
                
                <div className="space-y-2 text-xs mb-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Type</span>
                    <span>{channel.type}</span>
                  </div>
                  {channel.testStatus && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Test Status</span>
                      <span className={`${
                        channel.testStatus === 'success' ? 'text-green-500' :
                        channel.testStatus === 'failed' ? 'text-red-500' :
                        'text-yellow-500'
                      }`}>
                        {channel.testStatus.toUpperCase()}
                      </span>
                    </div>
                  )}
                  {channel.lastUsed && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Last Used</span>
                      <span>{channel.lastUsed}</span>
                    </div>
                  )}
                </div>
                
                <div className="flex space-x-2">
                  <button className="flex-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                    Test
                  </button>
                  <button className="flex-1 px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    Configure
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}