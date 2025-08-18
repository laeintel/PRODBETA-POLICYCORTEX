'use client';

import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, AlertCircle, CheckCircle, XCircle, Clock, User,
  MessageSquare, FileText, TrendingUp, Shield, Zap, Bell, Phone,
  Mail, Slack, Users, Calendar, Timer, Activity, BarChart, Filter,
  Search, Plus, Edit, Trash2, ExternalLink, ChevronRight, Info
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface Incident {
  id: string;
  title: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'open' | 'investigating' | 'identified' | 'monitoring' | 'resolved' | 'closed';
  type: 'security' | 'performance' | 'availability' | 'data' | 'compliance' | 'other';
  affectedServices: string[];
  impact: string;
  rootCause?: string;
  createdAt: string;
  updatedAt: string;
  resolvedAt?: string;
  assignee: string;
  team: string;
  timeline: TimelineEvent[];
  metrics: {
    timeToDetect: number;
    timeToRespond: number;
    timeToResolve?: number;
    affectedUsers: number;
    downtime: number;
  };
}

interface TimelineEvent {
  id: string;
  timestamp: string;
  type: 'created' | 'updated' | 'escalated' | 'comment' | 'resolved' | 'closed';
  user: string;
  message: string;
}

export default function IncidentResponse() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showNewIncident, setShowNewIncident] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    // Initialize with mock incidents
    setIncidents([
      {
        id: 'INC-001',
        title: 'Database Connection Pool Exhausted',
        description: 'Primary database experiencing connection pool exhaustion leading to service degradation',
        severity: 'critical',
        status: 'investigating',
        type: 'performance',
        affectedServices: ['API Gateway', 'User Service', 'Order Service'],
        impact: 'Users unable to complete transactions, 30% of requests failing',
        createdAt: '10:45 AM',
        updatedAt: '11:15 AM',
        assignee: 'John Smith',
        team: 'Database Team',
        timeline: [
          { id: '1', timestamp: '10:45 AM', type: 'created', user: 'System', message: 'Incident automatically created by monitoring' },
          { id: '2', timestamp: '10:47 AM', type: 'updated', user: 'John Smith', message: 'Acknowledged and investigating' },
          { id: '3', timestamp: '10:55 AM', type: 'comment', user: 'John Smith', message: 'Identified connection leak in order service' },
          { id: '4', timestamp: '11:05 AM', type: 'escalated', user: 'System', message: 'Escalated to P1 due to user impact' }
        ],
        metrics: {
          timeToDetect: 2,
          timeToRespond: 5,
          affectedUsers: 1250,
          downtime: 30
        }
      },
      {
        id: 'INC-002',
        title: 'Suspicious Login Attempts Detected',
        description: 'Multiple failed login attempts from unknown IP addresses',
        severity: 'high',
        status: 'identified',
        type: 'security',
        affectedServices: ['Authentication Service', 'User Portal'],
        impact: 'Potential security breach attempt, no confirmed compromise',
        rootCause: 'Brute force attack from botnet',
        createdAt: '9:30 AM',
        updatedAt: '10:00 AM',
        assignee: 'Sarah Johnson',
        team: 'Security Team',
        timeline: [
          { id: '1', timestamp: '9:30 AM', type: 'created', user: 'Security Monitor', message: 'Anomaly detected in login patterns' },
          { id: '2', timestamp: '9:32 AM', type: 'updated', user: 'Sarah Johnson', message: 'Investigating source IPs' },
          { id: '3', timestamp: '9:45 AM', type: 'comment', user: 'Sarah Johnson', message: 'Blocked suspicious IP ranges' },
          { id: '4', timestamp: '10:00 AM', type: 'updated', user: 'Sarah Johnson', message: 'Root cause identified as botnet attack' }
        ],
        metrics: {
          timeToDetect: 1,
          timeToRespond: 3,
          affectedUsers: 0,
          downtime: 0
        }
      },
      {
        id: 'INC-003',
        title: 'Storage Service Latency Spike',
        description: 'Increased latency in object storage service affecting file uploads',
        severity: 'medium',
        status: 'monitoring',
        type: 'performance',
        affectedServices: ['Storage Service', 'Media Service'],
        impact: 'File uploads taking 3x longer than normal',
        rootCause: 'Disk I/O bottleneck on storage nodes',
        createdAt: '8:15 AM',
        updatedAt: '9:45 AM',
        assignee: 'Mike Chen',
        team: 'Infrastructure Team',
        timeline: [
          { id: '1', timestamp: '8:15 AM', type: 'created', user: 'System', message: 'Latency threshold exceeded' },
          { id: '2', timestamp: '8:20 AM', type: 'updated', user: 'Mike Chen', message: 'Investigating storage cluster' },
          { id: '3', timestamp: '8:45 AM', type: 'comment', user: 'Mike Chen', message: 'Applied temporary fix, monitoring' }
        ],
        metrics: {
          timeToDetect: 5,
          timeToRespond: 8,
          timeToResolve: 45,
          affectedUsers: 450,
          downtime: 0
        }
      },
      {
        id: 'INC-004',
        title: 'GDPR Compliance Alert',
        description: 'User data retention policy violation detected',
        severity: 'high',
        status: 'resolved',
        type: 'compliance',
        affectedServices: ['Data Warehouse', 'Analytics Service'],
        impact: 'Non-compliance with data retention requirements',
        rootCause: 'Automated cleanup job failure',
        createdAt: 'Yesterday 4:30 PM',
        updatedAt: 'Yesterday 6:00 PM',
        resolvedAt: 'Yesterday 6:00 PM',
        assignee: 'Lisa Wong',
        team: 'Compliance Team',
        timeline: [
          { id: '1', timestamp: 'Yesterday 4:30 PM', type: 'created', user: 'Compliance Scanner', message: 'Data retention violation detected' },
          { id: '2', timestamp: 'Yesterday 4:35 PM', type: 'updated', user: 'Lisa Wong', message: 'Reviewing affected data' },
          { id: '3', timestamp: 'Yesterday 5:15 PM', type: 'comment', user: 'Lisa Wong', message: 'Cleanup job restarted' },
          { id: '4', timestamp: 'Yesterday 6:00 PM', type: 'resolved', user: 'Lisa Wong', message: 'Data purged, compliance restored' }
        ],
        metrics: {
          timeToDetect: 10,
          timeToRespond: 12,
          timeToResolve: 90,
          affectedUsers: 0,
          downtime: 0
        }
      },
      {
        id: 'INC-005',
        title: 'API Rate Limiting Triggered',
        description: 'Excessive API calls causing rate limiting for multiple clients',
        severity: 'low',
        status: 'closed',
        type: 'performance',
        affectedServices: ['API Gateway'],
        impact: 'Some API clients experiencing throttling',
        rootCause: 'Client misconfiguration causing request loops',
        createdAt: 'Yesterday 2:00 PM',
        updatedAt: 'Yesterday 3:30 PM',
        resolvedAt: 'Yesterday 3:30 PM',
        assignee: 'Tom Davis',
        team: 'API Team',
        timeline: [
          { id: '1', timestamp: 'Yesterday 2:00 PM', type: 'created', user: 'System', message: 'Rate limiting threshold exceeded' },
          { id: '2', timestamp: 'Yesterday 2:15 PM', type: 'updated', user: 'Tom Davis', message: 'Identified problematic clients' },
          { id: '3', timestamp: 'Yesterday 3:00 PM', type: 'comment', user: 'Tom Davis', message: 'Contacted client teams' },
          { id: '4', timestamp: 'Yesterday 3:30 PM', type: 'closed', user: 'Tom Davis', message: 'Issue resolved, clients fixed configuration' }
        ],
        metrics: {
          timeToDetect: 3,
          timeToRespond: 15,
          timeToResolve: 90,
          affectedUsers: 50,
          downtime: 0
        }
      }
    ]);
  }, []);

  const refreshData = () => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1000);
  };

  const filteredIncidents = incidents.filter(incident => {
    if (filterStatus !== 'all' && incident.status !== filterStatus) return false;
    if (filterSeverity !== 'all' && incident.severity !== filterSeverity) return false;
    if (searchQuery && !incident.title.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !incident.description.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const stats = {
    open: incidents.filter(i => ['open', 'investigating', 'identified', 'monitoring'].includes(i.status)).length,
    resolved: incidents.filter(i => i.status === 'resolved').length,
    closed: incidents.filter(i => i.status === 'closed').length,
    critical: incidents.filter(i => i.severity === 'critical').length,
    avgTimeToResolve: Math.round(
      incidents.filter(i => i.metrics.timeToResolve)
        .reduce((sum, i) => sum + (i.metrics.timeToResolve || 0), 0) / 
      incidents.filter(i => i.metrics.timeToResolve).length || 0
    )
  };

  const getSeverityColor = (severity: string) => {
    switch(severity) {
      case 'critical': return 'text-red-500 bg-red-900/20 border-red-900/30';
      case 'high': return 'text-orange-500 bg-orange-900/20 border-orange-900/30';
      case 'medium': return 'text-yellow-500 bg-yellow-900/20 border-yellow-900/30';
      case 'low': return 'text-blue-500 bg-blue-900/20 border-blue-900/30';
      default: return 'text-gray-500 bg-gray-900/20 border-gray-900/30';
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'open': return 'text-red-500';
      case 'investigating': return 'text-yellow-500';
      case 'identified': return 'text-orange-500';
      case 'monitoring': return 'text-blue-500';
      case 'resolved': return 'text-green-500';
      case 'closed': return 'text-gray-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'open': return <AlertCircle className="w-4 h-4" />;
      case 'investigating': return <Clock className="w-4 h-4" />;
      case 'identified': return <Info className="w-4 h-4" />;
      case 'monitoring': return <Activity className="w-4 h-4" />;
      case 'resolved': return <CheckCircle className="w-4 h-4" />;
      case 'closed': return <XCircle className="w-4 h-4" />;
      default: return null;
    }
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Incident Response Center</h1>
            <p className="text-sm text-gray-400 mt-1">Manage and track all system incidents</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={refreshData}
              className={`p-2 hover:bg-gray-800 rounded ${refreshing ? 'animate-spin' : ''}`}
            >
              <Activity className="w-4 h-4" />
            </button>
            
            <button
              onClick={() => setShowNewIncident(true)}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm flex items-center space-x-2"
            >
              <Plus className="w-4 h-4" />
              <span>New Incident</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Open Incidents</p>
              <p className="text-xl font-bold">{stats.open}</p>
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
            <XCircle className="w-5 h-5 text-gray-500" />
            <div>
              <p className="text-xs text-gray-400">Closed</p>
              <p className="text-xl font-bold">{stats.closed}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Critical</p>
              <p className="text-xl font-bold text-red-500">{stats.critical}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Timer className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Avg Resolution</p>
              <p className="text-xl font-bold">{stats.avgTimeToResolve} min</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Users className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">On-Call Team</p>
              <p className="text-xl font-bold">5 Active</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Incidents List */}
        <div className="flex-1 p-6">
          {/* Filters */}
          <div className="flex items-center space-x-3 mb-6">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search incidents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              />
            </div>
            
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
            >
              <option value="all">All Status</option>
              <option value="open">Open</option>
              <option value="investigating">Investigating</option>
              <option value="identified">Identified</option>
              <option value="monitoring">Monitoring</option>
              <option value="resolved">Resolved</option>
              <option value="closed">Closed</option>
            </select>
            
            <select
              value={filterSeverity}
              onChange={(e) => setFilterSeverity(e.target.value)}
              className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
            >
              <option value="all">All Severities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
            
            <button className="p-2 hover:bg-gray-800 rounded">
              <Filter className="w-4 h-4" />
            </button>
          </div>

          {/* Incidents Grid */}
          <div className="space-y-4">
            {filteredIncidents.map(incident => (
              <div
                key={incident.id}
                className={`bg-gray-900 border rounded-lg p-4 hover:bg-gray-850 cursor-pointer ${
                  selectedIncident?.id === incident.id ? 'ring-2 ring-blue-500' : ''
                } ${getSeverityColor(incident.severity).split(' ').slice(1).join(' ')}`}
                onClick={() => setSelectedIncident(incident)}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <span className={`px-2 py-1 text-xs rounded font-medium ${getSeverityColor(incident.severity)}`}>
                        {incident.severity.toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-500">{incident.id}</span>
                      <span className={`flex items-center space-x-1 text-xs ${getStatusColor(incident.status)}`}>
                        {getStatusIcon(incident.status)}
                        <span>{incident.status.toUpperCase()}</span>
                      </span>
                    </div>
                    <h3 className="text-sm font-bold mb-1">{incident.title}</h3>
                    <p className="text-xs text-gray-400 mb-2">{incident.description}</p>
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span className="flex items-center space-x-1">
                        <User className="w-3 h-3" />
                        <span>{incident.assignee}</span>
                      </span>
                      <span className="flex items-center space-x-1">
                        <Users className="w-3 h-3" />
                        <span>{incident.team}</span>
                      </span>
                      <span className="flex items-center space-x-1">
                        <Clock className="w-3 h-3" />
                        <span>{incident.createdAt}</span>
                      </span>
                    </div>
                  </div>
                  <ChevronRight className="w-5 h-5 text-gray-500" />
                </div>
                
                <div className="flex items-center justify-between pt-3 border-t border-gray-800">
                  <div className="flex flex-wrap gap-1">
                    {incident.affectedServices.map(service => (
                      <span key={service} className="px-2 py-0.5 bg-gray-800 rounded text-xs">
                        {service}
                      </span>
                    ))}
                  </div>
                  <div className="flex items-center space-x-3 text-xs">
                    <span className="text-gray-400">Impact:</span>
                    <span className="flex items-center space-x-1">
                      <Users className="w-3 h-3" />
                      <span>{incident.metrics.affectedUsers} users</span>
                    </span>
                    {incident.metrics.downtime > 0 && (
                      <span className="flex items-center space-x-1 text-red-500">
                        <Clock className="w-3 h-3" />
                        <span>{incident.metrics.downtime}m downtime</span>
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Incident Details Panel */}
        {selectedIncident && (
          <div className="w-96 bg-gray-900 border-l border-gray-800 p-6 overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold">Incident Details</h2>
              <button
                onClick={() => setSelectedIncident(null)}
                className="p-1 hover:bg-gray-800 rounded"
              >
                <XCircle className="w-4 h-4" />
              </button>
            </div>
            
            <div className="space-y-4">
              {/* Incident Info */}
              <div>
                <p className="text-xs text-gray-400 mb-1">Incident ID</p>
                <p className="font-mono text-sm">{selectedIncident.id}</p>
              </div>
              
              <div>
                <p className="text-xs text-gray-400 mb-1">Impact</p>
                <p className="text-sm">{selectedIncident.impact}</p>
              </div>
              
              {selectedIncident.rootCause && (
                <div>
                  <p className="text-xs text-gray-400 mb-1">Root Cause</p>
                  <p className="text-sm">{selectedIncident.rootCause}</p>
                </div>
              )}
              
              {/* Metrics */}
              <div className="bg-gray-800 rounded p-3">
                <p className="text-xs text-gray-400 mb-2">Response Metrics</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <p className="text-gray-400">Time to Detect</p>
                    <p className="font-bold">{selectedIncident.metrics.timeToDetect} min</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Time to Respond</p>
                    <p className="font-bold">{selectedIncident.metrics.timeToRespond} min</p>
                  </div>
                  {selectedIncident.metrics.timeToResolve && (
                    <div>
                      <p className="text-gray-400">Time to Resolve</p>
                      <p className="font-bold">{selectedIncident.metrics.timeToResolve} min</p>
                    </div>
                  )}
                  <div>
                    <p className="text-gray-400">Affected Users</p>
                    <p className="font-bold">{selectedIncident.metrics.affectedUsers}</p>
                  </div>
                </div>
              </div>
              
              {/* Timeline */}
              <div>
                <p className="text-xs text-gray-400 mb-2">Timeline</p>
                <div className="space-y-2">
                  {selectedIncident.timeline.map(event => (
                    <div key={event.id} className="flex items-start space-x-2 text-xs">
                      <span className="text-gray-500 w-16 flex-shrink-0">{event.timestamp}</span>
                      <div className="flex-1">
                        <p className="font-medium">{event.user}</p>
                        <p className="text-gray-400">{event.message}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Actions */}
              <div className="flex space-x-2 pt-4 border-t border-gray-800">
                <button className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                  Update Status
                </button>
                <button className="flex-1 px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                  Add Comment
                </button>
                <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                  <ExternalLink className="w-3 h-3" />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}