'use client';

import React, { useState, useEffect } from 'react';
import { 
  Activity, Play, Pause, Search, Filter, Calendar, Clock, AlertCircle,
  CheckCircle, XCircle, Info, Zap, Database, Server, Cloud, Shield,
  User, Users, Globe, Package, ChevronRight, ChevronDown, Eye,
  Bookmark, Download, Settings, RefreshCw, Bell, BellOff, Hash,
  ArrowUp, ArrowDown, ExternalLink, Copy, MoreVertical, Tag
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface EventStream {
  id: string;
  name: string;
  description: string;
  source: string;
  category: 'system' | 'application' | 'security' | 'business' | 'infrastructure' | 'user';
  status: 'active' | 'paused' | 'error' | 'maintenance';
  eventCount: number;
  eventsPerSecond: number;
  lastEvent: string;
  retentionPeriod: string;
  consumers: number;
  partition: number;
  offset: number;
  lag: number;
}

interface StreamEvent {
  id: string;
  streamId: string;
  timestamp: string;
  eventType: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  source: string;
  target?: string;
  user?: string;
  sessionId?: string;
  correlationId?: string;
  payload: {
    action: string;
    resource: string;
    details: { [key: string]: any };
    metadata?: { [key: string]: any };
  };
  tags: string[];
  processed: boolean;
  acknowledged: boolean;
}

interface EventPattern {
  id: string;
  name: string;
  description: string;
  pattern: string;
  matches: number;
  lastMatch: string;
  severity: 'info' | 'warning' | 'critical';
  enabled: boolean;
  actions: {
    type: 'alert' | 'webhook' | 'email' | 'slack';
    configuration: any;
  }[];
}

interface EventFilter {
  streams: string[];
  eventTypes: string[];
  severities: string[];
  timeRange: string;
  searchQuery: string;
  correlationId?: string;
  userId?: string;
}

export default function EventStream() {
  const [streams, setStreams] = useState<EventStream[]>([]);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [patterns, setPatterns] = useState<EventPattern[]>([]);
  const [isStreaming, setIsStreaming] = useState(true);
  const [selectedStream, setSelectedStream] = useState<string>('all');
  const [filters, setFilters] = useState<EventFilter>({
    streams: [],
    eventTypes: [],
    severities: [],
    timeRange: '1h',
    searchQuery: ''
  });
  const [expandedEvents, setExpandedEvents] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<'events' | 'streams' | 'patterns'>('events');
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    // Initialize with mock stream data
    setStreams([
      {
        id: 'STREAM-001',
        name: 'Application Events',
        description: 'Core application lifecycle and business events',
        source: 'app-cluster',
        category: 'application',
        status: 'active',
        eventCount: 125847,
        eventsPerSecond: 45.2,
        lastEvent: '2 seconds ago',
        retentionPeriod: '7 days',
        consumers: 3,
        partition: 12,
        offset: 125847,
        lag: 0
      },
      {
        id: 'STREAM-002',
        name: 'Security Audit Log',
        description: 'Authentication, authorization, and security events',
        source: 'security-gateway',
        category: 'security',
        status: 'active',
        eventCount: 89234,
        eventsPerSecond: 12.8,
        lastEvent: '5 seconds ago',
        retentionPeriod: '90 days',
        consumers: 2,
        partition: 8,
        offset: 89234,
        lag: 3
      },
      {
        id: 'STREAM-003',
        name: 'Infrastructure Metrics',
        description: 'System health, performance, and infrastructure events',
        source: 'monitoring-agent',
        category: 'infrastructure',
        status: 'active',
        eventCount: 234567,
        eventsPerSecond: 78.9,
        lastEvent: '1 second ago',
        retentionPeriod: '30 days',
        consumers: 5,
        partition: 16,
        offset: 234567,
        lag: 12
      },
      {
        id: 'STREAM-004',
        name: 'User Activity',
        description: 'User interactions, sessions, and behavior events',
        source: 'frontend-tracking',
        category: 'user',
        status: 'active',
        eventCount: 456789,
        eventsPerSecond: 156.3,
        lastEvent: '1 second ago',
        retentionPeriod: '14 days',
        consumers: 4,
        partition: 20,
        offset: 456789,
        lag: 8
      },
      {
        id: 'STREAM-005',
        name: 'Business Intelligence',
        description: 'Business metrics, KPIs, and analytics events',
        source: 'analytics-pipeline',
        category: 'business',
        status: 'paused',
        eventCount: 67834,
        eventsPerSecond: 0,
        lastEvent: '45 minutes ago',
        retentionPeriod: '365 days',
        consumers: 1,
        partition: 4,
        offset: 67834,
        lag: 150
      }
    ]);

    setEvents([
      {
        id: 'EVT-001',
        streamId: 'STREAM-001',
        timestamp: new Date().toISOString(),
        eventType: 'user.login.success',
        severity: 'low',
        source: 'auth-service',
        user: 'john.doe@company.com',
        sessionId: 'sess_abc123',
        correlationId: 'corr_xyz789',
        payload: {
          action: 'user_login',
          resource: 'authentication_system',
          details: {
            userId: 'user_12345',
            loginMethod: 'sso',
            ipAddress: '192.168.1.100',
            userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            mfaEnabled: true
          },
          metadata: {
            geolocation: 'New York, US',
            deviceType: 'desktop',
            riskScore: 0.2
          }
        },
        tags: ['authentication', 'success', 'sso'],
        processed: true,
        acknowledged: false
      },
      {
        id: 'EVT-002',
        streamId: 'STREAM-002',
        timestamp: new Date(Date.now() - 30000).toISOString(),
        eventType: 'security.alert.high',
        severity: 'high',
        source: 'security-monitor',
        correlationId: 'corr_sec456',
        payload: {
          action: 'anomaly_detection',
          resource: 'network_traffic',
          details: {
            alertType: 'unusual_data_transfer',
            sourceIp: '10.0.0.15',
            targetIp: '203.0.113.42',
            transferVolume: '2.5GB',
            threshold: '1GB',
            confidence: 0.89
          },
          metadata: {
            detectionModel: 'ml_anomaly_v2.1',
            falsePositiveRate: 0.05,
            investigationRequired: true
          }
        },
        tags: ['security', 'anomaly', 'data-exfiltration'],
        processed: false,
        acknowledged: false
      },
      {
        id: 'EVT-003',
        streamId: 'STREAM-003',
        timestamp: new Date(Date.now() - 60000).toISOString(),
        eventType: 'system.resource.warning',
        severity: 'medium',
        source: 'infrastructure-monitor',
        payload: {
          action: 'resource_threshold_exceeded',
          resource: 'compute_cluster_01',
          details: {
            resourceType: 'cpu',
            currentUsage: 85.7,
            threshold: 80,
            duration: '15 minutes',
            affectedServices: ['api-gateway', 'user-service']
          }
        },
        tags: ['infrastructure', 'performance', 'cpu'],
        processed: true,
        acknowledged: true
      },
      {
        id: 'EVT-004',
        streamId: 'STREAM-004',
        timestamp: new Date(Date.now() - 90000).toISOString(),
        eventType: 'user.action.purchase',
        severity: 'low',
        source: 'e-commerce-api',
        user: 'customer123@email.com',
        sessionId: 'sess_def456',
        payload: {
          action: 'purchase_completed',
          resource: 'shopping_cart',
          details: {
            orderId: 'ORD-789456',
            totalAmount: 299.99,
            currency: 'USD',
            itemCount: 3,
            paymentMethod: 'credit_card',
            shippingMethod: 'express'
          },
          metadata: {
            customerId: 'CUST-54321',
            campaignId: 'SUMMER2024',
            discount: 29.99
          }
        },
        tags: ['ecommerce', 'purchase', 'revenue'],
        processed: true,
        acknowledged: false
      },
      {
        id: 'EVT-005',
        streamId: 'STREAM-001',
        timestamp: new Date(Date.now() - 120000).toISOString(),
        eventType: 'system.error.database',
        severity: 'critical',
        source: 'database-cluster',
        correlationId: 'corr_db789',
        payload: {
          action: 'connection_failure',
          resource: 'primary_database',
          details: {
            errorCode: 'CONN_TIMEOUT',
            errorMessage: 'Connection to primary database timed out after 30 seconds',
            affectedQueries: 15,
            fallbackActivated: true,
            estimatedDowntime: '2 minutes'
          },
          metadata: {
            dbHost: 'db-primary-01',
            connectionPool: 'main_pool',
            recoveryAction: 'failover_initiated'
          }
        },
        tags: ['database', 'error', 'failover'],
        processed: true,
        acknowledged: true
      }
    ]);

    setPatterns([
      {
        id: 'PATTERN-001',
        name: 'Failed Login Attempts',
        description: 'Detect multiple failed login attempts from same IP',
        pattern: 'eventType="user.login.failed" | stats count by sourceIp | where count > 5',
        matches: 23,
        lastMatch: '15 minutes ago',
        severity: 'warning',
        enabled: true,
        actions: [
          {
            type: 'alert',
            configuration: { severity: 'high', channel: 'security' }
          },
          {
            type: 'webhook',
            configuration: { url: 'https://security.company.com/webhook' }
          }
        ]
      },
      {
        id: 'PATTERN-002',
        name: 'High Error Rate',
        description: 'Alert when error rate exceeds 5% in 5-minute window',
        pattern: 'severity="high" OR severity="critical" | stats count by bin(5m) | where count > 50',
        matches: 8,
        lastMatch: '2 hours ago',
        severity: 'critical',
        enabled: true,
        actions: [
          {
            type: 'slack',
            configuration: { channel: '#alerts-critical', webhook: 'slack_webhook_url' }
          }
        ]
      },
      {
        id: 'PATTERN-003',
        name: 'Unusual Data Access',
        description: 'Detect access to sensitive data outside business hours',
        pattern: 'resource contains "sensitive" AND hour < 8 OR hour > 18',
        matches: 3,
        lastMatch: '5 days ago',
        severity: 'critical',
        enabled: true,
        actions: [
          {
            type: 'email',
            configuration: { recipients: ['security@company.com'] }
          }
        ]
      }
    ]);

    // Simulate real-time event streaming
    if (isStreaming) {
      const interval = setInterval(() => {
        const eventTypes = [
          'user.login.success', 'user.logout', 'api.request.completed',
          'system.health.check', 'database.query.slow', 'cache.miss',
          'security.scan.completed', 'backup.job.success'
        ];
        const sources = ['auth-service', 'api-gateway', 'database', 'cache-service', 'monitor'];
        const severities: StreamEvent['severity'][] = ['low', 'medium', 'high', 'critical'];
        
        const newEvent: StreamEvent = {
          id: `EVT-${Date.now()}`,
          streamId: streams[Math.floor(Math.random() * streams.length)]?.id || 'STREAM-001',
          timestamp: new Date().toISOString(),
          eventType: eventTypes[Math.floor(Math.random() * eventTypes.length)],
          severity: severities[Math.floor(Math.random() * severities.length)],
          source: sources[Math.floor(Math.random() * sources.length)],
          payload: {
            action: 'auto_generated_event',
            resource: 'system',
            details: {
              randomValue: Math.random(),
              timestamp: Date.now()
            }
          },
          tags: ['auto-generated', 'demo'],
          processed: Math.random() > 0.3,
          acknowledged: Math.random() > 0.7
        };

        setEvents(prev => [newEvent, ...prev].slice(0, 100));
        
        // Update stream statistics
        setStreams(prev => prev.map(stream => ({
          ...stream,
          eventCount: stream.eventCount + (Math.random() > 0.5 ? 1 : 0),
          eventsPerSecond: Math.max(0, stream.eventsPerSecond + (Math.random() - 0.5) * 5),
          lag: Math.max(0, stream.lag + Math.floor((Math.random() - 0.7) * 3))
        })));
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [isStreaming, streams]);

  const getCategoryIcon = (category: string) => {
    switch(category) {
      case 'system': return <Server className="w-4 h-4 text-blue-500" />;
      case 'application': return <Package className="w-4 h-4 text-green-500" />;
      case 'security': return <Shield className="w-4 h-4 text-red-500" />;
      case 'business': return <Zap className="w-4 h-4 text-purple-500" />;
      case 'infrastructure': return <Cloud className="w-4 h-4 text-orange-500" />;
      case 'user': return <Users className="w-4 h-4 text-cyan-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch(severity) {
      case 'low': return 'text-blue-500 bg-blue-900/20';
      case 'medium': return 'text-yellow-500 bg-yellow-900/20';
      case 'high': return 'text-orange-500 bg-orange-900/20';
      case 'critical': return 'text-red-500 bg-red-900/20';
      case 'info': return 'text-blue-500 bg-blue-900/20';
      case 'warning': return 'text-yellow-500 bg-yellow-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': return 'text-green-500 bg-green-900/20';
      case 'paused': return 'text-yellow-500 bg-yellow-900/20';
      case 'error': return 'text-red-500 bg-red-900/20';
      case 'maintenance': return 'text-orange-500 bg-orange-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const toggleEventExpansion = (eventId: string) => {
    setExpandedEvents(prev => {
      const newSet = new Set(prev);
      if (newSet.has(eventId)) {
        newSet.delete(eventId);
      } else {
        newSet.add(eventId);
      }
      return newSet;
    });
  };

  const filteredEvents = events.filter(event => {
    if (selectedStream !== 'all' && event.streamId !== selectedStream) return false;
    if (filters.searchQuery && !event.eventType.toLowerCase().includes(filters.searchQuery.toLowerCase()) &&
        !event.payload.action.toLowerCase().includes(filters.searchQuery.toLowerCase())) return false;
    return true;
  });

  const totalEvents = streams.reduce((sum, s) => sum + s.eventCount, 0);
  const totalEventsPerSecond = streams.reduce((sum, s) => sum + s.eventsPerSecond, 0);
  const activeStreams = streams.filter(s => s.status === 'active').length;
  const totalConsumers = streams.reduce((sum, s) => sum + s.consumers, 0);
  const unprocessedEvents = events.filter(e => !e.processed).length;
  const criticalEvents = events.filter(e => e.severity === 'critical').length;

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Event Stream</h1>
            <p className="text-sm text-gray-400 mt-1">Real-time event monitoring and stream analytics</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsStreaming(!isStreaming)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                isStreaming ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              {isStreaming ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              <span>{isStreaming ? 'Streaming' : 'Paused'}</span>
            </button>
            
            <button
              onClick={() => setAutoScroll(!autoScroll)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                autoScroll ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              {autoScroll ? <ArrowDown className="w-4 h-4" /> : <ArrowUp className="w-4 h-4" />}
              <span>Auto Scroll</span>
            </button>
            
            <button
              onClick={() => setViewMode(
                viewMode === 'events' ? 'streams' : 
                viewMode === 'streams' ? 'patterns' : 'events'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
            >
              {viewMode === 'events' ? 'Streams' : viewMode === 'streams' ? 'Patterns' : 'Events'}
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <Activity className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Total Events</p>
              <p className="text-xl font-bold">{totalEvents.toLocaleString()}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Zap className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Events/sec</p>
              <p className="text-xl font-bold">{totalEventsPerSecond.toFixed(1)}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Active Streams</p>
              <p className="text-xl font-bold">{activeStreams}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Users className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Consumers</p>
              <p className="text-xl font-bold">{totalConsumers}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Clock className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Unprocessed</p>
              <p className="text-xl font-bold text-yellow-500">{unprocessedEvents}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Critical</p>
              <p className="text-xl font-bold text-red-500">{criticalEvents}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden flex">
        {/* Sidebar - Stream List */}
        <div className="w-80 border-r border-gray-800 p-4 overflow-y-auto">
          <div className="mb-4">
            <h3 className="text-xs font-bold text-gray-400 uppercase mb-3">Event Streams</h3>
            <div className="space-y-2">
              <button
                onClick={() => setSelectedStream('all')}
                className={`w-full text-left p-2 rounded text-xs ${
                  selectedStream === 'all' ? 'bg-blue-600 text-white' : 'bg-gray-900 hover:bg-gray-800'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-bold">All Streams</span>
                  <span>{totalEventsPerSecond.toFixed(1)}/s</span>
                </div>
                <div className="text-gray-400">{totalEvents.toLocaleString()} events</div>
              </button>
              
              {streams.map(stream => (
                <button
                  key={stream.id}
                  onClick={() => setSelectedStream(stream.id)}
                  className={`w-full text-left p-2 rounded text-xs ${
                    selectedStream === stream.id ? 'bg-blue-600 text-white' : 'bg-gray-900 hover:bg-gray-800'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center space-x-2">
                      {getCategoryIcon(stream.category)}
                      <span className="font-bold">{stream.name}</span>
                    </div>
                    <div className={`w-2 h-2 rounded-full ${
                      stream.status === 'active' ? 'bg-green-500' :
                      stream.status === 'paused' ? 'bg-yellow-500' :
                      stream.status === 'error' ? 'bg-red-500' : 'bg-gray-500'
                    }`} />
                  </div>
                  <div className="flex items-center justify-between text-gray-400">
                    <span>{stream.eventsPerSecond.toFixed(1)}/s</span>
                    <span>{stream.lag > 0 && `Lag: ${stream.lag}`}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Event Content */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {viewMode === 'events' && (
            <>
              {/* Search and Filters */}
              <div className="p-4 border-b border-gray-800">
                <div className="flex items-center space-x-3">
                  <div className="flex-1 relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <input
                      type="text"
                      value={filters.searchQuery}
                      onChange={(e) => setFilters(prev => ({ ...prev, searchQuery: e.target.value }))}
                      placeholder="Search events..."
                      className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                    />
                  </div>
                  
                  <select
                    value={filters.timeRange}
                    onChange={(e) => setFilters(prev => ({ ...prev, timeRange: e.target.value }))}
                    className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                  >
                    <option value="5m">Last 5 Minutes</option>
                    <option value="15m">Last 15 Minutes</option>
                    <option value="1h">Last Hour</option>
                    <option value="6h">Last 6 Hours</option>
                    <option value="24h">Last 24 Hours</option>
                  </select>
                </div>
              </div>

              {/* Events List */}
              <div className="flex-1 overflow-y-auto p-4">
                <div className="space-y-2">
                  {filteredEvents.map(event => (
                    <div key={event.id} className="bg-gray-900 border border-gray-800 rounded">
                      <div 
                        className="p-3 cursor-pointer hover:bg-gray-800/50"
                        onClick={() => toggleEventExpansion(event.id)}
                      >
                        <div className="flex items-start space-x-3">
                          <div className="flex items-center space-x-2">
                            {expandedEvents.has(event.id) ? 
                              <ChevronDown className="w-4 h-4 text-gray-500" /> : 
                              <ChevronRight className="w-4 h-4 text-gray-500" />
                            }
                            <span className={`px-2 py-1 text-xs rounded ${getSeverityColor(event.severity)}`}>
                              {event.severity.toUpperCase()}
                            </span>
                            {!event.processed && <Clock className="w-3 h-3 text-yellow-500" />}
                            {event.acknowledged && <CheckCircle className="w-3 h-3 text-green-500" />}
                          </div>
                          
                          <div className="flex-1">
                            <div className="flex items-center space-x-3 text-xs text-gray-500 mb-1">
                              <span>{new Date(event.timestamp).toLocaleTimeString()}</span>
                              <span>{event.source}</span>
                              {event.user && <span>User: {event.user.split('@')[0]}</span>}
                              {event.correlationId && (
                                <span className="font-mono bg-gray-800 px-1 rounded">{event.correlationId}</span>
                              )}
                            </div>
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-bold">{event.eventType}</span>
                              <span className="text-xs text-gray-500">{event.payload.action}</span>
                            </div>
                          </div>
                          
                          <div className="flex items-center space-x-2">
                            {event.tags.map(tag => (
                              <span key={tag} className="px-1 py-0.5 bg-gray-800 rounded text-xs">
                                {tag}
                              </span>
                            ))}
                            <button className="p-1 hover:bg-gray-700 rounded">
                              <Copy className="w-4 h-4 text-gray-500" />
                            </button>
                          </div>
                        </div>
                      </div>
                      
                      {expandedEvents.has(event.id) && (
                        <div className="px-3 pb-3 border-t border-gray-800">
                          <div className="mt-3 space-y-2">
                            <div>
                              <h4 className="text-xs font-bold text-gray-400 mb-1">Payload</h4>
                              <div className="bg-gray-800 rounded p-2 text-xs font-mono">
                                <div className="space-y-1">
                                  <div><span className="text-gray-500">Action:</span> <span className="text-green-400">{event.payload.action}</span></div>
                                  <div><span className="text-gray-500">Resource:</span> <span className="text-blue-400">{event.payload.resource}</span></div>
                                  <div><span className="text-gray-500">Details:</span></div>
                                  <pre className="text-gray-300 ml-2 whitespace-pre-wrap">
                                    {JSON.stringify(event.payload.details, null, 2)}
                                  </pre>
                                  {event.payload.metadata && (
                                    <>
                                      <div><span className="text-gray-500">Metadata:</span></div>
                                      <pre className="text-gray-300 ml-2 whitespace-pre-wrap">
                                        {JSON.stringify(event.payload.metadata, null, 2)}
                                      </pre>
                                    </>
                                  )}
                                </div>
                              </div>
                            </div>
                            
                            <div className="flex items-center justify-between pt-2">
                              <div className="flex items-center space-x-3 text-xs">
                                {event.sessionId && (
                                  <span className="text-gray-500">Session: <span className="font-mono">{event.sessionId}</span></span>
                                )}
                                <span className="text-gray-500">Stream: {event.streamId}</span>
                              </div>
                              <div className="flex items-center space-x-2">
                                <button className="px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                                  Acknowledge
                                </button>
                                <button className="px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                                  Correlate
                                </button>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {viewMode === 'streams' && (
            <div className="p-6">
              <div className="grid grid-cols-2 gap-4">
                {streams.map(stream => (
                  <div key={stream.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        {getCategoryIcon(stream.category)}
                        <h3 className="text-sm font-bold">{stream.name}</h3>
                      </div>
                      <span className={`px-2 py-1 text-xs rounded ${getStatusColor(stream.status)}`}>
                        {stream.status.toUpperCase()}
                      </span>
                    </div>
                    
                    <p className="text-xs text-gray-400 mb-3">{stream.description}</p>
                    
                    <div className="grid grid-cols-2 gap-3 text-xs mb-3">
                      <div>
                        <p className="text-gray-400">Events</p>
                        <p className="text-lg font-bold">{stream.eventCount.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Rate</p>
                        <p className="text-lg font-bold">{stream.eventsPerSecond.toFixed(1)}/s</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Consumers</p>
                        <p className="text-lg font-bold">{stream.consumers}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Lag</p>
                        <p className={`text-lg font-bold ${
                          stream.lag === 0 ? 'text-green-500' :
                          stream.lag < 10 ? 'text-yellow-500' : 'text-red-500'
                        }`}>
                          {stream.lag}
                        </p>
                      </div>
                    </div>
                    
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Source</span>
                        <span>{stream.source}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Retention</span>
                        <span>{stream.retentionPeriod}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Last Event</span>
                        <span>{stream.lastEvent}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {viewMode === 'patterns' && (
            <div className="p-6">
              <div className="mb-6">
                <h3 className="text-sm font-bold mb-3">Event Patterns</h3>
                <p className="text-sm text-gray-400">Automated pattern detection and alerting rules</p>
              </div>
              
              <div className="space-y-4">
                {patterns.map(pattern => (
                  <div key={pattern.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-start space-x-3">
                        <div className={`p-2 rounded ${
                          pattern.enabled ? 'bg-green-900/20 text-green-500' : 'bg-gray-800 text-gray-500'
                        }`}>
                          <Eye className="w-4 h-4" />
                        </div>
                        <div className="flex-1">
                          <h4 className="text-sm font-bold mb-1">{pattern.name}</h4>
                          <p className="text-xs text-gray-400 mb-2">{pattern.description}</p>
                          <div className="bg-gray-800 rounded p-2 text-xs font-mono">
                            {pattern.pattern}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-1 text-xs rounded ${getSeverityColor(pattern.severity)}`}>
                          {pattern.severity.toUpperCase()}
                        </span>
                        <button className="p-1 hover:bg-gray-800 rounded">
                          <Settings className="w-4 h-4 text-gray-500" />
                        </button>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4 text-xs mb-3">
                      <div>
                        <p className="text-gray-400">Matches</p>
                        <p className="text-lg font-bold">{pattern.matches}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Last Match</p>
                        <p className="text-sm">{pattern.lastMatch}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Actions</p>
                        <div className="flex space-x-1">
                          {pattern.actions.map((action, idx) => (
                            <span key={idx} className="px-1 py-0.5 bg-gray-800 rounded text-xs">
                              {action.type}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}