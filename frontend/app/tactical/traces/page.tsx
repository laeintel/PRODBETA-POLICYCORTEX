'use client';

import React, { useState, useEffect } from 'react';
import { 
  GitBranch, Clock, Activity, AlertTriangle, CheckCircle, XCircle,
  Network, Database, Server, Globe, Zap, Eye, Filter, Search,
  BarChart, LineChart, Timer, MapPin, Route, Layers, Target,
  ArrowRight, ArrowDown, ChevronRight, RefreshCw, Download,
  Settings, Info, TrendingUp, TrendingDown, Hash, Code
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface DistributedTrace {
  id: string;
  traceId: string;
  operationName: string;
  serviceName: string;
  duration: number;
  timestamp: string;
  status: 'success' | 'error' | 'timeout' | 'partial';
  spanCount: number;
  errorCount: number;
  services: string[];
  tags: { [key: string]: string };
  spans: TraceSpan[];
}

interface TraceSpan {
  id: string;
  traceId: string;
  parentId?: string;
  operationName: string;
  serviceName: string;
  startTime: number;
  duration: number;
  status: 'success' | 'error' | 'timeout';
  tags: { [key: string]: string };
  logs: {
    timestamp: number;
    level: 'info' | 'warn' | 'error' | 'debug';
    message: string;
  }[];
  depth: number;
}

interface ServiceDependency {
  source: string;
  target: string;
  callCount: number;
  avgDuration: number;
  errorRate: number;
  throughput: number;
}

interface PerformanceBottleneck {
  id: string;
  type: 'slowest_operation' | 'error_hotspot' | 'high_latency' | 'dependency_issue';
  service: string;
  operation: string;
  impact: 'high' | 'medium' | 'low';
  avgDuration: number;
  frequency: number;
  recommendation: string;
}

export default function TraceAnalysis() {
  const [traces, setTraces] = useState<DistributedTrace[]>([]);
  const [dependencies, setDependencies] = useState<ServiceDependency[]>([]);
  const [bottlenecks, setBottlenecks] = useState<PerformanceBottleneck[]>([]);
  const [selectedTrace, setSelectedTrace] = useState<DistributedTrace | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedService, setSelectedService] = useState('all');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [viewMode, setViewMode] = useState<'traces' | 'dependencies' | 'bottlenecks' | 'timeline'>('traces');
  const [timeRange, setTimeRange] = useState('1h');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    // Initialize with mock distributed tracing data
    setTraces([
      {
        id: 'TRACE-001',
        traceId: 'abc123-def456-ghi789',
        operationName: 'POST /api/orders',
        serviceName: 'order-service',
        duration: 485,
        timestamp: '2 minutes ago',
        status: 'success',
        spanCount: 12,
        errorCount: 0,
        services: ['order-service', 'payment-service', 'inventory-service', 'notification-service'],
        tags: {
          'user.id': '12345',
          'order.type': 'express',
          'region': 'us-east-1'
        },
        spans: [
          {
            id: 'span-001',
            traceId: 'abc123-def456-ghi789',
            operationName: 'POST /api/orders',
            serviceName: 'order-service',
            startTime: 0,
            duration: 485,
            status: 'success',
            tags: { 'http.method': 'POST', 'http.url': '/api/orders' },
            logs: [
              { timestamp: 0, level: 'info', message: 'Order creation started' },
              { timestamp: 485, level: 'info', message: 'Order created successfully' }
            ],
            depth: 0
          },
          {
            id: 'span-002',
            traceId: 'abc123-def456-ghi789',
            parentId: 'span-001',
            operationName: 'validate_payment',
            serviceName: 'payment-service',
            startTime: 45,
            duration: 125,
            status: 'success',
            tags: { 'payment.method': 'credit_card' },
            logs: [
              { timestamp: 45, level: 'info', message: 'Payment validation started' },
              { timestamp: 170, level: 'info', message: 'Payment validated' }
            ],
            depth: 1
          },
          {
            id: 'span-003',
            traceId: 'abc123-def456-ghi789',
            parentId: 'span-001',
            operationName: 'check_inventory',
            serviceName: 'inventory-service',
            startTime: 180,
            duration: 95,
            status: 'success',
            tags: { 'product.id': 'PROD-789' },
            logs: [
              { timestamp: 180, level: 'info', message: 'Inventory check started' },
              { timestamp: 275, level: 'info', message: 'Inventory available' }
            ],
            depth: 1
          }
        ]
      },
      {
        id: 'TRACE-002',
        traceId: 'xyz987-uvw654-rst321',
        operationName: 'GET /api/users/profile',
        serviceName: 'user-service',
        duration: 1250,
        timestamp: '5 minutes ago',
        status: 'error',
        spanCount: 8,
        errorCount: 2,
        services: ['user-service', 'auth-service', 'profile-service'],
        tags: {
          'user.id': '67890',
          'auth.type': 'jwt',
          'region': 'us-west-2'
        },
        spans: [
          {
            id: 'span-004',
            traceId: 'xyz987-uvw654-rst321',
            operationName: 'GET /api/users/profile',
            serviceName: 'user-service',
            startTime: 0,
            duration: 1250,
            status: 'error',
            tags: { 'http.method': 'GET', 'http.status_code': '500' },
            logs: [
              { timestamp: 0, level: 'info', message: 'Profile request started' },
              { timestamp: 1250, level: 'error', message: 'Database timeout error' }
            ],
            depth: 0
          }
        ]
      },
      {
        id: 'TRACE-003',
        traceId: 'mno456-pqr123-stu789',
        operationName: 'POST /api/analytics/events',
        serviceName: 'analytics-service',
        duration: 320,
        timestamp: '8 minutes ago',
        status: 'success',
        spanCount: 6,
        errorCount: 0,
        services: ['analytics-service', 'event-store', 'cache-service'],
        tags: {
          'event.type': 'user_action',
          'batch.size': '50',
          'region': 'eu-west-1'
        },
        spans: []
      },
      {
        id: 'TRACE-004',
        traceId: 'def789-ghi012-jkl345',
        operationName: 'GET /api/recommendations',
        serviceName: 'recommendation-service',
        duration: 2150,
        timestamp: '12 minutes ago',
        status: 'timeout',
        spanCount: 15,
        errorCount: 1,
        services: ['recommendation-service', 'ml-service', 'feature-store', 'cache-service'],
        tags: {
          'user.id': '11111',
          'model.version': 'v2.1',
          'region': 'us-east-1'
        },
        spans: []
      },
      {
        id: 'TRACE-005',
        traceId: 'ghi012-jkl345-mno678',
        operationName: 'DELETE /api/sessions',
        serviceName: 'auth-service',
        duration: 85,
        timestamp: '15 minutes ago',
        status: 'success',
        spanCount: 3,
        errorCount: 0,
        services: ['auth-service', 'session-store'],
        tags: {
          'session.type': 'logout',
          'user.id': '22222',
          'region': 'us-central-1'
        },
        spans: []
      }
    ]);

    setDependencies([
      {
        source: 'order-service',
        target: 'payment-service',
        callCount: 1250,
        avgDuration: 125,
        errorRate: 2.1,
        throughput: 45.2
      },
      {
        source: 'order-service',
        target: 'inventory-service',
        callCount: 1180,
        avgDuration: 95,
        errorRate: 0.8,
        throughput: 42.1
      },
      {
        source: 'user-service',
        target: 'auth-service',
        callCount: 3200,
        avgDuration: 65,
        errorRate: 1.5,
        throughput: 115.7
      },
      {
        source: 'user-service',
        target: 'profile-service',
        callCount: 2850,
        avgDuration: 180,
        errorRate: 5.2,
        throughput: 103.2
      },
      {
        source: 'recommendation-service',
        target: 'ml-service',
        callCount: 850,
        avgDuration: 450,
        errorRate: 3.8,
        throughput: 28.5
      },
      {
        source: 'analytics-service',
        target: 'event-store',
        callCount: 4500,
        avgDuration: 25,
        errorRate: 0.3,
        throughput: 162.3
      }
    ]);

    setBottlenecks([
      {
        id: 'BTL-001',
        type: 'slowest_operation',
        service: 'recommendation-service',
        operation: 'GET /api/recommendations',
        impact: 'high',
        avgDuration: 2150,
        frequency: 850,
        recommendation: 'Implement caching for ML model predictions'
      },
      {
        id: 'BTL-002',
        type: 'error_hotspot',
        service: 'profile-service',
        operation: 'database_query',
        impact: 'high',
        avgDuration: 1200,
        frequency: 148,
        recommendation: 'Add database connection pooling and query optimization'
      },
      {
        id: 'BTL-003',
        type: 'high_latency',
        service: 'payment-service',
        operation: 'external_api_call',
        impact: 'medium',
        avgDuration: 380,
        frequency: 320,
        recommendation: 'Implement circuit breaker for external payment API'
      },
      {
        id: 'BTL-004',
        type: 'dependency_issue',
        service: 'order-service',
        operation: 'inventory_check',
        impact: 'medium',
        avgDuration: 250,
        frequency: 1180,
        recommendation: 'Add timeout and retry logic for inventory service calls'
      }
    ]);

    // Simulate real-time updates
    if (autoRefresh) {
      const interval = setInterval(() => {
        setTraces(prevTraces => 
          prevTraces.map(trace => ({
            ...trace,
            duration: Math.max(50, trace.duration + (Math.random() - 0.5) * 100)
          }))
        );
      }, 10000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'success': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error': return <XCircle className="w-4 h-4 text-red-500" />;
      case 'timeout': return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'partial': return <AlertTriangle className="w-4 h-4 text-orange-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'success': return 'text-green-500 bg-green-900/20';
      case 'error': return 'text-red-500 bg-red-900/20';
      case 'timeout': return 'text-yellow-500 bg-yellow-900/20';
      case 'partial': return 'text-orange-500 bg-orange-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getServiceIcon = (serviceName: string) => {
    if (serviceName.includes('auth')) return <Server className="w-4 h-4 text-blue-500" />;
    if (serviceName.includes('payment')) return <Database className="w-4 h-4 text-green-500" />;
    if (serviceName.includes('ml') || serviceName.includes('recommendation')) return <Zap className="w-4 h-4 text-purple-500" />;
    if (serviceName.includes('analytics')) return <BarChart className="w-4 h-4 text-cyan-500" />;
    return <Globe className="w-4 h-4 text-gray-500" />;
  };

  const getImpactColor = (impact: string) => {
    switch(impact) {
      case 'high': return 'text-red-500';
      case 'medium': return 'text-yellow-500';
      case 'low': return 'text-blue-500';
      default: return 'text-gray-500';
    }
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const filteredTraces = traces.filter(trace => {
    if (searchQuery && !trace.operationName.toLowerCase().includes(searchQuery.toLowerCase()) && 
        !trace.traceId.includes(searchQuery)) return false;
    if (selectedService !== 'all' && !trace.services.includes(selectedService)) return false;
    if (selectedStatus !== 'all' && trace.status !== selectedStatus) return false;
    return true;
  });

  const uniqueServices = [...new Set(traces.flatMap(t => t.services))];
  const errorRate = (traces.filter(t => t.status === 'error').length / traces.length) * 100;
  const avgDuration = traces.reduce((sum, t) => sum + t.duration, 0) / traces.length;
  const totalSpans = traces.reduce((sum, t) => sum + t.spanCount, 0);

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Trace Analysis</h1>
            <p className="text-sm text-gray-400 mt-1">Distributed tracing and performance analysis</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                autoRefresh ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              <span>{autoRefresh ? 'Live' : 'Paused'}</span>
            </button>
            
            <button
              onClick={() => setViewMode(
                viewMode === 'traces' ? 'dependencies' : 
                viewMode === 'dependencies' ? 'bottlenecks' : 
                viewMode === 'bottlenecks' ? 'timeline' : 'traces'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
            >
              {viewMode === 'traces' ? 'Dependencies' : 
               viewMode === 'dependencies' ? 'Bottlenecks' : 
               viewMode === 'bottlenecks' ? 'Timeline' : 'Traces'}
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export Traces</span>
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
              <p className="text-xs text-gray-400">Total Traces</p>
              <p className="text-xl font-bold">{traces.length}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XCircle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Error Rate</p>
              <p className="text-xl font-bold text-red-500">{errorRate.toFixed(1)}%</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Timer className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Avg Duration</p>
              <p className="text-xl font-bold">{formatDuration(avgDuration)}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Layers className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Total Spans</p>
              <p className="text-xl font-bold">{totalSpans}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Network className="w-5 h-5 text-cyan-500" />
            <div>
              <p className="text-xs text-gray-400">Services</p>
              <p className="text-xl font-bold">{uniqueServices.length}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Target className="w-5 h-5 text-orange-500" />
            <div>
              <p className="text-xs text-gray-400">Bottlenecks</p>
              <p className="text-xl font-bold text-orange-500">{bottlenecks.length}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Filters */}
        <div className="flex items-center space-x-3 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by operation name or trace ID..."
              className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
            />
          </div>
          
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="15m">Last 15 minutes</option>
            <option value="1h">Last hour</option>
            <option value="6h">Last 6 hours</option>
            <option value="24h">Last 24 hours</option>
          </select>
          
          <select
            value={selectedService}
            onChange={(e) => setSelectedService(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Services</option>
            {uniqueServices.map(service => (
              <option key={service} value={service}>{service}</option>
            ))}
          </select>
          
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Status</option>
            <option value="success">Success</option>
            <option value="error">Error</option>
            <option value="timeout">Timeout</option>
            <option value="partial">Partial</option>
          </select>
        </div>

        {viewMode === 'traces' && (
          <div className="space-y-3">
            {filteredTraces.map(trace => (
              <div 
                key={trace.id} 
                className={`bg-gray-900 border border-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-800/50 ${
                  selectedTrace?.id === trace.id ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => setSelectedTrace(selectedTrace?.id === trace.id ? null : trace)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(trace.status)}
                    <div>
                      <h3 className="text-sm font-bold">{trace.operationName}</h3>
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>Trace: {trace.traceId}</span>
                        <span>Service: {trace.serviceName}</span>
                        <span>{trace.timestamp}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <p className="text-sm font-bold">{formatDuration(trace.duration)}</p>
                      <p className="text-xs text-gray-500">{trace.spanCount} spans</p>
                    </div>
                    
                    <span className={`px-2 py-1 text-xs rounded ${getStatusColor(trace.status)}`}>
                      {trace.status.toUpperCase()}
                    </span>
                    
                    {trace.errorCount > 0 && (
                      <div className="text-right">
                        <p className="text-xs text-red-500">{trace.errorCount} errors</p>
                      </div>
                    )}
                    
                    <ChevronRight className={`w-4 h-4 text-gray-500 transition-transform ${
                      selectedTrace?.id === trace.id ? 'rotate-90' : ''
                    }`} />
                  </div>
                </div>
                
                <div className="mt-3 flex items-center space-x-4">
                  <div className="flex items-center space-x-1">
                    <Network className="w-3 h-3 text-gray-500" />
                    <span className="text-xs text-gray-500">
                      {trace.services.length} services: {trace.services.slice(0, 3).join(', ')}
                      {trace.services.length > 3 && ` +${trace.services.length - 3} more`}
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {Object.entries(trace.tags).slice(0, 2).map(([key, value]) => (
                      <span key={key} className="px-2 py-1 bg-gray-800 rounded text-xs">
                        {key}: {value}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Expanded Trace Details */}
                {selectedTrace?.id === trace.id && trace.spans.length > 0 && (
                  <div className="mt-4 border-t border-gray-700 pt-4">
                    <h4 className="text-sm font-bold mb-3">Span Timeline</h4>
                    <div className="space-y-2">
                      {trace.spans.map(span => (
                        <div key={span.id} className="bg-gray-800 rounded p-3" style={{ marginLeft: `${span.depth * 16}px` }}>
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              {getServiceIcon(span.serviceName)}
                              <span className="text-sm font-bold">{span.operationName}</span>
                              <span className="text-xs text-gray-500">({span.serviceName})</span>
                            </div>
                            <div className="flex items-center space-x-3">
                              <span className="text-xs">{formatDuration(span.duration)}</span>
                              {getStatusIcon(span.status)}
                            </div>
                          </div>
                          
                          {span.logs.length > 0 && (
                            <div className="mt-2 space-y-1">
                              {span.logs.map((log, idx) => (
                                <div key={idx} className="text-xs text-gray-400 flex items-center space-x-2">
                                  <span className={`w-1 h-1 rounded-full ${
                                    log.level === 'error' ? 'bg-red-500' :
                                    log.level === 'warn' ? 'bg-yellow-500' :
                                    'bg-blue-500'
                                  }`} />
                                  <span>+{log.timestamp}ms</span>
                                  <span>{log.message}</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {viewMode === 'dependencies' && (
          <>
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-3">Service Dependencies</h3>
              <p className="text-sm text-gray-400">Analyze service call patterns and performance</p>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              {dependencies.map((dep, idx) => (
                <div key={idx} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      {getServiceIcon(dep.source)}
                      <span className="text-sm font-bold">{dep.source}</span>
                      <ArrowRight className="w-4 h-4 text-gray-500" />
                      {getServiceIcon(dep.target)}
                      <span className="text-sm">{dep.target}</span>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <p className="text-gray-400">Call Count</p>
                      <p className="text-lg font-bold">{dep.callCount.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Avg Duration</p>
                      <p className="text-lg font-bold">{formatDuration(dep.avgDuration)}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Error Rate</p>
                      <p className={`text-lg font-bold ${dep.errorRate > 5 ? 'text-red-500' : dep.errorRate > 2 ? 'text-yellow-500' : 'text-green-500'}`}>
                        {dep.errorRate.toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-400">Throughput</p>
                      <p className="text-lg font-bold">{dep.throughput.toFixed(1)} req/s</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'bottlenecks' && (
          <>
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-3">Performance Bottlenecks</h3>
              <p className="text-sm text-gray-400">Identify and resolve performance issues</p>
            </div>
            
            <div className="space-y-3">
              {bottlenecks.map(bottleneck => (
                <div key={bottleneck.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <span className={`px-2 py-1 text-xs rounded ${
                        bottleneck.type === 'slowest_operation' ? 'bg-red-900/20 text-red-500' :
                        bottleneck.type === 'error_hotspot' ? 'bg-orange-900/20 text-orange-500' :
                        bottleneck.type === 'high_latency' ? 'bg-yellow-900/20 text-yellow-500' :
                        'bg-blue-900/20 text-blue-500'
                      }`}>
                        {bottleneck.type.replace('_', ' ').toUpperCase()}
                      </span>
                      <div>
                        <h4 className="text-sm font-bold">{bottleneck.service}</h4>
                        <p className="text-xs text-gray-400">{bottleneck.operation}</p>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <p className={`text-lg font-bold ${getImpactColor(bottleneck.impact)}`}>
                        {bottleneck.impact.toUpperCase()}
                      </p>
                      <p className="text-xs text-gray-500">Impact</p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 mb-3 text-xs">
                    <div>
                      <p className="text-gray-400">Avg Duration</p>
                      <p className="text-lg font-bold">{formatDuration(bottleneck.avgDuration)}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Frequency</p>
                      <p className="text-lg font-bold">{bottleneck.frequency.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Type</p>
                      <p className="text-sm">{bottleneck.type.replace('_', ' ')}</p>
                    </div>
                  </div>
                  
                  <div className="p-3 bg-gray-800 rounded text-xs">
                    <Info className="w-3 h-3 inline mr-1" />
                    <span className="text-gray-400">Recommendation: </span>
                    <span>{bottleneck.recommendation}</span>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'timeline' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="text-center py-12">
              <LineChart className="w-12 h-12 text-gray-500 mx-auto mb-4" />
              <h3 className="text-lg font-bold mb-2">Trace Timeline View</h3>
              <p className="text-sm text-gray-400 mb-6">
                Interactive timeline visualization would be implemented here
              </p>
              <div className="flex justify-center space-x-3">
                <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                  View Timeline
                </button>
                <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm">
                  Configure View
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}