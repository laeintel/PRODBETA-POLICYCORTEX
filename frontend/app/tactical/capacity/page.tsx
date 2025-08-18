'use client';

import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, HardDrive, Cpu, Server, Database, Cloud, Network,
  AlertTriangle, CheckCircle, XCircle, Info, Calendar, Clock,
  BarChart, LineChart, PieChart, Activity, Zap, Package,
  ChevronRight, ArrowUp, ArrowDown, Settings, Download, RefreshCw,
  Gauge, Timer, Target, AlertCircle, TrendingDown
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface ResourceCapacity {
  id: string;
  name: string;
  type: 'compute' | 'storage' | 'memory' | 'network' | 'database' | 'container';
  current: {
    used: number;
    total: number;
    percentage: number;
  };
  predicted: {
    thirtyDays: number;
    sixtyDays: number;
    ninetyDays: number;
  };
  trend: 'increasing' | 'decreasing' | 'stable';
  trendRate: number; // percentage per day
  threshold: {
    warning: number;
    critical: number;
  };
  daysUntilFull?: number;
  recommendation?: string;
  historicalData: {
    timestamp: string;
    used: number;
    total: number;
  }[];
}

interface CapacityAlert {
  id: string;
  resourceId: string;
  resourceName: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  predictedDate?: string;
  recommendation: string;
  timestamp: string;
}

interface ScalingRecommendation {
  id: string;
  resource: string;
  action: 'scale-up' | 'scale-out' | 'optimize' | 'migrate';
  urgency: 'immediate' | 'soon' | 'planned';
  impact: string;
  costImpact: number;
  implementationTime: string;
  benefits: string[];
}

export default function CapacityPlanning() {
  const [resources, setResources] = useState<ResourceCapacity[]>([]);
  const [alerts, setAlerts] = useState<CapacityAlert[]>([]);
  const [recommendations, setRecommendations] = useState<ScalingRecommendation[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState('30d');
  const [selectedResourceType, setSelectedResourceType] = useState('all');
  const [viewMode, setViewMode] = useState<'overview' | 'detailed' | 'predictions'>('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    // Initialize with mock capacity data
    setResources([
      {
        id: 'RES-001',
        name: 'Production Compute Cluster',
        type: 'compute',
        current: {
          used: 2800,
          total: 4000,
          percentage: 70
        },
        predicted: {
          thirtyDays: 78,
          sixtyDays: 86,
          ninetyDays: 94
        },
        trend: 'increasing',
        trendRate: 0.27,
        threshold: {
          warning: 80,
          critical: 90
        },
        daysUntilFull: 111,
        recommendation: 'Consider scaling up compute resources within 60 days',
        historicalData: [
          { timestamp: '30d ago', used: 2200, total: 4000 },
          { timestamp: '20d ago', used: 2400, total: 4000 },
          { timestamp: '10d ago', used: 2600, total: 4000 },
          { timestamp: 'Today', used: 2800, total: 4000 }
        ]
      },
      {
        id: 'RES-002',
        name: 'Primary Storage Array',
        type: 'storage',
        current: {
          used: 8500,
          total: 10000,
          percentage: 85
        },
        predicted: {
          thirtyDays: 89,
          sixtyDays: 93,
          ninetyDays: 97
        },
        trend: 'increasing',
        trendRate: 0.13,
        threshold: {
          warning: 85,
          critical: 95
        },
        daysUntilFull: 46,
        recommendation: 'URGENT: Storage capacity will reach critical levels within 45 days',
        historicalData: [
          { timestamp: '30d ago', used: 7800, total: 10000 },
          { timestamp: '20d ago', used: 8000, total: 10000 },
          { timestamp: '10d ago', used: 8250, total: 10000 },
          { timestamp: 'Today', used: 8500, total: 10000 }
        ]
      },
      {
        id: 'RES-003',
        name: 'Application Memory Pool',
        type: 'memory',
        current: {
          used: 48,
          total: 64,
          percentage: 75
        },
        predicted: {
          thirtyDays: 77,
          sixtyDays: 79,
          ninetyDays: 81
        },
        trend: 'stable',
        trendRate: 0.07,
        threshold: {
          warning: 80,
          critical: 90
        },
        recommendation: 'Memory usage is stable, monitor for sudden spikes',
        historicalData: [
          { timestamp: '30d ago', used: 46, total: 64 },
          { timestamp: '20d ago', used: 47, total: 64 },
          { timestamp: '10d ago', used: 47, total: 64 },
          { timestamp: 'Today', used: 48, total: 64 }
        ]
      },
      {
        id: 'RES-004',
        name: 'Network Bandwidth',
        type: 'network',
        current: {
          used: 6.8,
          total: 10,
          percentage: 68
        },
        predicted: {
          thirtyDays: 72,
          sixtyDays: 76,
          ninetyDays: 80
        },
        trend: 'increasing',
        trendRate: 0.13,
        threshold: {
          warning: 75,
          critical: 85
        },
        recommendation: 'Network usage trending up, consider bandwidth upgrade',
        historicalData: [
          { timestamp: '30d ago', used: 6.0, total: 10 },
          { timestamp: '20d ago', used: 6.3, total: 10 },
          { timestamp: '10d ago', used: 6.5, total: 10 },
          { timestamp: 'Today', used: 6.8, total: 10 }
        ]
      },
      {
        id: 'RES-005',
        name: 'Database Connections',
        type: 'database',
        current: {
          used: 850,
          total: 1000,
          percentage: 85
        },
        predicted: {
          thirtyDays: 88,
          sixtyDays: 91,
          ninetyDays: 94
        },
        trend: 'increasing',
        trendRate: 0.10,
        threshold: {
          warning: 85,
          critical: 95
        },
        daysUntilFull: 60,
        recommendation: 'Database connection pool nearing limit, optimize queries or increase pool size',
        historicalData: [
          { timestamp: '30d ago', used: 780, total: 1000 },
          { timestamp: '20d ago', used: 800, total: 1000 },
          { timestamp: '10d ago', used: 825, total: 1000 },
          { timestamp: 'Today', used: 850, total: 1000 }
        ]
      },
      {
        id: 'RES-006',
        name: 'Container Registry',
        type: 'container',
        current: {
          used: 420,
          total: 800,
          percentage: 52.5
        },
        predicted: {
          thirtyDays: 55,
          sixtyDays: 58,
          ninetyDays: 61
        },
        trend: 'stable',
        trendRate: 0.10,
        threshold: {
          warning: 70,
          critical: 85
        },
        recommendation: 'Container storage healthy, implement retention policy for old images',
        historicalData: [
          { timestamp: '30d ago', used: 400, total: 800 },
          { timestamp: '20d ago', used: 408, total: 800 },
          { timestamp: '10d ago', used: 414, total: 800 },
          { timestamp: 'Today', used: 420, total: 800 }
        ]
      }
    ]);

    setAlerts([
      {
        id: 'ALERT-001',
        resourceId: 'RES-002',
        resourceName: 'Primary Storage Array',
        severity: 'high',
        message: 'Storage capacity will exceed warning threshold in 15 days',
        predictedDate: '15 days',
        recommendation: 'Provision additional storage or archive old data',
        timestamp: '5 minutes ago'
      },
      {
        id: 'ALERT-002',
        resourceId: 'RES-005',
        resourceName: 'Database Connections',
        severity: 'medium',
        message: 'Connection pool at warning threshold',
        recommendation: 'Review and optimize database queries',
        timestamp: '1 hour ago'
      },
      {
        id: 'ALERT-003',
        resourceId: 'RES-001',
        resourceName: 'Production Compute Cluster',
        severity: 'low',
        message: 'Projected to reach 90% capacity in 90 days',
        predictedDate: '90 days',
        recommendation: 'Plan compute expansion for Q2',
        timestamp: '3 hours ago'
      }
    ]);

    setRecommendations([
      {
        id: 'REC-001',
        resource: 'Primary Storage Array',
        action: 'scale-up',
        urgency: 'soon',
        impact: 'Add 5TB additional storage capacity',
        costImpact: 5000,
        implementationTime: '2 hours',
        benefits: ['Prevent storage outage', 'Maintain performance', 'Support growth']
      },
      {
        id: 'REC-002',
        resource: 'Database Connections',
        action: 'optimize',
        urgency: 'immediate',
        impact: 'Reduce connection usage by 20%',
        costImpact: 0,
        implementationTime: '4 hours',
        benefits: ['Improve performance', 'Reduce resource usage', 'Prevent connection exhaustion']
      },
      {
        id: 'REC-003',
        resource: 'Production Compute',
        action: 'scale-out',
        urgency: 'planned',
        impact: 'Add 2 additional compute nodes',
        costImpact: 3000,
        implementationTime: '1 day',
        benefits: ['Increase capacity', 'Improve redundancy', 'Better load distribution']
      }
    ]);

    // Simulate real-time updates
    if (autoRefresh) {
      const interval = setInterval(() => {
        setResources(prevResources => 
          prevResources.map(resource => ({
            ...resource,
            current: {
              ...resource.current,
              used: resource.current.used + (Math.random() - 0.3) * 10,
              percentage: ((resource.current.used + (Math.random() - 0.3) * 10) / resource.current.total) * 100
            }
          }))
        );
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getTypeIcon = (type: string) => {
    switch(type) {
      case 'compute': return <Cpu className="w-4 h-4 text-purple-500" />;
      case 'storage': return <HardDrive className="w-4 h-4 text-blue-500" />;
      case 'memory': return <Server className="w-4 h-4 text-green-500" />;
      case 'network': return <Network className="w-4 h-4 text-cyan-500" />;
      case 'database': return <Database className="w-4 h-4 text-orange-500" />;
      case 'container': return <Package className="w-4 h-4 text-pink-500" />;
      default: return <Server className="w-4 h-4 text-gray-500" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch(severity) {
      case 'low': return 'text-blue-500 bg-blue-900/20';
      case 'medium': return 'text-yellow-500 bg-yellow-900/20';
      case 'high': return 'text-orange-500 bg-orange-900/20';
      case 'critical': return 'text-red-500 bg-red-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch(urgency) {
      case 'immediate': return 'text-red-500';
      case 'soon': return 'text-yellow-500';
      case 'planned': return 'text-blue-500';
      default: return 'text-gray-500';
    }
  };

  const getCapacityColor = (percentage: number, warning: number, critical: number) => {
    if (percentage >= critical) return 'text-red-500';
    if (percentage >= warning) return 'text-yellow-500';
    return 'text-green-500';
  };

  const filteredResources = selectedResourceType === 'all' 
    ? resources 
    : resources.filter(r => r.type === selectedResourceType);

  const criticalResources = resources.filter(r => r.current.percentage >= r.threshold.critical).length;
  const warningResources = resources.filter(r => r.current.percentage >= r.threshold.warning && r.current.percentage < r.threshold.critical).length;
  const avgUtilization = resources.reduce((sum, r) => sum + r.current.percentage, 0) / resources.length;
  const resourcesNeedingAction = resources.filter(r => r.daysUntilFull && r.daysUntilFull < 60).length;

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Capacity Planning</h1>
            <p className="text-sm text-gray-400 mt-1">Resource utilization and growth predictions</p>
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
                viewMode === 'overview' ? 'detailed' : 
                viewMode === 'detailed' ? 'predictions' : 'overview'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
            >
              {viewMode === 'overview' ? 'Detailed' : viewMode === 'detailed' ? 'Predictions' : 'Overview'}
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export Report</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <Gauge className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Avg Utilization</p>
              <p className="text-xl font-bold">{avgUtilization.toFixed(1)}%</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XCircle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Critical</p>
              <p className="text-xl font-bold text-red-500">{criticalResources}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Warning</p>
              <p className="text-xl font-bold text-yellow-500">{warningResources}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Clock className="w-5 h-5 text-orange-500" />
            <div>
              <p className="text-xs text-gray-400">Action Needed</p>
              <p className="text-xl font-bold text-orange-500">{resourcesNeedingAction}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <TrendingUp className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Growth Rate</p>
              <p className="text-xl font-bold">+2.3%/mo</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Target className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Optimized</p>
              <p className="text-xl font-bold">{resources.length - criticalResources - warningResources}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Filters */}
        <div className="flex items-center space-x-3 mb-6">
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
            <option value="180d">Last 180 Days</option>
          </select>
          
          <select
            value={selectedResourceType}
            onChange={(e) => setSelectedResourceType(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Resources</option>
            <option value="compute">Compute</option>
            <option value="storage">Storage</option>
            <option value="memory">Memory</option>
            <option value="network">Network</option>
            <option value="database">Database</option>
            <option value="container">Container</option>
          </select>
        </div>

        {viewMode === 'overview' && (
          <>
            {/* Resource Cards */}
            <div className="grid grid-cols-3 gap-4 mb-6">
              {filteredResources.map(resource => (
                <div key={resource.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      {getTypeIcon(resource.type)}
                      <h3 className="text-sm font-bold">{resource.name}</h3>
                    </div>
                    {resource.trend === 'increasing' ? 
                      <TrendingUp className="w-4 h-4 text-yellow-500" /> : 
                      resource.trend === 'decreasing' ?
                      <TrendingDown className="w-4 h-4 text-green-500" /> :
                      <Activity className="w-4 h-4 text-gray-500" />
                    }
                  </div>
                  
                  <div className="mb-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-400">Current Usage</span>
                      <span className={`text-lg font-bold ${
                        getCapacityColor(resource.current.percentage, resource.threshold.warning, resource.threshold.critical)
                      }`}>
                        {resource.current.percentage.toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          resource.current.percentage >= resource.threshold.critical ? 'bg-red-500' :
                          resource.current.percentage >= resource.threshold.warning ? 'bg-yellow-500' :
                          'bg-green-500'
                        }`}
                        style={{ width: `${resource.current.percentage}%` }}
                      />
                    </div>
                    <div className="flex justify-between mt-1 text-xs text-gray-500">
                      <span>{resource.current.used.toFixed(0)}</span>
                      <span>{resource.current.total.toFixed(0)}</span>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-2 text-xs mb-3">
                    <div>
                      <p className="text-gray-400">30 Days</p>
                      <p className="font-bold">{resource.predicted.thirtyDays}%</p>
                    </div>
                    <div>
                      <p className="text-gray-400">60 Days</p>
                      <p className="font-bold">{resource.predicted.sixtyDays}%</p>
                    </div>
                    <div>
                      <p className="text-gray-400">90 Days</p>
                      <p className="font-bold">{resource.predicted.ninetyDays}%</p>
                    </div>
                  </div>
                  
                  {resource.daysUntilFull && (
                    <div className={`text-xs p-2 rounded ${
                      resource.daysUntilFull < 30 ? 'bg-red-900/20 text-red-500' :
                      resource.daysUntilFull < 60 ? 'bg-yellow-900/20 text-yellow-500' :
                      'bg-blue-900/20 text-blue-500'
                    }`}>
                      <AlertCircle className="w-3 h-3 inline mr-1" />
                      Full in {resource.daysUntilFull} days
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Alerts */}
            {alerts.length > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-bold mb-3">Capacity Alerts</h3>
                <div className="space-y-2">
                  {alerts.map(alert => (
                    <div key={alert.id} className="bg-gray-900 border border-gray-800 rounded-lg p-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <span className={`px-2 py-1 text-xs rounded ${getSeverityColor(alert.severity)}`}>
                            {alert.severity.toUpperCase()}
                          </span>
                          <div>
                            <p className="text-sm font-bold">{alert.resourceName}</p>
                            <p className="text-xs text-gray-400">{alert.message}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-xs text-gray-500">{alert.timestamp}</p>
                          {alert.predictedDate && (
                            <p className="text-xs text-yellow-500">In {alert.predictedDate}</p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {viewMode === 'detailed' && (
          <div className="space-y-4">
            {filteredResources.map(resource => (
              <div key={resource.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    {getTypeIcon(resource.type)}
                    <div>
                      <h3 className="text-sm font-bold">{resource.name}</h3>
                      <p className="text-xs text-gray-400">ID: {resource.id}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-500">Growth: {resource.trendRate.toFixed(2)}%/day</span>
                    {resource.trend === 'increasing' ? 
                      <TrendingUp className="w-4 h-4 text-yellow-500" /> : 
                      resource.trend === 'decreasing' ?
                      <TrendingDown className="w-4 h-4 text-green-500" /> :
                      <Activity className="w-4 h-4 text-gray-500" />
                    }
                  </div>
                </div>
                
                <div className="grid grid-cols-4 gap-4 mb-4">
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Current Usage</p>
                    <p className="text-2xl font-bold">
                      {resource.current.percentage.toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500">
                      {resource.current.used.toFixed(0)} / {resource.current.total.toFixed(0)}
                    </p>
                  </div>
                  
                  <div>
                    <p className="text-xs text-gray-400 mb-1">30-Day Forecast</p>
                    <p className={`text-lg font-bold ${
                      resource.predicted.thirtyDays >= resource.threshold.warning ? 'text-yellow-500' : ''
                    }`}>
                      {resource.predicted.thirtyDays}%
                    </p>
                  </div>
                  
                  <div>
                    <p className="text-xs text-gray-400 mb-1">60-Day Forecast</p>
                    <p className={`text-lg font-bold ${
                      resource.predicted.sixtyDays >= resource.threshold.critical ? 'text-red-500' :
                      resource.predicted.sixtyDays >= resource.threshold.warning ? 'text-yellow-500' : ''
                    }`}>
                      {resource.predicted.sixtyDays}%
                    </p>
                  </div>
                  
                  <div>
                    <p className="text-xs text-gray-400 mb-1">90-Day Forecast</p>
                    <p className={`text-lg font-bold ${
                      resource.predicted.ninetyDays >= resource.threshold.critical ? 'text-red-500' :
                      resource.predicted.ninetyDays >= resource.threshold.warning ? 'text-yellow-500' : ''
                    }`}>
                      {resource.predicted.ninetyDays}%
                    </p>
                  </div>
                </div>
                
                {/* Mini Chart */}
                <div className="mb-4">
                  <p className="text-xs text-gray-400 mb-2">Historical Trend</p>
                  <div className="h-20 flex items-end space-x-2">
                    {resource.historicalData.map((point, idx) => (
                      <div key={idx} className="flex-1 flex flex-col items-center">
                        <div 
                          className="w-full bg-blue-500 rounded-t"
                          style={{
                            height: `${(point.used / point.total) * 80}px`
                          }}
                        />
                        <span className="text-xs text-gray-500 mt-1">{point.timestamp}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                {resource.recommendation && (
                  <div className="p-3 bg-gray-800 rounded text-xs">
                    <Info className="w-3 h-3 inline mr-1" />
                    {resource.recommendation}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {viewMode === 'predictions' && (
          <>
            {/* Scaling Recommendations */}
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-3">Scaling Recommendations</h3>
              <div className="grid grid-cols-3 gap-4">
                {recommendations.map(rec => (
                  <div key={rec.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm font-bold">{rec.resource}</h4>
                      <span className={`text-xs font-bold ${getUrgencyColor(rec.urgency)}`}>
                        {rec.urgency.toUpperCase()}
                      </span>
                    </div>
                    
                    <div className="mb-3">
                      <span className="px-2 py-1 bg-blue-900/20 text-blue-500 rounded text-xs">
                        {rec.action.replace('-', ' ').toUpperCase()}
                      </span>
                    </div>
                    
                    <p className="text-xs text-gray-400 mb-3">{rec.impact}</p>
                    
                    <div className="space-y-1 text-xs mb-3">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Cost Impact</span>
                        <span>${rec.costImpact}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Time</span>
                        <span>{rec.implementationTime}</span>
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      {rec.benefits.map((benefit, idx) => (
                        <div key={idx} className="flex items-center space-x-1 text-xs text-green-500">
                          <CheckCircle className="w-3 h-3" />
                          <span>{benefit}</span>
                        </div>
                      ))}
                    </div>
                    
                    <button className="w-full mt-3 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                      Implement
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* Growth Projections */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Growth Projections</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-gray-400 mb-2">Resource Growth Trends</p>
                  <div className="space-y-2">
                    {filteredResources.map(resource => (
                      <div key={resource.id} className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          {getTypeIcon(resource.type)}
                          <span className="text-xs">{resource.name}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-xs text-gray-500">+{resource.trendRate.toFixed(2)}%/day</span>
                          {resource.daysUntilFull && (
                            <span className="text-xs text-yellow-500">Full: {resource.daysUntilFull}d</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div>
                  <p className="text-xs text-gray-400 mb-2">Capacity Planning Timeline</p>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-2 bg-gray-800 rounded">
                      <span className="text-xs">Next 30 Days</span>
                      <span className="text-xs text-yellow-500">{warningResources} resources at warning</span>
                    </div>
                    <div className="flex items-center justify-between p-2 bg-gray-800 rounded">
                      <span className="text-xs">Next 60 Days</span>
                      <span className="text-xs text-orange-500">{resourcesNeedingAction} need action</span>
                    </div>
                    <div className="flex items-center justify-between p-2 bg-gray-800 rounded">
                      <span className="text-xs">Next 90 Days</span>
                      <span className="text-xs text-red-500">{criticalResources} critical</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}