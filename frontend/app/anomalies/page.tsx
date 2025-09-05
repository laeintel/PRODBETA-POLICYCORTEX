'use client';

import { Card } from '@/components/ui/card';
import { AlertTriangle, Activity, TrendingUp, Clock, Shield, Zap, AlertCircle } from 'lucide-react';
import { useState, useEffect } from 'react';

interface Anomaly {
  id: string;
  type: 'performance' | 'security' | 'cost' | 'availability';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  resource: string;
  detectedAt: string;
  status: 'new' | 'investigating' | 'resolved';
  deviation: string;
  baseline: string;
  current: string;
}

export default function AnomaliesPage() {
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [filter, setFilter] = useState<string>('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock anomaly data
    const mockAnomalies: Anomaly[] = [
      {
        id: 'anom-001',
        type: 'cost',
        severity: 'high',
        title: 'Unusual spike in compute costs',
        description: 'Detected 340% increase in VM costs compared to 7-day average',
        resource: 'Production VM Scale Set',
        detectedAt: '15 minutes ago',
        status: 'new',
        deviation: '+340%',
        baseline: '$450/day',
        current: '$1,980/day'
      },
      {
        id: 'anom-002',
        type: 'security',
        severity: 'critical',
        title: 'Abnormal login pattern detected',
        description: 'Multiple failed login attempts from unusual geographic location',
        resource: 'Azure AD - User: admin@company.com',
        detectedAt: '1 hour ago',
        status: 'investigating',
        deviation: '15 attempts',
        baseline: '0-2 failures/day',
        current: '15 failures/hour'
      },
      {
        id: 'anom-003',
        type: 'performance',
        severity: 'medium',
        title: 'Database response time degradation',
        description: 'Query response times increased by 250% in the last hour',
        resource: 'SQL Database - prod-db-01',
        detectedAt: '2 hours ago',
        status: 'new',
        deviation: '+250%',
        baseline: '120ms',
        current: '420ms'
      },
      {
        id: 'anom-004',
        type: 'availability',
        severity: 'critical',
        title: 'Service availability below SLA',
        description: 'API Gateway availability dropped to 95.2% (SLA: 99.9%)',
        resource: 'API Gateway - main-gateway',
        detectedAt: '30 minutes ago',
        status: 'investigating',
        deviation: '-4.7%',
        baseline: '99.9%',
        current: '95.2%'
      },
      {
        id: 'anom-005',
        type: 'cost',
        severity: 'medium',
        title: 'Unexpected data transfer charges',
        description: 'Egress data transfer 5x higher than normal patterns',
        resource: 'Storage Account - backupstorage01',
        detectedAt: '3 hours ago',
        status: 'resolved',
        deviation: '+500%',
        baseline: '100GB/day',
        current: '500GB/day'
      },
      {
        id: 'anom-006',
        type: 'performance',
        severity: 'low',
        title: 'CPU utilization anomaly',
        description: 'Consistent low CPU usage might indicate over-provisioning',
        resource: 'VM - app-server-03',
        detectedAt: '5 hours ago',
        status: 'new',
        deviation: '-85%',
        baseline: '40-60%',
        current: '5-8%'
      }
    ];

    setTimeout(() => {
      setAnomalies(mockAnomalies);
      setLoading(false);
    }, 800);
  }, []);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'performance':
        return <Activity className="h-5 w-5 text-blue-500" />;
      case 'security':
        return <Shield className="h-5 w-5 text-red-500" />;
      case 'cost':
        return <TrendingUp className="h-5 w-5 text-green-500" />;
      case 'availability':
        return <Zap className="h-5 w-5 text-yellow-500" />;
      default:
        return <AlertCircle className="h-5 w-5 text-gray-500" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 text-red-700 border-red-300 dark:bg-red-950 dark:text-red-400 dark:border-red-800';
      case 'high':
        return 'bg-orange-100 text-orange-700 border-orange-300 dark:bg-orange-950 dark:text-orange-400 dark:border-orange-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-700 border-yellow-300 dark:bg-yellow-950 dark:text-yellow-400 dark:border-yellow-800';
      case 'low':
        return 'bg-green-100 text-green-700 border-green-300 dark:bg-green-950 dark:text-green-400 dark:border-green-800';
      default:
        return 'bg-gray-100 text-gray-700 border-gray-300 dark:bg-gray-950 dark:text-gray-400 dark:border-gray-800';
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'new':
        return 'bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-400';
      case 'investigating':
        return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-400';
      case 'resolved':
        return 'bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-400';
      default:
        return 'bg-gray-100 text-gray-700 dark:bg-gray-950 dark:text-gray-400';
    }
  };

  const filteredAnomalies = filter === 'all' 
    ? anomalies 
    : anomalies.filter(a => a.type === filter);

  const stats = {
    total: anomalies.length,
    critical: anomalies.filter(a => a.severity === 'critical').length,
    investigating: anomalies.filter(a => a.status === 'investigating').length,
    new: anomalies.filter(a => a.status === 'new').length
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <AlertTriangle className="h-8 w-8 text-orange-600" />
            Anomaly Detection
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            AI-powered detection of unusual patterns and behaviors
          </p>
        </div>
        <button className="px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700">
          Configure Alerts
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Anomalies</p>
              <p className="text-2xl font-bold">{stats.total}</p>
            </div>
            <AlertTriangle className="h-8 w-8 text-orange-400" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Critical</p>
              <p className="text-2xl font-bold text-red-600">{stats.critical}</p>
            </div>
            <AlertCircle className="h-8 w-8 text-red-400" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Investigating</p>
              <p className="text-2xl font-bold text-yellow-600">{stats.investigating}</p>
            </div>
            <Clock className="h-8 w-8 text-yellow-400" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">New</p>
              <p className="text-2xl font-bold text-blue-600">{stats.new}</p>
            </div>
            <Activity className="h-8 w-8 text-blue-400" />
          </div>
        </Card>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2">
        {['all', 'performance', 'security', 'cost', 'availability'].map(type => (
          <button
            key={type}
            onClick={() => setFilter(type)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              filter === type
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            {type.charAt(0).toUpperCase() + type.slice(1)}
          </button>
        ))}
      </div>

      {/* Anomalies List */}
      <div className="space-y-4">
        {loading ? (
          [...Array(3)].map((_, i) => (
            <Card key={i} className="p-6 animate-pulse">
              <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-4"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-2"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
            </Card>
          ))
        ) : filteredAnomalies.length === 0 ? (
          <Card className="p-12 text-center">
            <AlertTriangle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-400">No anomalies detected</p>
          </Card>
        ) : (
          filteredAnomalies.map(anomaly => (
            <Card key={anomaly.id} className={`p-6 border-2 ${getSeverityColor(anomaly.severity)}`}>
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-start gap-3 flex-1">
                  {getTypeIcon(anomaly.type)}
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-semibold text-lg">{anomaly.title}</h3>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(anomaly.severity)}`}>
                        {anomaly.severity.toUpperCase()}
                      </span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBadge(anomaly.status)}`}>
                        {anomaly.status.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-gray-600 dark:text-gray-400 mb-3">{anomaly.description}</p>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-gray-500 dark:text-gray-400">Resource</p>
                        <p className="font-medium">{anomaly.resource}</p>
                      </div>
                      <div>
                        <p className="text-gray-500 dark:text-gray-400">Deviation</p>
                        <p className="font-medium text-orange-600">{anomaly.deviation}</p>
                      </div>
                      <div>
                        <p className="text-gray-500 dark:text-gray-400">Baseline</p>
                        <p className="font-medium">{anomaly.baseline}</p>
                      </div>
                      <div>
                        <p className="text-gray-500 dark:text-gray-400">Current</p>
                        <p className="font-medium">{anomaly.current}</p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="text-right">
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                    <Clock className="h-4 w-4 inline mr-1" />
                    {anomaly.detectedAt}
                  </p>
                  <div className="flex gap-2">
                    <button className="px-3 py-1 text-sm border border-current rounded-md hover:bg-black hover:bg-opacity-5">
                      Investigate
                    </button>
                    <button className="px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700">
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}