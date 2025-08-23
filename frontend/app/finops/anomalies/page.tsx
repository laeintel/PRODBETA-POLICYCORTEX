'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  AlertTriangle,
  TrendingUp,
  Clock,
  DollarSign,
  Activity,
  Bell,
  Filter,
  Download,
  RefreshCw,
  ArrowLeft,
  Brain,
  Zap,
  AlertCircle,
  CheckCircle,
  XCircle
} from 'lucide-react';
import { MLPredictionEngine, PredictionResult } from '@/lib/ml-predictions';

interface Anomaly {
  id: string;
  timestamp: Date;
  service: string;
  region: string;
  type: 'spike' | 'unusual_pattern' | 'new_resource' | 'config_change';
  severity: 'critical' | 'high' | 'medium' | 'low';
  cost: number;
  normalCost: number;
  deviation: number;
  mlConfidence: number;
  explanation: string;
  status: 'active' | 'investigating' | 'resolved' | 'false_positive';
  assignee?: string;
}

export default function CostAnomaliesPage() {
  const router = useRouter();
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [selectedAnomaly, setSelectedAnomaly] = useState<Anomaly | null>(null);
  const [filter, setFilter] = useState<'all' | 'active' | 'critical'>('all');
  const [realTimeMode, setRealTimeMode] = useState(true);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load ML predictions
    const loadPredictions = async () => {
      const prediction = await MLPredictionEngine.predictCostSpike('production');
      setPredictions([prediction]);
    };
    loadPredictions();

    // Simulate real-time anomaly detection
    const generateAnomaly = (): Anomaly => {
      const services = ['EC2', 'RDS', 'Lambda', 'S3', 'CloudFront', 'EKS'];
      const regions = ['us-east-1', 'eu-west-1', 'ap-southeast-1', 'us-west-2'];
      const types: Anomaly['type'][] = ['spike', 'unusual_pattern', 'new_resource', 'config_change'];
      const severities: Anomaly['severity'][] = ['critical', 'high', 'medium', 'low'];
      
      const normalCost = Math.floor(Math.random() * 5000) + 1000;
      const deviation = Math.floor(Math.random() * 200) + 50;
      const cost = normalCost * (1 + deviation / 100);

      return {
        id: `ANM-${Date.now()}`,
        timestamp: new Date(),
        service: services[Math.floor(Math.random() * services.length)],
        region: regions[Math.floor(Math.random() * regions.length)],
        type: types[Math.floor(Math.random() * types.length)],
        severity: severities[Math.floor(Math.random() * severities.length)],
        cost,
        normalCost,
        deviation,
        mlConfidence: Math.floor(Math.random() * 30) + 70,
        explanation: 'ML detected unusual spending pattern compared to historical baseline',
        status: 'active'
      };
    };

    // Initial anomalies
    setAnomalies([
      {
        id: 'ANM-001',
        timestamp: new Date(Date.now() - 3600000),
        service: 'EC2',
        region: 'us-east-1',
        type: 'spike',
        severity: 'critical',
        cost: 45000,
        normalCost: 15000,
        deviation: 200,
        mlConfidence: 94,
        explanation: 'Auto-scaling group triggered unexpectedly, 3x normal instance count',
        status: 'active'
      },
      {
        id: 'ANM-002',
        timestamp: new Date(Date.now() - 7200000),
        service: 'RDS',
        region: 'eu-west-1',
        type: 'config_change',
        severity: 'high',
        cost: 8500,
        normalCost: 3000,
        deviation: 183,
        mlConfidence: 89,
        explanation: 'Database instance class upgraded from db.t3.medium to db.r5.2xlarge',
        status: 'investigating',
        assignee: 'sarah.chen@company.com'
      },
      {
        id: 'ANM-003',
        timestamp: new Date(Date.now() - 10800000),
        service: 'Lambda',
        region: 'us-west-2',
        type: 'unusual_pattern',
        severity: 'medium',
        cost: 2200,
        normalCost: 800,
        deviation: 175,
        mlConfidence: 78,
        explanation: 'Function invocations increased 10x during off-peak hours',
        status: 'resolved'
      }
    ]);

    setLoading(false);

    // Real-time anomaly generation
    const interval = setInterval(() => {
      if (realTimeMode) {
        setAnomalies(prev => [generateAnomaly(), ...prev].slice(0, 20));
      }
    }, 30000); // Every 30 seconds

    return () => clearInterval(interval);
  }, [realTimeMode]);

  const filteredAnomalies = anomalies.filter(a => {
    if (filter === 'all') return true;
    if (filter === 'active') return a.status === 'active' || a.status === 'investigating';
    if (filter === 'critical') return a.severity === 'critical' || a.severity === 'high';
    return true;
  });

  const getSeverityColor = (severity: Anomaly['severity']) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50 dark:bg-red-900/20';
      case 'high': return 'text-orange-600 bg-orange-50 dark:bg-orange-900/20';
      case 'medium': return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20';
      case 'low': return 'text-blue-600 bg-blue-50 dark:bg-blue-900/20';
    }
  };

  const getStatusIcon = (status: Anomaly['status']) => {
    switch (status) {
      case 'active': return <AlertCircle className="h-5 w-5 text-red-500" />;
      case 'investigating': return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'resolved': return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'false_positive': return <XCircle className="h-5 w-5 text-gray-500" />;
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <button
            onClick={() => router.push('/finops')}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-2">
              <AlertTriangle className="h-8 w-8 text-orange-600" />
              Real-Time Cost Anomaly Detection
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              AI detects cost spikes within minutes, not days
            </p>
          </div>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setRealTimeMode(!realTimeMode)}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              realTimeMode 
                ? 'bg-green-600 text-white' 
                : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            <Activity className="h-4 w-4" />
            {realTimeMode ? 'Live' : 'Paused'}
          </button>
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
            <Bell className="h-4 w-4" />
            Configure Alerts
          </button>
        </div>
      </div>

      {/* ML Prediction Alert */}
      {predictions.length > 0 && (
        <div className="mb-6 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Brain className="h-6 w-6 text-amber-600 dark:text-amber-400 mt-1" />
            <div className="flex-1">
              <h3 className="font-semibold text-amber-900 dark:text-amber-100">
                AI Prediction: {predictions[0].prediction}
              </h3>
              <p className="text-amber-700 dark:text-amber-300 mt-1">
                {predictions[0].explanation}
              </p>
              <div className="flex items-center gap-4 mt-2 text-sm text-amber-600 dark:text-amber-400">
                <span>Confidence: {(predictions[0].confidence * 100).toFixed(0)}%</span>
                <span>Time to Event: {predictions[0].timeToEvent}</span>
                <span>Estimated Impact: ${predictions[0].impactEstimate?.financial?.toLocaleString()}</span>
              </div>
            </div>
            <button className="px-3 py-1 bg-amber-600 text-white rounded-md text-sm hover:bg-amber-700">
              Take Action
            </button>
          </div>
        </div>
      )}

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Active Anomalies</p>
              <p className="text-2xl font-bold">{anomalies.filter(a => a.status === 'active').length}</p>
            </div>
            <AlertTriangle className="h-8 w-8 text-orange-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Cost Impact</p>
              <p className="text-2xl font-bold">
                ${anomalies.reduce((sum, a) => sum + (a.cost - a.normalCost), 0).toLocaleString()}
              </p>
            </div>
            <DollarSign className="h-8 w-8 text-red-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Avg Detection Time</p>
              <p className="text-2xl font-bold">3.2 min</p>
            </div>
            <Zap className="h-8 w-8 text-yellow-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">ML Accuracy</p>
              <p className="text-2xl font-bold">92%</p>
            </div>
            <Brain className="h-8 w-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4 mb-6">
        <div className="flex items-center gap-2">
          <Filter className="h-5 w-5 text-gray-500" />
          <span className="text-sm text-gray-600 dark:text-gray-400">Filter:</span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setFilter('all')}
            className={`px-3 py-1 rounded-md text-sm ${
              filter === 'all' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            All ({anomalies.length})
          </button>
          <button
            onClick={() => setFilter('active')}
            className={`px-3 py-1 rounded-md text-sm ${
              filter === 'active' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            Active ({anomalies.filter(a => a.status === 'active' || a.status === 'investigating').length})
          </button>
          <button
            onClick={() => setFilter('critical')}
            className={`px-3 py-1 rounded-md text-sm ${
              filter === 'critical' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            Critical ({anomalies.filter(a => a.severity === 'critical' || a.severity === 'high').length})
          </button>
        </div>
        <div className="ml-auto">
          <button className="px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded-md text-sm flex items-center gap-2">
            <Download className="h-4 w-4" />
            Export
          </button>
        </div>
      </div>

      {/* Anomalies List */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm">
        <div className="p-6">
          <h2 className="text-xl font-semibold mb-4">Detected Anomalies</h2>
          <div className="space-y-3">
            {filteredAnomalies.map((anomaly) => (
              <div
                key={anomaly.id}
                className="border dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                onClick={() => setSelectedAnomaly(anomaly)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    {getStatusIcon(anomaly.status)}
                    <div>
                      <div className="flex items-center gap-3">
                        <h3 className="font-semibold">{anomaly.service} - {anomaly.region}</h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(anomaly.severity)}`}>
                          {anomaly.severity.toUpperCase()}
                        </span>
                        <span className="text-xs text-gray-500">
                          {new Date(anomaly.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {anomaly.explanation}
                      </p>
                      <div className="flex items-center gap-4 mt-2">
                        <span className="text-sm">
                          Cost: <span className="font-semibold text-red-600">${anomaly.cost.toLocaleString()}</span>
                          {' '}(Normal: ${anomaly.normalCost.toLocaleString()})
                        </span>
                        <span className="text-sm">
                          Deviation: <span className="font-semibold text-orange-600">+{anomaly.deviation}%</span>
                        </span>
                        <span className="text-sm">
                          ML Confidence: <span className="font-semibold">{anomaly.mlConfidence}%</span>
                        </span>
                      </div>
                      {anomaly.assignee && (
                        <p className="text-xs text-gray-500 mt-2">
                          Assigned to: {anomaly.assignee}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        // Handle investigate action
                      }}
                      className="px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700"
                    >
                      Investigate
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        // Handle suppress action
                      }}
                      className="px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded-md text-sm hover:bg-gray-300 dark:hover:bg-gray-600"
                    >
                      Suppress
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}