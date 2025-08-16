/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

'use client';

// ML Model Monitoring Dashboard
// Real-time monitoring of model performance, drift detection, and prediction analytics

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { AlertCircle, TrendingUp, TrendingDown, Activity, Brain, Clock, CheckCircle, XCircle } from 'lucide-react';

interface ModelMetrics {
  modelId: string;
  version: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
  predictionCount: number;
  averageLatency: number;
  driftDetected: boolean;
  lastTrainingDate: string;
  driftHistory: DriftPoint[];
  recentPredictions: PredictionSummary[];
  performanceHistory: PerformancePoint[];
  confusionMatrix: number[][];
}

interface DriftPoint {
  timestamp: string;
  driftScore: number;
  threshold: number;
}

interface PredictionSummary {
  timestamp: string;
  resourceId: string;
  prediction: string;
  confidence: number;
  actual?: string;
  correct?: boolean;
}

interface PerformancePoint {
  date: string;
  accuracy: number;
  precision: number;
  recall: number;
}

export default function MLMonitoringDashboard() {
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch(`/api/v1/ml/metrics?timeRange=${selectedTimeRange}`);
        if (response.ok) {
          const data = await response.json();
          setModelMetrics(data);
        }
      } catch (error) {
        console.error('Failed to fetch model metrics:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    
    // Auto-refresh every 5 seconds if enabled
    const interval = autoRefresh ? setInterval(fetchMetrics, 5000) : null;
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [selectedTimeRange, autoRefresh]);

  const MetricCard: React.FC<{
    title: string;
    value: number | string;
    target?: number;
    format?: 'percent' | 'number' | 'ms';
    icon?: React.ReactNode;
    trend?: 'up' | 'down' | 'stable';
  }> = ({ title, value, target, format = 'percent', icon, trend }) => {
    const formatValue = (val: number | string) => {
      if (typeof val === 'string') return val;
      switch (format) {
        case 'percent':
          return `${(val * 100).toFixed(1)}%`;
        case 'ms':
          return `${val.toFixed(0)}ms`;
        default:
          return val.toFixed(2);
      }
    };

    const isGood = target ? (
      format === 'percent' ? 
        (typeof value === 'number' && value >= target) :
        (typeof value === 'number' && value <= target)
    ) : true;

    return (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">{title}</CardTitle>
          {icon}
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {formatValue(value)}
          </div>
          {target && (
            <p className={`text-xs ${isGood ? 'text-green-500' : 'text-red-500'}`}>
              Target: {formatValue(target)}
            </p>
          )}
          {trend && (
            <div className="flex items-center text-xs mt-1">
              {trend === 'up' && <TrendingUp className="h-3 w-3 text-green-500 mr-1" />}
              {trend === 'down' && <TrendingDown className="h-3 w-3 text-red-500 mr-1" />}
              {trend === 'stable' && <Activity className="h-3 w-3 text-gray-500 mr-1" />}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const DriftChart: React.FC<{ data: DriftPoint[] }> = ({ data }) => {
    if (!data || data.length === 0) return null;

    const chartData = data.map(point => ({
      time: new Date(point.timestamp).toLocaleTimeString(),
      drift: point.driftScore,
      threshold: point.threshold,
    }));

    return (
      <Card className="col-span-2">
        <CardHeader>
          <CardTitle>Model Drift Detection</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="drift" 
                stroke="#8884d8" 
                name="Drift Score"
                strokeWidth={2}
              />
              <Line 
                type="monotone" 
                dataKey="threshold" 
                stroke="#ff0000" 
                name="Threshold"
                strokeDasharray="5 5"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    );
  };

  const PredictionTimeline: React.FC<{ predictions: PredictionSummary[] }> = ({ predictions }) => {
    if (!predictions || predictions.length === 0) return null;

    return (
      <Card className="col-span-2">
        <CardHeader>
          <CardTitle>Recent Predictions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {predictions.map((pred, idx) => (
              <div 
                key={idx} 
                className="flex items-center justify-between p-2 border rounded hover:bg-gray-50"
              >
                <div className="flex items-center space-x-2">
                  {pred.correct !== undefined && (
                    pred.correct ? 
                      <CheckCircle className="h-4 w-4 text-green-500" /> :
                      <XCircle className="h-4 w-4 text-red-500" />
                  )}
                  <div>
                    <p className="text-sm font-medium">{pred.resourceId}</p>
                    <p className="text-xs text-gray-500">
                      {new Date(pred.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm">{pred.prediction}</p>
                  <p className="text-xs text-gray-500">
                    Confidence: {(pred.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  };

  const PerformanceChart: React.FC<{ data: PerformancePoint[] }> = ({ data }) => {
    if (!data || data.length === 0) return null;

    return (
      <Card className="col-span-2">
        <CardHeader>
          <CardTitle>Performance History</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[0, 1]} />
              <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
              <Legend />
              <Line type="monotone" dataKey="accuracy" stroke="#8884d8" name="Accuracy" />
              <Line type="monotone" dataKey="precision" stroke="#82ca9d" name="Precision" />
              <Line type="monotone" dataKey="recall" stroke="#ffc658" name="Recall" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    );
  };

  const ConfusionMatrixDisplay: React.FC<{ matrix: number[][] }> = ({ matrix }) => {
    if (!matrix || matrix.length === 0) return null;

    const labels = ['Compliant', 'Violation'];
    const total = matrix.flat().reduce((a, b) => a + b, 0);

    return (
      <Card>
        <CardHeader>
          <CardTitle>Confusion Matrix</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div></div>
            {labels.map(label => (
              <div key={label} className="text-xs font-semibold">
                Predicted {label}
              </div>
            ))}
            {matrix.map((row, i) => (
              <React.Fragment key={i}>
                <div className="text-xs font-semibold">
                  Actual {labels[i]}
                </div>
                {row.map((cell, j) => (
                  <div 
                    key={j}
                    className={`p-2 rounded ${
                      i === j ? 'bg-green-100' : 'bg-red-100'
                    }`}
                  >
                    <div className="font-bold">{cell}</div>
                    <div className="text-xs text-gray-600">
                      {((cell / total) * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">ML Model Monitoring</h1>
          {modelMetrics && (
            <p className="text-gray-500">
              Model: {modelMetrics.modelId} (v{modelMetrics.version})
            </p>
          )}
        </div>
        <div className="flex space-x-2">
          <select 
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="px-3 py-2 border rounded"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-4 py-2 rounded ${
              autoRefresh ? 'bg-blue-500 text-white' : 'bg-gray-200'
            }`}
          >
            {autoRefresh ? 'Auto-Refresh ON' : 'Auto-Refresh OFF'}
          </button>
        </div>
      </div>

      {/* Drift Alert */}
      {modelMetrics?.driftDetected && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
          <div className="flex">
            <AlertCircle className="h-5 w-5 text-yellow-400" />
            <div className="ml-3">
              <p className="text-sm text-yellow-700">
                Model drift detected! Consider retraining the model to maintain accuracy.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Key Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          title="Accuracy"
          value={modelMetrics?.accuracy || 0}
          target={0.90}
          icon={<Brain className="h-4 w-4 text-blue-500" />}
          trend="up"
        />
        <MetricCard
          title="Precision"
          value={modelMetrics?.precision || 0}
          target={0.85}
          icon={<CheckCircle className="h-4 w-4 text-green-500" />}
        />
        <MetricCard
          title="Recall"
          value={modelMetrics?.recall || 0}
          target={0.85}
          icon={<Activity className="h-4 w-4 text-orange-500" />}
        />
        <MetricCard
          title="F1 Score"
          value={modelMetrics?.f1Score || 0}
          target={0.85}
          icon={<TrendingUp className="h-4 w-4 text-purple-500" />}
        />
      </div>

      {/* Error Rates */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          title="False Positive Rate"
          value={modelMetrics?.falsePositiveRate || 0}
          target={0.05}
          icon={<XCircle className="h-4 w-4 text-red-500" />}
        />
        <MetricCard
          title="Average Latency"
          value={modelMetrics?.averageLatency || 0}
          target={100}
          format="ms"
          icon={<Clock className="h-4 w-4 text-blue-500" />}
        />
        <MetricCard
          title="Total Predictions"
          value={modelMetrics?.predictionCount || 0}
          format="number"
          icon={<Activity className="h-4 w-4 text-gray-500" />}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-4">
        <DriftChart data={modelMetrics?.driftHistory || []} />
        <PerformanceChart data={modelMetrics?.performanceHistory || []} />
        <PredictionTimeline predictions={modelMetrics?.recentPredictions || []} />
        <ConfusionMatrixDisplay matrix={modelMetrics?.confusionMatrix || []} />
      </div>

      {/* Model Info */}
      <Card>
        <CardHeader>
          <CardTitle>Model Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Model ID</p>
              <p className="font-medium">{modelMetrics?.modelId}</p>
            </div>
            <div>
              <p className="text-gray-500">Version</p>
              <p className="font-medium">{modelMetrics?.version}</p>
            </div>
            <div>
              <p className="text-gray-500">Last Training</p>
              <p className="font-medium">
                {modelMetrics?.lastTrainingDate && 
                  new Date(modelMetrics.lastTrainingDate).toLocaleDateString()}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}