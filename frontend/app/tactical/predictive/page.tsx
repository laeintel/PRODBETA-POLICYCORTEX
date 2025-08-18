'use client';

import React, { useState, useEffect } from 'react';
import { 
  LineChart, TrendingUp, TrendingDown, Brain, Activity, AlertTriangle,
  Target, BarChart, PieChart, Zap, Clock, Calendar, Server, Database,
  CheckCircle, XCircle, Info, ChevronRight, ArrowUp, ArrowDown,
  Settings, Download, RefreshCw, Eye, Filter, Search, Gauge,
  Shield, Bug, Wifi, HardDrive, Cpu, Users, Globe, Package
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface PredictionModel {
  id: string;
  name: string;
  type: 'capacity' | 'performance' | 'anomaly' | 'failure' | 'demand' | 'cost';
  accuracy: number;
  confidence: number;
  lastTrained: string;
  status: 'active' | 'training' | 'stale' | 'error';
  predictions: {
    shortTerm: { value: number; probability: number; timeframe: string; };
    mediumTerm: { value: number; probability: number; timeframe: string; };
    longTerm: { value: number; probability: number; timeframe: string; };
  };
  metrics: {
    meanError: number;
    r2Score: number;
    precision: number;
    recall: number;
  };
}

interface PredictiveAlert {
  id: string;
  modelId: string;
  modelName: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  prediction: string;
  probability: number;
  expectedDate: string;
  impact: string;
  recommendation: string;
  timestamp: string;
  status: 'active' | 'acknowledged' | 'resolved';
}

interface AnomalyDetection {
  id: string;
  source: string;
  metric: string;
  timestamp: string;
  severity: 'minor' | 'moderate' | 'severe';
  anomalyScore: number;
  expectedValue: number;
  actualValue: number;
  deviation: number;
  pattern: string;
  rootCause?: string;
}

interface ForecastData {
  category: string;
  historical: { timestamp: string; value: number; }[];
  predicted: { timestamp: string; value: number; confidence: number; }[];
  trend: 'increasing' | 'decreasing' | 'stable';
  seasonality: boolean;
}

export default function PredictiveAnalytics() {
  const [models, setModels] = useState<PredictionModel[]>([]);
  const [alerts, setAlerts] = useState<PredictiveAlert[]>([]);
  const [anomalies, setAnomalies] = useState<AnomalyDetection[]>([]);
  const [forecasts, setForecasts] = useState<ForecastData[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState('30d');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [viewMode, setViewMode] = useState<'overview' | 'models' | 'forecasts' | 'anomalies'>('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    // Initialize with mock predictive analytics data
    setModels([
      {
        id: 'MODEL-001',
        name: 'CPU Usage Predictor',
        type: 'performance',
        accuracy: 94.5,
        confidence: 87.2,
        lastTrained: '2 hours ago',
        status: 'active',
        predictions: {
          shortTerm: { value: 78, probability: 0.92, timeframe: '1 hour' },
          mediumTerm: { value: 85, probability: 0.78, timeframe: '24 hours' },
          longTerm: { value: 92, probability: 0.65, timeframe: '7 days' }
        },
        metrics: {
          meanError: 2.3,
          r2Score: 0.945,
          precision: 0.91,
          recall: 0.88
        }
      },
      {
        id: 'MODEL-002',
        name: 'Storage Capacity Forecaster',
        type: 'capacity',
        accuracy: 91.8,
        confidence: 93.1,
        lastTrained: '6 hours ago',
        status: 'active',
        predictions: {
          shortTerm: { value: 82, probability: 0.95, timeframe: '1 day' },
          mediumTerm: { value: 89, probability: 0.87, timeframe: '1 week' },
          longTerm: { value: 96, probability: 0.71, timeframe: '1 month' }
        },
        metrics: {
          meanError: 1.8,
          r2Score: 0.918,
          precision: 0.94,
          recall: 0.89
        }
      },
      {
        id: 'MODEL-003',
        name: 'Failure Prediction Engine',
        type: 'failure',
        accuracy: 89.2,
        confidence: 85.7,
        lastTrained: '12 hours ago',
        status: 'active',
        predictions: {
          shortTerm: { value: 2, probability: 0.15, timeframe: '6 hours' },
          mediumTerm: { value: 8, probability: 0.32, timeframe: '24 hours' },
          longTerm: { value: 25, probability: 0.58, timeframe: '7 days' }
        },
        metrics: {
          meanError: 3.2,
          r2Score: 0.892,
          precision: 0.87,
          recall: 0.85
        }
      },
      {
        id: 'MODEL-004',
        name: 'Demand Forecasting',
        type: 'demand',
        accuracy: 87.6,
        confidence: 82.4,
        lastTrained: '1 day ago',
        status: 'active',
        predictions: {
          shortTerm: { value: 1250, probability: 0.89, timeframe: '1 hour' },
          mediumTerm: { value: 1580, probability: 0.74, timeframe: '4 hours' },
          longTerm: { value: 2100, probability: 0.61, timeframe: '24 hours' }
        },
        metrics: {
          meanError: 45.3,
          r2Score: 0.876,
          precision: 0.82,
          recall: 0.79
        }
      },
      {
        id: 'MODEL-005',
        name: 'Anomaly Detector',
        type: 'anomaly',
        accuracy: 93.7,
        confidence: 89.8,
        lastTrained: '4 hours ago',
        status: 'active',
        predictions: {
          shortTerm: { value: 0.15, probability: 0.85, timeframe: '10 minutes' },
          mediumTerm: { value: 0.23, probability: 0.72, timeframe: '1 hour' },
          longTerm: { value: 0.41, probability: 0.58, timeframe: '6 hours' }
        },
        metrics: {
          meanError: 0.08,
          r2Score: 0.937,
          precision: 0.95,
          recall: 0.91
        }
      }
    ]);

    setAlerts([
      {
        id: 'PRED-001',
        modelId: 'MODEL-002',
        modelName: 'Storage Capacity Forecaster',
        severity: 'high',
        prediction: 'Storage will exceed 95% capacity',
        probability: 0.89,
        expectedDate: 'In 18 days',
        impact: 'Service degradation and potential outage',
        recommendation: 'Provision additional storage or implement data archiving',
        timestamp: '5 minutes ago',
        status: 'active'
      },
      {
        id: 'PRED-002',
        modelId: 'MODEL-003',
        modelName: 'Failure Prediction Engine',
        severity: 'critical',
        prediction: 'Database node failure likely',
        probability: 0.73,
        expectedDate: 'In 6 hours',
        impact: 'Potential database outage affecting all services',
        recommendation: 'Schedule immediate maintenance and prepare failover',
        timestamp: '10 minutes ago',
        status: 'active'
      },
      {
        id: 'PRED-003',
        modelId: 'MODEL-001',
        modelName: 'CPU Usage Predictor',
        severity: 'medium',
        prediction: 'CPU usage will exceed warning threshold',
        probability: 0.78,
        expectedDate: 'In 4 hours',
        impact: 'Degraded response times for API services',
        recommendation: 'Scale up compute resources or optimize workloads',
        timestamp: '25 minutes ago',
        status: 'acknowledged'
      }
    ]);

    setAnomalies([
      {
        id: 'ANOM-001',
        source: 'api-gateway',
        metric: 'response_time',
        timestamp: '2 minutes ago',
        severity: 'severe',
        anomalyScore: 0.92,
        expectedValue: 145,
        actualValue: 890,
        deviation: 513.8,
        pattern: 'sudden_spike',
        rootCause: 'Database connection pool exhaustion'
      },
      {
        id: 'ANOM-002',
        source: 'cache-service',
        metric: 'hit_ratio',
        timestamp: '15 minutes ago',
        severity: 'moderate',
        anomalyScore: 0.78,
        expectedValue: 0.85,
        actualValue: 0.42,
        deviation: -0.43,
        pattern: 'gradual_decline'
      },
      {
        id: 'ANOM-003',
        source: 'network',
        metric: 'packet_loss',
        timestamp: '30 minutes ago',
        severity: 'minor',
        anomalyScore: 0.65,
        expectedValue: 0.01,
        actualValue: 0.08,
        deviation: 0.07,
        pattern: 'intermittent_spikes'
      }
    ]);

    setForecasts([
      {
        category: 'CPU Usage',
        historical: [
          { timestamp: '7d ago', value: 65 },
          { timestamp: '6d ago', value: 68 },
          { timestamp: '5d ago', value: 70 },
          { timestamp: '4d ago', value: 72 },
          { timestamp: '3d ago', value: 74 },
          { timestamp: '2d ago', value: 76 },
          { timestamp: '1d ago', value: 78 }
        ],
        predicted: [
          { timestamp: 'Today', value: 80, confidence: 0.92 },
          { timestamp: '1d', value: 82, confidence: 0.89 },
          { timestamp: '2d', value: 84, confidence: 0.85 },
          { timestamp: '3d', value: 86, confidence: 0.81 },
          { timestamp: '4d', value: 88, confidence: 0.77 },
          { timestamp: '5d', value: 90, confidence: 0.73 },
          { timestamp: '6d', value: 92, confidence: 0.68 }
        ],
        trend: 'increasing',
        seasonality: false
      },
      {
        category: 'Request Volume',
        historical: [
          { timestamp: '7d ago', value: 1200 },
          { timestamp: '6d ago', value: 1180 },
          { timestamp: '5d ago', value: 1250 },
          { timestamp: '4d ago', value: 1220 },
          { timestamp: '3d ago', value: 1290 },
          { timestamp: '2d ago', value: 1340 },
          { timestamp: '1d ago', value: 1380 }
        ],
        predicted: [
          { timestamp: 'Today', value: 1420, confidence: 0.88 },
          { timestamp: '1d', value: 1460, confidence: 0.84 },
          { timestamp: '2d', value: 1500, confidence: 0.79 },
          { timestamp: '3d', value: 1540, confidence: 0.74 },
          { timestamp: '4d', value: 1580, confidence: 0.69 },
          { timestamp: '5d', value: 1620, confidence: 0.64 },
          { timestamp: '6d', value: 1660, confidence: 0.59 }
        ],
        trend: 'increasing',
        seasonality: true
      }
    ]);

    // Simulate real-time updates
    if (autoRefresh) {
      const interval = setInterval(() => {
        setModels(prevModels => 
          prevModels.map(model => ({
            ...model,
            accuracy: Math.max(80, Math.min(99, model.accuracy + (Math.random() - 0.5) * 0.5)),
            confidence: Math.max(70, Math.min(95, model.confidence + (Math.random() - 0.5) * 0.8))
          }))
        );
      }, 8000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getModelTypeIcon = (type: string) => {
    switch(type) {
      case 'capacity': return <HardDrive className="w-4 h-4 text-blue-500" />;
      case 'performance': return <Cpu className="w-4 h-4 text-purple-500" />;
      case 'anomaly': return <Eye className="w-4 h-4 text-yellow-500" />;
      case 'failure': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'demand': return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'cost': return <Target className="w-4 h-4 text-orange-500" />;
      default: return <Brain className="w-4 h-4 text-gray-500" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch(severity) {
      case 'low':
      case 'minor': return 'text-blue-500 bg-blue-900/20';
      case 'medium':
      case 'moderate': return 'text-yellow-500 bg-yellow-900/20';
      case 'high':
      case 'severe': return 'text-orange-500 bg-orange-900/20';
      case 'critical': return 'text-red-500 bg-red-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': return 'text-green-500 bg-green-900/20';
      case 'training': return 'text-yellow-500 bg-yellow-900/20';
      case 'stale': return 'text-orange-500 bg-orange-900/20';
      case 'error': return 'text-red-500 bg-red-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 95) return 'text-green-500';
    if (accuracy >= 90) return 'text-blue-500';
    if (accuracy >= 85) return 'text-yellow-500';
    return 'text-red-500';
  };

  const filteredModels = selectedCategory === 'all' 
    ? models 
    : models.filter(m => m.type === selectedCategory);

  const activeModels = models.filter(m => m.status === 'active').length;
  const avgAccuracy = models.reduce((sum, m) => sum + m.accuracy, 0) / models.length;
  const activeAlerts = alerts.filter(a => a.status === 'active').length;
  const severeAnomalies = anomalies.filter(a => a.severity === 'severe').length;

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Predictive Analytics</h1>
            <p className="text-sm text-gray-400 mt-1">AI-powered predictions and anomaly detection</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                autoRefresh ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              <span>{autoRefresh ? 'Auto Update' : 'Paused'}</span>
            </button>
            
            <button
              onClick={() => setViewMode(
                viewMode === 'overview' ? 'models' : 
                viewMode === 'models' ? 'forecasts' : 
                viewMode === 'forecasts' ? 'anomalies' : 'overview'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
            >
              {viewMode === 'overview' ? 'Models' : 
               viewMode === 'models' ? 'Forecasts' : 
               viewMode === 'forecasts' ? 'Anomalies' : 'Overview'}
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export Analysis</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <Brain className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Active Models</p>
              <p className="text-xl font-bold">{activeModels}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Target className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Avg Accuracy</p>
              <p className="text-xl font-bold">{avgAccuracy.toFixed(1)}%</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Active Alerts</p>
              <p className="text-xl font-bold text-red-500">{activeAlerts}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Eye className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Anomalies</p>
              <p className="text-xl font-bold text-yellow-500">{severeAnomalies}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <LineChart className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Predictions</p>
              <p className="text-xl font-bold">127</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Activity className="w-5 h-5 text-cyan-500" />
            <div>
              <p className="text-xs text-gray-400">Confidence</p>
              <p className="text-xl font-bold">87.3%</p>
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
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
          </select>
          
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Categories</option>
            <option value="capacity">Capacity</option>
            <option value="performance">Performance</option>
            <option value="anomaly">Anomaly Detection</option>
            <option value="failure">Failure Prediction</option>
            <option value="demand">Demand Forecasting</option>
            <option value="cost">Cost Optimization</option>
          </select>
        </div>

        {viewMode === 'overview' && (
          <>
            {/* Predictive Alerts */}
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-3">Predictive Alerts</h3>
              <div className="space-y-2">
                {alerts.filter(a => a.status === 'active').map(alert => (
                  <div key={alert.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-3">
                        <span className={`px-2 py-1 text-xs rounded ${getSeverityColor(alert.severity)}`}>
                          {alert.severity.toUpperCase()}
                        </span>
                        <div className="flex-1">
                          <h4 className="text-sm font-bold mb-1">{alert.prediction}</h4>
                          <p className="text-xs text-gray-400 mb-2">{alert.impact}</p>
                          <div className="flex items-center space-x-4 text-xs text-gray-500">
                            <span>Model: {alert.modelName}</span>
                            <span>Probability: {(alert.probability * 100).toFixed(0)}%</span>
                            <span>{alert.expectedDate}</span>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-gray-500">{alert.timestamp}</p>
                        <button className="mt-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                          View Details
                        </button>
                      </div>
                    </div>
                    <div className="mt-3 p-2 bg-gray-800 rounded text-xs">
                      <Info className="w-3 h-3 inline mr-1" />
                      <span className="text-gray-400">Recommendation: </span>
                      <span>{alert.recommendation}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model Performance Overview */}
            <div className="grid grid-cols-3 gap-4">
              {models.slice(0, 3).map(model => (
                <div key={model.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      {getModelTypeIcon(model.type)}
                      <h3 className="text-sm font-bold">{model.name}</h3>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded ${getStatusColor(model.status)}`}>
                      {model.status.toUpperCase()}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <div>
                      <p className="text-xs text-gray-400">Accuracy</p>
                      <p className={`text-lg font-bold ${getAccuracyColor(model.accuracy)}`}>
                        {model.accuracy.toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Confidence</p>
                      <p className="text-lg font-bold">{model.confidence.toFixed(1)}%</p>
                    </div>
                  </div>
                  
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">1h Prediction</span>
                      <span>{model.predictions.shortTerm.value} ({(model.predictions.shortTerm.probability * 100).toFixed(0)}%)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">24h Prediction</span>
                      <span>{model.predictions.mediumTerm.value} ({(model.predictions.mediumTerm.probability * 100).toFixed(0)}%)</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'models' && (
          <div className="space-y-4">
            {filteredModels.map(model => (
              <div key={model.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    {getModelTypeIcon(model.type)}
                    <div>
                      <h3 className="text-sm font-bold">{model.name}</h3>
                      <p className="text-xs text-gray-400">ID: {model.id} • Type: {model.type}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className={`px-2 py-1 text-xs rounded ${getStatusColor(model.status)}`}>
                      {model.status.toUpperCase()}
                    </span>
                    <span className="text-xs text-gray-500">Last trained: {model.lastTrained}</span>
                  </div>
                </div>
                
                <div className="grid grid-cols-4 gap-4 mb-4">
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Accuracy</p>
                    <p className={`text-xl font-bold ${getAccuracyColor(model.accuracy)}`}>
                      {model.accuracy.toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Confidence</p>
                    <p className="text-xl font-bold">{model.confidence.toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Precision</p>
                    <p className="text-xl font-bold">{(model.metrics.precision * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">R² Score</p>
                    <p className="text-xl font-bold">{model.metrics.r2Score.toFixed(3)}</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-4 p-3 bg-gray-800 rounded">
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Short Term ({model.predictions.shortTerm.timeframe})</p>
                    <p className="text-sm font-bold">
                      {model.predictions.shortTerm.value} 
                      <span className="text-xs text-gray-500 ml-1">
                        ({(model.predictions.shortTerm.probability * 100).toFixed(0)}%)
                      </span>
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Medium Term ({model.predictions.mediumTerm.timeframe})</p>
                    <p className="text-sm font-bold">
                      {model.predictions.mediumTerm.value}
                      <span className="text-xs text-gray-500 ml-1">
                        ({(model.predictions.mediumTerm.probability * 100).toFixed(0)}%)
                      </span>
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Long Term ({model.predictions.longTerm.timeframe})</p>
                    <p className="text-sm font-bold">
                      {model.predictions.longTerm.value}
                      <span className="text-xs text-gray-500 ml-1">
                        ({(model.predictions.longTerm.probability * 100).toFixed(0)}%)
                      </span>
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'forecasts' && (
          <div className="space-y-6">
            {forecasts.map((forecast, idx) => (
              <div key={idx} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold">{forecast.category} Forecast</h3>
                  <div className="flex items-center space-x-2">
                    {forecast.trend === 'increasing' ? 
                      <TrendingUp className="w-4 h-4 text-red-500" /> : 
                      forecast.trend === 'decreasing' ?
                      <TrendingDown className="w-4 h-4 text-green-500" /> :
                      <Activity className="w-4 h-4 text-gray-500" />
                    }
                    <span className="text-xs text-gray-500">Trend: {forecast.trend}</span>
                    {forecast.seasonality && <span className="px-2 py-1 bg-blue-900/20 text-blue-500 rounded text-xs">Seasonal</span>}
                  </div>
                </div>
                
                {/* Historical vs Predicted Chart */}
                <div className="h-32 flex items-end space-x-1">
                  {/* Historical Data */}
                  {forecast.historical.map((point, idx) => (
                    <div key={idx} className="flex-1 flex flex-col items-center">
                      <div 
                        className="w-full bg-blue-500 rounded-t"
                        style={{
                          height: `${(point.value / Math.max(...forecast.historical.map(h => h.value), ...forecast.predicted.map(p => p.value))) * 128}px`
                        }}
                      />
                      <span className="text-xs text-gray-500 mt-1">{point.timestamp}</span>
                    </div>
                  ))}
                  
                  {/* Predicted Data */}
                  {forecast.predicted.map((point, idx) => (
                    <div key={idx} className="flex-1 flex flex-col items-center">
                      <div className="relative w-full">
                        <div 
                          className="w-full bg-green-500 rounded-t opacity-70"
                          style={{
                            height: `${(point.value / Math.max(...forecast.historical.map(h => h.value), ...forecast.predicted.map(p => p.value))) * 128}px`
                          }}
                        />
                        <div className="absolute -top-6 left-1/2 transform -translate-x-1/2">
                          <span className="text-xs text-green-500">{(point.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                      <span className="text-xs text-gray-500 mt-1">{point.timestamp}</span>
                    </div>
                  ))}
                </div>
                
                <div className="mt-3 flex items-center space-x-4 text-xs">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-blue-500 rounded" />
                    <span className="text-gray-400">Historical</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-green-500 rounded opacity-70" />
                    <span className="text-gray-400">Predicted</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'anomalies' && (
          <div className="space-y-3">
            {anomalies.map(anomaly => (
              <div key={anomaly.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    <span className={`px-2 py-1 text-xs rounded ${getSeverityColor(anomaly.severity)}`}>
                      {anomaly.severity.toUpperCase()}
                    </span>
                    <div className="flex-1">
                      <h4 className="text-sm font-bold mb-1">
                        Anomaly in {anomaly.metric} ({anomaly.source})
                      </h4>
                      <div className="grid grid-cols-3 gap-4 text-xs mb-2">
                        <div>
                          <span className="text-gray-400">Expected: </span>
                          <span>{anomaly.expectedValue}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Actual: </span>
                          <span className={anomaly.actualValue > anomaly.expectedValue ? 'text-red-500' : 'text-blue-500'}>
                            {anomaly.actualValue}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-400">Deviation: </span>
                          <span className={anomaly.deviation > 0 ? 'text-red-500' : 'text-blue-500'}>
                            {anomaly.deviation > 0 ? '+' : ''}{anomaly.deviation.toFixed(2)}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>Score: {(anomaly.anomalyScore * 100).toFixed(0)}%</span>
                        <span>Pattern: {anomaly.pattern.replace('_', ' ')}</span>
                        <span>{anomaly.timestamp}</span>
                      </div>
                      {anomaly.rootCause && (
                        <div className="mt-2 p-2 bg-gray-800 rounded text-xs">
                          <span className="text-gray-400">Root Cause: </span>
                          <span>{anomaly.rootCause}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}