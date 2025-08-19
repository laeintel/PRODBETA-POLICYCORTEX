'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import AuthGuard from '../../../components/AuthGuard';
import { api } from '../../../lib/api-client';
import toast from 'react-hot-toast';
import { 
  Brain, Sparkles, TrendingUp, Activity, Zap, Target, MessageSquare, AlertTriangle, 
  Settings, Download, Upload, Play, Pause, RefreshCw, Database, Cpu, Eye,
  BarChart3, LineChart, PieChart, TrendingDown, Clock, Shield, Users, 
  GitBranch, Code, Monitor, Server, Cloud, ChevronRight, Filter, Search,
  Calendar, Globe, Layers, Network, Bot, Microscope, FlaskConical
} from 'lucide-react';

interface AIModel {
  id: string;
  name: string;
  type: 'classification' | 'regression' | 'clustering' | 'nlp' | 'computer_vision';
  status: 'active' | 'training' | 'inactive' | 'error';
  accuracy: number;
  lastTrained: string;
  version: string;
  size: string;
  framework: string;
  deployment: string;
  usage: number;
  cost: number;
}

interface AIExperiment {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed' | 'queued';
  progress: number;
  startTime: string;
  estimatedCompletion: string;
  metrics: {
    accuracy: number;
    loss: number;
    f1Score: number;
    recall: number;
    precision: number;
  };
  hyperparameters: Record<string, any>;
}

interface AIMetrics {
  modelsActive: number;
  predictionsToday: number;
  accuracy: number;
  responseTime: number;
  totalModels: number;
  experimentsRunning: number;
  dataProcessed: number;
  computeUtilization: number;
  costThisMonth: number;
  predictionLatency: number;
  models: AIModel[];
  experiments: AIExperiment[];
  recommendations: Array<{
    id: string;
    type: string;
    confidence: number;
    impact: 'high' | 'medium' | 'low';
    description: string;
    status: 'pending' | 'applied' | 'rejected';
    estimatedSavings?: number;
    priority: number;
  }>;
  predictions: Array<{
    id: string;
    category: string;
    prediction: string;
    probability: number;
    timeframe: string;
    risk: 'critical' | 'high' | 'medium' | 'low';
    confidence: number;
    tags: string[];
  }>;
  training: {
    lastUpdate: string;
    dataPoints: number;
    modelVersion: string;
    nextTraining: string;
    trainingTime: number;
    epochs: number;
    batchSize: number;
  };
  performance: {
    cpuUsage: number[];
    memoryUsage: number[];
    gpuUsage: number[];
    timestamps: string[];
  };
  datasets: Array<{
    id: string;
    name: string;
    size: number;
    lastUpdated: string;
    quality: number;
    type: string;
  }>;
}

export default function AIAnalyticsCenter() {
  return (
    <AuthGuard requireAuth={true}>
      <AIAnalyticsCenterContent />
    </AuthGuard>
  );
}

function AIAnalyticsCenterContent() {
  const [metrics, setMetrics] = useState<AIMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'models' | 'experiments' | 'predictions' | 'chat' | 'datasets'>('dashboard');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [conversationInput, setConversationInput] = useState('');
  const [conversationHistory, setConversationHistory] = useState<Array<{role: string, message: string}>>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'accuracy' | 'lastTrained' | 'usage' | 'cost'>('accuracy');
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    fetchAIMetrics();
    const interval = setInterval(fetchAIMetrics, 15000);
    return () => clearInterval(interval);
  }, []);

  const fetchAIMetrics = async () => {
    try {
      const resp = await api.getAIPredictions()
      if (resp.error) {
        setMetrics(getMockAIMetrics());
      } else {
        setMetrics(processAIData(resp.data));
      }
    } catch (error) {
      setMetrics(getMockAIMetrics());
    } finally {
      setLoading(false);
    }
  };

  const triggerAction = async (actionType: string, params?: any) => {
    try {
      const resp = await api.createAction('global', actionType, params)
      if (resp.error || resp.status >= 400) {
        toast.error(`Action failed: ${actionType}`)
        return
      }
      toast.success(`${actionType.replace('_',' ')} started`)
      const id = resp.data?.action_id || resp.data?.id
      if (id) {
        const stop = api.streamActionEvents(String(id), (m) => console.log('[ai-action]', id, m))
        setTimeout(stop, 60000)
      }
    } catch (e) {
      toast.error(`Action error: ${actionType}`)
    }
  }

  const processAIData = (data: any): AIMetrics => {
    const base = getMockAIMetrics()
    return {
      ...base,
      // Optionally override selected fields from API data
      recommendations: data?.recommendations || base.recommendations,
      predictions: data?.predictions || base.predictions,
    }
  };

  const getMockAIMetrics = (): AIMetrics => ({
    modelsActive: 12,
    predictionsToday: 14847,
    accuracy: 94.7,
    responseTime: 87,
    totalModels: 24,
    experimentsRunning: 5,
    dataProcessed: 2847392,
    computeUtilization: 78,
    costThisMonth: 12450,
    predictionLatency: 23,
    models: [
      { id: 'm1', name: 'Cost Predictor v3.2', type: 'regression', status: 'active', accuracy: 96.4, lastTrained: '2 days ago', version: 'v3.2.1', size: '245MB', framework: 'TensorFlow', deployment: 'Azure ML', usage: 87, cost: 234.50 },
      { id: 'm2', name: 'Security Anomaly Detector', type: 'classification', status: 'active', accuracy: 94.2, lastTrained: '1 day ago', version: 'v2.1.0', size: '189MB', framework: 'PyTorch', deployment: 'AKS', usage: 92, cost: 189.25 },
      { id: 'm3', name: 'Compliance Risk Engine', type: 'classification', status: 'training', accuracy: 88.7, lastTrained: '5 days ago', version: 'v1.8.3', size: '312MB', framework: 'Scikit-learn', deployment: 'Azure Functions', usage: 65, cost: 156.75 },
      { id: 'm4', name: 'Resource Optimizer', type: 'clustering', status: 'active', accuracy: 91.3, lastTrained: '3 days ago', version: 'v2.5.2', size: '178MB', framework: 'XGBoost', deployment: 'Azure Container Instances', usage: 78, cost: 98.40 },
      { id: 'm5', name: 'Performance Predictor', type: 'regression', status: 'active', accuracy: 93.1, lastTrained: '1 day ago', version: 'v4.1.1', size: '267MB', framework: 'LightGBM', deployment: 'Azure ML', usage: 85, cost: 201.30 },
      { id: 'm6', name: 'Threat Intelligence NLP', type: 'nlp', status: 'inactive', accuracy: 89.6, lastTrained: '7 days ago', version: 'v1.2.4', size: '445MB', framework: 'Transformers', deployment: 'Azure Cognitive Services', usage: 23, cost: 67.20 }
    ],
    experiments: [
      { id: 'e1', name: 'Cost Model Optimization', status: 'running', progress: 67, startTime: '3 hours ago', estimatedCompletion: '2 hours', metrics: { accuracy: 89.2, loss: 0.23, f1Score: 0.91, recall: 0.88, precision: 0.94 }, hyperparameters: { learningRate: 0.001, batchSize: 64, epochs: 100 } },
      { id: 'e2', name: 'Security Pattern Recognition', status: 'queued', progress: 0, startTime: '', estimatedCompletion: '6 hours', metrics: { accuracy: 0, loss: 0, f1Score: 0, recall: 0, precision: 0 }, hyperparameters: { learningRate: 0.0005, batchSize: 32, epochs: 150 } },
      { id: 'e3', name: 'Multi-Modal Risk Assessment', status: 'completed', progress: 100, startTime: '2 days ago', estimatedCompletion: 'completed', metrics: { accuracy: 93.7, loss: 0.18, f1Score: 0.94, recall: 0.92, precision: 0.96 }, hyperparameters: { learningRate: 0.002, batchSize: 128, epochs: 200 } },
      { id: 'e4', name: 'Real-time Anomaly Detection', status: 'failed', progress: 45, startTime: '1 day ago', estimatedCompletion: 'failed', metrics: { accuracy: 76.3, loss: 0.45, f1Score: 0.72, recall: 0.68, precision: 0.79 }, hyperparameters: { learningRate: 0.01, batchSize: 256, epochs: 75 } },
      { id: 'e5', name: 'Predictive Scaling Model', status: 'running', progress: 23, startTime: '1 hour ago', estimatedCompletion: '8 hours', metrics: { accuracy: 82.1, loss: 0.31, f1Score: 0.79, recall: 0.75, precision: 0.84 }, hyperparameters: { learningRate: 0.0008, batchSize: 64, epochs: 300 } }
    ],
    datasets: [
      { id: 'd1', name: 'Azure Resource Metrics', size: 2.3, lastUpdated: '2 hours ago', quality: 98, type: 'Time Series' },
      { id: 'd2', name: 'Security Event Logs', size: 5.7, lastUpdated: '30 minutes ago', quality: 95, type: 'Log Data' },
      { id: 'd3', name: 'Cost & Billing Data', size: 1.8, lastUpdated: '4 hours ago', quality: 99, type: 'Structured' },
      { id: 'd4', name: 'Compliance Scan Results', size: 0.9, lastUpdated: '1 day ago', quality: 92, type: 'Structured' },
      { id: 'd5', name: 'Performance Benchmarks', size: 3.4, lastUpdated: '6 hours ago', quality: 97, type: 'Time Series' }
    ],
    recommendations: [
      { id: 'r1', type: 'Cost Optimization', confidence: 92, impact: 'high', description: 'Resize 12 overprovisioned VMs to save $3,450/month', status: 'pending', estimatedSavings: 3450, priority: 1 },
      { id: 'r2', type: 'Security Enhancement', confidence: 88, impact: 'high', description: 'Enable MFA for 5 privileged accounts', status: 'pending', priority: 1 },
      { id: 'r3', type: 'Performance', confidence: 85, impact: 'medium', description: 'Implement caching for frequently accessed storage', status: 'applied', priority: 2 },
      { id: 'r4', type: 'Compliance', confidence: 95, impact: 'high', description: 'Update 3 policies to meet new regulations', status: 'pending', priority: 1 },
      { id: 'r5', type: 'Resource Allocation', confidence: 78, impact: 'low', description: 'Redistribute workloads across regions', status: 'rejected', estimatedSavings: 1200, priority: 3 },
      { id: 'r6', type: 'Automation', confidence: 91, impact: 'medium', description: 'Automate 8 recurring maintenance tasks', status: 'pending', estimatedSavings: 2800, priority: 2 }
    ],
    predictions: [
      { id: 'p1', category: 'Cost', prediction: 'Budget overrun likely in Q4', probability: 87, timeframe: '7 days', risk: 'high', confidence: 94, tags: ['budget', 'quarterly', 'spending'] },
      { id: 'p2', category: 'Security', prediction: 'DDoS attack pattern detected', probability: 42, timeframe: '24 hours', risk: 'medium', confidence: 78, tags: ['ddos', 'attack', 'network'] },
      { id: 'p3', category: 'Compliance', prediction: 'Policy drift expected in 3 subscriptions', probability: 65, timeframe: '14 days', risk: 'medium', confidence: 82, tags: ['policy', 'drift', 'compliance'] },
      { id: 'p4', category: 'Performance', prediction: 'Resource bottleneck forming in West US', probability: 78, timeframe: '3 days', risk: 'high', confidence: 88, tags: ['performance', 'bottleneck', 'westus'] },
      { id: 'p5', category: 'Availability', prediction: 'Service degradation possible', probability: 23, timeframe: '48 hours', risk: 'low', confidence: 67, tags: ['availability', 'degradation', 'service'] },
      { id: 'p6', category: 'Storage', prediction: 'Storage quota breach imminent', probability: 91, timeframe: '5 days', risk: 'critical', confidence: 96, tags: ['storage', 'quota', 'capacity'] }
    ],
    training: {
      lastUpdate: '2 days ago',
      dataPoints: 2847392,
      modelVersion: 'v2.4.1',
      nextTraining: 'in 5 days',
      trainingTime: 4.2,
      epochs: 150,
      batchSize: 128
    },
    performance: {
      cpuUsage: [45, 52, 48, 61, 58, 72, 69, 73, 78, 75, 71, 68, 74, 79, 82, 78, 75, 73, 71, 69],
      memoryUsage: [38, 42, 45, 48, 52, 55, 58, 62, 65, 63, 60, 58, 61, 64, 67, 65, 62, 59, 57, 55],
      gpuUsage: [23, 28, 31, 35, 42, 48, 52, 58, 61, 59, 56, 53, 57, 62, 65, 63, 60, 57, 54, 51],
      timestamps: Array.from({ length: 20 }, (_, i) => new Date(Date.now() - (19 - i) * 60000).toLocaleTimeString())
    }
  });

  const handleConversation = async () => {
    if (!conversationInput.trim()) return;
    
    const userMessage = conversationInput;
    setConversationHistory(prev => [...prev, { role: 'user', message: userMessage }]);
    setConversationInput('');
    
    // Simulate AI response
    setTimeout(() => {
      const aiResponse = `Based on my analysis of your Azure environment, ${userMessage.toLowerCase().includes('cost') 
        ? 'I recommend implementing reserved instances for your production VMs, which could save approximately $12,450 per month.'
        : userMessage.toLowerCase().includes('security')
        ? 'I\'ve identified 3 critical security configurations that need immediate attention: enable MFA, update NSG rules, and rotate service principal keys.'
        : 'I\'ve analyzed your request and identified several optimization opportunities across your infrastructure.'}`;
      
      setConversationHistory(prev => [...prev, { role: 'ai', message: aiResponse }]);
    }, 1000);
  };

  // Add computed values for filtering and sorting
  const filteredModels = useMemo(() => {
    if (!metrics?.models) return [];
    
    let filtered = metrics.models.filter(model => 
      (filterStatus === 'all' || model.status === filterStatus) &&
      (searchQuery === '' || model.name.toLowerCase().includes(searchQuery.toLowerCase()))
    );
    
    return filtered.sort((a, b) => {
      switch (sortBy) {
        case 'accuracy': return b.accuracy - a.accuracy;
        case 'lastTrained': return new Date(b.lastTrained).getTime() - new Date(a.lastTrained).getTime();
        case 'usage': return b.usage - a.usage;
        case 'cost': return b.cost - a.cost;
        default: return 0;
      }
    });
  }, [metrics?.models, filterStatus, searchQuery, sortBy]);

  const renderPerformanceChart = () => {
    if (!metrics?.performance) return null;
    
    const maxValue = Math.max(
      ...metrics.performance.cpuUsage, 
      ...metrics.performance.memoryUsage, 
      ...metrics.performance.gpuUsage
    );
    
    return (
      <div className="h-48 flex items-end justify-between space-x-1">
        {metrics.performance.timestamps.map((timestamp, index) => (
          <div key={index} className="flex flex-col items-center space-y-1 flex-1">
            <div className="flex flex-col justify-end h-40 w-full space-y-0.5">
              <div 
                className="bg-blue-500 rounded-sm transition-all duration-300 hover:bg-blue-400"
                style={{ height: `${(metrics.performance.cpuUsage[index] / maxValue) * 100}%` }}
                title={`CPU: ${metrics.performance.cpuUsage[index]}%`}
              />
              <div 
                className="bg-green-500 rounded-sm transition-all duration-300 hover:bg-green-400"
                style={{ height: `${(metrics.performance.memoryUsage[index] / maxValue) * 100}%` }}
                title={`Memory: ${metrics.performance.memoryUsage[index]}%`}
              />
              <div 
                className="bg-purple-500 rounded-sm transition-all duration-300 hover:bg-purple-400"
                style={{ height: `${(metrics.performance.gpuUsage[index] / maxValue) * 100}%` }}
                title={`GPU: ${metrics.performance.gpuUsage[index]}%`}
              />
            </div>
            <span className="text-xs text-gray-500 transform rotate-45 whitespace-nowrap">
              {timestamp}
            </span>
          </div>
        ))}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 border-4 border-purple-600 border-t-transparent rounded-full animate-spin mx-auto mb-6" />
          <div className="space-y-2">
            <div className="flex items-center justify-center space-x-2">
              <Brain className="w-5 h-5 text-purple-500 animate-pulse" />
              <p className="text-lg font-bold text-purple-400">INITIALIZING AI SYSTEMS</p>
            </div>
            <p className="text-sm text-gray-500">Loading models and neural networks...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/tactical" className="text-gray-400 hover:text-white transition-colors flex items-center space-x-2">
                <ChevronRight className="w-4 h-4 rotate-180" />
                <span>TACTICAL</span>
              </Link>
              <div className="h-6 w-px bg-gray-700" />
              <div className="flex items-center space-x-3">
                <Brain className="w-6 h-6 text-purple-500 animate-pulse" />
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                  AI ANALYTICS CENTER
                </h1>
              </div>
              <div className="flex items-center space-x-2">
                <div className="px-3 py-1 bg-purple-900/30 text-purple-400 rounded-full text-xs font-bold animate-pulse border border-purple-800/30">
                  {metrics?.modelsActive} ACTIVE
                </div>
                <div className="px-3 py-1 bg-blue-900/30 text-blue-400 rounded-full text-xs font-bold border border-blue-800/30">
                  {metrics?.experimentsRunning} TRAINING
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`p-2 rounded-lg transition-all duration-200 ${
                    autoRefresh ? 'bg-green-900/30 text-green-400 border border-green-800/30' : 'bg-gray-800 text-gray-400 border border-gray-700'
                  }`}
                >
                  <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
                </button>
                <select
                  value={selectedTimeRange}
                  onChange={(e) => setSelectedTimeRange(e.target.value as any)}
                  className="px-3 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                >
                  <option value="1h">1 Hour</option>
                  <option value="24h">24 Hours</option>
                  <option value="7d">7 Days</option>
                  <option value="30d">30 Days</option>
                </select>
              </div>
              <button 
                onClick={() => triggerAction('train_models')} 
                className="px-6 py-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold rounded-lg transition-all duration-200 transform hover:scale-105 flex items-center space-x-2 shadow-lg shadow-purple-900/25"
              >
                <FlaskConical className="w-4 h-4" />
                <span>TRAIN MODELS</span>
              </button>
              <button 
                onClick={() => triggerAction('export_ai_insights')} 
                className="px-6 py-2 bg-gray-800 hover:bg-gray-700 text-white font-semibold rounded-lg border border-gray-700 transition-all duration-200 flex items-center space-x-2"
              >
                <Download className="w-4 h-4" />
                <span>EXPORT</span>
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 pb-4">
          <div className="flex space-x-1 bg-gray-800/50 rounded-lg p-1">
            {[
              { key: 'dashboard', label: 'Dashboard', icon: BarChart3 },
              { key: 'models', label: 'Models', icon: Brain },
              { key: 'experiments', label: 'Experiments', icon: FlaskConical },
              { key: 'predictions', label: 'Predictions', icon: Target },
              { key: 'chat', label: 'AI Chat', icon: MessageSquare },
              { key: 'datasets', label: 'Datasets', icon: Database }
            ].map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key as any)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                  activeTab === key
                    ? 'bg-purple-600 text-white shadow-lg'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="p-6 space-y-6">
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-purple-800/50 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-gray-500 uppercase font-semibold">Active Models</p>
                  <Brain className="w-5 h-5 text-purple-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-purple-400">{metrics?.modelsActive}</p>
                <p className="text-xs text-gray-500 mt-1">of {metrics?.totalModels} total</p>
              </div>
              
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-blue-800/50 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-gray-500 uppercase font-semibold">Predictions Today</p>
                  <Sparkles className="w-5 h-5 text-blue-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-blue-400">{metrics?.predictionsToday.toLocaleString()}</p>
                <p className="text-xs text-green-400 mt-1">↑ 23% from yesterday</p>
              </div>
              
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-green-800/50 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-gray-500 uppercase font-semibold">Avg Accuracy</p>
                  <Target className="w-5 h-5 text-green-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-green-400">{metrics?.accuracy}%</p>
                <div className="mt-2 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-green-600 to-green-400 rounded-full transition-all duration-1000" 
                    style={{ width: `${metrics?.accuracy}%` }} 
                  />
                </div>
              </div>
              
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-yellow-800/50 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-gray-500 uppercase font-semibold">Response Time</p>
                  <Zap className="w-5 h-5 text-yellow-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-yellow-400">{metrics?.responseTime}<span className="text-lg">ms</span></p>
                <p className="text-xs text-gray-500 mt-1">avg inference</p>
              </div>
              
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-red-800/50 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-gray-500 uppercase font-semibold">Monthly Cost</p>
                  <TrendingUp className="w-5 h-5 text-red-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-red-400">${(metrics?.costThisMonth || 0).toLocaleString()}</p>
                <p className="text-xs text-gray-500 mt-1">this month</p>
              </div>
              
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-cyan-800/50 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-gray-500 uppercase font-semibold">Data Processed</p>
                  <Database className="w-5 h-5 text-cyan-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-cyan-400">{((metrics?.dataProcessed || 0) / 1000000).toFixed(1)}<span className="text-lg">M</span></p>
                <p className="text-xs text-gray-500 mt-1">data points</p>
              </div>
            </div>

            {/* Performance Chart */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl">
              <div className="p-6 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-white">SYSTEM PERFORMANCE</h3>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2 text-sm">
                      <div className="w-3 h-3 bg-blue-500 rounded"></div>
                      <span className="text-gray-400">CPU</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm">
                      <div className="w-3 h-3 bg-green-500 rounded"></div>
                      <span className="text-gray-400">Memory</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm">
                      <div className="w-3 h-3 bg-purple-500 rounded"></div>
                      <span className="text-gray-400">GPU</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="p-6">
                {renderPerformanceChart()}
              </div>
            </div>

            {/* Quick Stats Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Top Predictions */}
              <div className="bg-gray-900 border border-gray-800 rounded-xl">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 text-red-500" />
                    <span>CRITICAL PREDICTIONS</span>
                  </h3>
                </div>
                <div className="divide-y divide-gray-800">
                  {metrics?.predictions.filter(p => p.risk === 'critical' || p.risk === 'high').slice(0, 4).map((prediction) => (
                    <div key={prediction.id} className="p-4 hover:bg-gray-800/50 transition-colors">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-2">
                            <span className={`text-xs px-2 py-1 rounded font-bold ${
                              prediction.risk === 'critical' ? 'bg-red-900/50 text-red-400' :
                              prediction.risk === 'high' ? 'bg-orange-900/50 text-orange-400' :
                              'bg-yellow-900/50 text-yellow-400'
                            }`}>
                              {prediction.category.toUpperCase()}
                            </span>
                            <span className="text-xs text-gray-500">{prediction.timeframe}</span>
                          </div>
                          <h4 className="font-medium text-white mb-1">{prediction.prediction}</h4>
                          <div className="flex flex-wrap gap-1">
                            {prediction.tags.map((tag, idx) => (
                              <span key={idx} className="text-xs px-2 py-0.5 bg-gray-800 text-gray-400 rounded">
                                #{tag}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div className="text-right ml-4">
                          <p className="text-2xl font-bold font-mono text-white">{prediction.probability}%</p>
                          <p className="text-xs text-gray-500">confidence</p>
                        </div>
                      </div>
                      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-1000 ${
                            prediction.probability >= 80 ? 'bg-red-500' :
                            prediction.probability >= 60 ? 'bg-yellow-500' :
                            'bg-blue-500'
                          }`}
                          style={{ width: `${prediction.probability}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Top Recommendations */}
              <div className="bg-gray-900 border border-gray-800 rounded-xl">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase flex items-center space-x-2">
                    <Sparkles className="w-4 h-4 text-purple-500" />
                    <span>AI RECOMMENDATIONS</span>
                  </h3>
                </div>
                <div className="divide-y divide-gray-800">
                  {metrics?.recommendations.filter(r => r.status === 'pending').slice(0, 4).map((rec) => (
                    <div key={rec.id} className="p-4 hover:bg-gray-800/50 transition-colors">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-sm font-medium text-white">{rec.type}</span>
                            <span className={`text-xs px-2 py-0.5 rounded font-bold ${
                              rec.impact === 'high' ? 'bg-red-900/50 text-red-400' :
                              rec.impact === 'medium' ? 'bg-yellow-900/50 text-yellow-400' :
                              'bg-gray-800 text-gray-400'
                            }`}>
                              {rec.impact.toUpperCase()}
                            </span>
                          </div>
                          <p className="text-sm text-gray-400 mb-2">{rec.description}</p>
                          <div className="flex items-center gap-4">
                            <div className="flex items-center gap-1">
                              <Brain className="w-3 h-3 text-purple-400" />
                              <span className="text-xs text-gray-500">{rec.confidence}% confidence</span>
                            </div>
                            {rec.estimatedSavings && (
                              <div className="flex items-center gap-1">
                                <TrendingUp className="w-3 h-3 text-green-400" />
                                <span className="text-xs text-green-400">${rec.estimatedSavings}/mo</span>
                              </div>
                            )}
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <button className="px-3 py-1 bg-green-900/30 hover:bg-green-900/50 border border-green-800 rounded text-green-400 text-xs transition-colors">
                            APPLY
                          </button>
                          <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-gray-400 text-xs transition-colors">
                            DISMISS
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Models Tab */}
        {activeTab === 'models' && (
          <div className="space-y-6">
            {/* Controls */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="flex flex-col lg:flex-row lg:items-center justify-between space-y-4 lg:space-y-0">
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500" />
                    <input
                      type="text"
                      placeholder="Search models..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm w-80"
                    />
                  </div>
                  <select
                    value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value)}
                    className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm"
                  >
                    <option value="all">All Status</option>
                    <option value="active">Active</option>
                    <option value="training">Training</option>
                    <option value="inactive">Inactive</option>
                    <option value="error">Error</option>
                  </select>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as any)}
                    className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm"
                  >
                    <option value="accuracy">Sort by Accuracy</option>
                    <option value="lastTrained">Sort by Last Trained</option>
                    <option value="usage">Sort by Usage</option>
                    <option value="cost">Sort by Cost</option>
                  </select>
                </div>
                <div className="flex items-center space-x-2">
                  <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center space-x-2">
                    <Upload className="w-4 h-4" />
                    <span>Deploy Model</span>
                  </button>
                  <button className="px-4 py-2 bg-gray-800 border border-gray-700 hover:bg-gray-700 text-white rounded-lg transition-colors">
                    <Settings className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Models Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              {filteredModels.map((model) => (
                <div key={model.id} className="bg-gray-900 border border-gray-800 rounded-xl hover:border-purple-800/50 transition-all duration-200">
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <h3 className="font-bold text-white text-lg mb-1">{model.name}</h3>
                        <div className="flex items-center space-x-2 mb-2">
                          <span className="text-xs px-2 py-1 bg-gray-800 text-gray-400 rounded">
                            {model.type.toUpperCase()}
                          </span>
                          <span className={`text-xs px-2 py-1 rounded font-bold ${
                            model.status === 'active' ? 'bg-green-900/50 text-green-400' :
                            model.status === 'training' ? 'bg-yellow-900/50 text-yellow-400' :
                            model.status === 'inactive' ? 'bg-gray-800 text-gray-500' :
                            'bg-red-900/50 text-red-400'
                          }`}>
                            {model.status.toUpperCase()}
                          </span>
                        </div>
                        <div className="space-y-1 text-sm text-gray-400">
                          <div>Framework: <span className="text-white">{model.framework}</span></div>
                          <div>Version: <span className="text-white">{model.version}</span></div>
                          <div>Size: <span className="text-white">{model.size}</span></div>
                          <div>Deployment: <span className="text-white">{model.deployment}</span></div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-xs text-gray-500">Accuracy</span>
                          <span className="text-xs font-mono text-white">{model.accuracy}%</span>
                        </div>
                        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-green-600 to-green-400 rounded-full transition-all duration-1000" 
                            style={{ width: `${model.accuracy}%` }} 
                          />
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-xs text-gray-500">Usage</span>
                          <span className="text-xs font-mono text-white">{model.usage}%</span>
                        </div>
                        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all duration-1000" 
                            style={{ width: `${model.usage}%` }} 
                          />
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between pt-2 border-t border-gray-800">
                        <div>
                          <p className="text-xs text-gray-500">Monthly Cost</p>
                          <p className="text-lg font-bold font-mono text-white">${model.cost}</p>
                        </div>
                        <div className="flex items-center space-x-2">
                          {model.status === 'active' && (
                            <button className="p-2 bg-red-900/30 hover:bg-red-900/50 border border-red-800 text-red-400 rounded-lg transition-colors">
                              <Pause className="w-4 h-4" />
                            </button>
                          )}
                          {model.status === 'inactive' && (
                            <button className="p-2 bg-green-900/30 hover:bg-green-900/50 border border-green-800 text-green-400 rounded-lg transition-colors">
                              <Play className="w-4 h-4" />
                            </button>
                          )}
                          <button className="p-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-400 rounded-lg transition-colors">
                            <Settings className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Experiments Tab */}
        {activeTab === 'experiments' && (
          <div className="space-y-6">
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold text-white">ACTIVE EXPERIMENTS</h2>
                <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center space-x-2">
                  <FlaskConical className="w-4 h-4" />
                  <span>New Experiment</span>
                </button>
              </div>
            </div>

            <div className="space-y-4">
              {metrics?.experiments.map((experiment) => (
                <div key={experiment.id} className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <h3 className="text-lg font-bold text-white">{experiment.name}</h3>
                        <span className={`text-xs px-2 py-1 rounded font-bold ${
                          experiment.status === 'running' ? 'bg-blue-900/50 text-blue-400 animate-pulse' :
                          experiment.status === 'completed' ? 'bg-green-900/50 text-green-400' :
                          experiment.status === 'failed' ? 'bg-red-900/50 text-red-400' :
                          'bg-gray-800 text-gray-400'
                        }`}>
                          {experiment.status.toUpperCase()}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500">Started:</span>
                          <p className="text-white">{experiment.startTime || 'Not started'}</p>
                        </div>
                        <div>
                          <span className="text-gray-500">Completion:</span>
                          <p className="text-white">{experiment.estimatedCompletion}</p>
                        </div>
                        <div>
                          <span className="text-gray-500">Learning Rate:</span>
                          <p className="text-white font-mono">{experiment.hyperparameters.learningRate}</p>
                        </div>
                        <div>
                          <span className="text-gray-500">Batch Size:</span>
                          <p className="text-white font-mono">{experiment.hyperparameters.batchSize}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Progress Bar */}
                  {experiment.status !== 'queued' && (
                    <div className="mb-4">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs text-gray-500">Progress</span>
                        <span className="text-xs font-mono text-white">{experiment.progress}%</span>
                      </div>
                      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className={`h-full rounded-full transition-all duration-1000 ${
                            experiment.status === 'failed' ? 'bg-red-500' :
                            experiment.status === 'completed' ? 'bg-green-500' :
                            'bg-blue-500'
                          }`}
                          style={{ width: `${experiment.progress}%` }} 
                        />
                      </div>
                    </div>
                  )}
                  
                  {/* Metrics */}
                  {experiment.metrics.accuracy > 0 && (
                    <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
                      <div className="text-center">
                        <p className="text-xs text-gray-500 uppercase">Accuracy</p>
                        <p className="text-lg font-bold font-mono text-green-400">{experiment.metrics.accuracy}%</p>
                      </div>
                      <div className="text-center">
                        <p className="text-xs text-gray-500 uppercase">Loss</p>
                        <p className="text-lg font-bold font-mono text-red-400">{experiment.metrics.loss}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-xs text-gray-500 uppercase">F1 Score</p>
                        <p className="text-lg font-bold font-mono text-blue-400">{experiment.metrics.f1Score}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-xs text-gray-500 uppercase">Recall</p>
                        <p className="text-lg font-bold font-mono text-purple-400">{experiment.metrics.recall}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-xs text-gray-500 uppercase">Precision</p>
                        <p className="text-lg font-bold font-mono text-yellow-400">{experiment.metrics.precision}</p>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Predictions Tab */}
        {activeTab === 'predictions' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
              {metrics?.predictions.map((prediction) => (
                <div key={prediction.id} className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-purple-800/50 transition-all duration-200">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <span className={`text-xs px-2 py-1 rounded font-bold ${
                        prediction.risk === 'critical' ? 'bg-red-900/50 text-red-400' :
                        prediction.risk === 'high' ? 'bg-orange-900/50 text-orange-400' :
                        prediction.risk === 'medium' ? 'bg-yellow-900/50 text-yellow-400' :
                        'bg-gray-800 text-gray-400'
                      }`}>
                        {prediction.category.toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-500">{prediction.timeframe}</span>
                    </div>
                    <div className="text-right">
                      <p className="text-2xl font-bold font-mono text-white">{prediction.probability}%</p>
                      <p className="text-xs text-gray-500">probability</p>
                    </div>
                  </div>
                  <h4 className="font-medium text-white mb-3">{prediction.prediction}</h4>
                  <div className="flex flex-wrap gap-1 mb-3">
                    {prediction.tags.map((tag, idx) => (
                      <span key={idx} className="text-xs px-2 py-0.5 bg-gray-800 text-gray-400 rounded">
                        #{tag}
                      </span>
                    ))}
                  </div>
                  <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-1000 ${
                        prediction.probability >= 80 ? 'bg-red-500' :
                        prediction.probability >= 60 ? 'bg-yellow-500' :
                        'bg-blue-500'
                      }`}
                      style={{ width: `${prediction.probability}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* AI Chat Tab */}
        {activeTab === 'chat' && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl">
            <div className="p-4 border-b border-gray-800 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Bot className="w-5 h-5 text-purple-500" />
                <h3 className="text-lg font-bold text-white">AI GOVERNANCE ASSISTANT</h3>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-xs text-green-400">ONLINE</span>
              </div>
            </div>
            <div className="p-4">
              <div className="h-96 overflow-y-auto mb-4 space-y-4 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-gray-800">
                {conversationHistory.length === 0 ? (
                  <div className="text-center py-12">
                    <Bot className="w-16 h-16 mx-auto mb-4 text-gray-600" />
                    <p className="text-gray-500 text-lg mb-2">Welcome to the AI Governance Assistant</p>
                    <p className="text-gray-600 text-sm">Ask me anything about your Azure environment, costs, security, compliance, or optimization opportunities.</p>
                  </div>
                ) : (
                  conversationHistory.map((msg, idx) => (
                    <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-2xl px-6 py-4 rounded-xl ${
                        msg.role === 'user' 
                          ? 'bg-blue-900/30 text-blue-300 border border-blue-800/30' 
                          : 'bg-purple-900/30 text-purple-300 border border-purple-800/30'
                      }`}>
                        <p className="text-sm leading-relaxed">{msg.message}</p>
                      </div>
                    </div>
                  ))
                )}
              </div>
              <div className="flex gap-3">
                <input
                  type="text"
                  value={conversationInput}
                  onChange={(e) => setConversationInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleConversation()}
                  placeholder="Ask about costs, security, compliance, predictions, or optimization..."
                  className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:border-purple-600 focus:outline-none"
                />
                <button 
                  onClick={handleConversation}
                  disabled={!conversationInput.trim()}
                  className="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-gray-800 disabled:to-gray-800 text-white font-semibold rounded-lg transition-all duration-200 transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed"
                >
                  ASK AI
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Datasets Tab */}
        {activeTab === 'datasets' && (
          <div className="space-y-6">
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold text-white">DATA SOURCES</h2>
                <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center space-x-2">
                  <Upload className="w-4 h-4" />
                  <span>Import Dataset</span>
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {metrics?.datasets.map((dataset) => (
                <div key={dataset.id} className="bg-gray-900 border border-gray-800 rounded-xl p-6 hover:border-cyan-800/50 transition-all duration-200">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <h3 className="text-lg font-bold text-white mb-1">{dataset.name}</h3>
                      <div className="flex items-center space-x-2 mb-2">
                        <span className="text-xs px-2 py-1 bg-gray-800 text-gray-400 rounded">
                          {dataset.type}
                        </span>
                        <span className="text-xs text-gray-500">
                          Updated {dataset.lastUpdated}
                        </span>
                      </div>
                    </div>
                    <Database className="w-6 h-6 text-cyan-500" />
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm text-gray-400">Data Quality</span>
                        <span className="text-sm font-mono text-white">{dataset.quality}%</span>
                      </div>
                      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-cyan-600 to-cyan-400 rounded-full transition-all duration-1000" 
                          style={{ width: `${dataset.quality}%` }} 
                        />
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between pt-2 border-t border-gray-800">
                      <div>
                        <p className="text-xs text-gray-500">Dataset Size</p>
                        <p className="text-xl font-bold font-mono text-white">{dataset.size} GB</p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <button className="p-2 bg-blue-900/30 hover:bg-blue-900/50 border border-blue-800 text-blue-400 rounded-lg transition-colors">
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="p-2 bg-green-900/30 hover:bg-green-900/50 border border-green-800 text-green-400 rounded-lg transition-colors">
                          <Download className="w-4 h-4" />
                        </button>
                        <button className="p-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-400 rounded-lg transition-colors">
                          <Settings className="w-4 h-4" />
                        </button>
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
  );
}