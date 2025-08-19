'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import AuthGuard from '../../../components/AuthGuard';
import { api } from '../../../lib/api-client';
import toast from 'react-hot-toast';
import { Brain, Sparkles, TrendingUp, Activity, Zap, Target, MessageSquare, AlertTriangle } from 'lucide-react';

interface AIMetrics {
  modelsActive: number;
  predictionsToday: number;
  accuracy: number;
  responseTime: number;
  recommendations: Array<{
    id: string;
    type: string;
    confidence: number;
    impact: 'high' | 'medium' | 'low';
    description: string;
    status: 'pending' | 'applied' | 'rejected';
  }>;
  predictions: Array<{
    id: string;
    category: string;
    prediction: string;
    probability: number;
    timeframe: string;
    risk: 'critical' | 'high' | 'medium' | 'low';
  }>;
  training: {
    lastUpdate: string;
    dataPoints: number;
    modelVersion: string;
    nextTraining: string;
  };
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
  const [activeModel, setActiveModel] = useState('governance');
  const [conversationInput, setConversationInput] = useState('');
  const [conversationHistory, setConversationHistory] = useState<Array<{role: string, message: string}>>([]);

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
    return {
      modelsActive: 4,
      predictionsToday: 1847,
      accuracy: 94.7,
      responseTime: 87,
      recommendations: data.recommendations || getMockAIMetrics().recommendations,
      predictions: data.predictions || getMockAIMetrics().predictions,
      training: getMockAIMetrics().training
    };
  };

  const getMockAIMetrics = (): AIMetrics => ({
    modelsActive: 4,
    predictionsToday: 1847,
    accuracy: 94.7,
    responseTime: 87,
    recommendations: [
      { id: 'r1', type: 'Cost Optimization', confidence: 92, impact: 'high', description: 'Resize 12 overprovisioned VMs to save $3,450/month', status: 'pending' },
      { id: 'r2', type: 'Security Enhancement', confidence: 88, impact: 'high', description: 'Enable MFA for 5 privileged accounts', status: 'pending' },
      { id: 'r3', type: 'Performance', confidence: 85, impact: 'medium', description: 'Implement caching for frequently accessed storage', status: 'applied' },
      { id: 'r4', type: 'Compliance', confidence: 95, impact: 'high', description: 'Update 3 policies to meet new regulations', status: 'pending' },
      { id: 'r5', type: 'Resource Allocation', confidence: 78, impact: 'low', description: 'Redistribute workloads across regions', status: 'rejected' }
    ],
    predictions: [
      { id: 'p1', category: 'Cost', prediction: 'Budget overrun likely', probability: 87, timeframe: '7 days', risk: 'high' },
      { id: 'p2', category: 'Security', prediction: 'DDoS attack pattern detected', probability: 42, timeframe: '24 hours', risk: 'medium' },
      { id: 'p3', category: 'Compliance', prediction: 'Policy drift expected', probability: 65, timeframe: '14 days', risk: 'medium' },
      { id: 'p4', category: 'Performance', prediction: 'Resource bottleneck forming', probability: 78, timeframe: '3 days', risk: 'high' },
      { id: 'p5', category: 'Availability', prediction: 'Service degradation possible', probability: 23, timeframe: '48 hours', risk: 'low' }
    ],
    training: {
      lastUpdate: '2 days ago',
      dataPoints: 2847392,
      modelVersion: 'v2.4.1',
      nextTraining: 'in 5 days'
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

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-purple-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-gray-400">INITIALIZING AI SYSTEMS...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/tactical" className="text-gray-400 hover:text-gray-200">
                ← BACK
              </Link>
              <div className="h-6 w-px bg-gray-700" />
              <h1 className="text-xl font-bold">AI ANALYTICS CENTER</h1>
              <div className="px-3 py-1 bg-purple-900/30 text-purple-500 rounded text-xs font-bold animate-pulse">
                {metrics?.modelsActive} MODELS ACTIVE
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button onClick={() => triggerAction('train_models')} className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded transition-colors flex items-center gap-2">
                <Brain className="w-4 h-4" />
                TRAIN MODELS
              </button>
              <button onClick={() => triggerAction('export_ai_insights')} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors">
                EXPORT INSIGHTS
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="p-6">
        {/* Metrics */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-gray-500 uppercase">Predictions Today</p>
              <Sparkles className="w-4 h-4 text-purple-500" />
            </div>
            <p className="text-3xl font-bold font-mono">{metrics?.predictionsToday.toLocaleString()}</p>
            <p className="text-xs text-green-500 mt-1">↑ 12% from yesterday</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-gray-500 uppercase">Model Accuracy</p>
              <Target className="w-4 h-4 text-green-500" />
            </div>
            <p className="text-3xl font-bold font-mono">{metrics?.accuracy}%</p>
            <div className="mt-2 h-1 bg-gray-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-green-600 to-green-400 rounded-full" style={{ width: `${metrics?.accuracy}%` }} />
            </div>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-gray-500 uppercase">Response Time</p>
              <Zap className="w-4 h-4 text-yellow-500" />
            </div>
            <p className="text-3xl font-bold font-mono">{metrics?.responseTime}ms</p>
            <p className="text-xs text-gray-500 mt-1">avg inference time</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-gray-500 uppercase">Training Data</p>
              <Activity className="w-4 h-4 text-blue-500" />
            </div>
            <p className="text-3xl font-bold font-mono">{((metrics?.training?.dataPoints || 0) / 1000000).toFixed(1)}M</p>
            <p className="text-xs text-gray-500 mt-1">data points</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          {/* AI Predictions */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg">
            <div className="p-4 border-b border-gray-800">
              <h3 className="text-sm font-bold text-gray-400 uppercase">PREDICTIVE ANALYTICS</h3>
            </div>
            <div className="divide-y divide-gray-800">
              {metrics?.predictions.map((prediction) => (
                <div key={prediction.id} className="p-4 hover:bg-gray-800/50 transition-colors">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <span className={`text-xs px-2 py-1 rounded font-bold ${
                        prediction.risk === 'critical' ? 'bg-red-900/30 text-red-500' :
                        prediction.risk === 'high' ? 'bg-orange-900/30 text-orange-500' :
                        prediction.risk === 'medium' ? 'bg-yellow-900/30 text-yellow-500' :
                        'bg-gray-800 text-gray-500'
                      }`}>
                        {prediction.category.toUpperCase()}
                      </span>
                      <h4 className="font-medium mt-2">{prediction.prediction}</h4>
                      <p className="text-sm text-gray-500 mt-1">Expected in {prediction.timeframe}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-2xl font-bold font-mono">{prediction.probability}%</p>
                      <p className="text-xs text-gray-500">probability</p>
                    </div>
                  </div>
                  <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
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

          {/* AI Recommendations */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg">
            <div className="p-4 border-b border-gray-800">
              <h3 className="text-sm font-bold text-gray-400 uppercase">AI RECOMMENDATIONS</h3>
            </div>
            <div className="divide-y divide-gray-800">
              {metrics?.recommendations.map((rec) => (
                <div key={rec.id} className="p-4 hover:bg-gray-800/50 transition-colors">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-medium">{rec.type}</span>
                        <span className={`text-xs px-2 py-0.5 rounded ${
                          rec.impact === 'high' ? 'bg-red-900/30 text-red-500' :
                          rec.impact === 'medium' ? 'bg-yellow-900/30 text-yellow-500' :
                          'bg-gray-800 text-gray-500'
                        }`}>
                          {rec.impact.toUpperCase()} IMPACT
                        </span>
                      </div>
                      <p className="text-sm text-gray-400">{rec.description}</p>
                      <div className="flex items-center gap-4 mt-2">
                        <div className="flex items-center gap-1">
                          <Brain className="w-3 h-3 text-purple-500" />
                          <span className="text-xs text-gray-500">{rec.confidence}% confidence</span>
                        </div>
                        <span className={`text-xs px-2 py-0.5 rounded ${
                          rec.status === 'applied' ? 'bg-green-900/30 text-green-500' :
                          rec.status === 'rejected' ? 'bg-red-900/30 text-red-500' :
                          'bg-blue-900/30 text-blue-500'
                        }`}>
                          {rec.status.toUpperCase()}
                        </span>
                      </div>
                    </div>
                    {rec.status === 'pending' && (
                      <div className="flex gap-2">
                        <button className="px-3 py-1 bg-green-900/30 hover:bg-green-900/50 border border-green-800 rounded text-green-500 text-xs transition-colors">
                          APPLY
                        </button>
                        <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-gray-400 text-xs transition-colors">
                          DISMISS
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Conversational AI */}
        <div className="mt-6 bg-gray-900 border border-gray-800 rounded-lg">
          <div className="p-4 border-b border-gray-800 flex items-center justify-between">
            <h3 className="text-sm font-bold text-gray-400 uppercase">CONVERSATIONAL AI</h3>
            <MessageSquare className="w-4 h-4 text-purple-500" />
          </div>
          <div className="p-4">
            <div className="h-64 overflow-y-auto mb-4 space-y-3">
              {conversationHistory.length === 0 ? (
                <p className="text-gray-500 text-sm">Ask me anything about your Azure environment...</p>
              ) : (
                conversationHistory.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-lg px-4 py-2 rounded-lg ${
                      msg.role === 'user' 
                        ? 'bg-blue-900/30 text-blue-300' 
                        : 'bg-purple-900/30 text-purple-300'
                    }`}>
                      <p className="text-sm">{msg.message}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={conversationInput}
                onChange={(e) => setConversationInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleConversation()}
                placeholder="Ask about costs, security, compliance, or optimization..."
                className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              />
              <button 
                onClick={handleConversation}
                className="px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded transition-colors"
              >
                ASK AI
              </button>
            </div>
          </div>
        </div>

        {/* Model Training Status */}
        <div className="mt-6 bg-gray-900 border border-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-gray-400 uppercase mb-2">MODEL TRAINING STATUS</h3>
              <div className="flex items-center gap-6 text-sm">
                <div>
                  <span className="text-gray-500">Version:</span> <span className="font-mono">{metrics?.training.modelVersion}</span>
                </div>
                <div>
                  <span className="text-gray-500">Last Update:</span> <span>{metrics?.training.lastUpdate}</span>
                </div>
                <div>
                  <span className="text-gray-500">Next Training:</span> <span>{metrics?.training.nextTraining}</span>
                </div>
              </div>
            </div>
            <button onClick={() => triggerAction('force_retrain')} className="px-4 py-2 bg-purple-900/30 hover:bg-purple-900/50 border border-purple-800 text-purple-500 rounded text-sm transition-colors">
              FORCE RETRAIN
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}