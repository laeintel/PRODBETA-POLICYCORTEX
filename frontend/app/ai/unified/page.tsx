'use client';

import { Card } from '@/components/ui/card';
import { Brain, Sparkles, Zap, TrendingUp, Target, Shield, AlertCircle, CheckCircle } from 'lucide-react';
import { useState, useEffect } from 'react';

interface AIInsight {
  id: string;
  type: 'optimization' | 'security' | 'compliance' | 'cost';
  title: string;
  description: string;
  impact: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  potentialSavings?: string;
  affectedResources: number;
}

export default function UnifiedAIPage() {
  const [insights, setInsights] = useState<AIInsight[]>([]);
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState({
    totalInsights: 0,
    highImpact: 0,
    potentialSavings: 0,
    resourcesAnalyzed: 0
  });

  useEffect(() => {
    // Mock AI insights
    const mockInsights: AIInsight[] = [
      {
        id: 'ai-001',
        type: 'cost',
        title: 'Underutilized VM instances detected',
        description: 'AI analysis shows 23 VMs with average CPU usage below 10% over the past 30 days.',
        impact: 'high',
        confidence: 92,
        potentialSavings: '$4,250/month',
        affectedResources: 23
      },
      {
        id: 'ai-002',
        type: 'security',
        title: 'Potential security vulnerability pattern detected',
        description: 'ML model identified resources with configuration similar to known vulnerability patterns.',
        impact: 'critical',
        confidence: 87,
        affectedResources: 8
      },
      {
        id: 'ai-003',
        type: 'compliance',
        title: 'Predicted compliance drift in 14 days',
        description: 'Predictive model shows 95% probability of compliance violation based on current trends.',
        impact: 'high',
        confidence: 95,
        affectedResources: 45
      },
      {
        id: 'ai-004',
        type: 'optimization',
        title: 'Database performance optimization opportunity',
        description: 'AI detected query patterns suggesting index optimization could improve performance by 40%.',
        impact: 'medium',
        confidence: 78,
        affectedResources: 5
      },
      {
        id: 'ai-005',
        type: 'cost',
        title: 'Reserved instance recommendations',
        description: 'Based on usage patterns, switching to reserved instances could save 35% on compute costs.',
        impact: 'high',
        confidence: 91,
        potentialSavings: '$12,500/month',
        affectedResources: 67
      }
    ];

    setTimeout(() => {
      setInsights(mockInsights);
      setMetrics({
        totalInsights: mockInsights.length,
        highImpact: mockInsights.filter(i => i.impact === 'high' || i.impact === 'critical').length,
        potentialSavings: 16750,
        resourcesAnalyzed: 1247
      });
      setLoading(false);
    }, 1000);
  }, []);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'cost':
        return <TrendingUp className="h-5 w-5 text-green-500" />;
      case 'security':
        return <Shield className="h-5 w-5 text-red-500" />;
      case 'compliance':
        return <CheckCircle className="h-5 w-5 text-blue-500" />;
      case 'optimization':
        return <Zap className="h-5 w-5 text-yellow-500" />;
      default:
        return <Sparkles className="h-5 w-5 text-purple-500" />;
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'critical':
        return 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-400';
      case 'high':
        return 'bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-400';
      case 'medium':
        return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-400';
      case 'low':
        return 'bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-400';
      default:
        return 'bg-gray-100 text-gray-700 dark:bg-gray-950 dark:text-gray-400';
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Brain className="h-8 w-8 text-purple-600" />
            Unified AI Intelligence
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            AI-powered insights and predictions across your cloud infrastructure
          </p>
        </div>
        <button className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 flex items-center gap-2">
          <Sparkles className="h-4 w-4" />
          Run Analysis
        </button>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Insights</p>
              <p className="text-2xl font-bold">{metrics.totalInsights}</p>
            </div>
            <Sparkles className="h-8 w-8 text-purple-400" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">High Impact</p>
              <p className="text-2xl font-bold text-orange-600">{metrics.highImpact}</p>
            </div>
            <AlertCircle className="h-8 w-8 text-orange-400" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Potential Savings</p>
              <p className="text-2xl font-bold text-green-600">${metrics.potentialSavings.toLocaleString()}</p>
            </div>
            <TrendingUp className="h-8 w-8 text-green-400" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Resources Analyzed</p>
              <p className="text-2xl font-bold">{metrics.resourcesAnalyzed.toLocaleString()}</p>
            </div>
            <Target className="h-8 w-8 text-blue-400" />
          </div>
        </Card>
      </div>

      {/* AI Model Status */}
      <Card className="p-6 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-950 dark:to-blue-950">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold mb-2">AI Models Status</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Predictive Compliance</p>
                <p className="font-semibold text-green-600">Active (99.2% accuracy)</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Anomaly Detection</p>
                <p className="font-semibold text-green-600">Active (97.8% accuracy)</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Cost Optimization</p>
                <p className="font-semibold text-green-600">Active (95.5% accuracy)</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Security Analysis</p>
                <p className="font-semibold text-green-600">Active (98.1% accuracy)</p>
              </div>
            </div>
          </div>
          <Brain className="h-16 w-16 text-purple-300" />
        </div>
      </Card>

      {/* AI Insights */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-6">AI-Generated Insights</h2>
        {loading ? (
          <div className="space-y-4">
            {[1, 2, 3].map(i => (
              <div key={i} className="animate-pulse">
                <div className="h-20 bg-gray-200 dark:bg-gray-700 rounded"></div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-4">
            {insights.map(insight => (
              <div key={insight.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3 flex-1">
                    {getTypeIcon(insight.type)}
                    <div className="flex-1">
                      <h3 className="font-semibold mb-1">{insight.title}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                        {insight.description}
                      </p>
                      <div className="flex items-center gap-4 text-sm">
                        <span className={`px-2 py-1 rounded-full ${getImpactColor(insight.impact)}`}>
                          {insight.impact.toUpperCase()} IMPACT
                        </span>
                        <span className="text-gray-600 dark:text-gray-400">
                          Confidence: {insight.confidence}%
                        </span>
                        {insight.potentialSavings && (
                          <span className="text-green-600 font-medium">
                            {insight.potentialSavings}
                          </span>
                        )}
                        <span className="text-gray-600 dark:text-gray-400">
                          {insight.affectedResources} resources
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-2 ml-4">
                    <button className="px-3 py-1 text-sm border border-blue-600 text-blue-600 rounded-md hover:bg-blue-50 dark:hover:bg-blue-950">
                      View Details
                    </button>
                    <button className="px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700">
                      Take Action
                    </button>
                  </div>
                </div>
                {/* Confidence Bar */}
                <div className="mt-3">
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                      style={{ width: `${insight.confidence}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}