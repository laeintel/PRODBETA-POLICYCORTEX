'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Brain,
  Target,
  Zap,
  CreditCard,
  PieChart,
  BarChart3,
  Activity,
  ArrowRight,
  RefreshCw,
  Download,
  Settings,
  Sparkles,
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react';
import { MLPredictionEngine, PredictionResult } from '@/lib/ml-predictions';
import ViewToggle from '@/components/ViewToggle';
import ChartContainer from '@/components/ChartContainer';
import MetricCard from '@/components/MetricCard';

interface FinOpsMetric {
  id: string;
  title: string;
  value: string | number;
  change: number;
  trend: 'up' | 'down' | 'stable';
  prediction?: PredictionResult;
  alert?: string;
  sparklineData?: number[];
}

export default function FinOpsCommandCenter() {
  const router = useRouter();
  const [view, setView] = useState<'cards' | 'visualizations'>('cards');
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [realTimeAlerts, setRealTimeAlerts] = useState<any[]>([]);
  const [optimizationRecommendations, setOptimizationRecommendations] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  // Real-time cost anomaly detection
  useEffect(() => {
    const loadPredictions = async () => {
      const costPrediction = await MLPredictionEngine.predictCostSpike('main-account');
      const wastePrediction = await MLPredictionEngine.predictResourceWaste('production-rg');
      const budgetPrediction = await MLPredictionEngine.predictBudgetOverrun('engineering');
      setPredictions([costPrediction, wastePrediction, budgetPrediction]);
      setLoading(false);
    };

    loadPredictions();

    // Simulate real-time alerts
    const interval = setInterval(() => {
      setRealTimeAlerts(prev => [...prev.slice(-4), {
        id: Date.now(),
        type: 'cost_anomaly',
        message: `Unusual spending detected in ${['us-east-1', 'eu-west-1', 'ap-south-1'][Math.floor(Math.random() * 3)]}`,
        severity: ['high', 'medium', 'low'][Math.floor(Math.random() * 3)],
        timestamp: new Date().toISOString(),
        amount: Math.floor(Math.random() * 5000) + 1000
      }]);
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  const metrics: FinOpsMetric[] = [
    {
      id: 'total-spend',
      title: 'Current Month Spend',
      value: '$287,432',
      change: 12.5,
      trend: 'up',
      sparklineData: [220000, 235000, 250000, 265000, 275000, 287432],
      alert: 'Trending 23% above budget'
    },
    {
      id: 'cost-savings',
      title: 'Identified Savings',
      value: '$45,230',
      change: 34.2,
      trend: 'up',
      sparklineData: [30000, 32000, 38000, 41000, 43000, 45230],
      prediction: predictions[1]
    },
    {
      id: 'waste-detected',
      title: 'Resource Waste',
      value: '$18,750',
      change: -15.3,
      trend: 'down',
      sparklineData: [25000, 23000, 21000, 20000, 19500, 18750]
    },
    {
      id: 'optimization-score',
      title: 'Optimization Score',
      value: '73%',
      change: 5.1,
      trend: 'up',
      sparklineData: [65, 67, 69, 70, 72, 73]
    }
  ];

  const finOpsFeatures = [
    {
      id: 'anomalies',
      title: 'Real-Time Anomaly Detection',
      description: 'AI-powered detection of cost spikes within minutes',
      icon: AlertTriangle,
      route: '/finops/anomalies',
      status: 'active',
      alerts: realTimeAlerts.length,
      color: 'red'
    },
    {
      id: 'optimization',
      title: 'Automated Rightsizing',
      description: 'ML-driven recommendations for resource optimization',
      icon: Target,
      route: '/finops/optimization',
      status: 'recommendations',
      count: 47,
      color: 'green'
    },
    {
      id: 'forecasting',
      title: 'Predictive Spend Forecasting',
      description: 'AI forecasts preventing budget overruns',
      icon: TrendingUp,
      route: '/finops/forecasting',
      predictions: predictions.length,
      color: 'blue'
    },
    {
      id: 'chargeback',
      title: 'Department Billing Integration',
      description: 'Automated show-back and charge-back to teams',
      icon: CreditCard,
      route: '/finops/chargeback',
      departments: 12,
      color: 'purple'
    },
    {
      id: 'savings-plans',
      title: 'Cross-Cloud Discount Optimizer',
      description: 'Maximize savings across AWS, Azure, GCP',
      icon: PieChart,
      route: '/finops/savings-plans',
      potential: '$125K/year',
      color: 'yellow'
    },
    {
      id: 'arbitrage',
      title: 'Multi-Cloud Arbitrage',
      description: 'AI suggests cheapest cloud for workloads',
      icon: Zap,
      route: '/finops/arbitrage',
      opportunities: 23,
      color: 'indigo'
    }
  ];

  const costOptimizationData = {
    labels: ['Compute', 'Storage', 'Network', 'Database', 'AI/ML', 'Other'],
    datasets: [{
      label: 'Current Spend',
      data: [45000, 23000, 18000, 35000, 42000, 15000],
      backgroundColor: 'rgba(59, 130, 246, 0.8)'
    }, {
      label: 'Optimized Spend',
      data: [38000, 19000, 15000, 28000, 35000, 12000],
      backgroundColor: 'rgba(34, 197, 94, 0.8)'
    }]
  };

  const anomalyTrendData = {
    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
    datasets: [{
      label: 'Normal Baseline',
      data: [1200, 1100, 1500, 2200, 2400, 2100, 1400],
      borderColor: 'rgba(156, 163, 175, 0.8)',
      fill: false
    }, {
      label: 'Actual Spend',
      data: [1250, 1150, 1600, 2300, 3800, 2200, 1450],
      borderColor: 'rgba(239, 68, 68, 0.8)',
      backgroundColor: 'rgba(239, 68, 68, 0.1)',
      fill: true
    }]
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold flex items-center gap-3">
            <DollarSign className="h-10 w-10 text-green-600" />
            FinOps Command Center
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            AI-powered cost optimization addressing the #1 cloud pain point
          </p>
        </div>
        <div className="flex gap-3">
          <ViewToggle view={view} onViewChange={setView} />
          <button
            onClick={() => router.push('/finops/settings')}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <Settings className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* AI Predictions Alert */}
      {predictions.length > 0 && predictions[0].riskLevel === 'high' && (
        <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Brain className="h-6 w-6 text-red-600 dark:text-red-400 mt-1" />
            <div className="flex-1">
              <h3 className="font-semibold text-red-900 dark:text-red-100">
                AI Cost Prediction Alert
              </h3>
              <p className="text-red-700 dark:text-red-300 mt-1">
                {predictions[0].prediction}
              </p>
              <p className="text-sm text-red-600 dark:text-red-400 mt-1">
                Confidence: {(predictions[0].confidence * 100).toFixed(0)}% | 
                Impact: ${predictions[0].impactEstimate?.financial?.toLocaleString()}
              </p>
              <div className="flex gap-2 mt-3">
                {predictions[0].recommendedActions.slice(0, 2).map((action, idx) => (
                  <button
                    key={idx}
                    className="px-3 py-1 bg-red-600 text-white rounded-md text-sm hover:bg-red-700"
                  >
                    {action}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        {metrics.map((metric) => (
          <MetricCard
            key={metric.id}
            title={metric.title}
            value={metric.value}
            change={metric.change}
            trend={metric.trend}
            sparklineData={metric.sparklineData}
            alert={metric.alert}
          />
        ))}
      </div>

      {view === 'cards' ? (
        <>
          {/* FinOps Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {finOpsFeatures.map((feature) => {
              const Icon = feature.icon;
              return (
                <div
                  key={feature.id}
                  className="bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-lg transition-all cursor-pointer transform hover:scale-[1.02]"
                  onClick={() => router.push(feature.route)}
                >
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className={`p-3 rounded-lg bg-${feature.color}-50 dark:bg-${feature.color}-900/20`}>
                        <Icon className={`h-8 w-8 text-${feature.color}-600 dark:text-${feature.color}-400`} />
                      </div>
                      {feature.alerts && (
                        <span className="px-2 py-1 bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300 rounded-full text-xs font-semibold">
                          {feature.alerts} alerts
                        </span>
                      )}
                      {feature.count && (
                        <span className="px-2 py-1 bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 rounded-full text-xs font-semibold">
                          {feature.count} ready
                        </span>
                      )}
                      {feature.potential && (
                        <span className="px-2 py-1 bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-300 rounded-full text-xs font-semibold">
                          {feature.potential}
                        </span>
                      )}
                    </div>
                    <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                    <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                      {feature.description}
                    </p>
                    <div className="flex items-center text-blue-600 dark:text-blue-400">
                      <span className="text-sm font-medium">View Details</span>
                      <ArrowRight className="h-4 w-4 ml-1" />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Real-Time Alerts */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Activity className="h-6 w-6 text-orange-600" />
              Real-Time Cost Alerts (Last 24h)
            </h2>
            <div className="space-y-3">
              {realTimeAlerts.slice(-5).reverse().map((alert) => (
                <div
                  key={alert.id}
                  className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    {alert.severity === 'high' ? (
                      <AlertCircle className="h-5 w-5 text-red-500" />
                    ) : alert.severity === 'medium' ? (
                      <AlertTriangle className="h-5 w-5 text-yellow-500" />
                    ) : (
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    )}
                    <div>
                      <p className="font-medium">{alert.message}</p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        ${alert.amount.toLocaleString()} • {new Date(alert.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                  <button className="px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700">
                    Investigate
                  </button>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <>
          {/* Visualization Mode */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartContainer
              title="Cost Optimization Opportunities"
              onDrillIn={() => router.push('/finops/optimization')}
            >
              <div className="p-4">
                {/* Placeholder for actual chart component */}
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                  <p className="text-gray-500">Cost optimization chart visualization</p>
                </div>
              </div>
            </ChartContainer>
            <ChartContainer
              title="Anomaly Detection (24h)"
              onDrillIn={() => router.push('/finops/anomalies')}
            >
              <div className="p-4">
                {/* Placeholder for actual chart component */}
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                  <p className="text-gray-500">Anomaly trend chart visualization</p>
                </div>
              </div>
            </ChartContainer>
          </div>

          {/* Optimization Recommendations */}
          <div className="mt-8 bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Sparkles className="h-6 w-6 text-purple-600" />
              AI-Powered Optimization Recommendations
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                {
                  title: 'Reserved Instance Optimization',
                  savings: '$32,400/month',
                  effort: 'Low',
                  confidence: 94
                },
                {
                  title: 'Orphaned Resource Cleanup',
                  savings: '$8,750/month',
                  effort: 'Low',
                  confidence: 89
                },
                {
                  title: 'Auto-scaling Configuration',
                  savings: '$15,200/month',
                  effort: 'Medium',
                  confidence: 87
                },
                {
                  title: 'Cross-Region Data Transfer',
                  savings: '$6,300/month',
                  effort: 'High',
                  confidence: 82
                }
              ].map((rec, idx) => (
                <div key={idx} className="p-4 border dark:border-gray-700 rounded-lg">
                  <h3 className="font-semibold">{rec.title}</h3>
                  <div className="flex justify-between mt-2 text-sm">
                    <span className="text-green-600 dark:text-green-400 font-semibold">
                      {rec.savings}
                    </span>
                    <span className="text-gray-500">
                      Effort: {rec.effort} • {rec.confidence}% confident
                    </span>
                  </div>
                  <button className="mt-3 w-full px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700">
                    Implement Now
                  </button>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}