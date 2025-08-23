'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  TrendingUp,
  Brain,
  Calendar,
  DollarSign,
  AlertTriangle,
  CheckCircle,
  ArrowLeft,
  Download,
  Settings,
  Eye,
  Clock,
  Target,
  Zap,
  ChevronRight
} from 'lucide-react';
import { MLPredictionEngine, PredictionResult } from '@/lib/ml-predictions';

interface ForecastData {
  month: string;
  actual?: number;
  predicted: number;
  optimistic: number;
  pessimistic: number;
  confidence: number;
}

interface BudgetAlert {
  id: string;
  department: string;
  currentSpend: number;
  projectedSpend: number;
  budget: number;
  overrunAmount: number;
  overrunPercent: number;
  daysUntilOverrun: number;
  severity: 'critical' | 'high' | 'medium' | 'low';
  recommendation: string;
}

export default function PredictiveSpendForecastingPage() {
  const router = useRouter();
  const [forecastData, setForecastData] = useState<ForecastData[]>([]);
  const [budgetAlerts, setBudgetAlerts] = useState<BudgetAlert[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [selectedDepartment, setSelectedDepartment] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<'3m' | '6m' | '12m'>('6m');
  const [showConfidenceIntervals, setShowConfidenceIntervals] = useState(true);

  useEffect(() => {
    // Load ML predictions
    const loadPredictions = async () => {
      const costPrediction = await MLPredictionEngine.predictCostSpike('main-account');
      const budgetPrediction = await MLPredictionEngine.predictBudgetOverrun('engineering');
      const capacityPrediction = await MLPredictionEngine.predictCapacityNeeds('main-app');
      setPredictions([costPrediction, budgetPrediction, capacityPrediction]);
    };
    loadPredictions();

    // Generate forecast data
    const generateForecast = () => {
      const months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
      ];
      
      const currentMonth = new Date().getMonth();
      const forecast: ForecastData[] = [];
      
      for (let i = 0; i < 12; i++) {
        const monthIndex = (currentMonth + i) % 12;
        const baseValue = 250000 + Math.random() * 50000;
        const trend = i * 5000; // Increasing trend
        const seasonality = Math.sin(i * Math.PI / 6) * 20000; // Seasonal variation
        
        const predicted = baseValue + trend + seasonality;
        
        forecast.push({
          month: months[monthIndex],
          actual: i < 3 ? predicted + (Math.random() - 0.5) * 10000 : undefined,
          predicted: predicted,
          optimistic: predicted * 0.85,
          pessimistic: predicted * 1.15,
          confidence: 95 - i * 2 // Confidence decreases over time
        });
      }
      
      return forecast;
    };

    setForecastData(generateForecast());

    // Generate budget alerts
    setBudgetAlerts([
      {
        id: 'alert-1',
        department: 'Engineering',
        currentSpend: 487000,
        projectedSpend: 625000,
        budget: 500000,
        overrunAmount: 125000,
        overrunPercent: 25,
        daysUntilOverrun: 7,
        severity: 'critical',
        recommendation: 'Implement immediate cost controls and defer non-critical projects'
      },
      {
        id: 'alert-2',
        department: 'Marketing',
        currentSpend: 89000,
        projectedSpend: 115000,
        budget: 100000,
        overrunAmount: 15000,
        overrunPercent: 15,
        daysUntilOverrun: 14,
        severity: 'high',
        recommendation: 'Review ad spend and optimize campaigns for efficiency'
      },
      {
        id: 'alert-3',
        department: 'Sales',
        currentSpend: 123000,
        projectedSpend: 138000,
        budget: 150000,
        overrunAmount: 0,
        overrunPercent: -8,
        daysUntilOverrun: 0,
        severity: 'low',
        recommendation: 'On track - continue monitoring'
      },
      {
        id: 'alert-4',
        department: 'Operations',
        currentSpend: 234000,
        projectedSpend: 285000,
        budget: 275000,
        overrunAmount: 10000,
        overrunPercent: 4,
        daysUntilOverrun: 21,
        severity: 'medium',
        recommendation: 'Consider rightsizing instances to stay within budget'
      }
    ]);
  }, []);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
      case 'high': return 'text-orange-600 bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800';
      case 'medium': return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
      case 'low': return 'text-green-600 bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
      default: return 'text-gray-600 bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800';
    }
  };

  const totalProjectedSpend = budgetAlerts.reduce((sum, alert) => sum + alert.projectedSpend, 0);
  const totalBudget = budgetAlerts.reduce((sum, alert) => sum + alert.budget, 0);
  const totalOverrun = budgetAlerts.reduce((sum, alert) => sum + Math.max(0, alert.overrunAmount), 0);

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
              <TrendingUp className="h-8 w-8 text-blue-600" />
              ML-Based Spend Forecasting
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              AI predicts and prevents budget overruns before they happen
            </p>
          </div>
        </div>
        <div className="flex gap-3">
          <button className="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Configure Alerts
          </button>
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
            <Download className="h-4 w-4" />
            Export Forecast
          </button>
        </div>
      </div>

      {/* ML Predictions Alert */}
      {predictions.length > 0 && predictions[0].riskLevel === 'high' && (
        <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Brain className="h-6 w-6 text-red-600 dark:text-red-400 mt-1" />
            <div className="flex-1">
              <h3 className="font-semibold text-red-900 dark:text-red-100">
                AI Budget Alert: {predictions[1].prediction}
              </h3>
              <p className="text-red-700 dark:text-red-300 mt-1">
                {predictions[1].explanation}
              </p>
              <div className="flex items-center gap-4 mt-2 text-sm text-red-600 dark:text-red-400">
                <span>Impact: ${predictions[1].impactEstimate?.financial?.toLocaleString()}</span>
                <span>Time to Event: {predictions[1].timeToEvent}</span>
                <span>Confidence: {(predictions[1].confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="flex gap-2 mt-3">
                <button className="px-3 py-1 bg-red-600 text-white rounded-md text-sm hover:bg-red-700">
                  Implement Cost Controls
                </button>
                <button className="px-3 py-1 bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300 rounded-md text-sm">
                  Review Details
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Projected Q1 Spend</p>
              <p className="text-2xl font-bold">${(totalProjectedSpend / 1000).toFixed(0)}K</p>
            </div>
            <TrendingUp className="h-8 w-8 text-blue-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Budget Remaining</p>
              <p className="text-2xl font-bold text-green-600">
                ${((totalBudget - totalProjectedSpend) / 1000).toFixed(0)}K
              </p>
            </div>
            <Target className="h-8 w-8 text-green-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">At-Risk Budget</p>
              <p className="text-2xl font-bold text-red-600">
                ${(totalOverrun / 1000).toFixed(0)}K
              </p>
            </div>
            <AlertTriangle className="h-8 w-8 text-red-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Forecast Accuracy</p>
              <p className="text-2xl font-bold">94.3%</p>
            </div>
            <Brain className="h-8 w-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* Time Range Selector */}
      <div className="flex items-center gap-4 mb-6">
        <span className="text-sm text-gray-600 dark:text-gray-400">Forecast Range:</span>
        <div className="flex gap-2">
          <button
            onClick={() => setTimeRange('3m')}
            className={`px-3 py-1 rounded-md text-sm ${
              timeRange === '3m' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            3 Months
          </button>
          <button
            onClick={() => setTimeRange('6m')}
            className={`px-3 py-1 rounded-md text-sm ${
              timeRange === '6m' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            6 Months
          </button>
          <button
            onClick={() => setTimeRange('12m')}
            className={`px-3 py-1 rounded-md text-sm ${
              timeRange === '12m' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            12 Months
          </button>
        </div>
        <div className="ml-auto">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={showConfidenceIntervals}
              onChange={(e) => setShowConfidenceIntervals(e.target.checked)}
              className="rounded"
            />
            Show Confidence Intervals
          </label>
        </div>
      </div>

      {/* Forecast Visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Spend Forecast</h2>
        <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div className="text-center">
            <TrendingUp className="h-16 w-16 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-500">Interactive forecast chart would be displayed here</p>
            <p className="text-sm text-gray-400 mt-2">
              Showing {timeRange === '3m' ? '3 month' : timeRange === '6m' ? '6 month' : '12 month'} projection
            </p>
          </div>
        </div>
        
        {/* Forecast Table */}
        <div className="mt-6 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b dark:border-gray-700">
                <th className="text-left py-2">Month</th>
                <th className="text-right py-2">Actual</th>
                <th className="text-right py-2">Predicted</th>
                {showConfidenceIntervals && (
                  <>
                    <th className="text-right py-2">Optimistic</th>
                    <th className="text-right py-2">Pessimistic</th>
                  </>
                )}
                <th className="text-right py-2">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {forecastData.slice(0, timeRange === '3m' ? 3 : timeRange === '6m' ? 6 : 12).map((data, idx) => (
                <tr key={idx} className="border-b dark:border-gray-700">
                  <td className="py-2">{data.month}</td>
                  <td className="text-right py-2">
                    {data.actual ? `$${(data.actual / 1000).toFixed(0)}K` : '-'}
                  </td>
                  <td className="text-right py-2 font-semibold">
                    ${(data.predicted / 1000).toFixed(0)}K
                  </td>
                  {showConfidenceIntervals && (
                    <>
                      <td className="text-right py-2 text-green-600">
                        ${(data.optimistic / 1000).toFixed(0)}K
                      </td>
                      <td className="text-right py-2 text-red-600">
                        ${(data.pessimistic / 1000).toFixed(0)}K
                      </td>
                    </>
                  )}
                  <td className="text-right py-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      data.confidence >= 90 ? 'bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300' :
                      data.confidence >= 80 ? 'bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-300' :
                      'bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300'
                    }`}>
                      {data.confidence}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Budget Alerts */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <h2 className="text-xl font-semibold mb-4">Department Budget Alerts</h2>
        <div className="space-y-3">
          {budgetAlerts.map((alert) => (
            <div
              key={alert.id}
              className={`border rounded-lg p-4 ${getSeverityColor(alert.severity)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="font-semibold">{alert.department}</h3>
                    {alert.severity === 'critical' && (
                      <span className="px-2 py-1 bg-red-600 text-white rounded-full text-xs font-medium animate-pulse">
                        CRITICAL - {alert.daysUntilOverrun} days
                      </span>
                    )}
                  </div>
                  <div className="grid grid-cols-4 gap-4 mb-2 text-sm">
                    <div>
                      <p className="text-gray-600 dark:text-gray-400">Current Spend</p>
                      <p className="font-semibold">${(alert.currentSpend / 1000).toFixed(0)}K</p>
                    </div>
                    <div>
                      <p className="text-gray-600 dark:text-gray-400">Projected</p>
                      <p className="font-semibold">${(alert.projectedSpend / 1000).toFixed(0)}K</p>
                    </div>
                    <div>
                      <p className="text-gray-600 dark:text-gray-400">Budget</p>
                      <p className="font-semibold">${(alert.budget / 1000).toFixed(0)}K</p>
                    </div>
                    <div>
                      <p className="text-gray-600 dark:text-gray-400">Overrun</p>
                      <p className={`font-semibold ${alert.overrunAmount > 0 ? 'text-red-600' : 'text-green-600'}`}>
                        {alert.overrunAmount > 0 ? '+' : ''}{alert.overrunPercent}%
                      </p>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    <span className="font-medium">Recommendation:</span> {alert.recommendation}
                  </p>
                </div>
                <button className="px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700 flex items-center gap-1">
                  Take Action
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}