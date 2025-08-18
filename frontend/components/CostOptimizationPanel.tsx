'use client';

import React, { useState } from 'react';
import { DollarSign, TrendingDown, TrendingUp, Calendar, Info } from 'lucide-react';

type TimeRange = 'weekly' | 'monthly';

// Mock ARIMA trend data
const generateTrendData = (range: TimeRange) => {
  const points = range === 'weekly' ? 7 : 30;
  const baseValue = 4200;
  const trend = -0.02; // 2% downward trend
  
  return Array.from({ length: points }, (_, i) => ({
    date: range === 'weekly' 
      ? `Day ${i + 1}`
      : `${Math.floor(i / 7) + 1}W-${(i % 7) + 1}`,
    actual: baseValue * (1 + trend * i / points) + (Math.random() - 0.5) * 200,
    predicted: baseValue * (1 + trend * (i + 1) / points),
    optimized: baseValue * (1 + trend * (i + 1) / points) * 0.85
  }));
};

export const CostOptimizationPanel: React.FC = () => {
  const [timeRange, setTimeRange] = useState<TimeRange>('monthly');
  const trendData = generateTrendData(timeRange);
  
  const currentCost = trendData[trendData.length - 1].actual;
  const predictedCost = trendData[trendData.length - 1].predicted;
  const potentialSavings = currentCost * 0.15;

  return (
    <div className="space-y-6">
      {/* Header with Toggle */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-green-500" />
            Cost Optimization Intelligence
          </h3>
          
          {/* Time Range Toggle */}
          <div className="flex items-center gap-2 bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
            <button
              onClick={() => setTimeRange('weekly')}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                timeRange === 'weekly'
                  ? 'bg-white dark:bg-gray-600 text-purple-600 dark:text-purple-400 shadow-sm'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              Weekly
            </button>
            <button
              onClick={() => setTimeRange('monthly')}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                timeRange === 'monthly'
                  ? 'bg-white dark:bg-gray-600 text-purple-600 dark:text-purple-400 shadow-sm'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              Monthly
            </button>
          </div>
        </div>

        {/* Cost Metrics */}
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Current Cost</span>
              <TrendingDown className="w-4 h-4 text-green-500" />
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              ${currentCost.toFixed(0)}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Per {timeRange === 'weekly' ? 'week' : 'month'}
            </p>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Predicted Next</span>
              <TrendingUp className="w-4 h-4 text-yellow-500" />
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              ${predictedCost.toFixed(0)}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              ARIMA forecast
            </p>
          </div>

          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-green-700 dark:text-green-400">Potential Savings</span>
              <DollarSign className="w-4 h-4 text-green-600" />
            </div>
            <p className="text-2xl font-bold text-green-700 dark:text-green-400">
              ${potentialSavings.toFixed(0)}
            </p>
            <p className="text-xs text-green-600 dark:text-green-500 mt-1">
              ~15% optimization
            </p>
          </div>
        </div>
      </div>

      {/* ARIMA Trend Visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h4 className="text-md font-semibold text-gray-900 dark:text-white mb-4">
          {timeRange === 'weekly' ? 'Weekly' : 'Monthly'} Cost Trend (ARIMA Model)
        </h4>
        
        <div className="h-64 flex items-end gap-1">
          {trendData.slice(-10).map((point, index) => (
            <div key={index} className="flex-1 flex flex-col items-center gap-1">
              <div className="w-full flex flex-col gap-1">
                {/* Optimized bar */}
                <div 
                  className="w-full bg-green-500/30 rounded-t"
                  style={{ height: `${(point.optimized / currentCost) * 100}px` }}
                  title={`Optimized: $${point.optimized.toFixed(0)}`}
                />
                {/* Predicted bar */}
                <div 
                  className="w-full bg-purple-500/50 rounded-t"
                  style={{ height: `${(point.predicted / currentCost) * 100}px` }}
                  title={`Predicted: $${point.predicted.toFixed(0)}`}
                />
                {/* Actual bar */}
                <div 
                  className="w-full bg-blue-500 rounded-t"
                  style={{ height: `${(point.actual / currentCost) * 100}px` }}
                  title={`Actual: $${point.actual.toFixed(0)}`}
                />
              </div>
              <span className="text-xs text-gray-500 dark:text-gray-400 mt-1 rotate-45 origin-left">
                {point.date}
              </span>
            </div>
          ))}
        </div>

        <div className="flex items-center gap-6 mt-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded" />
            <span className="text-gray-600 dark:text-gray-400">Actual Cost</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-purple-500/50 rounded" />
            <span className="text-gray-600 dark:text-gray-400">ARIMA Prediction</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500/30 rounded" />
            <span className="text-gray-600 dark:text-gray-400">Optimized Target</span>
          </div>
        </div>
      </div>

      {/* Optimization Recommendations */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h4 className="text-md font-semibold text-gray-900 dark:text-white mb-4">
          AI-Powered Recommendations
        </h4>
        
        <div className="space-y-3">
          <div className="flex items-start gap-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-white text-sm font-bold">1</span>
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                Rightsize underutilized VMs
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                23 VMs running at &lt;20% CPU. Potential savings: $1,200/{timeRange === 'weekly' ? 'week' : 'month'}
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-white text-sm font-bold">2</span>
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                Enable auto-shutdown for dev resources
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                45 dev/test resources running 24/7. Potential savings: $800/{timeRange === 'weekly' ? 'week' : 'month'}
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
            <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-white text-sm font-bold">3</span>
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                Optimize storage tiers
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                Move 2.3TB of cold data to archive tier. Potential savings: $450/{timeRange === 'weekly' ? 'week' : 'month'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Demo Mode Notice */}
      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-purple-600 dark:text-purple-400 mt-0.5" />
          <div className="text-xs text-purple-600 dark:text-purple-400">
            <p className="font-semibold mb-1">Demo Mode: ARIMA Cost Forecasting</p>
            <p>
              Showing simulated cost optimization using ARIMA time-series forecasting. 
              In production, this integrates with Azure Cost Management APIs for real-time analysis.
              Toggle between weekly and monthly views to see different forecast horizons.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};