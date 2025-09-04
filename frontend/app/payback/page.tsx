'use client'

import { useEffect } from 'react';
import { usePCGStore } from '@/stores/resourceStore';
import { DollarSign, TrendingUp, Award, BarChart3 } from 'lucide-react';

export default function PaybackPage() {
  const { roiMetrics, isLoading, error, fetchROIMetrics } = usePCGStore();

  useEffect(() => {
    fetchROIMetrics();
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-600 dark:text-red-400">Error: {error}</p>
      </div>
    );
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-US').format(value);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Payback</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          ROI tracking and cost optimization metrics
        </p>
      </div>

      {/* Primary ROI Card */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-blue-100 text-sm uppercase tracking-wider">Total ROI</p>
            <p className="text-5xl font-bold mt-2">
              {roiMetrics ? formatCurrency(roiMetrics.totalSavings) : '$0'}
            </p>
            <p className="text-blue-100 mt-2">
              Achieved through AI-powered governance optimization
            </p>
          </div>
          <DollarSign className="w-24 h-24 text-white/20" />
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded-lg">
              <TrendingUp className="w-6 h-6 text-green-600 dark:text-green-400" />
            </div>
            <span className="text-sm text-green-600 dark:text-green-400 font-medium">+23%</span>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400">Cost Avoidance</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
            {roiMetrics ? formatCurrency(roiMetrics.costAvoidance) : '$0'}
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <Award className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <span className="text-sm text-blue-600 dark:text-blue-400 font-medium">98%</span>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400">Compliance Score</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
            {roiMetrics ? `${roiMetrics.complianceScore}%` : '0%'}
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <BarChart3 className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            </div>
            <span className="text-sm text-purple-600 dark:text-purple-400 font-medium">-45%</span>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400">Risk Reduction</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
            {roiMetrics ? `${roiMetrics.riskReduction}%` : '0%'}
          </p>
        </div>
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Savings Breakdown */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Savings Breakdown</h2>
          </div>
          <div className="p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-600 dark:text-gray-400">Prevented Incidents</span>
              <span className="font-semibold text-gray-900 dark:text-white">
                {roiMetrics ? formatNumber(roiMetrics.preventedIncidents) : '0'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600 dark:text-gray-400">Automation Hours Saved</span>
              <span className="font-semibold text-gray-900 dark:text-white">
                {roiMetrics ? formatNumber(roiMetrics.automationHours) : '0'} hrs
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600 dark:text-gray-400">Compliance Score</span>
              <span className="font-semibold text-gray-900 dark:text-white">
                {roiMetrics ? `${roiMetrics.complianceScore}%` : '0%'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600 dark:text-gray-400">Risk Reduction</span>
              <span className="font-semibold text-gray-900 dark:text-white">
                {roiMetrics ? `${roiMetrics.riskReduction}%` : '0%'}
              </span>
            </div>
          </div>
        </div>

        {/* ROI Timeline */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Monthly Trend</h2>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {['Jan', 'Feb', 'Mar', 'Apr'].map((month, index) => {
                const value = roiMetrics 
                  ? roiMetrics.totalSavings * (0.7 + index * 0.1)
                  : 0;
                const percentage = roiMetrics 
                  ? (value / roiMetrics.totalSavings) * 100
                  : 0;
                
                return (
                  <div key={month}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm text-gray-600 dark:text-gray-400">{month}</span>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {formatCurrency(value)}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-blue-600 dark:bg-blue-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Export Actions */}
      <div className="flex justify-end gap-4">
        <button className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">
          Export CSV
        </button>
        <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
          Generate Report
        </button>
      </div>
    </div>
  );
}