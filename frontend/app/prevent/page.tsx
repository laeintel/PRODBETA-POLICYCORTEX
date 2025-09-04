'use client'

import { useEffect } from 'react';
import { usePCGStore } from '@/stores/resourceStore';
import { Shield, AlertTriangle, TrendingUp, Clock } from 'lucide-react';

export default function PreventPage() {
  const { predictions, isLoading, error, fetchPredictions } = usePCGStore();

  useEffect(() => {
    fetchPredictions();
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

  const highRiskCount = predictions.filter(p => p.impact === 'high').length;
  const compliancePredictions = predictions.filter(p => p.type === 'compliance');
  const avgConfidence = predictions.length > 0 
    ? Math.round(predictions.reduce((acc, p) => acc + p.confidence, 0) / predictions.length)
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Prevent</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          AI-powered predictive compliance and risk prevention
        </p>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Active Predictions</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                {predictions.length}
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">High Risk Items</p>
              <p className="text-2xl font-bold text-red-600 dark:text-red-400 mt-1">
                {highRiskCount}
              </p>
            </div>
            <AlertTriangle className="w-8 h-8 text-red-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Compliance Score</p>
              <p className="text-2xl font-bold text-green-600 dark:text-green-400 mt-1">
                {compliancePredictions.length}
              </p>
            </div>
            <Shield className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Avg Confidence</p>
              <p className="text-2xl font-bold text-purple-600 dark:text-purple-400 mt-1">
                {avgConfidence}%
              </p>
            </div>
            <Clock className="w-8 h-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* Predictions List */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Predictions</h2>
        </div>
        <div className="divide-y divide-gray-200 dark:divide-gray-700">
          {predictions.slice(0, 5).map((prediction) => (
            <div key={prediction.id} className="px-6 py-4">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      prediction.impact === 'high' 
                        ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                        : prediction.impact === 'medium'
                        ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                        : 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                    }`}>
                      {prediction.impact.toUpperCase()}
                    </span>
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                      {prediction.type}
                    </span>
                  </div>
                  <p className="mt-2 text-gray-900 dark:text-white">{prediction.prediction}</p>
                  <div className="mt-2 flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                    <span>Confidence: {prediction.confidence}%</span>
                    <span>Score: {prediction.score}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
        {predictions.length === 0 && (
          <div className="px-6 py-8 text-center text-gray-500 dark:text-gray-400">
            No predictions available
          </div>
        )}
      </div>
    </div>
  );
}