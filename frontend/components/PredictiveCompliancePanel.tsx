'use client';

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, TrendingUp, Shield, Activity, Clock, ChevronRight } from 'lucide-react';
import { useMLPredictions, useModelMetrics } from '@/lib/mlClient';
import type { PredictionResponse, ViolationForecast } from '@/lib/mlClient';

interface Props {
  tenantId: string;
}

export default function PredictiveCompliancePanel({ tenantId }: Props) {
  const { predictions, loading, error, getRiskAssessment } = useMLPredictions(tenantId);
  const { metrics } = useModelMetrics();
  const [selectedPrediction, setSelectedPrediction] = useState<PredictionResponse | null>(null);
  const [riskAssessment, setRiskAssessment] = useState<any>(null);

  // Filter high-risk predictions
  const highRiskPredictions = predictions.filter(
    p => p.riskLevel === 'critical' || p.riskLevel === 'high'
  );

  // Get recent violations forecast
  const violationForecasts = predictions
    .filter(p => p.timeToViolationHours && p.timeToViolationHours < 72)
    .sort((a, b) => (a.timeToViolationHours || 0) - (b.timeToViolationHours || 0));

  const handlePredictionClick = async (prediction: PredictionResponse) => {
    setSelectedPrediction(prediction);
    try {
      const assessment = await getRiskAssessment(prediction.resourceId);
      setRiskAssessment(assessment);
    } catch (err) {
      console.error('Failed to get risk assessment:', err);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'critical': return 'text-red-500 bg-red-50';
      case 'high': return 'text-orange-500 bg-orange-50';
      case 'medium': return 'text-yellow-500 bg-yellow-50';
      case 'low': return 'text-green-500 bg-green-50';
      default: return 'text-gray-500 bg-gray-50';
    }
  };

  const formatTimeToViolation = (hours?: number) => {
    if (!hours) return 'Unknown';
    if (hours < 24) return `${Math.round(hours)} hours`;
    return `${Math.round(hours / 24)} days`;
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="space-y-3">
            <div className="h-4 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded w-5/6"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="text-red-500">
          <AlertTriangle className="w-6 h-6 mb-2" />
          <p>Failed to load predictions: {error.message}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">Predictive Compliance Engine</h2>
          {metrics && (
            <div className="flex items-center space-x-4 text-sm">
              <span className="text-gray-500">Model Accuracy:</span>
              <span className={`font-semibold ${metrics.accuracy >= 0.992 ? 'text-green-600' : 'text-orange-600'}`}>
                {(metrics.accuracy * 100).toFixed(2)}%
              </span>
              <span className="text-gray-500">FPR:</span>
              <span className={`font-semibold ${metrics.falsePositiveRate < 0.02 ? 'text-green-600' : 'text-orange-600'}`}>
                {(metrics.falsePositiveRate * 100).toFixed(2)}%
              </span>
            </div>
          )}
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-50 rounded-lg p-4"
          >
            <div className="flex items-center justify-between">
              <AlertTriangle className="w-8 h-8 text-red-500" />
              <span className="text-2xl font-bold text-red-700">
                {highRiskPredictions.length}
              </span>
            </div>
            <p className="text-sm text-red-600 mt-2">High Risk Resources</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-orange-50 rounded-lg p-4"
          >
            <div className="flex items-center justify-between">
              <Clock className="w-8 h-8 text-orange-500" />
              <span className="text-2xl font-bold text-orange-700">
                {violationForecasts.length}
              </span>
            </div>
            <p className="text-sm text-orange-600 mt-2">Violations in 72h</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-blue-50 rounded-lg p-4"
          >
            <div className="flex items-center justify-between">
              <Activity className="w-8 h-8 text-blue-500" />
              <span className="text-2xl font-bold text-blue-700">
                {metrics ? `${metrics.inferenceTimeP95Ms.toFixed(0)}ms` : 'N/A'}
              </span>
            </div>
            <p className="text-sm text-blue-600 mt-2">P95 Latency</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-green-50 rounded-lg p-4"
          >
            <div className="flex items-center justify-between">
              <Shield className="w-8 h-8 text-green-500" />
              <span className="text-2xl font-bold text-green-700">
                {predictions.filter(p => p.confidenceScore > 0.9).length}
              </span>
            </div>
            <p className="text-sm text-green-600 mt-2">High Confidence</p>
          </motion.div>
        </div>

        {/* Violation Forecasts */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Upcoming Violations</h3>
          <div className="space-y-2">
            {violationForecasts.slice(0, 5).map((forecast, index) => (
              <motion.div
                key={forecast.predictionId}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                onClick={() => handlePredictionClick(forecast)}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${getRiskColor(forecast.riskLevel)}`}>
                    <AlertTriangle className="w-4 h-4" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">{forecast.resourceId}</p>
                    <p className="text-sm text-gray-500">
                      Violation in {formatTimeToViolation(forecast.timeToViolationHours)}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="text-right">
                    <p className="text-sm font-semibold text-gray-900">
                      {(forecast.violationProbability * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500">Probability</p>
                  </div>
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Recent Predictions */}
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Recent Predictions</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Resource
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Risk Level
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Violation Prob.
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Confidence
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Time to Violation
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Latency
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {predictions.slice(0, 10).map((prediction) => (
                  <tr
                    key={prediction.predictionId}
                    onClick={() => handlePredictionClick(prediction)}
                    className="hover:bg-gray-50 cursor-pointer"
                  >
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                      {prediction.resourceId}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getRiskColor(prediction.riskLevel)}`}>
                        {prediction.riskLevel}
                      </span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                      {(prediction.violationProbability * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                      {(prediction.confidenceScore * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                      {formatTimeToViolation(prediction.timeToViolationHours)}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                      {prediction.inferenceTimeMs.toFixed(1)}ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Selected Prediction Details */}
      {selectedPrediction && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="p-6 bg-gray-50 border-t border-gray-200"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Details</h3>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-gray-500 mb-2">Recommendations</h4>
              <ul className="space-y-2">
                {selectedPrediction.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start">
                    <ChevronRight className="w-4 h-4 text-gray-400 mt-0.5 mr-2 flex-shrink-0" />
                    <span className="text-sm text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
            {riskAssessment && (
              <div>
                <h4 className="text-sm font-medium text-gray-500 mb-2">Impact Factors</h4>
                <div className="space-y-2">
                  {Object.entries(riskAssessment.impactFactors).map(([factor, value]) => (
                    <div key={factor} className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 capitalize">{factor}</span>
                      <div className="flex items-center">
                        <div className="w-24 bg-gray-200 rounded-full h-2 mr-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${(value as number) * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium text-gray-900">
                          {((value as number) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500">
                Confidence Interval: [{selectedPrediction.confidenceInterval[0].toFixed(2)}, {selectedPrediction.confidenceInterval[1].toFixed(2)}]
              </span>
              <span className="text-gray-500">
                Model Version: {selectedPrediction.modelVersion}
              </span>
              <span className="text-gray-500">
                Prediction ID: {selectedPrediction.predictionId}
              </span>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}