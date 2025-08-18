'use client';

import React from 'react';

interface ExplainabilityChartProps {
  predictionId?: string;
  showStatic?: boolean;
}

export const ExplainabilityChart: React.FC<ExplainabilityChartProps> = ({ 
  predictionId, 
  showStatic = true 
}) => {
  // Static SHAP values for demo
  const staticShapData = {
    features: [
      { name: 'Resource Count', value: 0.23, impact: 'positive' },
      { name: 'Policy Violations', value: -0.18, impact: 'negative' },
      { name: 'Security Score', value: 0.15, impact: 'positive' },
      { name: 'Cost Trend', value: -0.12, impact: 'negative' },
      { name: 'Compliance History', value: 0.09, impact: 'positive' },
      { name: 'User Activity', value: 0.07, impact: 'positive' },
      { name: 'Network Exposure', value: -0.05, impact: 'negative' },
      { name: 'Data Sensitivity', value: -0.03, impact: 'negative' }
    ],
    baseValue: 0.65,
    prediction: 0.82,
    confidence: 0.91
  };

  const maxValue = Math.max(...staticShapData.features.map(f => Math.abs(f.value)));

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Model Explainability (SHAP)
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Feature importance for prediction confidence: {(staticShapData.confidence * 100).toFixed(0)}%
        </p>
      </div>

      {/* SHAP Waterfall Chart */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm mb-3">
          <span className="text-gray-600 dark:text-gray-400">Base value</span>
          <span className="font-mono">{staticShapData.baseValue.toFixed(3)}</span>
        </div>

        {staticShapData.features.map((feature, index) => (
          <div key={index} className="flex items-center gap-3">
            <div className="w-32 text-sm text-gray-600 dark:text-gray-400 truncate">
              {feature.name}
            </div>
            <div className="flex-1 flex items-center">
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-6 relative">
                <div
                  className={`absolute h-full rounded-full transition-all ${
                    feature.impact === 'positive' 
                      ? 'bg-green-500' 
                      : 'bg-red-500'
                  }`}
                  style={{
                    width: `${(Math.abs(feature.value) / maxValue) * 100}%`,
                    left: feature.impact === 'positive' ? '50%' : 'auto',
                    right: feature.impact === 'negative' ? '50%' : 'auto',
                  }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-xs font-mono text-gray-800 dark:text-gray-200">
                    {feature.value > 0 ? '+' : ''}{feature.value.toFixed(3)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}

        <div className="flex items-center justify-between text-sm mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
          <span className="text-gray-900 dark:text-white font-semibold">Final prediction</span>
          <span className="font-mono font-bold text-lg">
            {staticShapData.prediction.toFixed(3)}
          </span>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 flex items-center justify-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          <span className="text-gray-600 dark:text-gray-400">Increases prediction</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
          <span className="text-gray-600 dark:text-gray-400">Decreases prediction</span>
        </div>
      </div>

      {showStatic && (
        <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <p className="text-xs text-blue-600 dark:text-blue-400">
            Demo Mode: Showing static SHAP values. Connect to ML service for live explanations.
          </p>
        </div>
      )}
    </div>
  );
};