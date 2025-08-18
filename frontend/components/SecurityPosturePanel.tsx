'use client';

import React from 'react';
import { Shield, AlertTriangle, CheckCircle, Info, TrendingUp, TrendingDown } from 'lucide-react';

// Sample security posture data fixture
const securityPostureData = {
  overallScore: 88,
  trend: 'improving',
  lastUpdated: new Date().toISOString(),
  categories: {
    identity: {
      score: 92,
      findings: 3,
      critical: 0,
      high: 1,
      medium: 2
    },
    network: {
      score: 85,
      findings: 5,
      critical: 1,
      high: 2,
      medium: 2
    },
    data: {
      score: 90,
      findings: 2,
      critical: 0,
      high: 0,
      medium: 2
    },
    compute: {
      score: 86,
      findings: 4,
      critical: 0,
      high: 2,
      medium: 2
    }
  },
  topFindings: [
    {
      id: 'sec-001',
      severity: 'critical',
      title: 'Public storage account detected',
      resource: 'storageaccount01',
      recommendation: 'Enable private endpoints and disable public access'
    },
    {
      id: 'sec-002',
      severity: 'high',
      title: 'VM without managed disk encryption',
      resource: 'vm-prod-web-01',
      recommendation: 'Enable Azure Disk Encryption'
    },
    {
      id: 'sec-003',
      severity: 'high',
      title: 'Excessive permissions on service principal',
      resource: 'sp-automation',
      recommendation: 'Apply principle of least privilege'
    },
    {
      id: 'sec-004',
      severity: 'medium',
      title: 'Network security group allows RDP from internet',
      resource: 'nsg-frontend',
      recommendation: 'Restrict RDP access to specific IPs'
    }
  ],
  recommendations: [
    'Enable Azure Defender for all resource types',
    'Implement Just-In-Time VM access',
    'Configure Azure Policy for compliance enforcement',
    'Enable diagnostic logging for all resources'
  ]
};

export const SecurityPosturePanel: React.FC = () => {
  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-500';
    if (score >= 70) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
      case 'high':
        return 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
      default:
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400';
    }
  };

  return (
    <div className="space-y-6">
      {/* Overall Score Card */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            <Shield className="w-5 h-5 text-purple-500" />
            Security Posture Score
          </h3>
          <div className="flex items-center gap-2">
            {securityPostureData.trend === 'improving' ? (
              <TrendingUp className="w-4 h-4 text-green-500" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-500" />
            )}
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {securityPostureData.trend === 'improving' ? '+3%' : '-2%'} this week
            </span>
          </div>
        </div>

        <div className="flex items-center justify-center">
          <div className="relative w-32 h-32">
            <svg className="w-32 h-32 transform -rotate-90">
              <circle
                cx="64"
                cy="64"
                r="56"
                stroke="currentColor"
                strokeWidth="12"
                fill="none"
                className="text-gray-200 dark:text-gray-700"
              />
              <circle
                cx="64"
                cy="64"
                r="56"
                stroke="currentColor"
                strokeWidth="12"
                fill="none"
                strokeDasharray={`${2 * Math.PI * 56}`}
                strokeDashoffset={`${2 * Math.PI * 56 * (1 - securityPostureData.overallScore / 100)}`}
                className={getScoreColor(securityPostureData.overallScore)}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className={`text-3xl font-bold ${getScoreColor(securityPostureData.overallScore)}`}>
                {securityPostureData.overallScore}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Category Scores */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Security Categories
        </h3>
        <div className="grid grid-cols-2 gap-4">
          {Object.entries(securityPostureData.categories).map(([category, data]) => (
            <div key={category} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300 capitalize">
                  {category}
                </span>
                <span className={`text-lg font-bold ${getScoreColor(data.score)}`}>
                  {data.score}
                </span>
              </div>
              <div className="flex items-center gap-4 text-xs">
                <span className="text-red-500">{data.critical} Critical</span>
                <span className="text-orange-500">{data.high} High</span>
                <span className="text-yellow-500">{data.medium} Medium</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Top Security Findings */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Top Security Findings
        </h3>
        <div className="space-y-3">
          {securityPostureData.topFindings.map((finding) => (
            <div key={finding.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getSeverityColor(finding.severity)}`}>
                      {finding.severity.toUpperCase()}
                    </span>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {finding.title}
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                    Resource: <span className="font-mono">{finding.resource}</span>
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-300">
                    <strong>Recommendation:</strong> {finding.recommendation}
                  </p>
                </div>
                <button className="text-purple-500 hover:text-purple-600 text-sm">
                  Fix →
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recommendations */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-300 mb-3 flex items-center gap-2">
          <Info className="w-5 h-5" />
          Security Recommendations
        </h3>
        <ul className="space-y-2">
          {securityPostureData.recommendations.map((rec, index) => (
            <li key={index} className="flex items-start gap-2">
              <CheckCircle className="w-4 h-4 text-blue-500 mt-0.5" />
              <span className="text-sm text-blue-800 dark:text-blue-300">{rec}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Demo Mode Notice */}
      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
        <p className="text-xs text-purple-600 dark:text-purple-400">
          Demo Mode: Showing sample security posture data. In production, this integrates with Azure Defender and Security Center.
        </p>
      </div>
    </div>
  );
};