'use client'

import { useState } from 'react'
import { GitBranch, Shield, CheckCircle, XCircle, AlertTriangle, Play, Pause, RefreshCw, Clock, Zap } from 'lucide-react'
import MetricCard from '@/components/MetricCard'
import ChartContainer from '@/components/ChartContainer'

export default function DevSecOpsPipelinesPage() {
  const [selectedPipeline, setSelectedPipeline] = useState<string | null>(null)
  const [scanResults, setScanResults] = useState<boolean>(false)

  const pipelines = [
    {
      id: 'main-deploy',
      name: 'Main Branch Deploy',
      repo: 'policycortex/core',
      status: 'running',
      securityScore: 92,
      lastRun: '12 minutes ago',
      duration: '8m 34s',
      stages: [
        { name: 'Build', status: 'completed', duration: '2m 12s' },
        { name: 'Security Scan', status: 'running', duration: '1m 45s' },
        { name: 'Policy Check', status: 'pending', duration: '-' },
        { name: 'Deploy', status: 'pending', duration: '-' }
      ],
      vulnerabilities: {
        critical: 0,
        high: 2,
        medium: 5,
        low: 12
      }
    },
    {
      id: 'feature-pr',
      name: 'Feature PR #234',
      repo: 'policycortex/frontend',
      status: 'failed',
      securityScore: 78,
      lastRun: '1 hour ago',
      duration: '5m 21s',
      stages: [
        { name: 'Build', status: 'completed', duration: '1m 34s' },
        { name: 'Security Scan', status: 'failed', duration: '2m 10s' },
        { name: 'Policy Check', status: 'skipped', duration: '-' },
        { name: 'Deploy', status: 'skipped', duration: '-' }
      ],
      vulnerabilities: {
        critical: 1,
        high: 4,
        medium: 8,
        low: 15
      }
    },
    {
      id: 'hotfix',
      name: 'Hotfix Release',
      repo: 'policycortex/api',
      status: 'success',
      securityScore: 98,
      lastRun: '3 hours ago',
      duration: '6m 45s',
      stages: [
        { name: 'Build', status: 'completed', duration: '1m 23s' },
        { name: 'Security Scan', status: 'completed', duration: '2m 05s' },
        { name: 'Policy Check', status: 'completed', duration: '1m 12s' },
        { name: 'Deploy', status: 'completed', duration: '2m 05s' }
      ],
      vulnerabilities: {
        critical: 0,
        high: 0,
        medium: 2,
        low: 7
      }
    }
  ]

  const securityPolicies = [
    {
      name: 'No Critical Vulnerabilities',
      status: 'active',
      enforcement: 'blocking',
      violations: 1,
      lastViolation: '1 hour ago'
    },
    {
      name: 'OWASP Top 10 Compliance',
      status: 'active',
      enforcement: 'blocking',
      violations: 0,
      lastViolation: 'Never'
    },
    {
      name: 'Dependency License Check',
      status: 'active',
      enforcement: 'warning',
      violations: 3,
      lastViolation: '2 days ago'
    },
    {
      name: 'Container Image Signing',
      status: 'active',
      enforcement: 'blocking',
      violations: 0,
      lastViolation: 'Never'
    },
    {
      name: 'Secret Detection',
      status: 'active',
      enforcement: 'blocking',
      violations: 2,
      lastViolation: '5 days ago'
    }
  ]

  const integrations = [
    { name: 'GitHub Actions', status: 'connected', lastSync: '2 minutes ago' },
    { name: 'Azure DevOps', status: 'connected', lastSync: '5 minutes ago' },
    { name: 'GitLab CI', status: 'connected', lastSync: '10 minutes ago' },
    { name: 'Jenkins', status: 'disconnected', lastSync: 'Never' },
    { name: 'CircleCI', status: 'connected', lastSync: '1 hour ago' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-3 bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg">
              <GitBranch className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                DevSecOps Pipeline Integration
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Shift-left security with native CI/CD integration
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <MetricCard
            title="Security Score"
            value="89%"
            subtitle="Across all pipelines"
            trend="up"
            icon={<Shield className="w-5 h-5 text-green-500" />}
          />
          <MetricCard
            title="Vulnerabilities Blocked"
            value="234"
            subtitle="This week"
            icon={<XCircle className="w-5 h-5 text-red-500" />}
          />
          <MetricCard
            title="Policy Violations"
            value="6"
            subtitle="Requiring review"
            alert="Action needed"
            icon={<AlertTriangle className="w-5 h-5 text-orange-500" />}
          />
          <MetricCard
            title="Deploy Frequency"
            value="47/day"
            subtitle="15% faster"
            trend="up"
            icon={<Zap className="w-5 h-5 text-blue-500" />}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Active Pipelines</h3>
              <div className="space-y-4">
                {pipelines.map((pipeline) => (
                  <div
                    key={pipeline.id}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      selectedPipeline === pipeline.id
                        ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-purple-300'
                    }`}
                    onClick={() => setSelectedPipeline(pipeline.id)}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <h4 className="font-semibold">{pipeline.name}</h4>
                          <div className={`px-2 py-0.5 rounded text-xs font-medium ${
                            pipeline.status === 'running' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 animate-pulse' :
                            pipeline.status === 'success' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                            'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                          }`}>
                            {pipeline.status.toUpperCase()}
                          </div>
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {pipeline.repo} • {pipeline.lastRun} • {pipeline.duration}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-2xl font-bold ${
                          pipeline.securityScore >= 90 ? 'text-green-600' :
                          pipeline.securityScore >= 70 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {pipeline.securityScore}
                        </div>
                        <div className="text-xs text-gray-500">Security Score</div>
                      </div>
                    </div>
                    
                    <div className="flex gap-2 mb-3">
                      {pipeline.stages.map((stage, idx) => (
                        <div key={idx} className="flex-1">
                          <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">{stage.name}</div>
                          <div className={`h-2 rounded-full ${
                            stage.status === 'completed' ? 'bg-green-500' :
                            stage.status === 'running' ? 'bg-blue-500 animate-pulse' :
                            stage.status === 'failed' ? 'bg-red-500' :
                            stage.status === 'skipped' ? 'bg-gray-300' : 'bg-gray-200'
                          }`} />
                        </div>
                      ))}
                    </div>
                    
                    <div className="flex items-center gap-4 text-sm">
                      <span className={`${pipeline.vulnerabilities.critical > 0 ? 'text-red-600 font-medium' : 'text-gray-600 dark:text-gray-400'}`}>
                        {pipeline.vulnerabilities.critical} Critical
                      </span>
                      <span className={`${pipeline.vulnerabilities.high > 0 ? 'text-orange-600 font-medium' : 'text-gray-600 dark:text-gray-400'}`}>
                        {pipeline.vulnerabilities.high} High
                      </span>
                      <span className="text-gray-600 dark:text-gray-400">
                        {pipeline.vulnerabilities.medium} Medium
                      </span>
                      <span className="text-gray-600 dark:text-gray-400">
                        {pipeline.vulnerabilities.low} Low
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Security Policies</h3>
              <div className="space-y-3">
                {securityPolicies.map((policy, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                    <div className="flex items-center gap-3">
                      <Shield className={`w-5 h-5 ${
                        policy.violations === 0 ? 'text-green-500' : 'text-orange-500'
                      }`} />
                      <div>
                        <div className="font-medium">{policy.name}</div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {policy.enforcement === 'blocking' ? 'Blocking' : 'Warning'} • 
                          {policy.violations > 0 ? ` ${policy.violations} violations` : ' No violations'}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`px-2 py-1 rounded text-xs font-medium ${
                        policy.status === 'active' 
                          ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                          : 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400'
                      }`}>
                        {policy.status.toUpperCase()}
                      </div>
                      {policy.lastViolation !== 'Never' && (
                        <div className="text-xs text-gray-500 mt-1">{policy.lastViolation}</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">CI/CD Integrations</h3>
              <div className="space-y-3">
                {integrations.map((integration, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                    <div>
                      <div className="font-medium">{integration.name}</div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">
                        Last sync: {integration.lastSync}
                      </div>
                    </div>
                    <div className={`w-3 h-3 rounded-full ${
                      integration.status === 'connected' ? 'bg-green-500' : 'bg-red-500'
                    }`} />
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg p-6 text-white">
              <h3 className="text-lg font-semibold mb-2">Security Gates</h3>
              <p className="text-sm opacity-90 mb-4">
                Automatically block deployments with critical vulnerabilities
              </p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="opacity-75">Blocked this week:</span>
                  <span className="font-semibold">12 deployments</span>
                </div>
                <div className="flex justify-between">
                  <span className="opacity-75">Vulnerabilities prevented:</span>
                  <span className="font-semibold">234</span>
                </div>
                <div className="flex justify-between">
                  <span className="opacity-75">Compliance maintained:</span>
                  <span className="font-semibold">100%</span>
                </div>
              </div>
              <button 
                onClick={() => setScanResults(!scanResults)}
                className="w-full mt-4 px-4 py-2 bg-white text-purple-600 rounded-lg hover:bg-gray-100 transition-colors"
              >
                Configure Gates
              </button>
            </div>

            <ChartContainer title="Security Trend">
              <div className="h-48 flex items-center justify-center text-gray-500">
                Security score over time
              </div>
            </ChartContainer>
          </div>
        </div>
      </div>
    </div>
  )
}