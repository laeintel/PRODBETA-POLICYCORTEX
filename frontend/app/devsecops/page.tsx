'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  GitBranch,
  Shield,
  Code,
  Terminal,
  Lock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Zap,
  Package,
  FileCode,
  GitMerge,
  GitPullRequest,
  Activity,
  Settings,
  Play,
  Pause,
  RefreshCw,
  Download,
  Brain
} from 'lucide-react';
import { MLPredictionEngine, PredictionResult } from '@/lib/ml-predictions';

interface Pipeline {
  id: string;
  name: string;
  status: 'running' | 'passed' | 'failed' | 'pending';
  stage: string;
  securityGates: {
    sast: 'passed' | 'failed' | 'running' | 'pending';
    dast: 'passed' | 'failed' | 'running' | 'pending';
    secrets: 'passed' | 'failed' | 'running' | 'pending';
    compliance: 'passed' | 'failed' | 'running' | 'pending';
  };
  policyViolations: number;
  duration: string;
  trigger: string;
}

interface PolicyAsCode {
  id: string;
  name: string;
  description: string;
  language: 'rego' | 'sentinel' | 'yaml';
  status: 'active' | 'draft' | 'deprecated';
  appliedTo: string[];
  violations: number;
  lastModified: Date;
}

export default function DevSecOpsIntegrationHub() {
  const router = useRouter();
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [policies, setPolicies] = useState<PolicyAsCode[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [activeIntegrations, setActiveIntegrations] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load DevSecOps data
    const loadData = async () => {
      // Load ML predictions for security
      const securityPrediction = await MLPredictionEngine.predictSecurityBreach('main-pipeline');
      const compliancePrediction = await MLPredictionEngine.predictComplianceDrift('deployment-policy');
      setPredictions([securityPrediction, compliancePrediction]);

      // Mock pipeline data
      setPipelines([
        {
          id: 'pipe-1',
          name: 'main-service-deploy',
          status: 'running',
          stage: 'Security Scanning',
          securityGates: {
            sast: 'passed',
            dast: 'running',
            secrets: 'passed',
            compliance: 'pending'
          },
          policyViolations: 0,
          duration: '3m 42s',
          trigger: 'git push'
        },
        {
          id: 'pipe-2',
          name: 'api-gateway-release',
          status: 'failed',
          stage: 'Policy Check',
          securityGates: {
            sast: 'passed',
            dast: 'passed',
            secrets: 'failed',
            compliance: 'failed'
          },
          policyViolations: 3,
          duration: '5m 18s',
          trigger: 'merge request'
        },
        {
          id: 'pipe-3',
          name: 'frontend-preview',
          status: 'passed',
          stage: 'Deployed',
          securityGates: {
            sast: 'passed',
            dast: 'passed',
            secrets: 'passed',
            compliance: 'passed'
          },
          policyViolations: 0,
          duration: '7m 23s',
          trigger: 'scheduled'
        }
      ]);

      // Mock Policy-as-Code data
      setPolicies([
        {
          id: 'pol-1',
          name: 'require-encryption',
          description: 'All data at rest must be encrypted with AES-256',
          language: 'rego',
          status: 'active',
          appliedTo: ['production', 'staging'],
          violations: 2,
          lastModified: new Date(Date.now() - 86400000)
        },
        {
          id: 'pol-2',
          name: 'no-public-access',
          description: 'No resources should be publicly accessible without approval',
          language: 'sentinel',
          status: 'active',
          appliedTo: ['all-environments'],
          violations: 0,
          lastModified: new Date(Date.now() - 172800000)
        },
        {
          id: 'pol-3',
          name: 'compliance-tagging',
          description: 'All resources must have required compliance tags',
          language: 'yaml',
          status: 'active',
          appliedTo: ['production'],
          violations: 12,
          lastModified: new Date(Date.now() - 259200000)
        }
      ]);

      setActiveIntegrations(['github', 'gitlab', 'jenkins', 'azure-devops']);
      setLoading(false);
    };

    loadData();
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed': return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'failed': return <XCircle className="h-5 w-5 text-red-500" />;
      case 'running': return <Activity className="h-5 w-5 text-blue-500 animate-pulse" />;
      case 'pending': return <Clock className="h-5 w-5 text-gray-400" />;
      default: return null;
    }
  };

  const integrations = [
    {
      id: 'pipelines',
      title: 'CI/CD Pipeline Integration',
      description: 'Native integration with Jenkins, GitHub Actions, GitLab CI, Azure DevOps',
      icon: GitBranch,
      route: '/devsecops/pipelines',
      stats: { active: pipelines.length, failed: pipelines.filter(p => p.status === 'failed').length },
      color: 'blue'
    },
    {
      id: 'policy-code',
      title: 'Policy-as-Code Management',
      description: 'Git-based policy management with OPA, Sentinel, and custom DSL',
      icon: FileCode,
      route: '/devsecops/policy-code',
      stats: { policies: policies.length, violations: policies.reduce((sum, p) => sum + p.violations, 0) },
      color: 'purple'
    },
    {
      id: 'gates',
      title: 'Security Gate Configuration',
      description: 'Automated security checks at every stage of deployment',
      icon: Shield,
      route: '/devsecops/gates',
      stats: { gates: 8, blocked: 3 },
      color: 'red'
    },
    {
      id: 'ide-plugins',
      title: 'IDE Plugin Hub',
      description: 'VS Code, IntelliJ, and Eclipse plugins for inline policy checks',
      icon: Code,
      route: '/devsecops/ide-plugins',
      stats: { downloads: '12.3K', active: '3.2K' },
      color: 'green'
    }
  ];

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold flex items-center gap-3">
            <GitMerge className="h-10 w-10 text-purple-600" />
            DevSecOps Integration Hub
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Shift-left security with native CI/CD integration and policy-as-code
          </p>
        </div>
        <div className="flex gap-3">
          <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Configure Integrations
          </button>
        </div>
      </div>

      {/* ML Prediction Alert */}
      {predictions.length > 0 && predictions[0].riskLevel === 'critical' && (
        <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Brain className="h-6 w-6 text-red-600 dark:text-red-400 mt-1" />
            <div className="flex-1">
              <h3 className="font-semibold text-red-900 dark:text-red-100">
                Security Risk Prediction
              </h3>
              <p className="text-red-700 dark:text-red-300 mt-1">
                {predictions[0].prediction}
              </p>
              <p className="text-sm text-red-600 dark:text-red-400 mt-1">
                {predictions[0].explanation}
              </p>
              <div className="flex gap-2 mt-3">
                <button className="px-3 py-1 bg-red-600 text-white rounded-md text-sm hover:bg-red-700">
                  Block Deployments
                </button>
                <button className="px-3 py-1 bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300 rounded-md text-sm">
                  Review Policies
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Active Integrations */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Active Integrations</h2>
        <div className="flex gap-4 flex-wrap">
          {['GitHub Actions', 'GitLab CI', 'Jenkins', 'Azure DevOps', 'CircleCI', 'Bitbucket'].map((integration) => {
            const isActive = activeIntegrations.includes(integration.toLowerCase().replace(' ', '-'));
            return (
              <div
                key={integration}
                className={`px-4 py-2 rounded-lg border ${
                  isActive 
                    ? 'bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700' 
                    : 'bg-gray-50 dark:bg-gray-900/20 border-gray-300 dark:border-gray-700'
                }`}
              >
                <div className="flex items-center gap-2">
                  {isActive ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : (
                    <XCircle className="h-4 w-4 text-gray-400" />
                  )}
                  <span className={isActive ? 'font-medium' : 'text-gray-500'}>
                    {integration}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Integration Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {integrations.map((integration) => {
          const Icon = integration.icon;
          return (
            <div
              key={integration.id}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-lg transition-all cursor-pointer"
              onClick={() => router.push(integration.route)}
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className={`p-3 rounded-lg bg-${integration.color}-50 dark:bg-${integration.color}-900/20`}>
                    <Icon className={`h-8 w-8 text-${integration.color}-600 dark:text-${integration.color}-400`} />
                  </div>
                  <div className="text-right">
                    {Object.entries(integration.stats).map(([key, value]) => (
                      <p key={key} className="text-sm">
                        <span className="text-gray-500">{key}:</span>{' '}
                        <span className="font-semibold">{value}</span>
                      </p>
                    ))}
                  </div>
                </div>
                <h3 className="text-xl font-semibold mb-2">{integration.title}</h3>
                <p className="text-gray-600 dark:text-gray-400 text-sm">
                  {integration.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Active Pipelines */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Active Pipelines</h2>
          <button className="text-sm text-blue-600 dark:text-blue-400 hover:underline">
            View All Pipelines
          </button>
        </div>
        <div className="space-y-3">
          {pipelines.map((pipeline) => (
            <div
              key={pipeline.id}
              className="border dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {getStatusIcon(pipeline.status)}
                  <div>
                    <h3 className="font-semibold">{pipeline.name}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Stage: {pipeline.stage} • Duration: {pipeline.duration} • Trigger: {pipeline.trigger}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex gap-2">
                    <div className="text-center">
                      <p className="text-xs text-gray-500">SAST</p>
                      {getStatusIcon(pipeline.securityGates.sast)}
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-gray-500">DAST</p>
                      {getStatusIcon(pipeline.securityGates.dast)}
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-gray-500">Secrets</p>
                      {getStatusIcon(pipeline.securityGates.secrets)}
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-gray-500">Compliance</p>
                      {getStatusIcon(pipeline.securityGates.compliance)}
                    </div>
                  </div>
                  {pipeline.policyViolations > 0 && (
                    <span className="px-2 py-1 bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300 rounded-full text-xs font-semibold">
                      {pipeline.policyViolations} violations
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Policy-as-Code Overview */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Policy-as-Code Repository</h2>
          <button
            onClick={() => router.push('/devsecops/policy-code')}
            className="px-3 py-1 bg-purple-600 text-white rounded-md text-sm hover:bg-purple-700"
          >
            Manage Policies
          </button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {policies.map((policy) => (
            <div
              key={policy.id}
              className="border dark:border-gray-700 rounded-lg p-4"
            >
              <div className="flex items-start justify-between mb-2">
                <h3 className="font-semibold">{policy.name}</h3>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  policy.status === 'active' 
                    ? 'bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300'
                    : 'bg-gray-100 dark:bg-gray-900/50 text-gray-700 dark:text-gray-300'
                }`}>
                  {policy.status.toUpperCase()}
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                {policy.description}
              </p>
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>Language: {policy.language.toUpperCase()}</span>
                {policy.violations > 0 && (
                  <span className="text-red-600 font-medium">
                    {policy.violations} violations
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Add missing Clock import
import { Clock } from 'lucide-react';