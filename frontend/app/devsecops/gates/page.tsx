'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Shield,
  Lock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  Play,
  Pause,
  Settings,
  Plus,
  RefreshCw,
  Activity,
  Zap,
  Brain,
  Target,
  Search,
  Filter,
  GitBranch,
  Code,
  Database,
  Globe,
  Key,
  Bug,
  FileSearch,
  UserCheck,
  AlertCircle,
  TrendingUp,
  BarChart3,
  Package
} from 'lucide-react';
import ViewToggle from '@/components/ViewToggle';
import ChartContainer from '@/components/ChartContainer';
import MetricCard from '@/components/MetricCard';
import { MLPredictionEngine, PredictionResult } from '@/lib/ml-predictions';

interface SecurityGate {
  id: string;
  name: string;
  description: string;
  type: 'sast' | 'dast' | 'secrets' | 'dependencies' | 'compliance' | 'license' | 'container' | 'infrastructure';
  stage: 'pre-commit' | 'build' | 'test' | 'pre-deploy' | 'post-deploy';
  status: 'active' | 'inactive' | 'configuring';
  severity: 'low' | 'medium' | 'high' | 'critical';
  blockingEnabled: boolean;
  lastRun: Date;
  successRate: number;
  averageDuration: string;
  findings: number;
  falsePositives: number;
  configuration: Record<string, any>;
  integrations: string[];
}

interface GateExecution {
  id: string;
  gateId: string;
  gateName: string;
  pipelineId: string;
  pipelineName: string;
  status: 'running' | 'passed' | 'failed' | 'warning' | 'skipped';
  startTime: Date;
  duration?: string;
  findings: SecurityFinding[];
  blockingReason?: string;
}

interface SecurityFinding {
  id: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  file?: string;
  line?: number;
  remediation: string;
  cveId?: string;
  falsePositive: boolean;
}

export default function SecurityGatesPage() {
  const router = useRouter();
  const [view, setView] = useState<'cards' | 'visualizations'>('cards');
  const [gates, setGates] = useState<SecurityGate[]>([]);
  const [executions, setExecutions] = useState<GateExecution[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedGate, setSelectedGate] = useState<string | null>(null);
  const [filterStage, setFilterStage] = useState<string>('all');
  const [filterType, setFilterType] = useState<string>('all');

  useEffect(() => {
    loadGateData();
  }, []);

  const loadGateData = async () => {
    // Load ML predictions for security risks
    const securityPrediction = await MLPredictionEngine.predictSecurityBreach('security-gates');
    setPredictions([securityPrediction]);

    // Mock security gates
    const mockGates: SecurityGate[] = [
      {
        id: 'gate-001',
        name: 'Static Application Security Testing',
        description: 'Scans source code for security vulnerabilities using multiple SAST engines',
        type: 'sast',
        stage: 'build',
        status: 'active',
        severity: 'high',
        blockingEnabled: true,
        lastRun: new Date(Date.now() - 3600000),
        successRate: 94.2,
        averageDuration: '4m 32s',
        findings: 12,
        falsePositives: 3,
        configuration: {
          engines: ['SonarQube', 'Checkmarx', 'Veracode'],
          languages: ['JavaScript', 'TypeScript', 'Python', 'Java'],
          rules: 'OWASP Top 10',
          failureThreshold: 'high'
        },
        integrations: ['GitHub', 'Azure DevOps', 'Jenkins']
      },
      {
        id: 'gate-002',
        name: 'Dynamic Application Security Testing',
        description: 'Runtime security testing of running applications and APIs',
        type: 'dast',
        stage: 'test',
        status: 'active',
        severity: 'high',
        blockingEnabled: true,
        lastRun: new Date(Date.now() - 7200000),
        successRate: 89.7,
        averageDuration: '12m 15s',
        findings: 8,
        falsePositives: 1,
        configuration: {
          scanner: 'OWASP ZAP',
          testTypes: ['Spider', 'Active Scan', 'API Security'],
          targetUrls: ['staging.example.com', 'api.staging.example.com'],
          authMethod: 'JWT'
        },
        integrations: ['OWASP ZAP', 'Burp Suite']
      },
      {
        id: 'gate-003',
        name: 'Secrets Detection',
        description: 'Scans code and configuration for exposed secrets and API keys',
        type: 'secrets',
        stage: 'pre-commit',
        status: 'active',
        severity: 'critical',
        blockingEnabled: true,
        lastRun: new Date(Date.now() - 1800000),
        successRate: 98.5,
        averageDuration: '45s',
        findings: 2,
        falsePositives: 0,
        configuration: {
          scanner: 'GitLeaks',
          patterns: ['AWS Keys', 'API Keys', 'Private Keys', 'Database Passwords'],
          whitelistFiles: ['.env.example', 'test-fixtures'],
          entropy: true
        },
        integrations: ['GitLeaks', 'TruffleHog', 'Detect-Secrets']
      },
      {
        id: 'gate-004',
        name: 'Dependency Vulnerability Scan',
        description: 'Checks third-party dependencies for known security vulnerabilities',
        type: 'dependencies',
        stage: 'build',
        status: 'active',
        severity: 'medium',
        blockingEnabled: false,
        lastRun: new Date(Date.now() - 14400000),
        successRate: 92.1,
        averageDuration: '2m 18s',
        findings: 15,
        falsePositives: 5,
        configuration: {
          scanner: 'npm audit',
          databases: ['NPM Advisory', 'NVD', 'Snyk'],
          severity: 'medium',
          excludeDevDependencies: true
        },
        integrations: ['npm', 'Snyk', 'WhiteSource']
      },
      {
        id: 'gate-005',
        name: 'Container Security Scan',
        description: 'Scans container images for vulnerabilities and misconfigurations',
        type: 'container',
        stage: 'pre-deploy',
        status: 'active',
        severity: 'high',
        blockingEnabled: true,
        lastRun: new Date(Date.now() - 5400000),
        successRate: 87.3,
        averageDuration: '3m 42s',
        findings: 22,
        falsePositives: 8,
        configuration: {
          scanner: 'Trivy',
          registries: ['ACR', 'Docker Hub'],
          policies: ['CIS Benchmarks', 'NIST'],
          baseImageUpdates: true
        },
        integrations: ['Trivy', 'Aqua', 'Twistlock']
      },
      {
        id: 'gate-006',
        name: 'Infrastructure as Code Security',
        description: 'Validates Terraform and ARM templates for security misconfigurations',
        type: 'infrastructure',
        stage: 'build',
        status: 'active',
        severity: 'medium',
        blockingEnabled: false,
        lastRun: new Date(Date.now() - 10800000),
        successRate: 91.8,
        averageDuration: '1m 25s',
        findings: 7,
        falsePositives: 2,
        configuration: {
          scanner: 'Checkov',
          frameworks: ['Terraform', 'ARM', 'CloudFormation'],
          policies: ['CIS', 'NIST', 'Custom'],
          skipChecks: ['CKV_AWS_20']
        },
        integrations: ['Checkov', 'Terrascan', 'TFSec']
      }
    ];

    setGates(mockGates);

    // Mock recent executions
    const mockExecutions: GateExecution[] = [
      {
        id: 'exec-001',
        gateId: 'gate-001',
        gateName: 'Static Application Security Testing',
        pipelineId: 'pipeline-main',
        pipelineName: 'Main API Pipeline',
        status: 'failed',
        startTime: new Date(Date.now() - 3600000),
        duration: '4m 32s',
        findings: [
          {
            id: 'finding-001',
            type: 'SQL Injection',
            severity: 'high',
            title: 'Potential SQL Injection vulnerability',
            description: 'User input is directly concatenated into SQL query without sanitization',
            file: 'src/database/queries.js',
            line: 45,
            remediation: 'Use parameterized queries or prepared statements',
            falsePositive: false
          }
        ],
        blockingReason: 'High severity SQL Injection vulnerability detected'
      },
      {
        id: 'exec-002',
        gateId: 'gate-003',
        gateName: 'Secrets Detection',
        pipelineId: 'pipeline-api',
        pipelineName: 'API Gateway Pipeline',
        status: 'passed',
        startTime: new Date(Date.now() - 1800000),
        duration: '45s',
        findings: []
      },
      {
        id: 'exec-003',
        gateId: 'gate-002',
        gateName: 'Dynamic Application Security Testing',
        pipelineId: 'pipeline-frontend',
        pipelineName: 'Frontend Pipeline',
        status: 'running',
        startTime: new Date(Date.now() - 420000),
        findings: []
      }
    ];

    setExecutions(mockExecutions);
    setLoading(false);
  };

  const getGateTypeIcon = (type: string) => {
    switch (type) {
      case 'sast': return <Code className="h-5 w-5" />;
      case 'dast': return <Globe className="h-5 w-5" />;
      case 'secrets': return <Key className="h-5 w-5" />;
      case 'dependencies': return <Package className="h-5 w-5" />;
      case 'compliance': return <UserCheck className="h-5 w-5" />;
      case 'license': return <FileSearch className="h-5 w-5" />;
      case 'container': return <Database className="h-5 w-5" />;
      case 'infrastructure': return <Settings className="h-5 w-5" />;
      default: return <Shield className="h-5 w-5" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed': return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'failed': return <XCircle className="h-5 w-5 text-red-500" />;
      case 'running': return <Activity className="h-5 w-5 text-blue-500 animate-pulse" />;
      case 'warning': return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      case 'skipped': return <Clock className="h-5 w-5 text-gray-400" />;
      default: return null;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400';
      case 'high': return 'text-orange-600 bg-orange-50 dark:bg-orange-900/20 dark:text-orange-400';
      case 'medium': return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'low': return 'text-blue-600 bg-blue-50 dark:bg-blue-900/20 dark:text-blue-400';
      default: return 'text-gray-600 bg-gray-50 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const filteredGates = gates.filter(gate => {
    const stageMatch = filterStage === 'all' || gate.stage === filterStage;
    const typeMatch = filterType === 'all' || gate.type === filterType;
    return stageMatch && typeMatch;
  });

  const metrics = [
    {
      id: 'active-gates',
      title: 'Active Security Gates',
      value: gates.filter(g => g.status === 'active').length,
      change: 12.5,
      trend: 'up' as const,
      sparklineData: [6, 7, 8, 9, 8, gates.filter(g => g.status === 'active').length],
      alert: `${gates.filter(g => g.blockingEnabled).length} blocking enabled`
    },
    {
      id: 'blocked-deployments',
      title: 'Blocked Deployments (24h)',
      value: executions.filter(e => e.status === 'failed').length,
      change: -8.3,
      trend: 'down' as const,
      sparklineData: [5, 4, 6, 3, 2, executions.filter(e => e.status === 'failed').length]
    },
    {
      id: 'total-findings',
      title: 'Security Findings',
      value: gates.reduce((sum, g) => sum + g.findings, 0),
      change: -15.2,
      trend: 'down' as const,
      sparklineData: [78, 72, 68, 65, 62, gates.reduce((sum, g) => sum + g.findings, 0)]
    },
    {
      id: 'avg-success-rate',
      title: 'Average Success Rate',
      value: `${Math.round(gates.reduce((sum, g) => sum + g.successRate, 0) / gates.length || 0)}%`,
      change: 3.7,
      trend: 'up' as const,
      sparklineData: [89, 90, 91, 92, 91, Math.round(gates.reduce((sum, g) => sum + g.successRate, 0) / gates.length || 0)]
    }
  ];

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin text-blue-600" />
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold flex items-center gap-3">
            <Shield className="h-10 w-10 text-red-600" />
            Security Gates Configuration
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Automated security checks at every stage of the CI/CD pipeline
          </p>
        </div>
        <div className="flex gap-3">
          <ViewToggle view={view} onViewChange={setView} />
          <button
            onClick={() => router.push('/devsecops/gates/new')}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center gap-2"
          >
            <Plus className="h-5 w-5" />
            New Gate
          </button>
          <button
            onClick={() => loadGateData()}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <RefreshCw className="h-5 w-5" />
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
                Security Risk Prediction
              </h3>
              <p className="text-red-700 dark:text-red-300 mt-1">
                AI detected potential bypass attempt in security gate configuration
              </p>
              <button className="mt-2 px-3 py-1 bg-red-600 text-white rounded-md text-sm hover:bg-red-700">
                Review Gate Settings
              </button>
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
          {/* Filters */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-4 mb-6">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2">
                <Filter className="h-5 w-5 text-gray-500" />
                <span className="font-medium">Filters:</span>
              </div>
              <select
                value={filterStage}
                onChange={(e) => setFilterStage(e.target.value)}
                className="px-3 py-1 border dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm"
              >
                <option value="all">All Stages</option>
                <option value="pre-commit">Pre-commit</option>
                <option value="build">Build</option>
                <option value="test">Test</option>
                <option value="pre-deploy">Pre-deploy</option>
                <option value="post-deploy">Post-deploy</option>
              </select>
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="px-3 py-1 border dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm"
              >
                <option value="all">All Types</option>
                <option value="sast">SAST</option>
                <option value="dast">DAST</option>
                <option value="secrets">Secrets</option>
                <option value="dependencies">Dependencies</option>
                <option value="container">Container</option>
                <option value="infrastructure">Infrastructure</option>
              </select>
              <span className="text-sm text-gray-500">
                {filteredGates.length} of {gates.length} gates
              </span>
            </div>
          </div>

          {/* Security Gates */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Target className="h-6 w-6 text-red-600" />
              Security Gates
            </h2>
            <div className="space-y-4">
              {filteredGates.map((gate) => (
                <div
                  key={gate.id}
                  className={`border dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-all ${
                    selectedGate === gate.id ? 'border-red-500 bg-red-50 dark:bg-red-900/20' : ''
                  }`}
                  onClick={() => setSelectedGate(selectedGate === gate.id ? null : gate.id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <div className="p-2 bg-red-50 dark:bg-red-900/20 rounded-lg">
                          {getGateTypeIcon(gate.type)}
                        </div>
                        <h3 className="font-semibold text-lg">{gate.name}</h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          gate.status === 'active' ? 'text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-400' :
                          gate.status === 'inactive' ? 'text-gray-600 bg-gray-50 dark:bg-gray-900/20 dark:text-gray-400' :
                          'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 dark:text-yellow-400'
                        }`}>
                          {gate.status}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(gate.severity)}`}>
                          {gate.severity}
                        </span>
                        {gate.blockingEnabled && (
                          <span className="px-2 py-1 bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300 rounded-full text-xs font-semibold">
                            BLOCKING
                          </span>
                        )}
                      </div>
                      <p className="text-gray-600 dark:text-gray-400 mb-2">{gate.description}</p>
                      <div className="flex items-center gap-4 text-sm text-gray-500">
                        <span>Stage: {gate.stage}</span>
                        <span>Success Rate: {gate.successRate}%</span>
                        <span>Duration: {gate.averageDuration}</span>
                        <span>Findings: {gate.findings}</span>
                        <span>Last Run: {gate.lastRun.toLocaleString()}</span>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      {gate.status === 'active' ? (
                        <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-600 rounded text-orange-600">
                          <Pause className="h-4 w-4" />
                        </button>
                      ) : (
                        <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-600 rounded text-green-600">
                          <Play className="h-4 w-4" />
                        </button>
                      )}
                      <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-600 rounded">
                        <Settings className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  {selectedGate === gate.id && (
                    <div className="mt-4 pt-4 border-t dark:border-gray-700">
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div>
                          <h4 className="font-medium mb-2">Configuration:</h4>
                          <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded text-sm">
                            <pre className="text-xs overflow-x-auto">
                              {JSON.stringify(gate.configuration, null, 2)}
                            </pre>
                          </div>
                        </div>
                        <div className="space-y-4">
                          <div>
                            <h4 className="font-medium mb-2">Integrations:</h4>
                            <div className="flex flex-wrap gap-2">
                              {gate.integrations.map((integration, idx) => (
                                <span key={idx} className="px-2 py-1 bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded text-sm">
                                  {integration}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div>
                            <h4 className="font-medium mb-2">Statistics:</h4>
                            <div className="text-sm space-y-1">
                              <div className="flex justify-between">
                                <span>Success Rate:</span>
                                <span className="font-semibold">{gate.successRate}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span>False Positives:</span>
                                <span className="font-semibold">{gate.falsePositives}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Average Duration:</span>
                                <span className="font-semibold">{gate.averageDuration}</span>
                              </div>
                            </div>
                          </div>
                          <div className="flex gap-2">
                            <button className="px-3 py-1 bg-red-600 text-white rounded-md text-sm hover:bg-red-700">
                              Configure
                            </button>
                            <button className="px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700">
                              Test Gate
                            </button>
                            <button className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md text-sm hover:bg-gray-200 dark:hover:bg-gray-600">
                              View Logs
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Recent Executions */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Activity className="h-6 w-6 text-blue-600" />
              Recent Gate Executions
            </h2>
            <div className="space-y-3">
              {executions.map((execution) => (
                <div key={execution.id} className="border dark:border-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(execution.status)}
                      <div>
                        <h3 className="font-semibold">{execution.gateName}</h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          Pipeline: {execution.pipelineName} • Started: {execution.startTime.toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      {execution.duration && (
                        <div className="text-sm font-medium">{execution.duration}</div>
                      )}
                      {execution.findings.length > 0 && (
                        <div className="text-sm text-orange-600 dark:text-orange-400">
                          {execution.findings.length} findings
                        </div>
                      )}
                    </div>
                  </div>
                  {execution.blockingReason && (
                    <div className="mt-2 p-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded">
                      <p className="text-red-700 dark:text-red-300 text-sm">
                        <strong>Blocked:</strong> {execution.blockingReason}
                      </p>
                    </div>
                  )}
                  {execution.findings.length > 0 && (
                    <div className="mt-3 space-y-2">
                      {execution.findings.slice(0, 2).map((finding) => (
                        <div key={finding.id} className="p-2 bg-gray-50 dark:bg-gray-700 rounded text-sm">
                          <div className="flex items-center gap-2 mb-1">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(finding.severity)}`}>
                              {finding.severity.toUpperCase()}
                            </span>
                            <span className="font-medium">{finding.title}</span>
                            {finding.file && (
                              <span className="text-gray-500">in {finding.file}:{finding.line}</span>
                            )}
                          </div>
                          <p className="text-gray-600 dark:text-gray-400">{finding.description}</p>
                          <p className="text-blue-600 dark:text-blue-400 mt-1">
                            <span className="font-medium">Fix:</span> {finding.remediation}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <>
          {/* Visualization Mode */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <ChartContainer
              title="Gate Success Rate Trends"
              onDrillIn={() => router.push('/devsecops/gates/analytics')}
            >
              <div className="p-4">
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                  <p className="text-gray-500">Gate success rate trend visualization</p>
                </div>
              </div>
            </ChartContainer>
            <ChartContainer
              title="Security Findings by Type"
              onDrillIn={() => router.push('/devsecops/gates/findings')}
            >
              <div className="p-4">
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                  <p className="text-gray-500">Security findings breakdown chart</p>
                </div>
              </div>
            </ChartContainer>
          </div>

          {/* Gate Performance Analytics */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="h-6 w-6 text-purple-600" />
              Gate Performance Analytics
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="p-4 border dark:border-gray-700 rounded-lg">
                <h3 className="font-semibold mb-3">Fastest Gates</h3>
                <div className="space-y-2">
                  {gates
                    .sort((a, b) => parseFloat(a.averageDuration) - parseFloat(b.averageDuration))
                    .slice(0, 3)
                    .map((gate) => (
                      <div key={gate.id} className="flex justify-between items-center">
                        <span className="text-sm truncate">{gate.name}</span>
                        <span className="text-green-600 dark:text-green-400 font-semibold text-sm">
                          {gate.averageDuration}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
              <div className="p-4 border dark:border-gray-700 rounded-lg">
                <h3 className="font-semibold mb-3">Most Findings</h3>
                <div className="space-y-2">
                  {gates
                    .sort((a, b) => b.findings - a.findings)
                    .slice(0, 3)
                    .map((gate) => (
                      <div key={gate.id} className="flex justify-between items-center">
                        <span className="text-sm truncate">{gate.name}</span>
                        <span className="text-orange-600 dark:text-orange-400 font-semibold text-sm">
                          {gate.findings}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
              <div className="p-4 border dark:border-gray-700 rounded-lg">
                <h3 className="font-semibold mb-3">Highest Success Rate</h3>
                <div className="space-y-2">
                  {gates
                    .sort((a, b) => b.successRate - a.successRate)
                    .slice(0, 3)
                    .map((gate) => (
                      <div key={gate.id} className="flex justify-between items-center">
                        <span className="text-sm truncate">{gate.name}</span>
                        <span className="text-green-600 dark:text-green-400 font-semibold text-sm">
                          {gate.successRate}%
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}