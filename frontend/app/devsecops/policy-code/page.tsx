'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  FileCode,
  GitBranch,
  Shield,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  Edit,
  Eye,
  Download,
  Upload,
  Plus,
  RefreshCw,
  Settings,
  Code,
  Terminal,
  BookOpen,
  Activity,
  Zap,
  Brain,
  Target,
  Users,
  Search,
  Filter,
  ArrowRight
} from 'lucide-react';
import ViewToggle from '@/components/ViewToggle';
import ChartContainer from '@/components/ChartContainer';
import MetricCard from '@/components/MetricCard';
import { MLPredictionEngine, PredictionResult } from '@/lib/ml-predictions';

interface PolicyDefinition {
  id: string;
  name: string;
  description: string;
  language: 'rego' | 'sentinel' | 'yaml' | 'json' | 'hcl';
  category: 'security' | 'compliance' | 'cost' | 'governance' | 'operations';
  status: 'active' | 'draft' | 'deprecated' | 'testing';
  version: string;
  author: string;
  lastModified: Date;
  appliedTo: string[];
  violations: number;
  successRate: number;
  impactLevel: 'low' | 'medium' | 'high' | 'critical';
  codePreview: string;
  dependencies: string[];
  testCoverage: number;
}

interface PolicyViolation {
  id: string;
  policyId: string;
  policyName: string;
  resource: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  timestamp: Date;
  remediation: string;
  status: 'open' | 'investigating' | 'resolved' | 'suppressed';
}

export default function PolicyAsCodePage() {
  const router = useRouter();
  const [view, setView] = useState<'cards' | 'visualizations'>('cards');
  const [policies, setPolicies] = useState<PolicyDefinition[]>([]);
  const [violations, setViolations] = useState<PolicyViolation[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPolicy, setSelectedPolicy] = useState<string | null>(null);
  const [filterLanguage, setFilterLanguage] = useState<string>('all');
  const [filterCategory, setFilterCategory] = useState<string>('all');

  useEffect(() => {
    loadPolicyData();
  }, []);

  const loadPolicyData = async () => {
    // Load ML predictions for policy compliance
    const compliancePrediction = await MLPredictionEngine.predictComplianceDrift('policy-engine');
    setPredictions([compliancePrediction]);

    // Mock policy definitions
    const mockPolicies: PolicyDefinition[] = [
      {
        id: 'pol-001',
        name: 'require-encryption-at-rest',
        description: 'Ensures all storage resources are encrypted at rest with customer-managed keys',
        language: 'rego',
        category: 'security',
        status: 'active',
        version: '2.1.0',
        author: 'security-team',
        lastModified: new Date(Date.now() - 86400000),
        appliedTo: ['production', 'staging'],
        violations: 3,
        successRate: 94.2,
        impactLevel: 'critical',
        codePreview: `package azure.storage.encryption
        
import data.azure.resources

deny[msg] {
  resource := azure.resources[_]
  resource.type == "Microsoft.Storage/storageAccounts"
  not resource.properties.encryption.keySource == "Microsoft.Keyvault"
  msg := sprintf("Storage account '%s' must use customer-managed encryption keys", [resource.name])
}`,
        dependencies: ['azure-policy-lib', 'encryption-utils'],
        testCoverage: 98
      },
      {
        id: 'pol-002',
        name: 'network-security-groups',
        description: 'Validates NSG rules to prevent unrestricted inbound access',
        language: 'sentinel',
        category: 'security',
        status: 'active',
        version: '1.5.2',
        author: 'network-team',
        lastModified: new Date(Date.now() - 172800000),
        appliedTo: ['production', 'staging', 'development'],
        violations: 0,
        successRate: 100,
        impactLevel: 'high',
        codePreview: `import "tfplan/v2" as tfplan

deny_unrestricted_ingress = rule {
  all tfplan.resource_changes as _, changes {
    changes.type is "azurerm_network_security_rule" and
    changes.change.after.access is "Allow" and
    changes.change.after.direction is "Inbound" and
    changes.change.after.source_address_prefix is not "*"
  }
}

main = rule {
  deny_unrestricted_ingress
}`,
        dependencies: ['terraform-sentinel-policies'],
        testCoverage: 92
      },
      {
        id: 'pol-003',
        name: 'cost-budget-limits',
        description: 'Enforces spending limits and budget alerts for resource groups',
        language: 'yaml',
        category: 'cost',
        status: 'active',
        version: '1.2.1',
        author: 'finops-team',
        lastModified: new Date(Date.now() - 259200000),
        appliedTo: ['production'],
        violations: 12,
        successRate: 76.5,
        impactLevel: 'medium',
        codePreview: `apiVersion: v1
kind: Policy
metadata:
  name: cost-budget-limits
spec:
  rules:
    - name: enforce-budget-alerts
      match:
        resources:
          - Microsoft.Resources/resourceGroups
      validate:
        message: "Resource group must have budget alerts configured"
        pattern:
          properties:
            budget:
              alerts:
                - threshold: "<=80"
                - threshold: "<=100"`,
        dependencies: ['azure-resource-manager'],
        testCoverage: 85
      },
      {
        id: 'pol-004',
        name: 'tagging-compliance',
        description: 'Ensures all resources have required compliance and governance tags',
        language: 'rego',
        category: 'governance',
        status: 'active',
        version: '3.0.0',
        author: 'governance-team',
        lastModified: new Date(Date.now() - 432000000),
        appliedTo: ['production', 'staging'],
        violations: 8,
        successRate: 89.3,
        impactLevel: 'medium',
        codePreview: `package azure.tagging

required_tags := {"Environment", "Owner", "CostCenter", "Project", "Compliance"}

deny[msg] {
  resource := input.resource
  missing_tags := required_tags - set(object.get(resource, "tags", {}))
  count(missing_tags) > 0
  msg := sprintf("Resource '%s' is missing required tags: %v", [resource.name, missing_tags])
}`,
        dependencies: ['azure-policy-lib'],
        testCoverage: 96
      },
      {
        id: 'pol-005',
        name: 'kubernetes-security-baseline',
        description: 'Implements security baseline for Kubernetes workloads',
        language: 'rego',
        category: 'security',
        status: 'testing',
        version: '1.0.0-beta',
        author: 'k8s-team',
        lastModified: new Date(Date.now() - 86400000),
        appliedTo: ['development'],
        violations: 15,
        successRate: 67.8,
        impactLevel: 'high',
        codePreview: `package kubernetes.security

deny[msg] {
  input.request.kind.kind == "Pod"
  input.request.object.spec.securityContext.runAsRoot == true
  msg := "Pod must not run as root user"
}

deny[msg] {
  input.request.kind.kind == "Pod"
  container := input.request.object.spec.containers[_]
  not container.securityContext.allowPrivilegeEscalation == false
  msg := sprintf("Container '%s' must disable privilege escalation", [container.name])
}`,
        dependencies: ['opa-gatekeeper', 'k8s-policy-library'],
        testCoverage: 78
      }
    ];

    setPolicies(mockPolicies);

    // Mock violations
    const mockViolations: PolicyViolation[] = [
      {
        id: 'viol-001',
        policyId: 'pol-001',
        policyName: 'require-encryption-at-rest',
        resource: 'storage-account-prod-01',
        severity: 'critical',
        description: 'Storage account is using Microsoft-managed keys instead of customer-managed keys',
        timestamp: new Date(Date.now() - 3600000),
        remediation: 'Configure customer-managed encryption keys in Azure Key Vault',
        status: 'open'
      },
      {
        id: 'viol-002',
        policyId: 'pol-003',
        policyName: 'cost-budget-limits',
        resource: 'rg-analytics-prod',
        severity: 'medium',
        description: 'Resource group exceeds monthly budget threshold without alerts configured',
        timestamp: new Date(Date.now() - 7200000),
        remediation: 'Set up budget alerts at 80% and 100% thresholds',
        status: 'investigating'
      },
      {
        id: 'viol-003',
        policyId: 'pol-004',
        policyName: 'tagging-compliance',
        resource: 'vm-web-server-02',
        severity: 'low',
        description: 'Virtual machine missing required tags: CostCenter, Compliance',
        timestamp: new Date(Date.now() - 10800000),
        remediation: 'Add missing tags to the resource',
        status: 'open'
      }
    ];

    setViolations(mockViolations);
    setLoading(false);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-400';
      case 'testing': return 'text-blue-600 bg-blue-50 dark:bg-blue-900/20 dark:text-blue-400';
      case 'draft': return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'deprecated': return 'text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400';
      default: return 'text-gray-600 bg-gray-50 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const getLanguageColor = (language: string) => {
    switch (language) {
      case 'rego': return 'text-purple-600 bg-purple-50 dark:bg-purple-900/20 dark:text-purple-400';
      case 'sentinel': return 'text-blue-600 bg-blue-50 dark:bg-blue-900/20 dark:text-blue-400';
      case 'yaml': return 'text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-400';
      case 'json': return 'text-orange-600 bg-orange-50 dark:bg-orange-900/20 dark:text-orange-400';
      case 'hcl': return 'text-indigo-600 bg-indigo-50 dark:bg-indigo-900/20 dark:text-indigo-400';
      default: return 'text-gray-600 bg-gray-50 dark:bg-gray-900/20 dark:text-gray-400';
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

  const filteredPolicies = policies.filter(policy => {
    const languageMatch = filterLanguage === 'all' || policy.language === filterLanguage;
    const categoryMatch = filterCategory === 'all' || policy.category === filterCategory;
    return languageMatch && categoryMatch;
  });

  const metrics = [
    {
      id: 'total-policies',
      title: 'Active Policies',
      value: policies.filter(p => p.status === 'active').length,
      change: 8.5,
      trend: 'up' as const,
      sparklineData: [18, 19, 21, 22, 23, policies.filter(p => p.status === 'active').length],
      alert: `${policies.filter(p => p.status === 'testing').length} in testing`
    },
    {
      id: 'violations',
      title: 'Active Violations',
      value: violations.filter(v => v.status === 'open').length,
      change: -12.3,
      trend: 'down' as const,
      sparklineData: [28, 25, 22, 18, 15, violations.filter(v => v.status === 'open').length]
    },
    {
      id: 'coverage',
      title: 'Policy Coverage',
      value: '89.4%',
      change: 4.2,
      trend: 'up' as const,
      sparklineData: [82, 84, 86, 87, 88, 89.4]
    },
    {
      id: 'compliance',
      title: 'Compliance Score',
      value: `${Math.round(policies.reduce((sum, p) => sum + p.successRate, 0) / policies.length || 0)}%`,
      change: 2.1,
      trend: 'up' as const,
      sparklineData: [85, 86, 87, 88, 89, Math.round(policies.reduce((sum, p) => sum + p.successRate, 0) / policies.length || 0)]
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
            <FileCode className="h-10 w-10 text-blue-600" />
            Policy-as-Code Management
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Git-based policy management with OPA, Sentinel, and custom DSL support
          </p>
        </div>
        <div className="flex gap-3">
          <ViewToggle view={view} onViewChange={setView} />
          <button
            onClick={() => router.push('/devsecops/policy-code/new')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
          >
            <Plus className="h-5 w-5" />
            New Policy
          </button>
          <button
            onClick={() => loadPolicyData()}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <RefreshCw className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* AI Predictions Alert */}
      {predictions.length > 0 && predictions[0].riskLevel === 'high' && (
        <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Brain className="h-6 w-6 text-blue-600 dark:text-blue-400 mt-1" />
            <div className="flex-1">
              <h3 className="font-semibold text-blue-900 dark:text-blue-100">
                Policy Compliance Prediction
              </h3>
              <p className="text-blue-700 dark:text-blue-300 mt-1">
                AI detected potential policy drift in Kubernetes workloads
              </p>
              <button className="mt-2 px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700">
                Review Policy Updates
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
                value={filterLanguage}
                onChange={(e) => setFilterLanguage(e.target.value)}
                className="px-3 py-1 border dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm"
              >
                <option value="all">All Languages</option>
                <option value="rego">Rego</option>
                <option value="sentinel">Sentinel</option>
                <option value="yaml">YAML</option>
                <option value="json">JSON</option>
                <option value="hcl">HCL</option>
              </select>
              <select
                value={filterCategory}
                onChange={(e) => setFilterCategory(e.target.value)}
                className="px-3 py-1 border dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm"
              >
                <option value="all">All Categories</option>
                <option value="security">Security</option>
                <option value="compliance">Compliance</option>
                <option value="cost">Cost</option>
                <option value="governance">Governance</option>
                <option value="operations">Operations</option>
              </select>
              <span className="text-sm text-gray-500">
                {filteredPolicies.length} of {policies.length} policies
              </span>
            </div>
          </div>

          {/* Policy Definitions */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Code className="h-6 w-6 text-blue-600" />
              Policy Definitions
            </h2>
            <div className="space-y-4">
              {filteredPolicies.map((policy) => (
                <div
                  key={policy.id}
                  className={`border dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-all ${
                    selectedPolicy === policy.id ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : ''
                  }`}
                  onClick={() => setSelectedPolicy(selectedPolicy === policy.id ? null : policy.id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="font-semibold text-lg">{policy.name}</h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(policy.status)}`}>
                          {policy.status}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getLanguageColor(policy.language)}`}>
                          {policy.language.toUpperCase()}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          policy.impactLevel === 'critical' ? 'text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400' :
                          policy.impactLevel === 'high' ? 'text-orange-600 bg-orange-50 dark:bg-orange-900/20 dark:text-orange-400' :
                          policy.impactLevel === 'medium' ? 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 dark:text-yellow-400' :
                          'text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-400'
                        }`}>
                          {policy.impactLevel} impact
                        </span>
                      </div>
                      <p className="text-gray-600 dark:text-gray-400 mb-2">{policy.description}</p>
                      <div className="flex items-center gap-4 text-sm text-gray-500">
                        <span>v{policy.version}</span>
                        <span>by {policy.author}</span>
                        <span>Coverage: {policy.testCoverage}%</span>
                        <span>Success: {policy.successRate}%</span>
                        <span>Applied to: {policy.appliedTo.join(', ')}</span>
                      </div>
                    </div>
                    <div className="text-right">
                      {policy.violations > 0 && (
                        <div className="text-red-600 dark:text-red-400 font-semibold">
                          {policy.violations} violations
                        </div>
                      )}
                      <div className="flex gap-2 mt-2">
                        <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded">
                          <Eye className="h-4 w-4" />
                        </button>
                        <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded">
                          <Edit className="h-4 w-4" />
                        </button>
                        <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded">
                          <Download className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  </div>

                  {selectedPolicy === policy.id && (
                    <div className="mt-4 pt-4 border-t dark:border-gray-700">
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div>
                          <h4 className="font-medium mb-2">Policy Code Preview:</h4>
                          <pre className="bg-gray-900 text-green-400 p-4 rounded-lg text-xs overflow-x-auto">
                            <code>{policy.codePreview}</code>
                          </pre>
                        </div>
                        <div className="space-y-4">
                          <div>
                            <h4 className="font-medium mb-2">Dependencies:</h4>
                            <div className="flex flex-wrap gap-2">
                              {policy.dependencies.map((dep, idx) => (
                                <span key={idx} className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">
                                  {dep}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div>
                            <h4 className="font-medium mb-2">Applied Environments:</h4>
                            <div className="flex flex-wrap gap-2">
                              {policy.appliedTo.map((env, idx) => (
                                <span key={idx} className="px-2 py-1 bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded text-sm">
                                  {env}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div className="flex gap-2">
                            <button className="px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700">
                              Edit Policy
                            </button>
                            <button className="px-3 py-1 bg-green-600 text-white rounded-md text-sm hover:bg-green-700">
                              Deploy
                            </button>
                            <button className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md text-sm hover:bg-gray-200 dark:hover:bg-gray-600">
                              Test
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

          {/* Recent Violations */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <AlertTriangle className="h-6 w-6 text-orange-600" />
              Recent Policy Violations
            </h2>
            <div className="space-y-3">
              {violations.map((violation) => (
                <div key={violation.id} className="border dark:border-gray-700 rounded-lg p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(violation.severity)}`}>
                          {violation.severity.toUpperCase()}
                        </span>
                        <h3 className="font-semibold">{violation.policyName}</h3>
                        <span className="text-sm text-gray-500">on {violation.resource}</span>
                      </div>
                      <p className="text-gray-600 dark:text-gray-400 mb-2">{violation.description}</p>
                      <p className="text-sm text-blue-600 dark:text-blue-400">
                        <span className="font-medium">Remediation:</span> {violation.remediation}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-500">
                        {violation.timestamp.toLocaleString()}
                      </div>
                      <div className="flex gap-2 mt-2">
                        <button className="px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700">
                          Investigate
                        </button>
                        <button className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md text-sm hover:bg-gray-200 dark:hover:bg-gray-600">
                          Suppress
                        </button>
                      </div>
                    </div>
                  </div>
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
              title="Policy Compliance Trends"
              onDrillIn={() => router.push('/devsecops/policy-code/analytics')}
            >
              <div className="p-4">
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                  <p className="text-gray-500">Policy compliance trend visualization</p>
                </div>
              </div>
            </ChartContainer>
            <ChartContainer
              title="Violations by Category"
              onDrillIn={() => router.push('/devsecops/policy-code/violations')}
            >
              <div className="p-4">
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                  <p className="text-gray-500">Violations breakdown chart</p>
                </div>
              </div>
            </ChartContainer>
          </div>

          {/* Policy Analytics */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Target className="h-6 w-6 text-purple-600" />
              Policy Performance Analytics
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="p-4 border dark:border-gray-700 rounded-lg">
                <h3 className="font-semibold mb-3">Top Performing Policies</h3>
                <div className="space-y-2">
                  {policies
                    .sort((a, b) => b.successRate - a.successRate)
                    .slice(0, 3)
                    .map((policy) => (
                      <div key={policy.id} className="flex justify-between items-center">
                        <span className="text-sm truncate">{policy.name}</span>
                        <span className="text-green-600 dark:text-green-400 font-semibold text-sm">
                          {policy.successRate}%
                        </span>
                      </div>
                    ))}
                </div>
              </div>
              <div className="p-4 border dark:border-gray-700 rounded-lg">
                <h3 className="font-semibold mb-3">Most Violated Policies</h3>
                <div className="space-y-2">
                  {policies
                    .sort((a, b) => b.violations - a.violations)
                    .slice(0, 3)
                    .map((policy) => (
                      <div key={policy.id} className="flex justify-between items-center">
                        <span className="text-sm truncate">{policy.name}</span>
                        <span className="text-red-600 dark:text-red-400 font-semibold text-sm">
                          {policy.violations}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
              <div className="p-4 border dark:border-gray-700 rounded-lg">
                <h3 className="font-semibold mb-3">Language Distribution</h3>
                <div className="space-y-2">
                  {Object.entries(
                    policies.reduce((acc, policy) => {
                      acc[policy.language] = (acc[policy.language] || 0) + 1;
                      return acc;
                    }, {} as Record<string, number>)
                  ).map(([language, count]) => (
                    <div key={language} className="flex justify-between items-center">
                      <span className="text-sm capitalize">{language}</span>
                      <span className="font-semibold text-sm">{count}</span>
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