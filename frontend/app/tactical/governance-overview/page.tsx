'use client';

import React, { useState, useEffect } from 'react';
import { 
  Shield, FileText, AlertTriangle, CheckCircle, XCircle, TrendingUp, TrendingDown,
  Activity, BarChart3, PieChart, LineChart, Users, Building, Globe,
  Lock, Key, Eye, Settings, Filter, Search, Download, RefreshCw,
  ChevronRight, ChevronDown, MoreVertical, ExternalLink, Info,
  BookOpen, Scale, Gavel, Award, Target, Flag, Briefcase, FileCheck
} from 'lucide-react';

interface GovernanceMetric {
  id: string;
  name: string;
  value: number;
  target: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  status: 'good' | 'warning' | 'critical';
  category: string;
}

interface PolicyItem {
  id: string;
  name: string;
  type: 'regulatory' | 'security' | 'operational' | 'financial';
  status: 'active' | 'draft' | 'review' | 'archived';
  compliance: number;
  violations: number;
  lastUpdated: string;
  owner: string;
  scope: string[];
  frameworks: string[];
}

interface ComplianceFramework {
  id: string;
  name: string;
  acronym: string;
  score: number;
  requirements: number;
  compliant: number;
  inProgress: number;
  nonCompliant: number;
  lastAudit: string;
  nextAudit: string;
}

interface RiskItem {
  id: string;
  title: string;
  category: 'security' | 'compliance' | 'operational' | 'financial' | 'reputational';
  likelihood: 'rare' | 'unlikely' | 'possible' | 'likely' | 'certain';
  impact: 'negligible' | 'minor' | 'moderate' | 'major' | 'severe';
  riskScore: number;
  status: 'identified' | 'assessed' | 'mitigating' | 'accepted' | 'closed';
  owner: string;
  controls: string[];
}

export default function GovernanceOverview() {
  const [metrics, setMetrics] = useState<GovernanceMetric[]>([]);
  const [policies, setPolicies] = useState<PolicyItem[]>([]);
  const [frameworks, setFrameworks] = useState<ComplianceFramework[]>([]);
  const [risks, setRisks] = useState<RiskItem[]>([]);
  const [selectedView, setSelectedView] = useState<'dashboard' | 'policies' | 'compliance' | 'risks'>('dashboard');
  const [selectedFramework, setSelectedFramework] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState('30d');

  useEffect(() => {
    // Initialize with governance data
    setMetrics([
      {
        id: 'GM-001',
        name: 'Overall Compliance Score',
        value: 92,
        target: 95,
        unit: '%',
        trend: 'up',
        status: 'warning',
        category: 'compliance'
      },
      {
        id: 'GM-002',
        name: 'Policy Adherence',
        value: 88,
        target: 90,
        unit: '%',
        trend: 'stable',
        status: 'warning',
        category: 'policy'
      },
      {
        id: 'GM-003',
        name: 'Risk Score',
        value: 32,
        target: 40,
        unit: '/100',
        trend: 'down',
        status: 'good',
        category: 'risk'
      },
      {
        id: 'GM-004',
        name: 'Control Effectiveness',
        value: 94,
        target: 90,
        unit: '%',
        trend: 'up',
        status: 'good',
        category: 'controls'
      },
      {
        id: 'GM-005',
        name: 'Audit Findings',
        value: 12,
        target: 20,
        unit: 'open',
        trend: 'down',
        status: 'good',
        category: 'audit'
      },
      {
        id: 'GM-006',
        name: 'Resource Coverage',
        value: 78,
        target: 85,
        unit: '%',
        trend: 'up',
        status: 'warning',
        category: 'coverage'
      }
    ]);

    setPolicies([
      {
        id: 'POL-001',
        name: 'Data Protection Policy',
        type: 'regulatory',
        status: 'active',
        compliance: 95,
        violations: 3,
        lastUpdated: '2 weeks ago',
        owner: 'Security Team',
        scope: ['All Regions', 'All Data Types'],
        frameworks: ['GDPR', 'CCPA', 'HIPAA']
      },
      {
        id: 'POL-002',
        name: 'Access Control Policy',
        type: 'security',
        status: 'active',
        compliance: 92,
        violations: 8,
        lastUpdated: '1 month ago',
        owner: 'Identity Team',
        scope: ['Production', 'Staging'],
        frameworks: ['ISO 27001', 'NIST', 'CIS']
      },
      {
        id: 'POL-003',
        name: 'Cloud Resource Tagging',
        type: 'operational',
        status: 'review',
        compliance: 78,
        violations: 45,
        lastUpdated: '3 days ago',
        owner: 'Cloud Operations',
        scope: ['Azure', 'AWS', 'GCP'],
        frameworks: ['FinOps', 'Cloud Security Alliance']
      },
      {
        id: 'POL-004',
        name: 'Cost Management Policy',
        type: 'financial',
        status: 'active',
        compliance: 85,
        violations: 12,
        lastUpdated: '1 week ago',
        owner: 'Finance Team',
        scope: ['All Departments'],
        frameworks: ['FinOps', 'ISO 20000']
      },
      {
        id: 'POL-005',
        name: 'Incident Response Policy',
        type: 'security',
        status: 'draft',
        compliance: 0,
        violations: 0,
        lastUpdated: 'In Progress',
        owner: 'Security Operations',
        scope: ['Global'],
        frameworks: ['NIST IR', 'ISO 27035']
      }
    ]);

    setFrameworks([
      {
        id: 'FW-001',
        name: 'ISO 27001:2013',
        acronym: 'ISO 27001',
        score: 88,
        requirements: 114,
        compliant: 100,
        inProgress: 10,
        nonCompliant: 4,
        lastAudit: '3 months ago',
        nextAudit: 'in 3 months'
      },
      {
        id: 'FW-002',
        name: 'NIST Cybersecurity Framework',
        acronym: 'NIST CSF',
        score: 92,
        requirements: 98,
        compliant: 90,
        inProgress: 6,
        nonCompliant: 2,
        lastAudit: '1 month ago',
        nextAudit: 'in 5 months'
      },
      {
        id: 'FW-003',
        name: 'SOC 2 Type II',
        acronym: 'SOC 2',
        score: 95,
        requirements: 64,
        compliant: 61,
        inProgress: 2,
        nonCompliant: 1,
        lastAudit: '6 months ago',
        nextAudit: 'in 2 weeks'
      },
      {
        id: 'FW-004',
        name: 'PCI DSS v4.0',
        acronym: 'PCI DSS',
        score: 98,
        requirements: 384,
        compliant: 376,
        inProgress: 8,
        nonCompliant: 0,
        lastAudit: '2 months ago',
        nextAudit: 'in 4 months'
      },
      {
        id: 'FW-005',
        name: 'GDPR',
        acronym: 'GDPR',
        score: 94,
        requirements: 99,
        compliant: 93,
        inProgress: 4,
        nonCompliant: 2,
        lastAudit: '4 months ago',
        nextAudit: 'in 2 months'
      }
    ]);

    setRisks([
      {
        id: 'RISK-001',
        title: 'Unauthorized Data Access',
        category: 'security',
        likelihood: 'possible',
        impact: 'major',
        riskScore: 75,
        status: 'mitigating',
        owner: 'CISO',
        controls: ['MFA', 'Access Reviews', 'Encryption', 'DLP']
      },
      {
        id: 'RISK-002',
        title: 'Regulatory Non-Compliance',
        category: 'compliance',
        likelihood: 'unlikely',
        impact: 'severe',
        riskScore: 60,
        status: 'assessed',
        owner: 'Compliance Officer',
        controls: ['Regular Audits', 'Policy Updates', 'Training']
      },
      {
        id: 'RISK-003',
        title: 'Cloud Cost Overrun',
        category: 'financial',
        likelihood: 'likely',
        impact: 'moderate',
        riskScore: 65,
        status: 'mitigating',
        owner: 'CFO',
        controls: ['Budget Alerts', 'Resource Optimization', 'FinOps']
      },
      {
        id: 'RISK-004',
        title: 'Service Availability',
        category: 'operational',
        likelihood: 'possible',
        impact: 'major',
        riskScore: 70,
        status: 'mitigating',
        owner: 'VP Engineering',
        controls: ['HA Architecture', 'DR Plan', 'Monitoring']
      },
      {
        id: 'RISK-005',
        title: 'Third-Party Vendor Risk',
        category: 'operational',
        likelihood: 'possible',
        impact: 'moderate',
        riskScore: 55,
        status: 'accepted',
        owner: 'Procurement',
        controls: ['Vendor Assessment', 'Contracts', 'SLAs']
      }
    ]);
  }, []);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': case 'good': return 'text-green-500 bg-green-900/20';
      case 'draft': case 'warning': return 'text-yellow-500 bg-yellow-900/20';
      case 'review': return 'text-blue-500 bg-blue-900/20';
      case 'archived': case 'critical': return 'text-red-500 bg-red-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getRiskColor = (score: number) => {
    if (score >= 75) return 'text-red-500 bg-red-900/20';
    if (score >= 50) return 'text-orange-500 bg-orange-900/20';
    if (score >= 25) return 'text-yellow-500 bg-yellow-900/20';
    return 'text-green-500 bg-green-900/20';
  };

  const getLikelihoodColor = (likelihood: string) => {
    switch(likelihood) {
      case 'certain': return 'text-red-600';
      case 'likely': return 'text-orange-500';
      case 'possible': return 'text-yellow-500';
      case 'unlikely': return 'text-blue-500';
      case 'rare': return 'text-green-500';
      default: return 'text-gray-500';
    }
  };

  const getImpactColor = (impact: string) => {
    switch(impact) {
      case 'severe': return 'text-red-600';
      case 'major': return 'text-orange-500';
      case 'moderate': return 'text-yellow-500';
      case 'minor': return 'text-blue-500';
      case 'negligible': return 'text-green-500';
      default: return 'text-gray-500';
    }
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold flex items-center space-x-2">
              <Shield className="w-6 h-6 text-blue-500" />
              <span>Governance Overview</span>
            </h1>
            <p className="text-sm text-gray-400 mt-1">Unified governance, risk, and compliance management</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
            >
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
              <option value="90d">Last 90 Days</option>
              <option value="1y">Last Year</option>
            </select>

            <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>

            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export Report</span>
            </button>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="bg-gray-900/30 border-b border-gray-800 px-6">
        <div className="flex space-x-6">
          {['dashboard', 'policies', 'compliance', 'risks'].map(view => (
            <button
              key={view}
              onClick={() => setSelectedView(view as any)}
              className={`py-3 border-b-2 text-sm capitalize ${
                selectedView === view 
                  ? 'border-blue-500 text-blue-500' 
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              {view}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {selectedView === 'dashboard' && (
          <div className="space-y-6">
            {/* Metrics Grid */}
            <div className="grid grid-cols-6 gap-4">
              {metrics.map(metric => (
                <div key={metric.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="text-xs text-gray-400">{metric.name}</h3>
                    {metric.trend === 'up' ? (
                      <TrendingUp className="w-4 h-4 text-green-500" />
                    ) : metric.trend === 'down' ? (
                      <TrendingDown className="w-4 h-4 text-red-500" />
                    ) : (
                      <Activity className="w-4 h-4 text-gray-500" />
                    )}
                  </div>
                  <div className="flex items-baseline space-x-1">
                    <span className={`text-2xl font-bold ${
                      metric.status === 'good' ? 'text-green-500' :
                      metric.status === 'warning' ? 'text-yellow-500' :
                      'text-red-500'
                    }`}>
                      {metric.value}
                    </span>
                    <span className="text-sm text-gray-500">{metric.unit}</span>
                  </div>
                  <div className="mt-2">
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-500">Target</span>
                      <span>{metric.target}{metric.unit}</span>
                    </div>
                    <div className="w-full h-1 bg-gray-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${
                          metric.status === 'good' ? 'bg-green-500' :
                          metric.status === 'warning' ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}
                        style={{ width: `${(metric.value / metric.target) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Compliance Frameworks */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h2 className="text-sm font-bold mb-4">Compliance Frameworks</h2>
              <div className="grid grid-cols-5 gap-4">
                {frameworks.map(framework => (
                  <div key={framework.id} className="bg-gray-800 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-bold">{framework.acronym}</h3>
                      <span className={`text-xl font-bold ${
                        framework.score >= 95 ? 'text-green-500' :
                        framework.score >= 85 ? 'text-yellow-500' :
                        'text-red-500'
                      }`}>
                        {framework.score}%
                      </span>
                    </div>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Compliant</span>
                        <span className="text-green-500">{framework.compliant}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">In Progress</span>
                        <span className="text-yellow-500">{framework.inProgress}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Non-Compliant</span>
                        <span className="text-red-500">{framework.nonCompliant}</span>
                      </div>
                    </div>
                    <div className="mt-2 pt-2 border-t border-gray-700 text-xs text-gray-500">
                      Next audit {framework.nextAudit}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Risk Matrix */}
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h2 className="text-sm font-bold mb-4">Risk Distribution</h2>
                <div className="space-y-3">
                  {['security', 'compliance', 'operational', 'financial', 'reputational'].map(category => {
                    const categoryRisks = risks.filter(r => r.category === category);
                    const avgScore = categoryRisks.reduce((acc, r) => acc + r.riskScore, 0) / (categoryRisks.length || 1);
                    return (
                      <div key={category} className="flex items-center justify-between">
                        <span className="text-sm capitalize">{category}</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-32 h-2 bg-gray-800 rounded-full overflow-hidden">
                            <div 
                              className={`h-full ${getRiskColor(avgScore).replace('text-', 'bg-').replace('/20', '')}`}
                              style={{ width: `${avgScore}%` }}
                            />
                          </div>
                          <span className="text-xs w-12 text-right">{Math.round(avgScore)}%</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h2 className="text-sm font-bold mb-4">Recent Policy Updates</h2>
                <div className="space-y-2">
                  {policies.slice(0, 5).map(policy => (
                    <div key={policy.id} className="flex items-center justify-between p-2 bg-gray-800 rounded">
                      <div className="flex items-center space-x-2">
                        <FileText className="w-4 h-4 text-gray-500" />
                        <div>
                          <div className="text-xs font-medium">{policy.name}</div>
                          <div className="text-xs text-gray-500">{policy.lastUpdated}</div>
                        </div>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs ${getStatusColor(policy.status)}`}>
                        {policy.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedView === 'policies' && (
          <div className="space-y-4">
            {policies.map(policy => (
              <div key={policy.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center space-x-2 mb-1">
                      <h3 className="text-sm font-bold">{policy.name}</h3>
                      <span className={`px-2 py-1 rounded text-xs ${getStatusColor(policy.status)}`}>
                        {policy.status}
                      </span>
                      <span className="px-2 py-1 bg-gray-800 rounded text-xs">
                        {policy.type}
                      </span>
                    </div>
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span>Owner: {policy.owner}</span>
                      <span>Updated: {policy.lastUpdated}</span>
                    </div>
                  </div>
                  <button className="p-1 hover:bg-gray-800 rounded">
                    <MoreVertical className="w-4 h-4 text-gray-500" />
                  </button>
                </div>

                <div className="grid grid-cols-4 gap-4 mb-3">
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Compliance</div>
                    <div className="flex items-center space-x-2">
                      <div className="text-xl font-bold">{policy.compliance}%</div>
                      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${
                            policy.compliance >= 90 ? 'bg-green-500' :
                            policy.compliance >= 70 ? 'bg-yellow-500' :
                            'bg-red-500'
                          }`}
                          style={{ width: `${policy.compliance}%` }}
                        />
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Violations</div>
                    <div className={`text-xl font-bold ${
                      policy.violations > 10 ? 'text-red-500' :
                      policy.violations > 5 ? 'text-yellow-500' :
                      'text-green-500'
                    }`}>
                      {policy.violations}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Scope</div>
                    <div className="flex flex-wrap gap-1">
                      {policy.scope.map(s => (
                        <span key={s} className="px-2 py-1 bg-gray-800 rounded text-xs">
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Frameworks</div>
                    <div className="flex flex-wrap gap-1">
                      {policy.frameworks.map(f => (
                        <span key={f} className="px-2 py-1 bg-blue-900/20 text-blue-500 rounded text-xs">
                          {f}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="flex space-x-2">
                  <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                    View Details
                  </button>
                  <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    Edit Policy
                  </button>
                  <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    View Violations
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedView === 'compliance' && (
          <div className="space-y-4">
            {frameworks.map(framework => (
              <div key={framework.id} className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-bold">{framework.name}</h3>
                      <p className="text-sm text-gray-500">Total Requirements: {framework.requirements}</p>
                    </div>
                    <div className="text-center">
                      <div className={`text-3xl font-bold ${
                        framework.score >= 95 ? 'text-green-500' :
                        framework.score >= 85 ? 'text-yellow-500' :
                        'text-red-500'
                      }`}>
                        {framework.score}%
                      </div>
                      <div className="text-xs text-gray-500">Compliance Score</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="bg-gray-800 rounded p-3 text-center">
                      <CheckCircle className="w-6 h-6 text-green-500 mx-auto mb-1" />
                      <div className="text-2xl font-bold">{framework.compliant}</div>
                      <div className="text-xs text-gray-500">Compliant</div>
                    </div>
                    <div className="bg-gray-800 rounded p-3 text-center">
                      <Activity className="w-6 h-6 text-yellow-500 mx-auto mb-1" />
                      <div className="text-2xl font-bold">{framework.inProgress}</div>
                      <div className="text-xs text-gray-500">In Progress</div>
                    </div>
                    <div className="bg-gray-800 rounded p-3 text-center">
                      <XCircle className="w-6 h-6 text-red-500 mx-auto mb-1" />
                      <div className="text-2xl font-bold">{framework.nonCompliant}</div>
                      <div className="text-xs text-gray-500">Non-Compliant</div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center space-x-4">
                      <span className="text-gray-500">Last Audit: {framework.lastAudit}</span>
                      <span className="text-gray-500">Next Audit: {framework.nextAudit}</span>
                    </div>
                    <div className="flex space-x-2">
                      <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                        View Requirements
                      </button>
                      <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                        Audit Report
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedView === 'risks' && (
          <div className="space-y-4">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-4">
              <h2 className="text-sm font-bold mb-3">Risk Heat Map</h2>
              <div className="grid grid-cols-5 gap-2">
                {['rare', 'unlikely', 'possible', 'likely', 'certain'].reverse().map((likelihood, lIdx) => (
                  <div key={likelihood} className="space-y-2">
                    <div className="text-xs text-center capitalize">{likelihood}</div>
                    {['negligible', 'minor', 'moderate', 'major', 'severe'].map((impact, iIdx) => {
                      const cellRisks = risks.filter(r => r.likelihood === likelihood && r.impact === impact);
                      const riskLevel = (4 - lIdx) * (iIdx + 1);
                      return (
                        <div
                          key={impact}
                          className={`h-12 rounded flex items-center justify-center text-xs font-bold ${
                            riskLevel >= 15 ? 'bg-red-900/50' :
                            riskLevel >= 10 ? 'bg-orange-900/50' :
                            riskLevel >= 5 ? 'bg-yellow-900/50' :
                            'bg-green-900/50'
                          }`}
                        >
                          {cellRisks.length > 0 && cellRisks.length}
                        </div>
                      );
                    })}
                  </div>
                ))}
                <div className="col-span-5 flex justify-between text-xs mt-2">
                  <span>Negligible</span>
                  <span>Minor</span>
                  <span>Moderate</span>
                  <span>Major</span>
                  <span>Severe</span>
                </div>
              </div>
            </div>

            {risks.map(risk => (
              <div key={risk.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center space-x-2 mb-1">
                      <h3 className="text-sm font-bold">{risk.title}</h3>
                      <span className={`px-2 py-1 rounded text-xs ${getRiskColor(risk.riskScore)}`}>
                        Risk Score: {risk.riskScore}
                      </span>
                    </div>
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span>Category: {risk.category}</span>
                      <span>Owner: {risk.owner}</span>
                      <span>Status: {risk.status}</span>
                    </div>
                  </div>
                  <button className="p-1 hover:bg-gray-800 rounded">
                    <MoreVertical className="w-4 h-4 text-gray-500" />
                  </button>
                </div>

                <div className="grid grid-cols-4 gap-4 mb-3">
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Likelihood</div>
                    <div className={`text-sm font-bold capitalize ${getLikelihoodColor(risk.likelihood)}`}>
                      {risk.likelihood}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Impact</div>
                    <div className={`text-sm font-bold capitalize ${getImpactColor(risk.impact)}`}>
                      {risk.impact}
                    </div>
                  </div>
                  <div className="col-span-2">
                    <div className="text-xs text-gray-500 mb-1">Controls</div>
                    <div className="flex flex-wrap gap-1">
                      {risk.controls.map(control => (
                        <span key={control} className="px-2 py-1 bg-gray-800 rounded text-xs">
                          {control}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="flex space-x-2">
                  <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                    Risk Assessment
                  </button>
                  <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    Mitigation Plan
                  </button>
                  <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    Add Control
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}