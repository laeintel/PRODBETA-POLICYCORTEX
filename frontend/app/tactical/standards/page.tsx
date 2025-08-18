'use client';

import React, { useState, useEffect } from 'react';
import { 
  BookOpen, Shield, CheckCircle, XCircle, AlertTriangle, Info,
  FileText, Layers, Award, Target, Scale, Briefcase, Building,
  Globe, Users, Clock, Calendar, Timer, TrendingUp, TrendingDown,
  Search, Filter, RefreshCw, Download, Upload, Settings,
  ChevronRight, ChevronDown, MoreVertical, ExternalLink,
  Flag, Hash, Lock, Eye, Activity, BarChart3
} from 'lucide-react';

interface Standard {
  id: string;
  name: string;
  code: string;
  description: string;
  framework: string;
  category: 'security' | 'compliance' | 'operational' | 'technical' | 'governance';
  version: string;
  status: 'active' | 'draft' | 'deprecated' | 'review';
  effectiveDate: string;
  lastReview: string;
  nextReview: string;
  owner: string;
  department: string;
  requirements: Requirement[];
  controls: Control[];
  applicability: string[];
  complianceLevel: number;
  violations: number;
  exemptions: number;
}

interface Requirement {
  id: string;
  title: string;
  description: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  mandatory: boolean;
  verification: string;
  evidence: string[];
}

interface Control {
  id: string;
  name: string;
  type: 'preventive' | 'detective' | 'corrective';
  effectiveness: number;
  automated: boolean;
  implementation: string;
}

interface StandardCompliance {
  standardId: string;
  standardName: string;
  resources: number;
  compliant: number;
  partial: number;
  nonCompliant: number;
  percentage: number;
  trend: 'improving' | 'declining' | 'stable';
  lastAssessment: string;
}

interface StandardViolation {
  id: string;
  standardId: string;
  standardName: string;
  resource: string;
  requirement: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  detected: string;
  status: 'open' | 'investigating' | 'resolved';
  assignee: string;
}

export default function Standards() {
  const [standards, setStandards] = useState<Standard[]>([]);
  const [compliance, setCompliance] = useState<StandardCompliance[]>([]);
  const [violations, setViolations] = useState<StandardViolation[]>([]);
  const [viewMode, setViewMode] = useState<'catalog' | 'compliance' | 'violations' | 'mapping'>('catalog');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedFramework, setSelectedFramework] = useState('all');
  const [expandedStandard, setExpandedStandard] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    // Initialize with standards data
    setStandards([
      {
        id: 'STD-001',
        name: 'Cloud Security Standard',
        code: 'CSS-2024',
        description: 'Comprehensive cloud security requirements for all cloud resources',
        framework: 'ISO 27017',
        category: 'security',
        version: '3.2.0',
        status: 'active',
        effectiveDate: '2024-01-01',
        lastReview: '1 month ago',
        nextReview: 'in 11 months',
        owner: 'CISO',
        department: 'Information Security',
        requirements: [
          { id: 'R1', title: 'Encryption at Rest', description: 'All data must be encrypted at rest', priority: 'critical', mandatory: true, verification: 'Automated scan', evidence: ['Encryption report', 'Key management logs'] },
          { id: 'R2', title: 'Network Segmentation', description: 'Implement network micro-segmentation', priority: 'high', mandatory: true, verification: 'Network audit', evidence: ['Network diagram', 'Firewall rules'] },
          { id: 'R3', title: 'Access Control', description: 'Implement least privilege access', priority: 'critical', mandatory: true, verification: 'Access review', evidence: ['Access matrix', 'IAM policies'] }
        ],
        controls: [
          { id: 'C1', name: 'Encryption Control', type: 'preventive', effectiveness: 95, automated: true, implementation: 'Azure Policy' },
          { id: 'C2', name: 'Network Monitor', type: 'detective', effectiveness: 88, automated: true, implementation: 'Azure Monitor' },
          { id: 'C3', name: 'Access Review', type: 'detective', effectiveness: 82, automated: false, implementation: 'Manual review' }
        ],
        applicability: ['All Azure Resources', 'Production Environment'],
        complianceLevel: 87,
        violations: 23,
        exemptions: 5
      },
      {
        id: 'STD-002',
        name: 'Data Privacy Standard',
        code: 'DPS-2024',
        description: 'Data privacy and protection requirements aligned with GDPR',
        framework: 'GDPR',
        category: 'compliance',
        version: '2.1.0',
        status: 'active',
        effectiveDate: '2024-02-01',
        lastReview: '2 weeks ago',
        nextReview: 'in 6 months',
        owner: 'DPO',
        department: 'Legal & Compliance',
        requirements: [
          { id: 'R1', title: 'Data Minimization', description: 'Collect only necessary data', priority: 'high', mandatory: true, verification: 'Data audit', evidence: ['Data inventory', 'Processing records'] },
          { id: 'R2', title: 'Consent Management', description: 'Obtain and manage user consent', priority: 'critical', mandatory: true, verification: 'Consent audit', evidence: ['Consent logs', 'Opt-out records'] }
        ],
        controls: [
          { id: 'C1', name: 'Data Classification', type: 'preventive', effectiveness: 92, automated: true, implementation: 'DLP Solution' },
          { id: 'C2', name: 'Consent Tracking', type: 'detective', effectiveness: 96, automated: true, implementation: 'CRM System' }
        ],
        applicability: ['Customer Data', 'Personal Information'],
        complianceLevel: 94,
        violations: 8,
        exemptions: 2
      },
      {
        id: 'STD-003',
        name: 'Operational Excellence',
        code: 'OPS-2024',
        description: 'Standards for operational reliability and performance',
        framework: 'ITIL',
        category: 'operational',
        version: '1.8.0',
        status: 'active',
        effectiveDate: '2024-03-01',
        lastReview: '3 weeks ago',
        nextReview: 'in 9 months',
        owner: 'COO',
        department: 'Operations',
        requirements: [
          { id: 'R1', title: 'SLA Compliance', description: 'Meet 99.9% availability SLA', priority: 'critical', mandatory: true, verification: 'Performance metrics', evidence: ['Uptime reports', 'SLA dashboard'] },
          { id: 'R2', title: 'Incident Response', description: 'Respond within defined timeframes', priority: 'high', mandatory: true, verification: 'Incident reports', evidence: ['Response logs', 'Resolution times'] }
        ],
        controls: [
          { id: 'C1', name: 'Availability Monitor', type: 'detective', effectiveness: 98, automated: true, implementation: 'Monitoring tools' },
          { id: 'C2', name: 'Auto-scaling', type: 'corrective', effectiveness: 85, automated: true, implementation: 'Azure Autoscale' }
        ],
        applicability: ['Production Services', 'Critical Systems'],
        complianceLevel: 91,
        violations: 15,
        exemptions: 3
      },
      {
        id: 'STD-004',
        name: 'Development Standards',
        code: 'DEV-2024',
        description: 'Software development and deployment standards',
        framework: 'DevSecOps',
        category: 'technical',
        version: '2.5.0',
        status: 'active',
        effectiveDate: '2024-01-15',
        lastReview: '1 month ago',
        nextReview: 'in 5 months',
        owner: 'CTO',
        department: 'Engineering',
        requirements: [
          { id: 'R1', title: 'Code Review', description: 'All code must be peer reviewed', priority: 'high', mandatory: true, verification: 'PR reviews', evidence: ['Review logs', 'Approval records'] },
          { id: 'R2', title: 'Security Testing', description: 'Perform security scans before deployment', priority: 'critical', mandatory: true, verification: 'Scan reports', evidence: ['SAST results', 'DAST results'] }
        ],
        controls: [
          { id: 'C1', name: 'PR Gate Check', type: 'preventive', effectiveness: 94, automated: true, implementation: 'GitHub Actions' },
          { id: 'C2', name: 'Security Scanner', type: 'detective', effectiveness: 89, automated: true, implementation: 'SonarQube' }
        ],
        applicability: ['All Code Repositories', 'CI/CD Pipelines'],
        complianceLevel: 88,
        violations: 32,
        exemptions: 7
      },
      {
        id: 'STD-005',
        name: 'Cost Management Standard',
        code: 'FIN-2024',
        description: 'Financial governance and cost optimization standards',
        framework: 'FinOps',
        category: 'governance',
        version: '1.3.0',
        status: 'review',
        effectiveDate: '2024-04-01',
        lastReview: '1 week ago',
        nextReview: 'in 2 months',
        owner: 'CFO',
        department: 'Finance',
        requirements: [
          { id: 'R1', title: 'Budget Compliance', description: 'Stay within allocated budgets', priority: 'high', mandatory: true, verification: 'Budget reports', evidence: ['Cost reports', 'Budget alerts'] },
          { id: 'R2', title: 'Resource Tagging', description: 'Tag all resources for cost allocation', priority: 'medium', mandatory: true, verification: 'Tag audit', evidence: ['Tag compliance report'] }
        ],
        controls: [
          { id: 'C1', name: 'Budget Alerts', type: 'detective', effectiveness: 96, automated: true, implementation: 'Azure Cost Management' },
          { id: 'C2', name: 'Tag Enforcement', type: 'preventive', effectiveness: 91, automated: true, implementation: 'Azure Policy' }
        ],
        applicability: ['All Azure Subscriptions'],
        complianceLevel: 82,
        violations: 45,
        exemptions: 12
      },
      {
        id: 'STD-006',
        name: 'Business Continuity',
        code: 'BC-2024',
        description: 'Business continuity and disaster recovery standards',
        framework: 'ISO 22301',
        category: 'operational',
        version: '1.5.0',
        status: 'draft',
        effectiveDate: '2024-06-01',
        lastReview: 'In progress',
        nextReview: 'TBD',
        owner: 'Risk Manager',
        department: 'Risk Management',
        requirements: [
          { id: 'R1', title: 'Backup Strategy', description: 'Implement 3-2-1 backup strategy', priority: 'critical', mandatory: true, verification: 'Backup audit', evidence: ['Backup logs', 'Recovery tests'] },
          { id: 'R2', title: 'DR Testing', description: 'Test DR procedures quarterly', priority: 'high', mandatory: true, verification: 'DR test results', evidence: ['Test reports', 'RTO/RPO metrics'] }
        ],
        controls: [
          { id: 'C1', name: 'Backup Automation', type: 'preventive', effectiveness: 93, automated: true, implementation: 'Azure Backup' },
          { id: 'C2', name: 'Replication Monitor', type: 'detective', effectiveness: 87, automated: true, implementation: 'Azure Site Recovery' }
        ],
        applicability: ['Critical Systems', 'Production Data'],
        complianceLevel: 0,
        violations: 0,
        exemptions: 0
      }
    ]);

    setCompliance([
      {
        standardId: 'STD-001',
        standardName: 'Cloud Security Standard',
        resources: 450,
        compliant: 391,
        partial: 45,
        nonCompliant: 14,
        percentage: 87,
        trend: 'improving',
        lastAssessment: '2 days ago'
      },
      {
        standardId: 'STD-002',
        standardName: 'Data Privacy Standard',
        resources: 280,
        compliant: 263,
        partial: 12,
        nonCompliant: 5,
        percentage: 94,
        trend: 'stable',
        lastAssessment: '1 week ago'
      },
      {
        standardId: 'STD-003',
        standardName: 'Operational Excellence',
        resources: 350,
        compliant: 318,
        partial: 25,
        nonCompliant: 7,
        percentage: 91,
        trend: 'improving',
        lastAssessment: '3 days ago'
      },
      {
        standardId: 'STD-004',
        standardName: 'Development Standards',
        resources: 180,
        compliant: 158,
        partial: 15,
        nonCompliant: 7,
        percentage: 88,
        trend: 'declining',
        lastAssessment: '1 day ago'
      }
    ]);

    setViolations([
      {
        id: 'VIOL-001',
        standardId: 'STD-001',
        standardName: 'Cloud Security Standard',
        resource: 'VM-PROD-WEB-01',
        requirement: 'Encryption at Rest',
        severity: 'critical',
        detected: '2 hours ago',
        status: 'open',
        assignee: 'Security Team'
      },
      {
        id: 'VIOL-002',
        standardId: 'STD-004',
        standardName: 'Development Standards',
        resource: 'repo-frontend',
        requirement: 'Security Testing',
        severity: 'high',
        detected: '5 hours ago',
        status: 'investigating',
        assignee: 'DevOps Team'
      },
      {
        id: 'VIOL-003',
        standardId: 'STD-005',
        standardName: 'Cost Management Standard',
        resource: 'SUB-PROD-001',
        requirement: 'Budget Compliance',
        severity: 'medium',
        detected: '1 day ago',
        status: 'resolved',
        assignee: 'Finance Team'
      }
    ]);
  }, []);

  const getCategoryColor = (category: string) => {
    switch(category) {
      case 'security': return 'text-red-500 bg-red-900/20';
      case 'compliance': return 'text-blue-500 bg-blue-900/20';
      case 'operational': return 'text-green-500 bg-green-900/20';
      case 'technical': return 'text-purple-500 bg-purple-900/20';
      case 'governance': return 'text-yellow-500 bg-yellow-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': return 'text-green-500 bg-green-900/20';
      case 'draft': return 'text-yellow-500 bg-yellow-900/20';
      case 'deprecated': return 'text-red-500 bg-red-900/20';
      case 'review': return 'text-blue-500 bg-blue-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch(severity) {
      case 'critical': return 'text-red-500';
      case 'high': return 'text-orange-500';
      case 'medium': return 'text-yellow-500';
      case 'low': return 'text-blue-500';
      default: return 'text-gray-500';
    }
  };

  const totalStandards = standards.length;
  const activeStandards = standards.filter(s => s.status === 'active').length;
  const totalViolations = standards.reduce((acc, s) => acc + s.violations, 0);
  const avgCompliance = Math.round(compliance.reduce((acc, c) => acc + c.percentage, 0) / compliance.length);

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold flex items-center space-x-2">
              <BookOpen className="w-6 h-6 text-indigo-500" />
              <span>Standards</span>
            </h1>
            <p className="text-sm text-gray-400 mt-1">Organizational standards, requirements, and compliance tracking</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
            
            <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export</span>
            </button>
            
            <button className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded text-sm flex items-center space-x-2">
              <BookOpen className="w-4 h-4" />
              <span>New Standard</span>
            </button>
          </div>
        </div>
      </header>

      {/* Summary Stats */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="text-center">
            <div className="text-xs text-gray-500">Total Standards</div>
            <div className="text-2xl font-bold">{totalStandards}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Active</div>
            <div className="text-2xl font-bold text-green-500">{activeStandards}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Avg Compliance</div>
            <div className="text-2xl font-bold text-blue-500">{avgCompliance}%</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Violations</div>
            <div className="text-2xl font-bold text-orange-500">{totalViolations}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Requirements</div>
            <div className="text-2xl font-bold">
              {standards.reduce((acc, s) => acc + s.requirements.length, 0)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Controls</div>
            <div className="text-2xl font-bold">
              {standards.reduce((acc, s) => acc + s.controls.length, 0)}
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-gray-900/30 border-b border-gray-800 px-6">
        <div className="flex space-x-6">
          {['catalog', 'compliance', 'violations', 'mapping'].map(view => (
            <button
              key={view}
              onClick={() => setViewMode(view as any)}
              className={`py-3 border-b-2 text-sm capitalize ${
                viewMode === view 
                  ? 'border-indigo-500 text-indigo-500' 
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              {view === 'catalog' ? 'Standards Catalog' : view === 'mapping' ? 'Framework Mapping' : view}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {viewMode === 'catalog' && (
          <div className="space-y-4">
            {/* Search and Filters */}
            <div className="flex items-center space-x-3 mb-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
                <input
                  type="text"
                  placeholder="Search standards..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                />
              </div>
              
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Categories</option>
                <option value="security">Security</option>
                <option value="compliance">Compliance</option>
                <option value="operational">Operational</option>
                <option value="technical">Technical</option>
                <option value="governance">Governance</option>
              </select>

              <select
                value={selectedFramework}
                onChange={(e) => setSelectedFramework(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Frameworks</option>
                <option value="ISO">ISO</option>
                <option value="GDPR">GDPR</option>
                <option value="ITIL">ITIL</option>
                <option value="DevSecOps">DevSecOps</option>
                <option value="FinOps">FinOps</option>
              </select>
            </div>

            {/* Standards List */}
            {standards.map(standard => (
              <div key={standard.id} className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center space-x-2 mb-1">
                        <h3 className="text-sm font-bold">{standard.name}</h3>
                        <span className="text-xs text-gray-500">({standard.code})</span>
                        <span className="text-xs text-gray-500">v{standard.version}</span>
                        <span className={`px-2 py-1 rounded text-xs ${getStatusColor(standard.status)}`}>
                          {standard.status}
                        </span>
                        <span className={`px-2 py-1 rounded text-xs ${getCategoryColor(standard.category)}`}>
                          {standard.category}
                        </span>
                      </div>
                      <p className="text-xs text-gray-400 mb-2">{standard.description}</p>
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>Framework: {standard.framework}</span>
                        <span>Owner: {standard.owner}</span>
                        <span>Next Review: {standard.nextReview}</span>
                      </div>
                    </div>
                    <button 
                      onClick={() => setExpandedStandard(expandedStandard === standard.id ? null : standard.id)}
                      className="p-1 hover:bg-gray-800 rounded"
                    >
                      <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${
                        expandedStandard === standard.id ? 'rotate-180' : ''
                      }`} />
                    </button>
                  </div>

                  <div className="grid grid-cols-5 gap-3 text-xs">
                    <div className="bg-gray-800 rounded p-2">
                      <div className="text-gray-500">Compliance</div>
                      <div className={`text-lg font-bold ${
                        standard.complianceLevel >= 90 ? 'text-green-500' :
                        standard.complianceLevel >= 70 ? 'text-yellow-500' :
                        'text-red-500'
                      }`}>
                        {standard.complianceLevel}%
                      </div>
                    </div>
                    <div className="bg-gray-800 rounded p-2">
                      <div className="text-gray-500">Requirements</div>
                      <div className="text-lg font-bold">{standard.requirements.length}</div>
                    </div>
                    <div className="bg-gray-800 rounded p-2">
                      <div className="text-gray-500">Controls</div>
                      <div className="text-lg font-bold">{standard.controls.length}</div>
                    </div>
                    <div className="bg-gray-800 rounded p-2">
                      <div className="text-gray-500">Violations</div>
                      <div className={`text-lg font-bold ${standard.violations > 0 ? 'text-orange-500' : 'text-green-500'}`}>
                        {standard.violations}
                      </div>
                    </div>
                    <div className="bg-gray-800 rounded p-2">
                      <div className="text-gray-500">Exemptions</div>
                      <div className="text-lg font-bold">{standard.exemptions}</div>
                    </div>
                  </div>

                  {expandedStandard === standard.id && (
                    <div className="mt-4 pt-4 border-t border-gray-800 space-y-4">
                      <div>
                        <h4 className="text-xs font-bold mb-2">Key Requirements</h4>
                        <div className="space-y-2">
                          {standard.requirements.slice(0, 3).map(req => (
                            <div key={req.id} className="bg-gray-800 rounded p-2">
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-xs font-medium">{req.title}</span>
                                <span className={`px-2 py-1 rounded text-xs ${
                                  req.priority === 'critical' ? 'bg-red-900/20 text-red-500' :
                                  req.priority === 'high' ? 'bg-orange-900/20 text-orange-500' :
                                  req.priority === 'medium' ? 'bg-yellow-900/20 text-yellow-500' :
                                  'bg-blue-900/20 text-blue-500'
                                }`}>
                                  {req.priority}
                                </span>
                              </div>
                              <p className="text-xs text-gray-400">{req.description}</p>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="flex space-x-2">
                        <button className="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 rounded text-xs">
                          View Details
                        </button>
                        <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                          Assess Compliance
                        </button>
                        <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                          Request Exemption
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'compliance' && (
          <div className="space-y-4">
            {compliance.map(comp => (
              <div key={comp.standardId} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-sm font-bold">{comp.standardName}</h3>
                    <div className="flex items-center space-x-4 text-xs text-gray-500 mt-1">
                      <span>Last Assessment: {comp.lastAssessment}</span>
                      <span className={`flex items-center ${
                        comp.trend === 'improving' ? 'text-green-500' :
                        comp.trend === 'declining' ? 'text-red-500' :
                        'text-gray-500'
                      }`}>
                        {comp.trend === 'improving' ? <TrendingUp className="w-3 h-3 mr-1" /> :
                         comp.trend === 'declining' ? <TrendingDown className="w-3 h-3 mr-1" /> :
                         <Activity className="w-3 h-3 mr-1" />}
                        {comp.trend}
                      </span>
                    </div>
                  </div>
                  <div className="text-center">
                    <div className={`text-2xl font-bold ${
                      comp.percentage >= 90 ? 'text-green-500' :
                      comp.percentage >= 70 ? 'text-yellow-500' :
                      'text-red-500'
                    }`}>
                      {comp.percentage}%
                    </div>
                    <div className="text-xs text-gray-500">Compliance</div>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-3">
                  <div className="bg-gray-800 rounded p-2 text-center">
                    <div className="text-xs text-gray-500">Total Resources</div>
                    <div className="text-lg font-bold">{comp.resources}</div>
                  </div>
                  <div className="bg-gray-800 rounded p-2 text-center">
                    <div className="text-xs text-gray-500">Compliant</div>
                    <div className="text-lg font-bold text-green-500">{comp.compliant}</div>
                  </div>
                  <div className="bg-gray-800 rounded p-2 text-center">
                    <div className="text-xs text-gray-500">Partial</div>
                    <div className="text-lg font-bold text-yellow-500">{comp.partial}</div>
                  </div>
                  <div className="bg-gray-800 rounded p-2 text-center">
                    <div className="text-xs text-gray-500">Non-Compliant</div>
                    <div className="text-lg font-bold text-red-500">{comp.nonCompliant}</div>
                  </div>
                </div>

                <div className="mt-3">
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full flex">
                      <div className="bg-green-500" style={{ width: `${(comp.compliant/comp.resources)*100}%` }} />
                      <div className="bg-yellow-500" style={{ width: `${(comp.partial/comp.resources)*100}%` }} />
                      <div className="bg-red-500" style={{ width: `${(comp.nonCompliant/comp.resources)*100}%` }} />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'violations' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-800">
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Standard</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Resource</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Requirement</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Severity</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Detected</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Status</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Assignee</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Actions</th>
                </tr>
              </thead>
              <tbody>
                {violations.map(violation => (
                  <tr key={violation.id} className="border-t border-gray-800 hover:bg-gray-800/30">
                    <td className="px-4 py-3 text-sm">{violation.standardName}</td>
                    <td className="px-4 py-3 text-sm font-mono">{violation.resource}</td>
                    <td className="px-4 py-3 text-sm">{violation.requirement}</td>
                    <td className="px-4 py-3">
                      <span className={`text-xs font-bold ${getSeverityColor(violation.severity)}`}>
                        {violation.severity}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm">{violation.detected}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs ${
                        violation.status === 'resolved' ? 'bg-green-900/20 text-green-500' :
                        violation.status === 'investigating' ? 'bg-yellow-900/20 text-yellow-500' :
                        'bg-red-900/20 text-red-500'
                      }`}>
                        {violation.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm">{violation.assignee}</td>
                    <td className="px-4 py-3">
                      <button className="text-xs text-blue-500 hover:text-blue-400">
                        Remediate
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {viewMode === 'mapping' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h3 className="text-sm font-bold mb-4">Framework Mapping Matrix</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr>
                    <th className="px-3 py-2 text-left">Standard</th>
                    <th className="px-3 py-2 text-center">ISO 27001</th>
                    <th className="px-3 py-2 text-center">GDPR</th>
                    <th className="px-3 py-2 text-center">NIST</th>
                    <th className="px-3 py-2 text-center">SOC 2</th>
                    <th className="px-3 py-2 text-center">PCI DSS</th>
                  </tr>
                </thead>
                <tbody>
                  {standards.filter(s => s.status === 'active').map(standard => (
                    <tr key={standard.id} className="border-t border-gray-800">
                      <td className="px-3 py-2 font-medium">{standard.name}</td>
                      <td className="px-3 py-2 text-center">
                        {standard.framework.includes('ISO') ? (
                          <CheckCircle className="w-4 h-4 text-green-500 mx-auto" />
                        ) : (
                          <XCircle className="w-4 h-4 text-gray-600 mx-auto" />
                        )}
                      </td>
                      <td className="px-3 py-2 text-center">
                        {standard.framework === 'GDPR' ? (
                          <CheckCircle className="w-4 h-4 text-green-500 mx-auto" />
                        ) : (
                          <XCircle className="w-4 h-4 text-gray-600 mx-auto" />
                        )}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <XCircle className="w-4 h-4 text-gray-600 mx-auto" />
                      </td>
                      <td className="px-3 py-2 text-center">
                        <XCircle className="w-4 h-4 text-gray-600 mx-auto" />
                      </td>
                      <td className="px-3 py-2 text-center">
                        <XCircle className="w-4 h-4 text-gray-600 mx-auto" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </>
  );
}