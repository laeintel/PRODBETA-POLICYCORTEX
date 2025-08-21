'use client';

import React, { useState, useEffect } from 'react';
import {
  AlertTriangle, Shield, FileText, CheckCircle, XCircle,
  ChevronRight, ChevronDown, Clock, TrendingUp, TrendingDown,
  User, Calendar, Target, AlertCircle, Info, Settings,
  Download, RefreshCw, Filter, Search, BarChart3, Activity,
  Code, GitBranch, Terminal, Database, Layers, Zap
} from 'lucide-react';
import { toast } from '@/hooks/useToast'
import { motion, AnimatePresence } from 'framer-motion';
import { format, formatDistanceToNow } from 'date-fns';

interface ComplianceViolation {
  violationId: string;
  violationType: 'policy' | 'configuration' | 'security' | 'regulatory';
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'active' | 'remediated' | 'exempted' | 'pending';
  detectedAt: string;
  lastUpdated: string;
  title: string;
  description: string;
  affectedResources: AffectedResource[];
  violatedPolicies: ViolatedPolicy[];
  remediationSteps: RemediationStep[];
  riskScore: number;
  complianceFrameworks: string[];
  rootCause: RootCauseAnalysis;
  impactAnalysis: ImpactAnalysis;
  timeline: TimelineEvent[];
  relatedViolations: string[];
  automationAvailable: boolean;
}

interface AffectedResource {
  resourceId: string;
  resourceName: string;
  resourceType: string;
  subscription: string;
  resourceGroup: string;
  region: string;
  tags: Record<string, string>;
  complianceState: 'non_compliant' | 'compliant' | 'unknown';
  lastAssessed: string;
  configurationDrift: ConfigurationDrift[];
  resourceHealth: ResourceHealth;
  dependencies: ResourceDependency[];
  costImpact: number;
}

interface ViolatedPolicy {
  policyId: string;
  policyName: string;
  policyType: 'azure_policy' | 'custom' | 'regulatory' | 'best_practice';
  category: string;
  effect: string;
  parameters: Record<string, any>;
  enforcementMode: 'enforced' | 'disabled';
  assignmentScope: string;
  excludedScopes: string[];
  policyDefinition: PolicyDefinition;
  violationDetails: ViolationDetails;
  historicalCompliance: HistoricalCompliance;
}

interface RemediationStep {
  stepId: string;
  stepNumber: number;
  title: string;
  description: string;
  type: 'manual' | 'automated' | 'script' | 'terraform' | 'arm_template';
  estimatedTime: string;
  complexity: 'low' | 'medium' | 'high';
  requiredPermissions: string[];
  automationScript?: string;
  validationSteps: string[];
  rollbackProcedure?: string;
  documentation: string[];
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  assignee?: string;
  completedAt?: string;
}

interface ConfigurationDrift {
  property: string;
  expectedValue: any;
  actualValue: any;
  driftType: 'added' | 'modified' | 'removed';
  detectedAt: string;
  impact: 'critical' | 'high' | 'medium' | 'low';
}

interface DrillLevel {
  level: number;
  type: 'violation' | 'resource' | 'policy' | 'remediation';
  id: string;
  name: string;
  data?: any;
}

export default function ComplianceDeepDrill() {
  const [selectedViolation, setSelectedViolation] = useState<ComplianceViolation | null>(null);
  const [selectedResource, setSelectedResource] = useState<AffectedResource | null>(null);
  const [selectedPolicy, setSelectedPolicy] = useState<ViolatedPolicy | null>(null);
  const [drillPath, setDrillPath] = useState<DrillLevel[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'resources' | 'policies' | 'remediation' | 'timeline'>('overview');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['summary']));
  const [remediationProgress, setRemediationProgress] = useState<Map<string, number>>(new Map());
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  const fetchViolationDetails = async (violationId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/compliance/violations/${violationId}/deep-drill`);
      const data = await response.json();
      setSelectedViolation(data);
      setDrillPath([
        { level: 0, type: 'violation', id: violationId, name: data.title, data }
      ]);
    } catch (error) {
      console.error('Error fetching violation details:', error);
    } finally {
      setLoading(false);
    }
  };

  const drillIntoResource = (resource: AffectedResource) => {
    setSelectedResource(resource);
    setDrillPath([
      ...drillPath,
      { level: drillPath.length, type: 'resource', id: resource.resourceId, name: resource.resourceName, data: resource }
    ]);
  };

  const drillIntoPolicy = (policy: ViolatedPolicy) => {
    setSelectedPolicy(policy);
    setDrillPath([
      ...drillPath,
      { level: drillPath.length, type: 'policy', id: policy.policyId, name: policy.policyName, data: policy }
    ]);
  };

  const navigateToDrillLevel = (level: number) => {
    const newPath = drillPath.slice(0, level + 1);
    setDrillPath(newPath);
    
    // Reset selections based on level
    if (level === 0) {
      setSelectedResource(null);
      setSelectedPolicy(null);
    } else if (level === 1) {
      setSelectedPolicy(null);
    }
  };

  const executeRemediation = async (step: RemediationStep) => {
    if (step.type === 'automated' || step.type === 'script') {
      // Simulate remediation execution
      const interval = setInterval(() => {
        setRemediationProgress(prev => {
          const current = prev.get(step.stepId) || 0;
          if (current >= 100) {
            clearInterval(interval);
            return prev;
          }
          const newMap = new Map(prev);
          newMap.set(step.stepId, Math.min(current + 10, 100));
          return newMap;
        });
      }, 500);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-red-600 bg-red-100';
      case 'remediated': return 'text-green-600 bg-green-100';
      case 'exempted': return 'text-yellow-600 bg-yellow-100';
      case 'pending': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const renderBreadcrumb = () => (
    <div className="flex items-center space-x-2 text-sm mb-6 p-3 bg-gray-50 rounded-lg">
      <button type="button"
        onClick={() => {
          setDrillPath([]);
          setSelectedViolation(null);
          setSelectedResource(null);
          setSelectedPolicy(null);
        }}
        className="text-blue-600 hover:text-blue-800 font-medium"
      >
        Compliance Dashboard
      </button>
      {drillPath.map((path, index) => (
        <React.Fragment key={`${path.type}-${path.id}`}>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <button type="button"
            onClick={() => navigateToDrillLevel(index)}
            className={`hover:text-blue-800 font-medium ${
              index === drillPath.length - 1 ? 'text-gray-900' : 'text-blue-600'
            }`}
          >
            {path.name}
          </button>
        </React.Fragment>
      ))}
    </div>
  );

  const renderViolationOverview = () => {
    if (!selectedViolation) return null;

    return (
      <div className="space-y-6">
        {/* Violation Header */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex justify-between items-start mb-4">
            <div>
              <div className="flex items-center space-x-3 mb-2">
                <AlertTriangle className="w-6 h-6 text-red-600" />
                <h2 className="text-xl font-bold text-gray-900">{selectedViolation.title}</h2>
                <span className={`px-2 py-1 text-xs rounded-full ${getSeverityColor(selectedViolation.severity)}`}>
                  {selectedViolation.severity.toUpperCase()}
                </span>
                <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(selectedViolation.status)}`}>
                  {selectedViolation.status.toUpperCase()}
                </span>
              </div>
              <p className="text-gray-600">{selectedViolation.description}</p>
            </div>
            <div className="flex space-x-2">
              {selectedViolation.automationAvailable && (
                <button
                  type="button"
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center"
                  onClick={() => toast({ title: 'Auto-remediation', description: 'Queued automated remediation' })}
                >
                  <Zap className="w-4 h-4 mr-2" />
                  Auto-Remediate
                </button>
              )}
              <button
                type="button"
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
                onClick={() => toast({ title: 'Export', description: 'Exporting violation report...' })}
              >
                <Download className="w-4 h-4 inline mr-2" />
                Export
              </button>
            </div>
          </div>

          {/* Key Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-gray-900">
                {selectedViolation.affectedResources.length}
              </div>
              <div className="text-xs text-gray-600">Affected Resources</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-red-600">
                {selectedViolation.riskScore}%
              </div>
              <div className="text-xs text-gray-600">Risk Score</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-sm font-bold text-gray-900">
                {formatDistanceToNow(new Date(selectedViolation.detectedAt), { addSuffix: true })}
              </div>
              <div className="text-xs text-gray-600">First Detected</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-gray-900">
                {selectedViolation.violatedPolicies.length}
              </div>
              <div className="text-xs text-gray-600">Violated Policies</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-gray-900">
                {selectedViolation.remediationSteps.filter(s => s.status === 'completed').length}/
                {selectedViolation.remediationSteps.length}
              </div>
              <div className="text-xs text-gray-600">Steps Complete</div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6" aria-label="Tabs">
              {['overview', 'resources', 'policies', 'remediation', 'timeline'].map((tab) => (
                <button type="button"
                  key={tab}
                  onClick={() => setActiveTab(tab as any)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm capitalize ${
                    activeTab === tab
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </nav>
          </div>

          <div className="p-6">
            {activeTab === 'resources' && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Affected Resources ({selectedViolation.affectedResources.length})</h3>
                {selectedViolation.affectedResources.map((resource) => (
                  <motion.div
                    key={resource.resourceId}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                    onClick={() => drillIntoResource(resource)}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <Database className="w-4 h-4 text-gray-400" />
                          <h4 className="font-medium text-gray-900">{resource.resourceName}</h4>
                          <span className="text-xs text-gray-500">({resource.resourceType})</span>
                          <span className={`px-2 py-0.5 text-xs rounded-full ${
                            resource.complianceState === 'non_compliant' 
                              ? 'bg-red-100 text-red-600' 
                              : 'bg-green-100 text-green-600'
                          }`}>
                            {resource.complianceState.replace('_', ' ')}
                          </span>
                        </div>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-600">
                          <div>
                            <span className="font-medium">Subscription:</span> {resource.subscription}
                          </div>
                          <div>
                            <span className="font-medium">Resource Group:</span> {resource.resourceGroup}
                          </div>
                          <div>
                            <span className="font-medium">Region:</span> {resource.region}
                          </div>
                          <div>
                            <span className="font-medium">Cost Impact:</span> ${resource.costImpact}/mo
                          </div>
                        </div>

                        {resource.configurationDrift.length > 0 && (
                          <div className="mt-3 flex items-center text-sm text-orange-600">
                            <AlertCircle className="w-4 h-4 mr-1" />
                            {resource.configurationDrift.length} configuration drift{resource.configurationDrift.length > 1 ? 's' : ''} detected
                          </div>
                        )}
                      </div>
                      
                      <ChevronRight className="w-5 h-5 text-gray-400 mt-1" />
                    </div>
                  </motion.div>
                ))}
              </div>
            )}

            {activeTab === 'policies' && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Violated Policies ({selectedViolation.violatedPolicies.length})</h3>
                {selectedViolation.violatedPolicies.map((policy) => (
                  <motion.div
                    key={policy.policyId}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                    onClick={() => drillIntoPolicy(policy)}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <FileText className="w-4 h-4 text-gray-400" />
                          <h4 className="font-medium text-gray-900">{policy.policyName}</h4>
                          <span className="px-2 py-0.5 text-xs bg-blue-100 text-blue-600 rounded-full">
                            {policy.policyType.replace('_', ' ')}
                          </span>
                          <span className={`px-2 py-0.5 text-xs rounded-full ${
                            policy.enforcementMode === 'enforced' 
                              ? 'bg-green-100 text-green-600' 
                              : 'bg-yellow-100 text-yellow-600'
                          }`}>
                            {policy.enforcementMode}
                          </span>
                        </div>
                        
                        <div className="text-sm text-gray-600">
                          <p className="mb-2">{policy.category} - Effect: {policy.effect}</p>
                          <p>Assignment Scope: {policy.assignmentScope}</p>
                        </div>
                      </div>
                      
                      <ChevronRight className="w-5 h-5 text-gray-400 mt-1" />
                    </div>
                  </motion.div>
                ))}
              </div>
            )}

            {activeTab === 'remediation' && (
              <div className="space-y-4">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Remediation Steps ({selectedViolation.remediationSteps.filter(s => s.status === 'completed').length}/{selectedViolation.remediationSteps.length})
                  </h3>
                  <button
                    type="button"
                    className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                    onClick={() => toast({ title: 'Remediation', description: 'Executing automated steps...' })}
                  >
                    Execute All Automated Steps
                  </button>
                </div>

                {selectedViolation.remediationSteps.map((step, index) => (
                  <motion.div
                    key={step.stepId}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className={`border rounded-lg p-4 ${
                      step.status === 'completed' 
                        ? 'bg-green-50 border-green-200' 
                        : 'bg-white border-gray-200'
                    }`}
                  >
                    <div className="flex items-start space-x-4">
                      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                        step.status === 'completed' 
                          ? 'bg-green-600 text-white' 
                          : 'bg-gray-200 text-gray-600'
                      }`}>
                        {step.status === 'completed' ? <CheckCircle className="w-5 h-5" /> : step.stepNumber}
                      </div>
                      
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium text-gray-900">{step.title}</h4>
                          <div className="flex items-center space-x-2">
                            <span className={`px-2 py-1 text-xs rounded-full ${
                              step.complexity === 'high' 
                                ? 'bg-red-100 text-red-600'
                                : step.complexity === 'medium'
                                ? 'bg-yellow-100 text-yellow-600'
                                : 'bg-green-100 text-green-600'
                            }`}>
                              {step.complexity} complexity
                            </span>
                            <span className="text-xs text-gray-500">
                              <Clock className="w-3 h-3 inline mr-1" />
                              {step.estimatedTime}
                            </span>
                          </div>
                        </div>
                        
                        <p className="text-sm text-gray-600 mb-3">{step.description}</p>
                        
                        {step.type === 'automated' || step.type === 'script' ? (
                          <div className="space-y-2">
                            {step.automationScript && (
                              <div className="bg-gray-900 text-gray-100 rounded-md p-3 font-mono text-xs overflow-x-auto">
                                <pre>{step.automationScript}</pre>
                              </div>
                            )}
                            
                            {step.status !== 'completed' && (
                              <div className="flex items-center space-x-3">
                                <button type="button"
                                  onClick={() => executeRemediation(step)}
                                  className="px-3 py-1 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 flex items-center"
                                >
                                  <Terminal className="w-4 h-4 mr-1" />
                                  Execute
                                </button>
                                
                                {remediationProgress.has(step.stepId) && (
                                  <div className="flex-1">
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                      <div 
                                        className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                                        style={{ width: `${remediationProgress.get(step.stepId)}%` }}
                                      />
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="space-y-2">
                            <div className="text-sm">
                              <span className="font-medium text-gray-700">Required Permissions:</span>
                              <div className="flex flex-wrap gap-1 mt-1">
                                {step.requiredPermissions.map((perm, i) => (
                                  <span key={i} className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded">
                                    {perm}
                                  </span>
                                ))}
                              </div>
                            </div>
                            
                            {step.status !== 'completed' && (
                              <button
                                type="button"
                                className="px-3 py-1 bg-gray-600 text-white text-sm rounded-md hover:bg-gray-700"
                                onClick={() => toast({ title: 'Step', description: 'Marked as complete' })}
                              >
                                Mark as Complete
                              </button>
                            )}
                          </div>
                        )}
                        
                        {step.validationSteps.length > 0 && (
                          <div className="mt-3 p-3 bg-blue-50 rounded-md">
                            <h5 className="text-sm font-medium text-blue-900 mb-1">Validation Steps:</h5>
                            <ul className="list-disc list-inside text-sm text-blue-700 space-y-1">
                              {step.validationSteps.map((validation, i) => (
                                <li key={i}>{validation}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}

            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Root Cause Analysis */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Root Cause Analysis</h3>
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <div className="flex items-start">
                      <Info className="w-5 h-5 text-yellow-600 mr-3 mt-0.5" />
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2">Primary Cause Identified</h4>
                        <p className="text-sm text-gray-700">
                          {selectedViolation.rootCause?.description || 'Analysis in progress...'}
                        </p>
                        {selectedViolation.rootCause?.contributingFactors && (
                          <div className="mt-3">
                            <h5 className="text-sm font-medium text-gray-700 mb-1">Contributing Factors:</h5>
                            <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                              {selectedViolation.rootCause.contributingFactors.map((factor: string, i: number) => (
                                <li key={i}>{factor}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Impact Analysis */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Impact Analysis</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-red-50 rounded-lg p-4">
                      <h4 className="font-medium text-red-900 mb-2">Business Impact</h4>
                      <p className="text-sm text-red-700">
                        {selectedViolation.impactAnalysis?.businessImpact || 'High risk to business operations'}
                      </p>
                    </div>
                    <div className="bg-orange-50 rounded-lg p-4">
                      <h4 className="font-medium text-orange-900 mb-2">Security Impact</h4>
                      <p className="text-sm text-orange-700">
                        {selectedViolation.impactAnalysis?.securityImpact || 'Potential security vulnerability'}
                      </p>
                    </div>
                    <div className="bg-blue-50 rounded-lg p-4">
                      <h4 className="font-medium text-blue-900 mb-2">Compliance Impact</h4>
                      <p className="text-sm text-blue-700">
                        {selectedViolation.impactAnalysis?.complianceImpact || 'May affect regulatory compliance'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Compliance Frameworks */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Affected Compliance Frameworks</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedViolation.complianceFrameworks.map((framework, index) => (
                      <span 
                        key={index}
                        className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium"
                      >
                        {framework}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'timeline' && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Event Timeline</h3>
                <div className="relative">
                  <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-300"></div>
                  {selectedViolation.timeline?.map((event: any, index: number) => (
                    <div key={index} className="relative flex items-start mb-6">
                      <div className="absolute left-4 w-2 h-2 bg-blue-600 rounded-full -translate-x-1/2 mt-2"></div>
                      <div className="ml-10">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className="text-sm font-medium text-gray-900">{event.title}</span>
                          <span className="text-xs text-gray-500">
                            {format(new Date(event.timestamp), 'MMM dd, yyyy HH:mm')}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600">{event.description}</p>
                        {event.actor && (
                          <p className="text-xs text-gray-500 mt-1">
                            <User className="w-3 h-3 inline mr-1" />
                            {event.actor}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderResourceDetail = () => {
    if (!selectedResource) return null;

    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex justify-between items-start mb-6">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">{selectedResource.resourceName}</h2>
            <p className="text-sm text-gray-600 mt-1">
              {selectedResource.resourceType} • {selectedResource.resourceId}
            </p>
          </div>
          <div className="flex space-x-2">
            <button
              type="button"
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              onClick={() => toast({ title: 'Azure', description: 'Opening Azure portal link' })}
            >
              View in Azure Portal
            </button>
            <button
              type="button"
              className="px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
              onClick={() => toast({ title: 'Dependencies', description: 'Showing dependencies' })}
            >
              View Dependencies
            </button>
          </div>
        </div>

        {/* Configuration Drift */}
        {selectedResource.configurationDrift.length > 0 && (
          <div className="mb-6">
            <h3 className="font-semibold text-gray-900 mb-4">Configuration Drift Detected</h3>
            <div className="space-y-3">
              {selectedResource.configurationDrift.map((drift, index) => (
                <div key={index} className="bg-orange-50 border border-orange-200 rounded-lg p-4">
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-medium text-gray-900">{drift.property}</h4>
                      <div className="mt-2 grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">Expected:</span>
                          <code className="ml-2 px-2 py-1 bg-green-100 text-green-700 rounded">
                            {JSON.stringify(drift.expectedValue)}
                          </code>
                        </div>
                        <div>
                          <span className="text-gray-600">Actual:</span>
                          <code className="ml-2 px-2 py-1 bg-red-100 text-red-700 rounded">
                            {JSON.stringify(drift.actualValue)}
                          </code>
                        </div>
                      </div>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full ${getSeverityColor(drift.impact)}`}>
                      {drift.impact}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Resource Health */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-900 mb-4">Resource Details</h3>
            <div className="space-y-3">
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Subscription</span>
                <span className="text-sm font-medium">{selectedResource.subscription}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Resource Group</span>
                <span className="text-sm font-medium">{selectedResource.resourceGroup}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Region</span>
                <span className="text-sm font-medium">{selectedResource.region}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Last Assessed</span>
                <span className="text-sm font-medium">
                  {formatDistanceToNow(new Date(selectedResource.lastAssessed), { addSuffix: true })}
                </span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 mb-4">Tags</h3>
            <div className="flex flex-wrap gap-2">
              {Object.entries(selectedResource.tags).map(([key, value]) => (
                <span key={key} className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-sm">
                  {key}: {value}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900">Compliance Deep-Drill Analysis</h1>
          <p className="text-gray-600">Comprehensive violation analysis and remediation tracking</p>
        </div>

        {/* Breadcrumb */}
        {renderBreadcrumb()}

        {/* Main Content */}
        <AnimatePresence mode="wait">
          {loading ? (
            <div className="flex justify-center items-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
          ) : selectedResource ? (
            renderResourceDetail()
          ) : selectedViolation ? (
            renderViolationOverview()
          ) : (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
              <AlertTriangle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Select a Compliance Violation</h3>
              <p className="text-gray-600 mb-6">
                Choose a violation from the compliance dashboard to begin deep-drill analysis
              </p>
              <button type="button" 
                onClick={() => fetchViolationDetails('violation-123')}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Load Sample Violation
              </button>
            </div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// Type definitions for missing interfaces
interface RootCauseAnalysis {
  description: string;
  contributingFactors: string[];
  preventionMeasures: string[];
}

interface ImpactAnalysis {
  businessImpact: string;
  securityImpact: string;
  complianceImpact: string;
  financialImpact: number;
  affectedUsers: number;
}

interface TimelineEvent {
  eventId: string;
  timestamp: string;
  title: string;
  description: string;
  actor?: string;
  eventType: string;
}

interface ResourceHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  lastCheck: string;
  metrics: Record<string, any>;
}

interface ResourceDependency {
  resourceId: string;
  resourceName: string;
  dependencyType: string;
  impact: string;
}

interface PolicyDefinition {
  rules: any;
  parameters: Record<string, any>;
  metadata: Record<string, any>;
}

interface ViolationDetails {
  failureReason: string;
  evaluationDetails: any;
  remediationHint: string;
}

interface HistoricalCompliance {
  dataPoints: Array<{
    date: string;
    compliancePercentage: number;
    violationCount: number;
  }>;
  trend: 'improving' | 'stable' | 'degrading';
}