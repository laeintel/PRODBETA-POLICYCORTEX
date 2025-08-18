'use client';

import React, { useState, useEffect } from 'react';
import { 
  Percent, Shield, CheckCircle, XCircle, AlertTriangle, TrendingUp, TrendingDown,
  Activity, BarChart3, PieChart, LineChart, Award, Target, Flag,
  FileCheck, BookOpen, Scale, Clock, Calendar, Timer, RefreshCw,
  Download, Filter, Search, ChevronRight, ChevronDown, MoreVertical,
  Info, Eye, FileText, Users, Building, Globe, Briefcase
} from 'lucide-react';

interface ComplianceScore {
  id: string;
  framework: string;
  category: string;
  score: number;
  previousScore: number;
  target: number;
  trend: 'up' | 'down' | 'stable';
  status: 'compliant' | 'partial' | 'non-compliant';
  lastAssessment: string;
  nextAssessment: string;
  requirements: {
    total: number;
    met: number;
    partial: number;
    failed: number;
  };
  controls: {
    implemented: number;
    effective: number;
    ineffective: number;
  };
  findings: number;
  criticalFindings: number;
}

interface ComplianceRequirement {
  id: string;
  framework: string;
  requirement: string;
  description: string;
  status: 'met' | 'partial' | 'failed' | 'not-applicable';
  evidence: string[];
  controls: string[];
  lastReview: string;
  reviewer: string;
  notes: string;
}

interface ComplianceTrend {
  date: string;
  score: number;
  findings: number;
  requirements: number;
}

export default function ComplianceScores() {
  const [scores, setScores] = useState<ComplianceScore[]>([]);
  const [requirements, setRequirements] = useState<ComplianceRequirement[]>([]);
  const [trends, setTrends] = useState<ComplianceTrend[]>([]);
  const [selectedFramework, setSelectedFramework] = useState('all');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [viewMode, setViewMode] = useState<'scores' | 'requirements' | 'trends' | 'reports'>('scores');
  const [expandedScore, setExpandedScore] = useState<string | null>(null);

  useEffect(() => {
    // Initialize with compliance data
    setScores([
      {
        id: 'CS-001',
        framework: 'ISO 27001:2013',
        category: 'Information Security',
        score: 92,
        previousScore: 88,
        target: 95,
        trend: 'up',
        status: 'partial',
        lastAssessment: '2 weeks ago',
        nextAssessment: 'in 10 weeks',
        requirements: { total: 114, met: 105, partial: 7, failed: 2 },
        controls: { implemented: 93, effective: 88, ineffective: 5 },
        findings: 12,
        criticalFindings: 2
      },
      {
        id: 'CS-002',
        framework: 'SOC 2 Type II',
        category: 'Trust Services',
        score: 96,
        previousScore: 94,
        target: 95,
        trend: 'up',
        status: 'compliant',
        lastAssessment: '1 month ago',
        nextAssessment: 'in 5 months',
        requirements: { total: 64, met: 61, partial: 3, failed: 0 },
        controls: { implemented: 58, effective: 56, ineffective: 2 },
        findings: 5,
        criticalFindings: 0
      },
      {
        id: 'CS-003',
        framework: 'PCI DSS v4.0',
        category: 'Payment Card Security',
        score: 98,
        previousScore: 97,
        target: 100,
        trend: 'up',
        status: 'compliant',
        lastAssessment: '3 weeks ago',
        nextAssessment: 'in 3 months',
        requirements: { total: 384, met: 376, partial: 8, failed: 0 },
        controls: { implemented: 312, effective: 308, ineffective: 4 },
        findings: 8,
        criticalFindings: 0
      },
      {
        id: 'CS-004',
        framework: 'GDPR',
        category: 'Data Protection',
        score: 94,
        previousScore: 91,
        target: 95,
        trend: 'up',
        status: 'partial',
        lastAssessment: '1 week ago',
        nextAssessment: 'in 11 weeks',
        requirements: { total: 99, met: 93, partial: 4, failed: 2 },
        controls: { implemented: 82, effective: 77, ineffective: 5 },
        findings: 10,
        criticalFindings: 1
      },
      {
        id: 'CS-005',
        framework: 'HIPAA',
        category: 'Healthcare',
        score: 89,
        previousScore: 85,
        target: 95,
        trend: 'up',
        status: 'partial',
        lastAssessment: '2 months ago',
        nextAssessment: 'in 4 months',
        requirements: { total: 78, met: 69, partial: 7, failed: 2 },
        controls: { implemented: 65, effective: 58, ineffective: 7 },
        findings: 15,
        criticalFindings: 3
      },
      {
        id: 'CS-006',
        framework: 'NIST CSF',
        category: 'Cybersecurity',
        score: 91,
        previousScore: 90,
        target: 90,
        trend: 'up',
        status: 'compliant',
        lastAssessment: '1 month ago',
        nextAssessment: 'in 2 months',
        requirements: { total: 98, met: 89, partial: 8, failed: 1 },
        controls: { implemented: 76, effective: 72, ineffective: 4 },
        findings: 9,
        criticalFindings: 1
      },
      {
        id: 'CS-007',
        framework: 'CIS Controls',
        category: 'Security Best Practices',
        score: 85,
        previousScore: 82,
        target: 90,
        trend: 'up',
        status: 'partial',
        lastAssessment: '3 weeks ago',
        nextAssessment: 'in 9 weeks',
        requirements: { total: 153, met: 130, partial: 18, failed: 5 },
        controls: { implemented: 124, effective: 112, ineffective: 12 },
        findings: 23,
        criticalFindings: 5
      },
      {
        id: 'CS-008',
        framework: 'CCPA',
        category: 'California Privacy',
        score: 93,
        previousScore: 93,
        target: 95,
        trend: 'stable',
        status: 'partial',
        lastAssessment: '2 weeks ago',
        nextAssessment: 'in 10 weeks',
        requirements: { total: 45, met: 42, partial: 2, failed: 1 },
        controls: { implemented: 38, effective: 36, ineffective: 2 },
        findings: 6,
        criticalFindings: 1
      }
    ]);

    setRequirements([
      {
        id: 'REQ-001',
        framework: 'ISO 27001:2013',
        requirement: 'A.5.1.1',
        description: 'Information security policy document',
        status: 'met',
        evidence: ['Policy_v2.1.pdf', 'Board_Approval.pdf'],
        controls: ['DOC-001', 'REV-001'],
        lastReview: '1 month ago',
        reviewer: 'John Smith',
        notes: 'Policy updated and approved by board'
      },
      {
        id: 'REQ-002',
        framework: 'GDPR',
        requirement: 'Article 32',
        description: 'Security of processing',
        status: 'partial',
        evidence: ['Encryption_Report.pdf', 'Access_Controls.xlsx'],
        controls: ['ENC-001', 'ACC-001', 'MON-001'],
        lastReview: '2 weeks ago',
        reviewer: 'Jane Doe',
        notes: 'Need to implement additional monitoring controls'
      },
      {
        id: 'REQ-003',
        framework: 'PCI DSS v4.0',
        requirement: '8.3.1',
        description: 'Multi-factor authentication for all access',
        status: 'met',
        evidence: ['MFA_Implementation.pdf', 'Access_Logs.csv'],
        controls: ['MFA-001', 'LOG-001'],
        lastReview: '3 weeks ago',
        reviewer: 'Security Team',
        notes: 'MFA fully implemented across all systems'
      }
    ]);

    // Generate trend data
    const trendData: ComplianceTrend[] = [];
    for (let i = 11; i >= 0; i--) {
      trendData.push({
        date: `Month ${12 - i}`,
        score: 80 + Math.random() * 15,
        findings: Math.floor(5 + Math.random() * 20),
        requirements: Math.floor(90 + Math.random() * 10)
      });
    }
    setTrends(trendData);
  }, []);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'compliant': case 'met': return 'text-green-500 bg-green-900/20';
      case 'partial': return 'text-yellow-500 bg-yellow-900/20';
      case 'non-compliant': case 'failed': return 'text-red-500 bg-red-900/20';
      case 'not-applicable': return 'text-gray-500 bg-gray-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getScoreColor = (score: number, target: number) => {
    const percentage = (score / target) * 100;
    if (percentage >= 100) return 'text-green-500';
    if (percentage >= 90) return 'text-yellow-500';
    if (percentage >= 80) return 'text-orange-500';
    return 'text-red-500';
  };

  const overallScore = scores.reduce((acc, s) => acc + s.score, 0) / scores.length;
  const totalFindings = scores.reduce((acc, s) => acc + s.findings, 0);
  const criticalFindings = scores.reduce((acc, s) => acc + s.criticalFindings, 0);
  const compliantFrameworks = scores.filter(s => s.status === 'compliant').length;

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold flex items-center space-x-2">
              <Percent className="w-6 h-6 text-green-500" />
              <span>Compliance Scores</span>
            </h1>
            <p className="text-sm text-gray-400 mt-1">Framework compliance scoring and requirement tracking</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
            
            <button className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm flex items-center space-x-2">
              <FileCheck className="w-4 h-4" />
              <span>New Assessment</span>
            </button>
          </div>
        </div>
      </header>

      {/* Summary Stats */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="text-center">
            <div className="text-xs text-gray-500">Overall Score</div>
            <div className={`text-2xl font-bold ${getScoreColor(overallScore, 95)}`}>
              {overallScore.toFixed(1)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Frameworks</div>
            <div className="text-2xl font-bold">{scores.length}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Compliant</div>
            <div className="text-2xl font-bold text-green-500">{compliantFrameworks}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Partial</div>
            <div className="text-2xl font-bold text-yellow-500">
              {scores.filter(s => s.status === 'partial').length}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Total Findings</div>
            <div className="text-2xl font-bold text-orange-500">{totalFindings}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Critical</div>
            <div className="text-2xl font-bold text-red-500">{criticalFindings}</div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-gray-900/30 border-b border-gray-800 px-6">
        <div className="flex space-x-6">
          {['scores', 'requirements', 'trends', 'reports'].map(view => (
            <button
              key={view}
              onClick={() => setViewMode(view as any)}
              className={`py-3 border-b-2 text-sm capitalize ${
                viewMode === view 
                  ? 'border-green-500 text-green-500' 
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
        {viewMode === 'scores' && (
          <div className="space-y-4">
            {/* Filters */}
            <div className="flex items-center space-x-3 mb-4">
              <select
                value={selectedFramework}
                onChange={(e) => setSelectedFramework(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Frameworks</option>
                <option value="ISO 27001">ISO 27001</option>
                <option value="SOC 2">SOC 2</option>
                <option value="PCI DSS">PCI DSS</option>
                <option value="GDPR">GDPR</option>
                <option value="HIPAA">HIPAA</option>
              </select>

              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Status</option>
                <option value="compliant">Compliant</option>
                <option value="partial">Partial</option>
                <option value="non-compliant">Non-Compliant</option>
              </select>
            </div>

            {/* Compliance Scores Grid */}
            <div className="grid grid-cols-2 gap-4">
              {scores.map(score => (
                <div key={score.id} className="bg-gray-900 border border-gray-800 rounded-lg">
                  <div className="p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h3 className="text-sm font-bold">{score.framework}</h3>
                        <p className="text-xs text-gray-500">{score.category}</p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-1 rounded text-xs ${getStatusColor(score.status)}`}>
                          {score.status}
                        </span>
                        <button 
                          onClick={() => setExpandedScore(expandedScore === score.id ? null : score.id)}
                          className="p-1 hover:bg-gray-800 rounded"
                        >
                          <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${
                            expandedScore === score.id ? 'rotate-180' : ''
                          }`} />
                        </button>
                      </div>
                    </div>

                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-4">
                        <div>
                          <div className={`text-3xl font-bold ${getScoreColor(score.score, score.target)}`}>
                            {score.score}%
                          </div>
                          <div className="flex items-center space-x-1 text-xs text-gray-500">
                            <span>Target: {score.target}%</span>
                            {score.trend === 'up' ? (
                              <TrendingUp className="w-3 h-3 text-green-500" />
                            ) : score.trend === 'down' ? (
                              <TrendingDown className="w-3 h-3 text-red-500" />
                            ) : (
                              <Activity className="w-3 h-3 text-gray-500" />
                            )}
                            <span>from {score.previousScore}%</span>
                          </div>
                        </div>
                      </div>

                      <div className="text-right">
                        <div className="text-xs text-gray-500">Next Assessment</div>
                        <div className="text-sm font-medium">{score.nextAssessment}</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-3 text-xs">
                      <div className="bg-gray-800 rounded p-2">
                        <div className="text-gray-500">Requirements</div>
                        <div className="flex items-center justify-between mt-1">
                          <span className="text-green-500">{score.requirements.met}</span>
                          <span className="text-yellow-500">{score.requirements.partial}</span>
                          <span className="text-red-500">{score.requirements.failed}</span>
                        </div>
                        <div className="text-gray-600 text-center">of {score.requirements.total}</div>
                      </div>
                      <div className="bg-gray-800 rounded p-2">
                        <div className="text-gray-500">Controls</div>
                        <div className="text-lg font-bold">{score.controls.effective}</div>
                        <div className="text-gray-600">effective</div>
                      </div>
                      <div className="bg-gray-800 rounded p-2">
                        <div className="text-gray-500">Findings</div>
                        <div className="text-lg font-bold">{score.findings}</div>
                        <div className="text-red-500">{score.criticalFindings} critical</div>
                      </div>
                    </div>

                    {expandedScore === score.id && (
                      <div className="mt-4 pt-4 border-t border-gray-800">
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-gray-500">Last Assessment</span>
                            <span>{score.lastAssessment}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Implemented Controls</span>
                            <span>{score.controls.implemented}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Ineffective Controls</span>
                            <span className="text-orange-500">{score.controls.ineffective}</span>
                          </div>
                        </div>
                        <div className="flex space-x-2 mt-3">
                          <button className="flex-1 px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                            View Details
                          </button>
                          <button className="flex-1 px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                            Requirements
                          </button>
                          <button className="flex-1 px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                            Report
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {viewMode === 'requirements' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-800">
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Framework</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Requirement</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Description</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Status</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Evidence</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Last Review</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Actions</th>
                </tr>
              </thead>
              <tbody>
                {requirements.map(req => (
                  <tr key={req.id} className="border-t border-gray-800 hover:bg-gray-800/30">
                    <td className="px-4 py-3 text-sm">{req.framework}</td>
                    <td className="px-4 py-3 text-sm font-mono">{req.requirement}</td>
                    <td className="px-4 py-3 text-sm">{req.description}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs ${getStatusColor(req.status)}`}>
                        {req.status}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center space-x-1">
                        <FileText className="w-3 h-3 text-gray-500" />
                        <span className="text-xs">{req.evidence.length} files</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm">{req.lastReview}</td>
                    <td className="px-4 py-3">
                      <button className="text-xs text-blue-500 hover:text-blue-400">
                        Review
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {viewMode === 'trends' && (
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Compliance Score Trend</h3>
              <div className="h-48 flex items-end justify-between space-x-2">
                {trends.map((trend, idx) => (
                  <div key={idx} className="flex-1 flex flex-col items-center">
                    <div 
                      className="w-full bg-gradient-to-t from-green-600 to-green-400 rounded-t"
                      style={{ height: `${trend.score}%` }}
                    />
                    <span className="text-xs text-gray-500 mt-2 -rotate-45 origin-left">
                      {trend.date}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Findings Trend</h3>
              <div className="h-48 flex items-end justify-between space-x-2">
                {trends.map((trend, idx) => (
                  <div key={idx} className="flex-1 flex flex-col items-center">
                    <div 
                      className="w-full bg-gradient-to-t from-orange-600 to-orange-400 rounded-t"
                      style={{ height: `${(trend.findings / 30) * 100}%` }}
                    />
                    <span className="text-xs text-gray-500 mt-2 -rotate-45 origin-left">
                      {trend.date}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {viewMode === 'reports' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-8 text-center">
            <FileCheck className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-lg font-bold mb-2">Compliance Reports</h3>
            <p className="text-sm text-gray-500 mb-6">Generate and download compliance assessment reports</p>
            <div className="flex justify-center space-x-3">
              <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                Generate Report
              </button>
              <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm">
                Schedule Report
              </button>
              <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm">
                View History
              </button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}