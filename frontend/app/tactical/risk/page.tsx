'use client';

import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, Shield, TrendingUp, TrendingDown, Activity, Target,
  CheckCircle, XCircle, Info, Clock, Calendar, Timer, Users,
  FileText, BarChart3, PieChart, LineChart, Gauge, Thermometer,
  Eye, Lock, Zap, Database, Cloud, Server, Globe, Building,
  ChevronRight, ChevronDown, MoreVertical, ExternalLink, Download,
  RefreshCw, Filter, Search, Bell, Flag, Briefcase, Scale
} from 'lucide-react';

interface Risk {
  id: string;
  title: string;
  description: string;
  category: 'strategic' | 'operational' | 'financial' | 'compliance' | 'technology' | 'security' | 'reputational';
  likelihood: 1 | 2 | 3 | 4 | 5;
  impact: 1 | 2 | 3 | 4 | 5;
  inherentRisk: number;
  residualRisk: number;
  riskAppetite: number;
  status: 'identified' | 'assessing' | 'treating' | 'monitoring' | 'closed';
  owner: string;
  department: string;
  identifiedDate: string;
  lastReview: string;
  nextReview: string;
  controls: Control[];
  mitigations: Mitigation[];
  kris: string[]; // Key Risk Indicators
}

interface Control {
  id: string;
  name: string;
  type: 'preventive' | 'detective' | 'corrective';
  effectiveness: number;
  status: 'effective' | 'partial' | 'ineffective';
  implementationDate: string;
}

interface Mitigation {
  id: string;
  action: string;
  status: 'planned' | 'in-progress' | 'completed';
  dueDate: string;
  owner: string;
  cost: number;
}

interface RiskMatrix {
  likelihood: number;
  impact: number;
  count: number;
  risks: string[];
}

export default function RiskAssessment() {
  const [risks, setRisks] = useState<Risk[]>([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [viewMode, setViewMode] = useState<'matrix' | 'register' | 'trends' | 'controls'>('matrix');
  const [expandedRisk, setExpandedRisk] = useState<string | null>(null);
  const [riskMatrix, setRiskMatrix] = useState<RiskMatrix[]>([]);

  useEffect(() => {
    // Initialize with risk data
    const mockRisks: Risk[] = [
      {
        id: 'RISK-001',
        title: 'Data Breach',
        description: 'Unauthorized access to sensitive customer data',
        category: 'security',
        likelihood: 3,
        impact: 5,
        inherentRisk: 15,
        residualRisk: 8,
        riskAppetite: 5,
        status: 'treating',
        owner: 'CISO',
        department: 'Information Security',
        identifiedDate: '2024-01-15',
        lastReview: '2 weeks ago',
        nextReview: 'in 2 weeks',
        controls: [
          { id: 'C001', name: 'Encryption at Rest', type: 'preventive', effectiveness: 90, status: 'effective', implementationDate: '2024-01-20' },
          { id: 'C002', name: 'Access Controls', type: 'preventive', effectiveness: 85, status: 'effective', implementationDate: '2024-01-25' },
          { id: 'C003', name: 'Security Monitoring', type: 'detective', effectiveness: 80, status: 'effective', implementationDate: '2024-02-01' }
        ],
        mitigations: [
          { id: 'M001', action: 'Implement DLP Solution', status: 'in-progress', dueDate: '2024-12-31', owner: 'Security Team', cost: 50000 },
          { id: 'M002', action: 'Security Awareness Training', status: 'completed', dueDate: '2024-11-30', owner: 'HR Team', cost: 10000 }
        ],
        kris: ['Failed login attempts > 100/hour', 'Data export > 1GB/day', 'Privilege escalations > 5/week']
      },
      {
        id: 'RISK-002',
        title: 'Regulatory Non-Compliance',
        description: 'Failure to meet GDPR requirements',
        category: 'compliance',
        likelihood: 2,
        impact: 4,
        inherentRisk: 8,
        residualRisk: 4,
        riskAppetite: 3,
        status: 'monitoring',
        owner: 'Chief Compliance Officer',
        department: 'Legal & Compliance',
        identifiedDate: '2024-02-01',
        lastReview: '1 month ago',
        nextReview: 'in 1 month',
        controls: [
          { id: 'C004', name: 'Privacy Policy', type: 'preventive', effectiveness: 95, status: 'effective', implementationDate: '2024-02-15' },
          { id: 'C005', name: 'Consent Management', type: 'preventive', effectiveness: 88, status: 'effective', implementationDate: '2024-02-20' }
        ],
        mitigations: [
          { id: 'M003', action: 'Privacy Impact Assessment', status: 'completed', dueDate: '2024-10-31', owner: 'DPO', cost: 15000 }
        ],
        kris: ['Consent rate < 80%', 'Data retention > policy', 'Subject requests > 10/month']
      },
      {
        id: 'RISK-003',
        title: 'Cloud Service Outage',
        description: 'Major cloud provider service disruption',
        category: 'operational',
        likelihood: 2,
        impact: 5,
        inherentRisk: 10,
        residualRisk: 5,
        riskAppetite: 6,
        status: 'treating',
        owner: 'CTO',
        department: 'Infrastructure',
        identifiedDate: '2024-03-01',
        lastReview: '3 weeks ago',
        nextReview: 'in 1 week',
        controls: [
          { id: 'C006', name: 'Multi-Region Deployment', type: 'preventive', effectiveness: 85, status: 'effective', implementationDate: '2024-03-15' },
          { id: 'C007', name: 'Backup Strategy', type: 'corrective', effectiveness: 90, status: 'effective', implementationDate: '2024-03-20' },
          { id: 'C008', name: 'Disaster Recovery Plan', type: 'corrective', effectiveness: 75, status: 'partial', implementationDate: '2024-04-01' }
        ],
        mitigations: [
          { id: 'M004', action: 'Multi-Cloud Strategy', status: 'planned', dueDate: '2025-06-30', owner: 'Architecture Team', cost: 100000 }
        ],
        kris: ['SLA violations > 1/month', 'RTO > 4 hours', 'RPO > 1 hour']
      },
      {
        id: 'RISK-004',
        title: 'Budget Overrun',
        description: 'Project costs exceeding allocated budget',
        category: 'financial',
        likelihood: 4,
        impact: 3,
        inherentRisk: 12,
        residualRisk: 7,
        riskAppetite: 8,
        status: 'monitoring',
        owner: 'CFO',
        department: 'Finance',
        identifiedDate: '2024-01-01',
        lastReview: '1 week ago',
        nextReview: 'in 3 weeks',
        controls: [
          { id: 'C009', name: 'Budget Monitoring', type: 'detective', effectiveness: 80, status: 'effective', implementationDate: '2024-01-10' },
          { id: 'C010', name: 'Approval Workflow', type: 'preventive', effectiveness: 75, status: 'partial', implementationDate: '2024-01-15' }
        ],
        mitigations: [
          { id: 'M005', action: 'Implement FinOps', status: 'in-progress', dueDate: '2024-12-31', owner: 'Finance Team', cost: 25000 }
        ],
        kris: ['Cost variance > 10%', 'Unplanned expenses > $10k/month', 'Budget utilization > 90%']
      },
      {
        id: 'RISK-005',
        title: 'Key Person Dependency',
        description: 'Critical knowledge concentrated in single individuals',
        category: 'operational',
        likelihood: 3,
        impact: 3,
        inherentRisk: 9,
        residualRisk: 6,
        riskAppetite: 5,
        status: 'treating',
        owner: 'CHRO',
        department: 'Human Resources',
        identifiedDate: '2024-04-01',
        lastReview: '2 weeks ago',
        nextReview: 'in 2 weeks',
        controls: [
          { id: 'C011', name: 'Knowledge Documentation', type: 'preventive', effectiveness: 70, status: 'partial', implementationDate: '2024-04-15' },
          { id: 'C012', name: 'Cross-Training Program', type: 'preventive', effectiveness: 65, status: 'partial', implementationDate: '2024-05-01' }
        ],
        mitigations: [
          { id: 'M006', action: 'Succession Planning', status: 'in-progress', dueDate: '2025-03-31', owner: 'HR Team', cost: 20000 }
        ],
        kris: ['Single points of failure > 5', 'Documentation coverage < 70%', 'Cross-trained staff < 50%']
      },
      {
        id: 'RISK-006',
        title: 'Cyber Attack',
        description: 'Targeted ransomware or malware attack',
        category: 'security',
        likelihood: 3,
        impact: 5,
        inherentRisk: 15,
        residualRisk: 9,
        riskAppetite: 5,
        status: 'treating',
        owner: 'CISO',
        department: 'Information Security',
        identifiedDate: '2024-02-15',
        lastReview: '1 week ago',
        nextReview: 'in 1 week',
        controls: [
          { id: 'C013', name: 'EDR Solution', type: 'detective', effectiveness: 85, status: 'effective', implementationDate: '2024-03-01' },
          { id: 'C014', name: 'Network Segmentation', type: 'preventive', effectiveness: 80, status: 'effective', implementationDate: '2024-03-15' },
          { id: 'C015', name: 'Incident Response Plan', type: 'corrective', effectiveness: 75, status: 'partial', implementationDate: '2024-04-01' }
        ],
        mitigations: [
          { id: 'M007', action: 'Zero Trust Architecture', status: 'planned', dueDate: '2025-12-31', owner: 'Security Architecture', cost: 150000 }
        ],
        kris: ['Malware detections > 10/day', 'Phishing attempts > 100/week', 'Patch compliance < 95%']
      }
    ];

    setRisks(mockRisks);

    // Generate risk matrix data
    const matrixData: RiskMatrix[] = [];
    for (let likelihood = 1; likelihood <= 5; likelihood++) {
      for (let impact = 1; impact <= 5; impact++) {
        const risksInCell = mockRisks.filter(r => r.likelihood === likelihood && r.impact === impact);
        matrixData.push({
          likelihood,
          impact,
          count: risksInCell.length,
          risks: risksInCell.map(r => r.id)
        });
      }
    }
    setRiskMatrix(matrixData);
  }, []);

  const getRiskLevel = (score: number): { label: string; color: string } => {
    if (score <= 3) return { label: 'Low', color: 'bg-green-600' };
    if (score <= 6) return { label: 'Medium', color: 'bg-yellow-600' };
    if (score <= 12) return { label: 'High', color: 'bg-orange-600' };
    return { label: 'Critical', color: 'bg-red-600' };
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'identified': return 'text-blue-500 bg-blue-900/20';
      case 'assessing': return 'text-yellow-500 bg-yellow-900/20';
      case 'treating': return 'text-orange-500 bg-orange-900/20';
      case 'monitoring': return 'text-green-500 bg-green-900/20';
      case 'closed': return 'text-gray-500 bg-gray-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch(category) {
      case 'strategic': return <Target className="w-4 h-4" />;
      case 'operational': return <Activity className="w-4 h-4" />;
      case 'financial': return <Briefcase className="w-4 h-4" />;
      case 'compliance': return <Scale className="w-4 h-4" />;
      case 'technology': return <Server className="w-4 h-4" />;
      case 'security': return <Shield className="w-4 h-4" />;
      case 'reputational': return <Users className="w-4 h-4" />;
      default: return <AlertTriangle className="w-4 h-4" />;
    }
  };

  const totalRisks = risks.length;
  const highRisks = risks.filter(r => r.inherentRisk > 12).length;
  const risksAboveAppetite = risks.filter(r => r.residualRisk > r.riskAppetite).length;
  const avgResidualRisk = risks.reduce((acc, r) => acc + r.residualRisk, 0) / totalRisks;

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold flex items-center space-x-2">
              <AlertTriangle className="w-6 h-6 text-orange-500" />
              <span>Risk Assessment</span>
            </h1>
            <p className="text-sm text-gray-400 mt-1">Enterprise risk management and assessment</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
            
            <button className="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded text-sm flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4" />
              <span>New Risk</span>
            </button>
          </div>
        </div>
      </header>

      {/* Summary Stats */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="text-center">
            <div className="text-xs text-gray-500">Total Risks</div>
            <div className="text-2xl font-bold">{totalRisks}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">High/Critical</div>
            <div className="text-2xl font-bold text-red-500">{highRisks}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Above Appetite</div>
            <div className="text-2xl font-bold text-orange-500">{risksAboveAppetite}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Avg Residual Risk</div>
            <div className="text-2xl font-bold">{avgResidualRisk.toFixed(1)}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Being Treated</div>
            <div className="text-2xl font-bold text-yellow-500">
              {risks.filter(r => r.status === 'treating').length}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Total Controls</div>
            <div className="text-2xl font-bold text-blue-500">
              {risks.reduce((acc, r) => acc + r.controls.length, 0)}
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-gray-900/30 border-b border-gray-800 px-6">
        <div className="flex space-x-6">
          {['matrix', 'register', 'trends', 'controls'].map(view => (
            <button
              key={view}
              onClick={() => setViewMode(view as any)}
              className={`py-3 border-b-2 text-sm capitalize ${
                viewMode === view 
                  ? 'border-orange-500 text-orange-500' 
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              {view === 'matrix' ? 'Risk Matrix' : view === 'register' ? 'Risk Register' : view}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {viewMode === 'matrix' && (
          <div className="space-y-6">
            {/* Risk Heat Map */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h2 className="text-sm font-bold mb-4">Risk Heat Map</h2>
              <div className="flex">
                {/* Y-axis label */}
                <div className="flex flex-col justify-between mr-4 text-xs text-gray-500">
                  <div className="h-20 flex items-center">
                    <span className="-rotate-90 whitespace-nowrap">Likelihood →</span>
                  </div>
                </div>
                
                {/* Matrix Grid */}
                <div>
                  <div className="grid grid-cols-5 gap-1">
                    {[5, 4, 3, 2, 1].map(likelihood => (
                      [1, 2, 3, 4, 5].map(impact => {
                        const cell = riskMatrix.find(m => m.likelihood === likelihood && m.impact === impact);
                        const riskScore = likelihood * impact;
                        const riskLevel = getRiskLevel(riskScore);
                        
                        return (
                          <div
                            key={`${likelihood}-${impact}`}
                            className={`w-24 h-20 ${riskLevel.color} bg-opacity-50 border border-gray-700 rounded flex flex-col items-center justify-center cursor-pointer hover:bg-opacity-70 transition-all`}
                          >
                            <div className="text-2xl font-bold">{cell?.count || 0}</div>
                            <div className="text-xs">Score: {riskScore}</div>
                          </div>
                        );
                      })
                    ))}
                  </div>
                  
                  {/* X-axis label */}
                  <div className="flex justify-between mt-2 text-xs text-gray-500">
                    <span>1</span>
                    <span>2</span>
                    <span>3</span>
                    <span>4</span>
                    <span>5</span>
                  </div>
                  <div className="text-center mt-1 text-xs text-gray-500">Impact →</div>
                </div>
              </div>

              {/* Legend */}
              <div className="flex items-center justify-center space-x-6 mt-6">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-green-600 rounded"></div>
                  <span className="text-xs">Low (1-3)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-yellow-600 rounded"></div>
                  <span className="text-xs">Medium (4-6)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-orange-600 rounded"></div>
                  <span className="text-xs">High (7-12)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-red-600 rounded"></div>
                  <span className="text-xs">Critical (13-25)</span>
                </div>
              </div>
            </div>

            {/* Risk Categories */}
            <div className="grid grid-cols-4 gap-4">
              {['strategic', 'operational', 'financial', 'compliance', 'technology', 'security', 'reputational'].map(category => {
                const categoryRisks = risks.filter(r => r.category === category);
                const avgRisk = categoryRisks.reduce((acc, r) => acc + r.residualRisk, 0) / (categoryRisks.length || 1);
                
                return (
                  <div key={category} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {getCategoryIcon(category)}
                        <h3 className="text-sm font-bold capitalize">{category}</h3>
                      </div>
                      <span className="text-lg font-bold">{categoryRisks.length}</span>
                    </div>
                    <div className="text-xs text-gray-500 mb-2">Avg Risk Score: {avgRisk.toFixed(1)}</div>
                    <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${getRiskLevel(avgRisk).color}`}
                        style={{ width: `${(avgRisk / 25) * 100}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {viewMode === 'register' && (
          <div className="space-y-4">
            {/* Filters */}
            <div className="flex items-center space-x-3 mb-4">
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Categories</option>
                <option value="strategic">Strategic</option>
                <option value="operational">Operational</option>
                <option value="financial">Financial</option>
                <option value="compliance">Compliance</option>
                <option value="technology">Technology</option>
                <option value="security">Security</option>
                <option value="reputational">Reputational</option>
              </select>

              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Status</option>
                <option value="identified">Identified</option>
                <option value="assessing">Assessing</option>
                <option value="treating">Treating</option>
                <option value="monitoring">Monitoring</option>
                <option value="closed">Closed</option>
              </select>
            </div>

            {/* Risk List */}
            {risks.map(risk => (
              <div key={risk.id} className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start space-x-3">
                      <div className={`p-2 rounded ${getRiskLevel(risk.inherentRisk).color} bg-opacity-20`}>
                        {getCategoryIcon(risk.category)}
                      </div>
                      <div>
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className="text-sm font-bold">{risk.title}</h3>
                          <span className={`px-2 py-1 rounded text-xs ${getStatusColor(risk.status)}`}>
                            {risk.status}
                          </span>
                        </div>
                        <p className="text-xs text-gray-400 mb-2">{risk.description}</p>
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <span>Owner: {risk.owner}</span>
                          <span>Department: {risk.department}</span>
                          <span>Next Review: {risk.nextReview}</span>
                        </div>
                      </div>
                    </div>
                    <button 
                      onClick={() => setExpandedRisk(expandedRisk === risk.id ? null : risk.id)}
                      className="p-1 hover:bg-gray-800 rounded"
                    >
                      <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${
                        expandedRisk === risk.id ? 'rotate-180' : ''
                      }`} />
                    </button>
                  </div>

                  <div className="grid grid-cols-5 gap-3 text-xs">
                    <div className="bg-gray-800 rounded p-2">
                      <div className="text-gray-500">Likelihood</div>
                      <div className="text-lg font-bold">{risk.likelihood}/5</div>
                    </div>
                    <div className="bg-gray-800 rounded p-2">
                      <div className="text-gray-500">Impact</div>
                      <div className="text-lg font-bold">{risk.impact}/5</div>
                    </div>
                    <div className="bg-gray-800 rounded p-2">
                      <div className="text-gray-500">Inherent Risk</div>
                      <div className={`text-lg font-bold ${getRiskLevel(risk.inherentRisk).label === 'Critical' ? 'text-red-500' : getRiskLevel(risk.inherentRisk).label === 'High' ? 'text-orange-500' : getRiskLevel(risk.inherentRisk).label === 'Medium' ? 'text-yellow-500' : 'text-green-500'}`}>
                        {risk.inherentRisk}
                      </div>
                    </div>
                    <div className="bg-gray-800 rounded p-2">
                      <div className="text-gray-500">Residual Risk</div>
                      <div className={`text-lg font-bold ${risk.residualRisk > risk.riskAppetite ? 'text-red-500' : 'text-green-500'}`}>
                        {risk.residualRisk}
                      </div>
                    </div>
                    <div className="bg-gray-800 rounded p-2">
                      <div className="text-gray-500">Risk Appetite</div>
                      <div className="text-lg font-bold">{risk.riskAppetite}</div>
                    </div>
                  </div>

                  {expandedRisk === risk.id && (
                    <div className="mt-4 pt-4 border-t border-gray-800">
                      <div className="grid grid-cols-2 gap-4">
                        {/* Controls */}
                        <div>
                          <h4 className="text-xs font-bold mb-2">Controls ({risk.controls.length})</h4>
                          <div className="space-y-2">
                            {risk.controls.map(control => (
                              <div key={control.id} className="bg-gray-800 rounded p-2 text-xs">
                                <div className="flex items-center justify-between mb-1">
                                  <span className="font-medium">{control.name}</span>
                                  <span className={`px-2 py-1 rounded ${
                                    control.status === 'effective' ? 'bg-green-900/20 text-green-500' :
                                    control.status === 'partial' ? 'bg-yellow-900/20 text-yellow-500' :
                                    'bg-red-900/20 text-red-500'
                                  }`}>
                                    {control.effectiveness}%
                                  </span>
                                </div>
                                <div className="flex justify-between text-gray-500">
                                  <span>{control.type}</span>
                                  <span>{control.status}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Mitigations */}
                        <div>
                          <h4 className="text-xs font-bold mb-2">Mitigations ({risk.mitigations.length})</h4>
                          <div className="space-y-2">
                            {risk.mitigations.map(mitigation => (
                              <div key={mitigation.id} className="bg-gray-800 rounded p-2 text-xs">
                                <div className="font-medium mb-1">{mitigation.action}</div>
                                <div className="flex justify-between text-gray-500">
                                  <span className={`${
                                    mitigation.status === 'completed' ? 'text-green-500' :
                                    mitigation.status === 'in-progress' ? 'text-yellow-500' :
                                    'text-gray-500'
                                  }`}>
                                    {mitigation.status}
                                  </span>
                                  <span>Due: {mitigation.dueDate}</span>
                                </div>
                                <div className="flex justify-between text-gray-500 mt-1">
                                  <span>{mitigation.owner}</span>
                                  <span>${mitigation.cost.toLocaleString()}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>

                      {/* Key Risk Indicators */}
                      <div className="mt-4">
                        <h4 className="text-xs font-bold mb-2">Key Risk Indicators</h4>
                        <div className="flex flex-wrap gap-2">
                          {risk.kris.map((kri, idx) => (
                            <span key={idx} className="px-2 py-1 bg-gray-800 rounded text-xs">
                              {kri}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'trends' && (
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Risk Score Trend</h3>
              <div className="h-48 flex items-end justify-center">
                <LineChart className="w-32 h-32 text-gray-600" />
              </div>
              <p className="text-xs text-gray-500 text-center">Trend visualization coming soon</p>
            </div>

            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold mb-4">Risk Distribution</h3>
              <div className="h-48 flex items-end justify-center">
                <PieChart className="w-32 h-32 text-gray-600" />
              </div>
              <p className="text-xs text-gray-500 text-center">Distribution chart coming soon</p>
            </div>
          </div>
        )}

        {viewMode === 'controls' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-800">
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Control</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Type</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Effectiveness</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Status</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Risk</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Implementation</th>
                </tr>
              </thead>
              <tbody>
                {risks.flatMap(risk => 
                  risk.controls.map(control => (
                    <tr key={control.id} className="border-t border-gray-800 hover:bg-gray-800/30">
                      <td className="px-4 py-3 text-sm">{control.name}</td>
                      <td className="px-4 py-3 text-sm capitalize">{control.type}</td>
                      <td className="px-4 py-3">
                        <div className="flex items-center space-x-2">
                          <div className="w-24 h-2 bg-gray-800 rounded-full overflow-hidden">
                            <div 
                              className={`h-full ${
                                control.effectiveness >= 80 ? 'bg-green-500' :
                                control.effectiveness >= 60 ? 'bg-yellow-500' :
                                'bg-red-500'
                              }`}
                              style={{ width: `${control.effectiveness}%` }}
                            />
                          </div>
                          <span className="text-xs">{control.effectiveness}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 rounded text-xs ${
                          control.status === 'effective' ? 'bg-green-900/20 text-green-500' :
                          control.status === 'partial' ? 'bg-yellow-900/20 text-yellow-500' :
                          'bg-red-900/20 text-red-500'
                        }`}>
                          {control.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm">{risk.title}</td>
                      <td className="px-4 py-3 text-sm">{control.implementationDate}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}