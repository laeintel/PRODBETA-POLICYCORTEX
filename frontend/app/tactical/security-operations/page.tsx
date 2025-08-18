'use client';

import React, { useState, useEffect } from 'react';
import { 
  Shield, Lock, Key, AlertTriangle, UserX, Eye, EyeOff, 
  Fingerprint, Scan, Bug, ShieldCheck, ShieldAlert, ShieldOff,
  Activity, Terminal, Code, FileWarning, UserCheck, Users,
  Globe, Wifi, WifiOff, Server, Database, Cloud, Network,
  TrendingUp, TrendingDown, BarChart3, PieChart, LineChart,
  Clock, Calendar, Timer, RefreshCw, Download, Upload, Search,
  Filter, Settings, MoreVertical, ChevronRight, ExternalLink,
  Zap, AlertCircle, CheckCircle, XCircle, Info, Bell, BellOff,
  Package, Archive, Folder, File, FileText, Hash, GitBranch
} from 'lucide-react';

interface SecurityThreat {
  id: string;
  type: 'malware' | 'phishing' | 'ddos' | 'intrusion' | 'data_breach' | 'insider' | 'vulnerability';
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'active' | 'investigating' | 'mitigated' | 'resolved';
  source: string;
  target: string;
  description: string;
  detectedAt: string;
  indicators: string[];
  affectedAssets: string[];
  mitigationSteps: string[];
  assignee?: string;
  riskScore: number;
  confidence: number;
}

interface SecurityIncident {
  id: string;
  title: string;
  category: string;
  priority: 'P1' | 'P2' | 'P3' | 'P4';
  status: 'open' | 'in_progress' | 'resolved' | 'closed';
  createdAt: string;
  updatedAt: string;
  responseTime: number;
  resolutionTime?: number;
  team: string;
  playbook?: string;
  artifacts: {
    type: string;
    name: string;
    hash?: string;
  }[];
}

interface SecurityControl {
  id: string;
  name: string;
  type: 'preventive' | 'detective' | 'corrective' | 'compensating';
  category: string;
  status: 'active' | 'inactive' | 'degraded' | 'testing';
  effectiveness: number;
  coverage: number;
  lastTested: string;
  complianceFrameworks: string[];
  automationLevel: 'full' | 'partial' | 'manual';
}

interface VulnerabilityItem {
  id: string;
  cve?: string;
  title: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  cvss: number;
  asset: string;
  discovered: string;
  status: 'new' | 'triaged' | 'patching' | 'mitigated' | 'accepted';
  exploitability: 'high' | 'medium' | 'low' | 'none';
  patchAvailable: boolean;
  affectedSystems: number;
}

export default function SecurityOperationsCenter() {
  const [threats, setThreats] = useState<SecurityThreat[]>([]);
  const [incidents, setIncidents] = useState<SecurityIncident[]>([]);
  const [controls, setControls] = useState<SecurityControl[]>([]);
  const [vulnerabilities, setVulnerabilities] = useState<VulnerabilityItem[]>([]);
  const [selectedView, setSelectedView] = useState<'dashboard' | 'threats' | 'incidents' | 'controls' | 'vulnerabilities'>('dashboard');
  const [timeRange, setTimeRange] = useState('24h');
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    // Initialize with security data
    setThreats([
      {
        id: 'THR-001',
        type: 'intrusion',
        severity: 'critical',
        status: 'active',
        source: '185.220.101.45',
        target: 'Production API Gateway',
        description: 'Suspicious login attempts detected from unknown IP address',
        detectedAt: '5 minutes ago',
        indicators: ['Multiple failed login attempts', 'Unusual user agent', 'TOR exit node'],
        affectedAssets: ['api-gateway-01', 'auth-service'],
        mitigationSteps: ['Block IP address', 'Enable MFA', 'Review access logs'],
        riskScore: 92,
        confidence: 88
      },
      {
        id: 'THR-002',
        type: 'vulnerability',
        severity: 'high',
        status: 'investigating',
        source: 'Internal Scan',
        target: 'Database Server',
        description: 'Unpatched SQL injection vulnerability detected',
        detectedAt: '2 hours ago',
        indicators: ['CVE-2024-1234', 'OWASP Top 10'],
        affectedAssets: ['db-prod-01', 'db-prod-02'],
        mitigationSteps: ['Apply security patch', 'Enable WAF rules', 'Review query patterns'],
        assignee: 'security-team@company.com',
        riskScore: 78,
        confidence: 95
      },
      {
        id: 'THR-003',
        type: 'malware',
        severity: 'medium',
        status: 'mitigated',
        source: 'Email Gateway',
        target: 'User Workstation',
        description: 'Trojan detected in email attachment',
        detectedAt: '6 hours ago',
        indicators: ['Suspicious file hash', 'Known malware signature'],
        affectedAssets: ['WS-USER-142'],
        mitigationSteps: ['Quarantine file', 'Run full system scan', 'Update signatures'],
        riskScore: 65,
        confidence: 100
      },
      {
        id: 'THR-004',
        type: 'ddos',
        severity: 'high',
        status: 'active',
        source: 'Multiple IPs',
        target: 'Public Website',
        description: 'Distributed denial of service attack in progress',
        detectedAt: '10 minutes ago',
        indicators: ['Traffic spike 500%', 'Botnet signature', 'UDP flood'],
        affectedAssets: ['www.company.com', 'cdn-edge-servers'],
        mitigationSteps: ['Enable DDoS protection', 'Rate limiting', 'Traffic filtering'],
        riskScore: 85,
        confidence: 92
      },
      {
        id: 'THR-005',
        type: 'data_breach',
        severity: 'critical',
        status: 'resolved',
        source: 'Internal Investigation',
        target: 'Customer Database',
        description: 'Unauthorized data access attempt blocked',
        detectedAt: '12 hours ago',
        indicators: ['Unusual query patterns', 'After-hours access', 'Large data export attempt'],
        affectedAssets: ['customer-db', 'reporting-service'],
        mitigationSteps: ['Revoke access', 'Audit logs review', 'Notify stakeholders'],
        assignee: 'incident-response@company.com',
        riskScore: 95,
        confidence: 87
      }
    ]);

    setIncidents([
      {
        id: 'INC-2024-001',
        title: 'Critical Infrastructure Attack',
        category: 'Security Breach',
        priority: 'P1',
        status: 'in_progress',
        createdAt: '2 hours ago',
        updatedAt: '5 minutes ago',
        responseTime: 5,
        team: 'SOC Team Alpha',
        playbook: 'PB-SEC-001',
        artifacts: [
          { type: 'log', name: 'firewall.log', hash: 'sha256:abcd1234' },
          { type: 'pcap', name: 'network_capture.pcap' },
          { type: 'memory', name: 'memory_dump.bin' }
        ]
      },
      {
        id: 'INC-2024-002',
        title: 'Ransomware Detection',
        category: 'Malware',
        priority: 'P1',
        status: 'resolved',
        createdAt: '24 hours ago',
        updatedAt: '12 hours ago',
        responseTime: 3,
        resolutionTime: 180,
        team: 'Incident Response',
        playbook: 'PB-MAL-003',
        artifacts: [
          { type: 'malware', name: 'sample.exe', hash: 'sha256:efgh5678' },
          { type: 'report', name: 'analysis_report.pdf' }
        ]
      },
      {
        id: 'INC-2024-003',
        title: 'Phishing Campaign',
        category: 'Social Engineering',
        priority: 'P2',
        status: 'open',
        createdAt: '30 minutes ago',
        updatedAt: '30 minutes ago',
        responseTime: 0,
        team: 'Email Security',
        artifacts: [
          { type: 'email', name: 'phishing_email.eml' },
          { type: 'url', name: 'malicious_urls.txt' }
        ]
      }
    ]);

    setControls([
      {
        id: 'CTRL-001',
        name: 'Web Application Firewall',
        type: 'preventive',
        category: 'Network Security',
        status: 'active',
        effectiveness: 94,
        coverage: 88,
        lastTested: '2 days ago',
        complianceFrameworks: ['PCI-DSS', 'ISO 27001', 'SOC 2'],
        automationLevel: 'full'
      },
      {
        id: 'CTRL-002',
        name: 'Intrusion Detection System',
        type: 'detective',
        category: 'Network Security',
        status: 'active',
        effectiveness: 89,
        coverage: 92,
        lastTested: '1 week ago',
        complianceFrameworks: ['NIST', 'ISO 27001'],
        automationLevel: 'full'
      },
      {
        id: 'CTRL-003',
        name: 'Multi-Factor Authentication',
        type: 'preventive',
        category: 'Access Control',
        status: 'active',
        effectiveness: 98,
        coverage: 75,
        lastTested: '1 month ago',
        complianceFrameworks: ['NIST', 'CIS', 'ISO 27001'],
        automationLevel: 'partial'
      },
      {
        id: 'CTRL-004',
        name: 'Data Loss Prevention',
        type: 'preventive',
        category: 'Data Security',
        status: 'degraded',
        effectiveness: 72,
        coverage: 65,
        lastTested: '3 days ago',
        complianceFrameworks: ['GDPR', 'CCPA', 'HIPAA'],
        automationLevel: 'partial'
      },
      {
        id: 'CTRL-005',
        name: 'Security Information Event Management',
        type: 'detective',
        category: 'Monitoring',
        status: 'active',
        effectiveness: 91,
        coverage: 95,
        lastTested: '1 day ago',
        complianceFrameworks: ['SOC 2', 'ISO 27001', 'PCI-DSS'],
        automationLevel: 'full'
      },
      {
        id: 'CTRL-006',
        name: 'Endpoint Detection & Response',
        type: 'detective',
        category: 'Endpoint Security',
        status: 'active',
        effectiveness: 87,
        coverage: 82,
        lastTested: '5 days ago',
        complianceFrameworks: ['NIST', 'CIS'],
        automationLevel: 'full'
      }
    ]);

    setVulnerabilities([
      {
        id: 'VULN-001',
        cve: 'CVE-2024-1234',
        title: 'SQL Injection in User API',
        severity: 'critical',
        cvss: 9.8,
        asset: 'api.company.com',
        discovered: '2 days ago',
        status: 'patching',
        exploitability: 'high',
        patchAvailable: true,
        affectedSystems: 3
      },
      {
        id: 'VULN-002',
        cve: 'CVE-2024-5678',
        title: 'Outdated SSL/TLS Configuration',
        severity: 'high',
        cvss: 7.5,
        asset: 'mail.company.com',
        discovered: '1 week ago',
        status: 'triaged',
        exploitability: 'medium',
        patchAvailable: true,
        affectedSystems: 5
      },
      {
        id: 'VULN-003',
        title: 'Weak Password Policy',
        severity: 'medium',
        cvss: 5.3,
        asset: 'Active Directory',
        discovered: '2 weeks ago',
        status: 'mitigated',
        exploitability: 'low',
        patchAvailable: false,
        affectedSystems: 150
      },
      {
        id: 'VULN-004',
        cve: 'CVE-2024-9012',
        title: 'Remote Code Execution in Web Server',
        severity: 'critical',
        cvss: 10.0,
        asset: 'web-prod-01',
        discovered: '6 hours ago',
        status: 'new',
        exploitability: 'high',
        patchAvailable: true,
        affectedSystems: 2
      },
      {
        id: 'VULN-005',
        title: 'Missing Security Headers',
        severity: 'low',
        cvss: 3.1,
        asset: 'blog.company.com',
        discovered: '1 month ago',
        status: 'accepted',
        exploitability: 'none',
        patchAvailable: false,
        affectedSystems: 1
      }
    ]);
  }, []);

  const getThreatIcon = (type: string) => {
    switch(type) {
      case 'malware': return <Bug className="w-4 h-4" />;
      case 'phishing': return <UserX className="w-4 h-4" />;
      case 'ddos': return <Wifi className="w-4 h-4" />;
      case 'intrusion': return <ShieldAlert className="w-4 h-4" />;
      case 'data_breach': return <Database className="w-4 h-4" />;
      case 'insider': return <UserCheck className="w-4 h-4" />;
      case 'vulnerability': return <FileWarning className="w-4 h-4" />;
      default: return <AlertTriangle className="w-4 h-4" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch(severity) {
      case 'critical': return 'text-red-600 bg-red-900/20';
      case 'high': return 'text-orange-500 bg-orange-900/20';
      case 'medium': return 'text-yellow-500 bg-yellow-900/20';
      case 'low': return 'text-blue-500 bg-blue-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': case 'open': case 'new': return 'text-red-500 bg-red-900/20';
      case 'investigating': case 'in_progress': case 'triaged': return 'text-yellow-500 bg-yellow-900/20';
      case 'mitigated': case 'patching': return 'text-blue-500 bg-blue-900/20';
      case 'resolved': case 'closed': case 'accepted': return 'text-green-500 bg-green-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch(priority) {
      case 'P1': return 'text-red-600 bg-red-900/20';
      case 'P2': return 'text-orange-500 bg-orange-900/20';
      case 'P3': return 'text-yellow-500 bg-yellow-900/20';
      case 'P4': return 'text-blue-500 bg-blue-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const securityMetrics = {
    threatScore: 78,
    activeThreats: threats.filter(t => t.status === 'active').length,
    openIncidents: incidents.filter(i => i.status === 'open' || i.status === 'in_progress').length,
    criticalVulnerabilities: vulnerabilities.filter(v => v.severity === 'critical').length,
    controlEffectiveness: Math.round(controls.reduce((acc, c) => acc + c.effectiveness, 0) / controls.length),
    securityPosture: 'Medium Risk',
    meanTimeToDetect: '5 min',
    meanTimeToRespond: '12 min',
    patchCompliance: 82
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold flex items-center space-x-2">
              <Shield className="w-6 h-6 text-blue-500" />
              <span>Security Operations Center</span>
            </h1>
            <p className="text-sm text-gray-400 mt-1">Real-time threat monitoring and incident response</p>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* Time Range Selector */}
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>

            {/* Auto Refresh */}
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                autoRefresh ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              <span>{autoRefresh ? 'Live' : 'Paused'}</span>
            </button>

            <button className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4" />
              <span>Incident Response</span>
            </button>
          </div>
        </div>
      </header>

      {/* Security Metrics Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-8 gap-4">
          <div className="text-center">
            <div className="text-xs text-gray-500">Threat Score</div>
            <div className={`text-2xl font-bold ${
              securityMetrics.threatScore > 80 ? 'text-red-500' :
              securityMetrics.threatScore > 60 ? 'text-yellow-500' :
              'text-green-500'
            }`}>{securityMetrics.threatScore}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Active Threats</div>
            <div className="text-2xl font-bold text-red-500">{securityMetrics.activeThreats}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Open Incidents</div>
            <div className="text-2xl font-bold text-orange-500">{securityMetrics.openIncidents}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Critical Vulns</div>
            <div className="text-2xl font-bold text-red-600">{securityMetrics.criticalVulnerabilities}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Control Effect.</div>
            <div className="text-2xl font-bold text-blue-500">{securityMetrics.controlEffectiveness}%</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">MTTD</div>
            <div className="text-xl font-bold">{securityMetrics.meanTimeToDetect}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">MTTR</div>
            <div className="text-xl font-bold">{securityMetrics.meanTimeToRespond}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Patch Compliance</div>
            <div className="text-2xl font-bold text-green-500">{securityMetrics.patchCompliance}%</div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-gray-900/30 border-b border-gray-800 px-6">
        <div className="flex space-x-6">
          {['dashboard', 'threats', 'incidents', 'controls', 'vulnerabilities'].map(view => (
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
            {/* Threat Map */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h2 className="text-sm font-bold mb-4">Global Threat Map</h2>
              <div className="grid grid-cols-4 gap-4">
                <div className="col-span-3 bg-gray-800 rounded-lg p-8 text-center">
                  <Globe className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <p className="text-sm text-gray-500">Interactive threat visualization</p>
                  <div className="grid grid-cols-3 gap-4 mt-6">
                    <div className="bg-gray-900 rounded p-3">
                      <div className="text-2xl font-bold text-red-500">142</div>
                      <div className="text-xs text-gray-500">Blocked IPs</div>
                    </div>
                    <div className="bg-gray-900 rounded p-3">
                      <div className="text-2xl font-bold text-yellow-500">23</div>
                      <div className="text-xs text-gray-500">Countries</div>
                    </div>
                    <div className="bg-gray-900 rounded p-3">
                      <div className="text-2xl font-bold text-blue-500">5.2M</div>
                      <div className="text-xs text-gray-500">Requests Analyzed</div>
                    </div>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="bg-gray-800 rounded p-3">
                    <div className="text-xs text-gray-500 mb-1">Top Attack Type</div>
                    <div className="font-bold">DDoS</div>
                    <div className="text-xs text-red-500">45% of attacks</div>
                  </div>
                  <div className="bg-gray-800 rounded p-3">
                    <div className="text-xs text-gray-500 mb-1">Top Target</div>
                    <div className="font-bold">API Gateway</div>
                    <div className="text-xs text-orange-500">28 attempts</div>
                  </div>
                  <div className="bg-gray-800 rounded p-3">
                    <div className="text-xs text-gray-500 mb-1">Risk Level</div>
                    <div className="font-bold text-yellow-500">ELEVATED</div>
                    <div className="text-xs">Score: 78/100</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Threats */}
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold mb-3">Recent Threats</h3>
                <div className="space-y-2">
                  {threats.slice(0, 3).map(threat => (
                    <div key={threat.id} className="flex items-center justify-between p-2 bg-gray-800 rounded">
                      <div className="flex items-center space-x-2">
                        <div className={`p-1 rounded ${getSeverityColor(threat.severity)}`}>
                          {getThreatIcon(threat.type)}
                        </div>
                        <div>
                          <div className="text-xs font-medium">{threat.target}</div>
                          <div className="text-xs text-gray-500">{threat.detectedAt}</div>
                        </div>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs ${getSeverityColor(threat.severity)}`}>
                        {threat.severity}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold mb-3">Security Controls</h3>
                <div className="space-y-2">
                  {controls.slice(0, 3).map(control => (
                    <div key={control.id} className="flex items-center justify-between p-2 bg-gray-800 rounded">
                      <div className="flex items-center space-x-2">
                        <ShieldCheck className={`w-4 h-4 ${
                          control.status === 'active' ? 'text-green-500' :
                          control.status === 'degraded' ? 'text-yellow-500' :
                          'text-gray-500'
                        }`} />
                        <div>
                          <div className="text-xs font-medium">{control.name}</div>
                          <div className="text-xs text-gray-500">{control.category}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs font-bold">{control.effectiveness}%</div>
                        <div className="text-xs text-gray-500">effect.</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedView === 'threats' && (
          <div className="space-y-4">
            {/* Threat Filters */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <select
                  value={filterSeverity}
                  onChange={(e) => setFilterSeverity(e.target.value)}
                  className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                >
                  <option value="all">All Severities</option>
                  <option value="critical">Critical</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
                <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
                  <Filter className="w-4 h-4" />
                  <span>More Filters</span>
                </button>
              </div>
              <div className="flex items-center space-x-2">
                <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm">
                  Export
                </button>
              </div>
            </div>

            {/* Threats List */}
            <div className="space-y-3">
              {threats.map(threat => (
                <div key={threat.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start space-x-3">
                      <div className={`p-2 rounded ${getSeverityColor(threat.severity)}`}>
                        {getThreatIcon(threat.type)}
                      </div>
                      <div>
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className="text-sm font-bold">{threat.target}</h3>
                          <span className={`px-2 py-1 rounded text-xs ${getStatusColor(threat.status)}`}>
                            {threat.status}
                          </span>
                        </div>
                        <p className="text-xs text-gray-400 mb-2">{threat.description}</p>
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <span>Source: {threat.source}</span>
                          <span>Detected: {threat.detectedAt}</span>
                          <span>Risk Score: {threat.riskScore}</span>
                          <span>Confidence: {threat.confidence}%</span>
                        </div>
                      </div>
                    </div>
                    <button className="p-1 hover:bg-gray-800 rounded">
                      <MoreVertical className="w-4 h-4 text-gray-500" />
                    </button>
                  </div>

                  <div className="grid grid-cols-3 gap-4 text-xs">
                    <div>
                      <span className="text-gray-500">Indicators:</span>
                      <div className="mt-1 space-y-1">
                        {threat.indicators.map((indicator, idx) => (
                          <div key={idx} className="flex items-center space-x-1">
                            <ChevronRight className="w-3 h-3 text-gray-600" />
                            <span>{indicator}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-500">Affected Assets:</span>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {threat.affectedAssets.map(asset => (
                          <span key={asset} className="px-2 py-1 bg-gray-800 rounded">
                            {asset}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-500">Mitigation:</span>
                      <div className="mt-1 space-y-1">
                        {threat.mitigationSteps.slice(0, 2).map((step, idx) => (
                          <div key={idx} className="flex items-center space-x-1">
                            <CheckCircle className="w-3 h-3 text-green-500" />
                            <span>{step}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 pt-3 border-t border-gray-800 flex justify-between">
                    <div className="flex space-x-2">
                      <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                        Investigate
                      </button>
                      <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                        Create Incident
                      </button>
                    </div>
                    {threat.assignee && (
                      <span className="text-xs text-gray-500">Assigned to: {threat.assignee}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedView === 'incidents' && (
          <div className="space-y-4">
            {incidents.map(incident => (
              <div key={incident.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center space-x-3 mb-1">
                      <h3 className="text-sm font-bold">{incident.title}</h3>
                      <span className={`px-2 py-1 rounded text-xs ${getPriorityColor(incident.priority)}`}>
                        {incident.priority}
                      </span>
                      <span className={`px-2 py-1 rounded text-xs ${getStatusColor(incident.status)}`}>
                        {incident.status}
                      </span>
                    </div>
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span>ID: {incident.id}</span>
                      <span>Category: {incident.category}</span>
                      <span>Team: {incident.team}</span>
                      {incident.playbook && <span>Playbook: {incident.playbook}</span>}
                    </div>
                  </div>
                  <button className="p-1 hover:bg-gray-800 rounded">
                    <MoreVertical className="w-4 h-4 text-gray-500" />
                  </button>
                </div>

                <div className="grid grid-cols-4 gap-4 mb-3 text-xs">
                  <div>
                    <span className="text-gray-500">Created</span>
                    <p>{incident.createdAt}</p>
                  </div>
                  <div>
                    <span className="text-gray-500">Updated</span>
                    <p>{incident.updatedAt}</p>
                  </div>
                  <div>
                    <span className="text-gray-500">Response Time</span>
                    <p className="text-green-500">{incident.responseTime} min</p>
                  </div>
                  <div>
                    <span className="text-gray-500">Resolution Time</span>
                    <p className={incident.resolutionTime ? 'text-blue-500' : 'text-gray-500'}>
                      {incident.resolutionTime ? `${incident.resolutionTime} min` : 'In Progress'}
                    </p>
                  </div>
                </div>

                {incident.artifacts.length > 0 && (
                  <div className="flex items-center space-x-2 text-xs">
                    <span className="text-gray-500">Artifacts:</span>
                    {incident.artifacts.map((artifact, idx) => (
                      <span key={idx} className="px-2 py-1 bg-gray-800 rounded flex items-center space-x-1">
                        <File className="w-3 h-3" />
                        <span>{artifact.name}</span>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {selectedView === 'controls' && (
          <div className="grid grid-cols-2 gap-4">
            {controls.map(control => (
              <div key={control.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="text-sm font-bold">{control.name}</h3>
                    <p className="text-xs text-gray-500">{control.category} • {control.type}</p>
                  </div>
                  <div className={`px-2 py-1 rounded text-xs ${
                    control.status === 'active' ? 'bg-green-900/20 text-green-500' :
                    control.status === 'degraded' ? 'bg-yellow-900/20 text-yellow-500' :
                    'bg-gray-900/20 text-gray-500'
                  }`}>
                    {control.status}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 mb-3">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-500">Effectiveness</span>
                      <span className="font-bold">{control.effectiveness}%</span>
                    </div>
                    <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${
                          control.effectiveness >= 90 ? 'bg-green-500' :
                          control.effectiveness >= 70 ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}
                        style={{ width: `${control.effectiveness}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-500">Coverage</span>
                      <span className="font-bold">{control.coverage}%</span>
                    </div>
                    <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-blue-500"
                        style={{ width: `${control.coverage}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Last tested: {control.lastTested}</span>
                  <span className={`px-2 py-1 rounded ${
                    control.automationLevel === 'full' ? 'bg-green-900/20 text-green-500' :
                    control.automationLevel === 'partial' ? 'bg-yellow-900/20 text-yellow-500' :
                    'bg-gray-900/20 text-gray-500'
                  }`}>
                    {control.automationLevel} automation
                  </span>
                </div>

                <div className="mt-3 pt-3 border-t border-gray-800">
                  <div className="flex flex-wrap gap-1">
                    {control.complianceFrameworks.map(framework => (
                      <span key={framework} className="px-2 py-1 bg-gray-800 rounded text-xs">
                        {framework}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedView === 'vulnerabilities' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-800">
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Vulnerability</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Asset</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Severity</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">CVSS</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Status</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Exploit</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Patch</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Systems</th>
                  <th className="px-4 py-3 text-left text-xs uppercase text-gray-500">Actions</th>
                </tr>
              </thead>
              <tbody>
                {vulnerabilities.map(vuln => (
                  <tr key={vuln.id} className="border-t border-gray-800 hover:bg-gray-800/30">
                    <td className="px-4 py-3">
                      <div>
                        <div className="text-sm font-medium">{vuln.title}</div>
                        {vuln.cve && <div className="text-xs text-gray-500">{vuln.cve}</div>}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm">{vuln.asset}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs ${getSeverityColor(vuln.severity)}`}>
                        {vuln.severity}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`text-sm font-bold ${
                        vuln.cvss >= 9 ? 'text-red-500' :
                        vuln.cvss >= 7 ? 'text-orange-500' :
                        vuln.cvss >= 4 ? 'text-yellow-500' :
                        'text-blue-500'
                      }`}>
                        {vuln.cvss}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs ${getStatusColor(vuln.status)}`}>
                        {vuln.status}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`text-xs ${
                        vuln.exploitability === 'high' ? 'text-red-500' :
                        vuln.exploitability === 'medium' ? 'text-yellow-500' :
                        vuln.exploitability === 'low' ? 'text-blue-500' :
                        'text-gray-500'
                      }`}>
                        {vuln.exploitability}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      {vuln.patchAvailable ? (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      ) : (
                        <XCircle className="w-4 h-4 text-gray-500" />
                      )}
                    </td>
                    <td className="px-4 py-3 text-sm">{vuln.affectedSystems}</td>
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
      </div>
    </>
  );
}