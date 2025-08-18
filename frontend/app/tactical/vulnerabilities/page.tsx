'use client';

import React, { useState, useEffect } from 'react';
import { 
  ScanLine, AlertTriangle, Shield, Target, Server, Database, Globe,
  CheckCircle, XCircle, Clock, Calendar, TrendingUp, BarChart, Activity,
  Search, Filter, RefreshCw, Download, Settings, Info, Eye, Play,
  Pause, RotateCcw, FileText, Hash, Network, Bug, Zap, Package
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface Vulnerability {
  id: string;
  cveId?: string;
  title: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  cvssScore: number;
  cvssVector: string;
  category: 'network' | 'web_app' | 'system' | 'database' | 'container' | 'cloud';
  affectedAssets: {
    id: string;
    name: string;
    type: string;
    ip?: string;
  }[];
  discoveredDate: string;
  lastSeen: string;
  status: 'open' | 'investigating' | 'patched' | 'mitigated' | 'false_positive' | 'accepted';
  exploitAvailable: boolean;
  patchAvailable: boolean;
  patchComplexity: 'low' | 'medium' | 'high';
  businessImpact: 'low' | 'medium' | 'high' | 'critical';
  solution: string;
  references: string[];
  tags: string[];
}

interface ScanResult {
  id: string;
  scanName: string;
  target: string;
  scanType: 'full' | 'quick' | 'targeted' | 'compliance';
  status: 'running' | 'completed' | 'failed' | 'queued';
  startTime: string;
  endTime?: string;
  duration?: number;
  vulnerabilitiesFound: {
    critical: number;
    high: number;
    medium: number;
    low: number;
    info: number;
  };
  progress: number;
  scanner: string;
}

interface VulnerabilityMetrics {
  totalVulnerabilities: number;
  criticalVulnerabilities: number;
  patchedLastWeek: number;
  avgTimeToRemediate: number;
  exposureScore: number;
  complianceStatus: number;
}

export default function VulnerabilityScanner() {
  const [vulnerabilities, setVulnerabilities] = useState<Vulnerability[]>([]);
  const [scanResults, setScanResults] = useState<ScanResult[]>([]);
  const [metrics, setMetrics] = useState<VulnerabilityMetrics | null>(null);
  const [selectedVuln, setSelectedVuln] = useState<Vulnerability | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSeverity, setSelectedSeverity] = useState('all');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [viewMode, setViewMode] = useState<'vulnerabilities' | 'scans' | 'reports' | 'assets'>('vulnerabilities');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [scanRunning, setScanRunning] = useState(false);

  useEffect(() => {
    // Initialize with mock vulnerability data
    setVulnerabilities([
      {
        id: 'VULN-001',
        cveId: 'CVE-2024-0001',
        title: 'Remote Code Execution in Apache Struts',
        description: 'A critical vulnerability allows remote attackers to execute arbitrary code via specially crafted HTTP requests',
        severity: 'critical',
        cvssScore: 9.8,
        cvssVector: 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H',
        category: 'web_app',
        affectedAssets: [
          { id: 'asset-001', name: 'Web Server 01', type: 'Linux Server', ip: '192.168.1.10' },
          { id: 'asset-002', name: 'Web Server 02', type: 'Linux Server', ip: '192.168.1.11' }
        ],
        discoveredDate: '2024-01-20T10:30:00Z',
        lastSeen: '2024-01-22T14:30:00Z',
        status: 'open',
        exploitAvailable: true,
        patchAvailable: true,
        patchComplexity: 'medium',
        businessImpact: 'critical',
        solution: 'Update Apache Struts to version 2.5.32 or later. Apply security patches immediately.',
        references: [
          'https://nvd.nist.gov/vuln/detail/CVE-2024-0001',
          'https://struts.apache.org/announce.html'
        ],
        tags: ['rce', 'web-application', 'apache', 'urgent']
      },
      {
        id: 'VULN-002',
        cveId: 'CVE-2023-4567',
        title: 'SQL Injection in Custom Application',
        description: 'SQL injection vulnerability in user authentication module allows attackers to bypass authentication',
        severity: 'high',
        cvssScore: 8.1,
        cvssVector: 'CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:N',
        category: 'web_app',
        affectedAssets: [
          { id: 'asset-003', name: 'Database Server', type: 'MySQL Server', ip: '192.168.1.20' }
        ],
        discoveredDate: '2024-01-18T08:15:00Z',
        lastSeen: '2024-01-22T14:30:00Z',
        status: 'investigating',
        exploitAvailable: false,
        patchAvailable: false,
        patchComplexity: 'high',
        businessImpact: 'high',
        solution: 'Implement parameterized queries and input validation. Review all database interactions.',
        references: [
          'https://owasp.org/www-community/attacks/SQL_Injection'
        ],
        tags: ['sql-injection', 'authentication', 'database']
      },
      {
        id: 'VULN-003',
        cveId: 'CVE-2023-8901',
        title: 'Outdated SSL/TLS Configuration',
        description: 'Web server supports deprecated TLS 1.0 and weak cipher suites, allowing man-in-the-middle attacks',
        severity: 'medium',
        cvssScore: 5.3,
        cvssVector: 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N',
        category: 'network',
        affectedAssets: [
          { id: 'asset-001', name: 'Web Server 01', type: 'Linux Server', ip: '192.168.1.10' },
          { id: 'asset-004', name: 'Load Balancer', type: 'F5 Device', ip: '192.168.1.30' }
        ],
        discoveredDate: '2024-01-15T16:45:00Z',
        lastSeen: '2024-01-22T14:30:00Z',
        status: 'patched',
        exploitAvailable: false,
        patchAvailable: true,
        patchComplexity: 'low',
        businessImpact: 'medium',
        solution: 'Update SSL/TLS configuration to disable TLS 1.0 and enable only strong cipher suites.',
        references: [
          'https://tools.ietf.org/html/rfc7525',
          'https://ssl-config.mozilla.org/'
        ],
        tags: ['ssl', 'tls', 'encryption', 'configuration']
      },
      {
        id: 'VULN-004',
        title: 'Unpatched Operating System',
        description: 'Multiple critical security updates missing on Windows servers',
        severity: 'high',
        cvssScore: 7.8,
        cvssVector: 'CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H',
        category: 'system',
        affectedAssets: [
          { id: 'asset-005', name: 'Windows Server 01', type: 'Windows Server', ip: '192.168.1.50' },
          { id: 'asset-006', name: 'Windows Server 02', type: 'Windows Server', ip: '192.168.1.51' }
        ],
        discoveredDate: '2024-01-10T12:00:00Z',
        lastSeen: '2024-01-22T14:30:00Z',
        status: 'mitigated',
        exploitAvailable: true,
        patchAvailable: true,
        patchComplexity: 'low',
        businessImpact: 'high',
        solution: 'Install latest Windows security updates and enable automatic updates.',
        references: [
          'https://support.microsoft.com/en-us/windows/windows-update'
        ],
        tags: ['windows', 'patches', 'operating-system']
      },
      {
        id: 'VULN-005',
        title: 'Docker Container Vulnerability',
        description: 'Base image contains known vulnerabilities in system libraries',
        severity: 'medium',
        cvssScore: 6.2,
        cvssVector: 'CVSS:3.1/AV:L/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H',
        category: 'container',
        affectedAssets: [
          { id: 'asset-007', name: 'Docker Host 01', type: 'Container Host', ip: '192.168.1.60' }
        ],
        discoveredDate: '2024-01-22T09:00:00Z',
        lastSeen: '2024-01-22T14:30:00Z',
        status: 'open',
        exploitAvailable: false,
        patchAvailable: true,
        patchComplexity: 'medium',
        businessImpact: 'low',
        solution: 'Update base Docker image to latest version and rebuild containers.',
        references: [
          'https://docs.docker.com/engine/security/security/'
        ],
        tags: ['docker', 'container', 'base-image']
      }
    ]);

    setScanResults([
      {
        id: 'SCAN-001',
        scanName: 'Weekly Infrastructure Scan',
        target: '192.168.1.0/24',
        scanType: 'full',
        status: 'completed',
        startTime: '2024-01-22T02:00:00Z',
        endTime: '2024-01-22T04:30:00Z',
        duration: 150,
        vulnerabilitiesFound: {
          critical: 1,
          high: 2,
          medium: 8,
          low: 15,
          info: 23
        },
        progress: 100,
        scanner: 'Nessus'
      },
      {
        id: 'SCAN-002',
        scanName: 'Web Application Security Test',
        target: 'https://app.company.com',
        scanType: 'targeted',
        status: 'running',
        startTime: '2024-01-22T14:00:00Z',
        vulnerabilitiesFound: {
          critical: 0,
          high: 1,
          medium: 3,
          low: 7,
          info: 12
        },
        progress: 65,
        scanner: 'OWASP ZAP'
      },
      {
        id: 'SCAN-003',
        scanName: 'Compliance Audit Scan',
        target: 'Production Environment',
        scanType: 'compliance',
        status: 'queued',
        startTime: '2024-01-22T18:00:00Z',
        vulnerabilitiesFound: {
          critical: 0,
          high: 0,
          medium: 0,
          low: 0,
          info: 0
        },
        progress: 0,
        scanner: 'Rapid7'
      }
    ]);

    setMetrics({
      totalVulnerabilities: 247,
      criticalVulnerabilities: 3,
      patchedLastWeek: 18,
      avgTimeToRemediate: 12.5,
      exposureScore: 7.2,
      complianceStatus: 87
    });

    // Simulate real-time updates
    if (autoRefresh) {
      const interval = setInterval(() => {
        setScanResults(prev => 
          prev.map(scan => {
            if (scan.status === 'running') {
              const newProgress = Math.min(100, scan.progress + Math.random() * 5);
              return {
                ...scan,
                progress: newProgress,
                status: newProgress >= 100 ? 'completed' : 'running'
              };
            }
            return scan;
          })
        );
      }, 3000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getSeverityColor = (severity: string) => {
    switch(severity) {
      case 'critical': return 'text-red-500 bg-red-900/20 border-red-800';
      case 'high': return 'text-orange-500 bg-orange-900/20 border-orange-800';
      case 'medium': return 'text-yellow-500 bg-yellow-900/20 border-yellow-800';
      case 'low': return 'text-blue-500 bg-blue-900/20 border-blue-800';
      case 'info': return 'text-gray-500 bg-gray-900/20 border-gray-800';
      default: return 'text-gray-500 bg-gray-900/20 border-gray-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'open': return 'text-red-500 bg-red-900/20';
      case 'investigating': return 'text-yellow-500 bg-yellow-900/20';
      case 'patched': return 'text-green-500 bg-green-900/20';
      case 'mitigated': return 'text-blue-500 bg-blue-900/20';
      case 'false_positive': return 'text-gray-500 bg-gray-900/20';
      case 'accepted': return 'text-purple-500 bg-purple-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch(category) {
      case 'network': return <Network className="w-4 h-4 text-blue-500" />;
      case 'web_app': return <Globe className="w-4 h-4 text-green-500" />;
      case 'system': return <Server className="w-4 h-4 text-purple-500" />;
      case 'database': return <Database className="w-4 h-4 text-cyan-500" />;
      case 'container': return <Package className="w-4 h-4 text-orange-500" />;
      case 'cloud': return <Target className="w-4 h-4 text-pink-500" />;
      default: return <Bug className="w-4 h-4 text-gray-500" />;
    }
  };

  const filteredVulnerabilities = vulnerabilities.filter(vuln => {
    if (searchQuery && !vuln.title.toLowerCase().includes(searchQuery.toLowerCase()) && 
        !vuln.description.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !(vuln.cveId && vuln.cveId.toLowerCase().includes(searchQuery.toLowerCase()))) return false;
    if (selectedSeverity !== 'all' && vuln.severity !== selectedSeverity) return false;
    if (selectedCategory !== 'all' && vuln.category !== selectedCategory) return false;
    if (selectedStatus !== 'all' && vuln.status !== selectedStatus) return false;
    return true;
  });

  const criticalVulns = vulnerabilities.filter(v => v.severity === 'critical').length;
  const openVulns = vulnerabilities.filter(v => v.status === 'open').length;
  const runningScans = scanResults.filter(s => s.status === 'running').length;

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Vulnerability Scanner</h1>
            <p className="text-sm text-gray-400 mt-1">Comprehensive vulnerability assessment and management</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setScanRunning(!scanRunning)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                scanRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {scanRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              <span>{scanRunning ? 'Stop Scan' : 'Start Scan'}</span>
            </button>
            
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                autoRefresh ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              <span>{autoRefresh ? 'Live' : 'Paused'}</span>
            </button>
            
            <button
              onClick={() => setViewMode(
                viewMode === 'vulnerabilities' ? 'scans' : 
                viewMode === 'scans' ? 'reports' : 
                viewMode === 'reports' ? 'assets' : 'vulnerabilities'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
            >
              {viewMode === 'vulnerabilities' ? 'Scans' : 
               viewMode === 'scans' ? 'Reports' : 
               viewMode === 'reports' ? 'Assets' : 'Vulnerabilities'}
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export Report</span>
            </button>
          </div>
        </div>
      </header>

      {/* Metrics Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Critical Vulns</p>
              <p className="text-xl font-bold text-red-500">{criticalVulns}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XCircle className="w-5 h-5 text-orange-500" />
            <div>
              <p className="text-xs text-gray-400">Open Issues</p>
              <p className="text-xl font-bold text-orange-500">{openVulns}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <ScanLine className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Running Scans</p>
              <p className="text-xl font-bold">{runningScans}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Patched/Week</p>
              <p className="text-xl font-bold">{metrics?.patchedLastWeek}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Clock className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Avg Remediation</p>
              <p className="text-xl font-bold">{metrics?.avgTimeToRemediate}d</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Shield className="w-5 h-5 text-cyan-500" />
            <div>
              <p className="text-xs text-gray-400">Exposure Score</p>
              <p className="text-xl font-bold">{metrics?.exposureScore}/10</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Filters */}
        <div className="flex items-center space-x-3 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search vulnerabilities by title, CVE ID, or description..."
              className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
            />
          </div>
          
          <select
            value={selectedSeverity}
            onChange={(e) => setSelectedSeverity(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
            <option value="info">Info</option>
          </select>
          
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Categories</option>
            <option value="network">Network</option>
            <option value="web_app">Web Application</option>
            <option value="system">System</option>
            <option value="database">Database</option>
            <option value="container">Container</option>
            <option value="cloud">Cloud</option>
          </select>
          
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Status</option>
            <option value="open">Open</option>
            <option value="investigating">Investigating</option>
            <option value="patched">Patched</option>
            <option value="mitigated">Mitigated</option>
            <option value="false_positive">False Positive</option>
            <option value="accepted">Accepted</option>
          </select>
        </div>

        {viewMode === 'vulnerabilities' && (
          <div className="space-y-4">
            {filteredVulnerabilities.map(vuln => (
              <div 
                key={vuln.id} 
                className={`bg-gray-900 border border-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-800/50 ${
                  selectedVuln?.id === vuln.id ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => setSelectedVuln(selectedVuln?.id === vuln.id ? null : vuln)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    {getCategoryIcon(vuln.category)}
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h3 className="text-sm font-bold">{vuln.title}</h3>
                        {vuln.cveId && (
                          <span className="px-2 py-1 bg-gray-800 text-gray-400 rounded text-xs font-mono">
                            {vuln.cveId}
                          </span>
                        )}
                        <span className={`px-2 py-1 text-xs rounded border ${getSeverityColor(vuln.severity)}`}>
                          {vuln.severity.toUpperCase()}
                        </span>
                        <span className={`px-2 py-1 text-xs rounded ${getStatusColor(vuln.status)}`}>
                          {vuln.status.replace('_', ' ').toUpperCase()}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400 mb-2">{vuln.description}</p>
                      <div className="grid grid-cols-4 gap-4 text-xs text-gray-500">
                        <div>
                          <span className="text-gray-400">CVSS Score: </span>
                          <span className={`font-bold ${vuln.cvssScore >= 9 ? 'text-red-500' : vuln.cvssScore >= 7 ? 'text-orange-500' : vuln.cvssScore >= 4 ? 'text-yellow-500' : 'text-blue-500'}`}>
                            {vuln.cvssScore}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-400">Assets: </span>
                          <span className="font-bold">{vuln.affectedAssets.length}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Exploit: </span>
                          <span className={`font-bold ${vuln.exploitAvailable ? 'text-red-500' : 'text-green-500'}`}>
                            {vuln.exploitAvailable ? 'Available' : 'None'}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-400">Patch: </span>
                          <span className={`font-bold ${vuln.patchAvailable ? 'text-green-500' : 'text-red-500'}`}>
                            {vuln.patchAvailable ? 'Available' : 'None'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <p className="text-xs text-gray-500">Discovered: {new Date(vuln.discoveredDate).toLocaleDateString()}</p>
                    <div className="flex items-center space-x-2 mt-2">
                      {vuln.exploitAvailable && (
                        <span className="px-2 py-1 bg-red-900/20 text-red-500 rounded text-xs">
                          EXPLOITABLE
                        </span>
                      )}
                      {vuln.businessImpact === 'critical' && (
                        <span className="px-2 py-1 bg-purple-900/20 text-purple-500 rounded text-xs">
                          BUSINESS CRITICAL
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Expanded Details */}
                {selectedVuln?.id === vuln.id && (
                  <div className="mt-4 border-t border-gray-700 pt-4">
                    <div className="grid grid-cols-2 gap-6">
                      {/* Affected Assets */}
                      <div>
                        <h4 className="text-sm font-bold mb-3">Affected Assets</h4>
                        <div className="space-y-2">
                          {vuln.affectedAssets.map(asset => (
                            <div key={asset.id} className="flex items-center space-x-3 p-2 bg-gray-800 rounded">
                              <Server className="w-4 h-4 text-gray-500" />
                              <div className="flex-1">
                                <p className="text-sm font-bold">{asset.name}</p>
                                <p className="text-xs text-gray-500">{asset.type} {asset.ip && `• ${asset.ip}`}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Solution and References */}
                      <div>
                        <h4 className="text-sm font-bold mb-3">Solution</h4>
                        <p className="text-sm text-gray-400 mb-4">{vuln.solution}</p>
                        
                        {vuln.references.length > 0 && (
                          <>
                            <h4 className="text-sm font-bold mb-2">References</h4>
                            <div className="space-y-1">
                              {vuln.references.map((ref, idx) => (
                                <a 
                                  key={idx} 
                                  href={ref} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="block text-xs text-blue-400 hover:text-blue-300 break-all"
                                >
                                  {ref}
                                </a>
                              ))}
                            </div>
                          </>
                        )}
                      </div>
                    </div>

                    {/* CVSS Details */}
                    <div className="mt-4 p-3 bg-gray-800 rounded">
                      <h4 className="text-sm font-bold mb-2">CVSS Details</h4>
                      <p className="text-xs text-gray-400 font-mono">{vuln.cvssVector}</p>
                    </div>

                    {/* Tags */}
                    {vuln.tags.length > 0 && (
                      <div className="mt-4">
                        <div className="flex flex-wrap gap-2">
                          {vuln.tags.map(tag => (
                            <span key={tag} className="px-2 py-1 bg-blue-900/20 text-blue-500 rounded text-xs">
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="mt-4 flex items-center space-x-3">
                      <button className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm">
                        Mark Patched
                      </button>
                      <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                        Investigate
                      </button>
                      <button className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-sm">
                        Accept Risk
                      </button>
                      <button className="px-3 py-1 bg-gray-600 hover:bg-gray-700 rounded text-sm">
                        False Positive
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {viewMode === 'scans' && (
          <>
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-3">Vulnerability Scans</h3>
              <p className="text-sm text-gray-400">Manage and monitor vulnerability scanning activities</p>
            </div>
            
            <div className="space-y-4">
              {scanResults.map(scan => (
                <div key={scan.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="text-sm font-bold">{scan.scanName}</h4>
                      <p className="text-xs text-gray-500">Target: {scan.target} • Scanner: {scan.scanner}</p>
                    </div>
                    <div className="flex items-center space-x-3">
                      <span className={`px-2 py-1 text-xs rounded ${
                        scan.status === 'completed' ? 'bg-green-900/20 text-green-500' :
                        scan.status === 'running' ? 'bg-blue-900/20 text-blue-500' :
                        scan.status === 'failed' ? 'bg-red-900/20 text-red-500' :
                        'bg-yellow-900/20 text-yellow-500'
                      }`}>
                        {scan.status.toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-500">
                        {scan.status === 'running' ? `${scan.progress}%` : 
                         scan.duration ? `${scan.duration}m` : ''}
                      </span>
                    </div>
                  </div>
                  
                  {scan.status === 'running' && (
                    <div className="mb-3">
                      <div className="w-full bg-gray-800 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${scan.progress}%` }}
                        />
                      </div>
                    </div>
                  )}
                  
                  <div className="grid grid-cols-5 gap-3 text-xs">
                    <div className="text-center">
                      <p className="text-red-500 font-bold text-lg">{scan.vulnerabilitiesFound.critical}</p>
                      <p className="text-gray-400">Critical</p>
                    </div>
                    <div className="text-center">
                      <p className="text-orange-500 font-bold text-lg">{scan.vulnerabilitiesFound.high}</p>
                      <p className="text-gray-400">High</p>
                    </div>
                    <div className="text-center">
                      <p className="text-yellow-500 font-bold text-lg">{scan.vulnerabilitiesFound.medium}</p>
                      <p className="text-gray-400">Medium</p>
                    </div>
                    <div className="text-center">
                      <p className="text-blue-500 font-bold text-lg">{scan.vulnerabilitiesFound.low}</p>
                      <p className="text-gray-400">Low</p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-500 font-bold text-lg">{scan.vulnerabilitiesFound.info}</p>
                      <p className="text-gray-400">Info</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'reports' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="text-center py-12">
              <FileText className="w-12 h-12 text-gray-500 mx-auto mb-4" />
              <h3 className="text-lg font-bold mb-2">Vulnerability Reports</h3>
              <p className="text-sm text-gray-400 mb-6">
                Detailed vulnerability reports and compliance assessments would be displayed here
              </p>
              <div className="flex justify-center space-x-3">
                <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                  Generate Report
                </button>
                <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm">
                  Schedule Reports
                </button>
              </div>
            </div>
          </div>
        )}

        {viewMode === 'assets' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="text-center py-12">
              <Server className="w-12 h-12 text-gray-500 mx-auto mb-4" />
              <h3 className="text-lg font-bold mb-2">Asset Vulnerability Overview</h3>
              <p className="text-sm text-gray-400 mb-6">
                Asset-centric vulnerability view and risk assessment would be implemented here
              </p>
              <div className="flex justify-center space-x-3">
                <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                  View Assets
                </button>
                <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm">
                  Risk Assessment
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}