'use client';

import React, { useState, useEffect } from 'react';
import { 
  XOctagon, Shield, AlertTriangle, Eye, Target, Zap, Globe, Server,
  Database, Network, Lock, Wifi, Clock, TrendingUp, BarChart, Activity,
  Search, Filter, RefreshCw, Download, Settings, Info, Users, MapPin,
  Hash, FileText, ChevronRight, ArrowRight, Play, Pause, RotateCcw
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface ThreatIntelligence {
  id: string;
  type: 'malware' | 'phishing' | 'ddos' | 'brute_force' | 'insider_threat' | 'apt' | 'ransomware' | 'data_breach';
  name: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  confidence: number;
  source: string;
  targetAssets: string[];
  indicators: {
    ips: string[];
    domains: string[];
    fileHashes: string[];
    urls: string[];
  };
  attackVectors: string[];
  timestamp: string;
  status: 'active' | 'investigating' | 'mitigated' | 'false_positive';
  affectedSystems: number;
  riskScore: number;
  description: string;
  mitigation: string[];
  timeline: {
    detected: string;
    confirmed: string;
    contained?: string;
    resolved?: string;
  };
}

interface ThreatPattern {
  id: string;
  pattern: string;
  category: 'behavioral' | 'network' | 'file' | 'registry' | 'process';
  frequency: number;
  lastSeen: string;
  riskLevel: 'high' | 'medium' | 'low';
  associatedThreats: string[];
}

interface SecurityMetrics {
  totalThreats: number;
  blockedAttacks: number;
  activeThreatHunts: number;
  threatsLastHour: number;
  avgResponseTime: number;
  falsePositiveRate: number;
}

export default function ThreatDetection() {
  const [threats, setThreats] = useState<ThreatIntelligence[]>([]);
  const [patterns, setPatterns] = useState<ThreatPattern[]>([]);
  const [metrics, setMetrics] = useState<SecurityMetrics | null>(null);
  const [selectedThreat, setSelectedThreat] = useState<ThreatIntelligence | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSeverity, setSelectedSeverity] = useState('all');
  const [selectedType, setSelectedType] = useState('all');
  const [viewMode, setViewMode] = useState<'threats' | 'patterns' | 'intelligence' | 'hunting'>('threats');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [threatHuntingActive, setThreatHuntingActive] = useState(false);

  useEffect(() => {
    // Initialize with mock threat intelligence data
    setThreats([
      {
        id: 'THR-001',
        type: 'apt',
        name: 'Advanced Persistent Threat - Operation ShadowNet',
        severity: 'critical',
        confidence: 95,
        source: 'MITRE ATT&CK',
        targetAssets: ['Production Database', 'Web Servers', 'File Shares'],
        indicators: {
          ips: ['185.220.101.45', '194.169.175.23', '203.128.45.67'],
          domains: ['malicious-update.com', 'fake-auth.net'],
          fileHashes: ['a1b2c3d4e5f6...', 'f6e5d4c3b2a1...'],
          urls: ['/admin/backdoor.php', '/api/v1/exfiltrate']
        },
        attackVectors: ['Spear Phishing', 'Lateral Movement', 'Data Exfiltration'],
        timestamp: '5 minutes ago',
        status: 'active',
        affectedSystems: 12,
        riskScore: 9.2,
        description: 'Sophisticated APT campaign targeting sensitive data with multi-stage attack',
        mitigation: [
          'Block identified IP addresses',
          'Quarantine affected systems',
          'Reset compromised credentials',
          'Deploy additional monitoring'
        ],
        timeline: {
          detected: '2024-01-22T14:30:00Z',
          confirmed: '2024-01-22T14:45:00Z'
        }
      },
      {
        id: 'THR-002',
        type: 'ransomware',
        name: 'LockCrypt Ransomware Variant',
        severity: 'critical',
        confidence: 88,
        source: 'Internal Detection',
        targetAssets: ['File Servers', 'Backup Systems'],
        indicators: {
          ips: ['45.142.214.112'],
          domains: ['payment-portal.onion'],
          fileHashes: ['b3c4d5e6f7g8...', 'g8f7e6d5c4b3...'],
          urls: []
        },
        attackVectors: ['Email Attachment', 'File Encryption'],
        timestamp: '15 minutes ago',
        status: 'investigating',
        affectedSystems: 3,
        riskScore: 8.8,
        description: 'New variant of LockCrypt ransomware with improved evasion techniques',
        mitigation: [
          'Isolate affected systems',
          'Activate backup recovery',
          'Deploy decryption tools',
          'Coordinate with law enforcement'
        ],
        timeline: {
          detected: '2024-01-22T14:15:00Z',
          confirmed: '2024-01-22T14:25:00Z'
        }
      },
      {
        id: 'THR-003',
        type: 'brute_force',
        name: 'Distributed SSH Brute Force Campaign',
        severity: 'high',
        confidence: 92,
        source: 'Honeypot Network',
        targetAssets: ['Linux Servers', 'SSH Services'],
        indicators: {
          ips: ['103.45.67.89', '198.51.100.45', '172.16.254.1'],
          domains: [],
          fileHashes: [],
          urls: []
        },
        attackVectors: ['SSH Brute Force', 'Dictionary Attack'],
        timestamp: '1 hour ago',
        status: 'mitigated',
        affectedSystems: 8,
        riskScore: 6.5,
        description: 'Coordinated brute force attack against SSH services using common credentials',
        mitigation: [
          'Implement fail2ban rules',
          'Enable two-factor authentication',
          'Update password policies',
          'Monitor authentication logs'
        ],
        timeline: {
          detected: '2024-01-22T13:30:00Z',
          confirmed: '2024-01-22T13:35:00Z',
          contained: '2024-01-22T13:50:00Z',
          resolved: '2024-01-22T14:10:00Z'
        }
      },
      {
        id: 'THR-004',
        type: 'phishing',
        name: 'CEO Impersonation Campaign',
        severity: 'high',
        confidence: 85,
        source: 'Email Security Gateway',
        targetAssets: ['Email System', 'Finance Department'],
        indicators: {
          ips: ['208.67.222.123'],
          domains: ['company-update.org', 'urgent-memo.net'],
          fileHashes: ['c5d6e7f8g9h0...'],
          urls: ['/login/verify-account', '/secure/update-payment']
        },
        attackVectors: ['Business Email Compromise', 'Social Engineering'],
        timestamp: '2 hours ago',
        status: 'investigating',
        affectedSystems: 25,
        riskScore: 7.2,
        description: 'Sophisticated phishing campaign impersonating company executives',
        mitigation: [
          'Block malicious domains',
          'Quarantine suspicious emails',
          'User awareness training',
          'Implement DMARC policy'
        ],
        timeline: {
          detected: '2024-01-22T12:30:00Z',
          confirmed: '2024-01-22T12:45:00Z'
        }
      },
      {
        id: 'THR-005',
        type: 'insider_threat',
        name: 'Anomalous Data Access Pattern',
        severity: 'medium',
        confidence: 78,
        source: 'User Behavior Analytics',
        targetAssets: ['Customer Database', 'Financial Records'],
        indicators: {
          ips: ['192.168.1.45'],
          domains: [],
          fileHashes: [],
          urls: []
        },
        attackVectors: ['Privilege Abuse', 'Data Exfiltration'],
        timestamp: '3 hours ago',
        status: 'investigating',
        affectedSystems: 1,
        riskScore: 5.8,
        description: 'Employee accessing unusual amounts of sensitive data outside normal hours',
        mitigation: [
          'Review access permissions',
          'Monitor user activity',
          'Conduct security interview',
          'Implement data loss prevention'
        ],
        timeline: {
          detected: '2024-01-22T11:30:00Z',
          confirmed: '2024-01-22T12:00:00Z'
        }
      }
    ]);

    setPatterns([
      {
        id: 'PAT-001',
        pattern: 'Multiple failed login attempts from single IP',
        category: 'behavioral',
        frequency: 1847,
        lastSeen: '2 minutes ago',
        riskLevel: 'high',
        associatedThreats: ['THR-003']
      },
      {
        id: 'PAT-002',
        pattern: 'Unusual outbound traffic to known C2 servers',
        category: 'network',
        frequency: 23,
        lastSeen: '15 minutes ago',
        riskLevel: 'high',
        associatedThreats: ['THR-001']
      },
      {
        id: 'PAT-003',
        pattern: 'Suspicious PowerShell execution patterns',
        category: 'process',
        frequency: 78,
        lastSeen: '1 hour ago',
        riskLevel: 'medium',
        associatedThreats: ['THR-002']
      },
      {
        id: 'PAT-004',
        pattern: 'File encryption activities during off-hours',
        category: 'file',
        frequency: 5,
        lastSeen: '3 hours ago',
        riskLevel: 'high',
        associatedThreats: ['THR-002']
      }
    ]);

    setMetrics({
      totalThreats: 127,
      blockedAttacks: 2456,
      activeThreatHunts: 8,
      threatsLastHour: 15,
      avgResponseTime: 12.5,
      falsePositiveRate: 3.2
    });

    // Simulate real-time updates
    if (autoRefresh) {
      const interval = setInterval(() => {
        setMetrics(prev => prev ? {
          ...prev,
          totalThreats: prev.totalThreats + Math.floor(Math.random() * 3),
          blockedAttacks: prev.blockedAttacks + Math.floor(Math.random() * 10),
          threatsLastHour: Math.max(0, prev.threatsLastHour + Math.floor(Math.random() * 3) - 1)
        } : null);
      }, 8000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getThreatTypeIcon = (type: string) => {
    switch(type) {
      case 'malware': return <XOctagon className="w-4 h-4 text-red-500" />;
      case 'phishing': return <Globe className="w-4 h-4 text-orange-500" />;
      case 'ddos': return <Zap className="w-4 h-4 text-yellow-500" />;
      case 'brute_force': return <Lock className="w-4 h-4 text-purple-500" />;
      case 'insider_threat': return <Users className="w-4 h-4 text-pink-500" />;
      case 'apt': return <Target className="w-4 h-4 text-red-600" />;
      case 'ransomware': return <Shield className="w-4 h-4 text-red-700" />;
      case 'data_breach': return <Database className="w-4 h-4 text-blue-500" />;
      default: return <AlertTriangle className="w-4 h-4 text-gray-500" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch(severity) {
      case 'critical': return 'text-red-500 bg-red-900/20 border-red-800';
      case 'high': return 'text-orange-500 bg-orange-900/20 border-orange-800';
      case 'medium': return 'text-yellow-500 bg-yellow-900/20 border-yellow-800';
      case 'low': return 'text-blue-500 bg-blue-900/20 border-blue-800';
      default: return 'text-gray-500 bg-gray-900/20 border-gray-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': return 'text-red-500 bg-red-900/20';
      case 'investigating': return 'text-yellow-500 bg-yellow-900/20';
      case 'mitigated': return 'text-green-500 bg-green-900/20';
      case 'false_positive': return 'text-gray-500 bg-gray-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const filteredThreats = threats.filter(threat => {
    if (searchQuery && !threat.name.toLowerCase().includes(searchQuery.toLowerCase()) && 
        !threat.description.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    if (selectedSeverity !== 'all' && threat.severity !== selectedSeverity) return false;
    if (selectedType !== 'all' && threat.type !== selectedType) return false;
    return true;
  });

  const criticalThreats = threats.filter(t => t.severity === 'critical').length;
  const activeThreats = threats.filter(t => t.status === 'active').length;

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Threat Detection</h1>
            <p className="text-sm text-gray-400 mt-1">Advanced threat intelligence and detection</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setThreatHuntingActive(!threatHuntingActive)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                threatHuntingActive ? 'bg-red-600 hover:bg-red-700' : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              <Eye className="w-4 h-4" />
              <span>{threatHuntingActive ? 'Stop Hunt' : 'Start Hunt'}</span>
            </button>
            
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                autoRefresh ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              <span>{autoRefresh ? 'Live' : 'Paused'}</span>
            </button>
            
            <button
              onClick={() => setViewMode(
                viewMode === 'threats' ? 'patterns' : 
                viewMode === 'patterns' ? 'intelligence' : 
                viewMode === 'intelligence' ? 'hunting' : 'threats'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
            >
              {viewMode === 'threats' ? 'Patterns' : 
               viewMode === 'patterns' ? 'Intelligence' : 
               viewMode === 'intelligence' ? 'Hunting' : 'Threats'}
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export Intel</span>
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
              <p className="text-xs text-gray-400">Critical Threats</p>
              <p className="text-xl font-bold text-red-500">{criticalThreats}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XOctagon className="w-5 h-5 text-orange-500" />
            <div>
              <p className="text-xs text-gray-400">Active Threats</p>
              <p className="text-xl font-bold text-orange-500">{activeThreats}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Shield className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Blocked Attacks</p>
              <p className="text-xl font-bold">{metrics?.blockedAttacks.toLocaleString()}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Eye className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Threat Hunts</p>
              <p className="text-xl font-bold">{metrics?.activeThreatHunts}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Clock className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Avg Response</p>
              <p className="text-xl font-bold">{metrics?.avgResponseTime}m</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <TrendingUp className="w-5 h-5 text-cyan-500" />
            <div>
              <p className="text-xs text-gray-400">Last Hour</p>
              <p className="text-xl font-bold">{metrics?.threatsLastHour}</p>
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
              placeholder="Search threats by name or description..."
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
          </select>
          
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Types</option>
            <option value="apt">APT</option>
            <option value="ransomware">Ransomware</option>
            <option value="phishing">Phishing</option>
            <option value="brute_force">Brute Force</option>
            <option value="insider_threat">Insider Threat</option>
            <option value="malware">Malware</option>
            <option value="ddos">DDoS</option>
          </select>
        </div>

        {viewMode === 'threats' && (
          <div className="space-y-4">
            {filteredThreats.map(threat => (
              <div 
                key={threat.id} 
                className={`bg-gray-900 border border-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-800/50 ${
                  selectedThreat?.id === threat.id ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => setSelectedThreat(selectedThreat?.id === threat.id ? null : threat)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    {getThreatTypeIcon(threat.type)}
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h3 className="text-sm font-bold">{threat.name}</h3>
                        <span className={`px-2 py-1 text-xs rounded border ${getSeverityColor(threat.severity)}`}>
                          {threat.severity.toUpperCase()}
                        </span>
                        <span className={`px-2 py-1 text-xs rounded ${getStatusColor(threat.status)}`}>
                          {threat.status.replace('_', ' ').toUpperCase()}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400 mb-2">{threat.description}</p>
                      <div className="grid grid-cols-4 gap-4 text-xs text-gray-500">
                        <div>
                          <span className="text-gray-400">Confidence: </span>
                          <span className="font-bold">{threat.confidence}%</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Risk Score: </span>
                          <span className="font-bold text-red-500">{threat.riskScore}/10</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Affected: </span>
                          <span className="font-bold">{threat.affectedSystems} systems</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Source: </span>
                          <span className="font-bold">{threat.source}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <p className="text-xs text-gray-500">{threat.timestamp}</p>
                    <ChevronRight className={`w-4 h-4 text-gray-500 mt-2 transition-transform ${
                      selectedThreat?.id === threat.id ? 'rotate-90' : ''
                    }`} />
                  </div>
                </div>

                {/* Expanded Details */}
                {selectedThreat?.id === threat.id && (
                  <div className="mt-4 border-t border-gray-700 pt-4">
                    <div className="grid grid-cols-2 gap-6">
                      {/* Indicators of Compromise */}
                      <div>
                        <h4 className="text-sm font-bold mb-3">Indicators of Compromise</h4>
                        <div className="space-y-2">
                          {threat.indicators.ips.length > 0 && (
                            <div>
                              <p className="text-xs text-gray-400 mb-1">IP Addresses:</p>
                              <div className="flex flex-wrap gap-1">
                                {threat.indicators.ips.map(ip => (
                                  <span key={ip} className="px-2 py-1 bg-red-900/20 text-red-500 rounded text-xs font-mono">
                                    {ip}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          {threat.indicators.domains.length > 0 && (
                            <div>
                              <p className="text-xs text-gray-400 mb-1">Domains:</p>
                              <div className="flex flex-wrap gap-1">
                                {threat.indicators.domains.map(domain => (
                                  <span key={domain} className="px-2 py-1 bg-orange-900/20 text-orange-500 rounded text-xs font-mono">
                                    {domain}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          {threat.indicators.fileHashes.length > 0 && (
                            <div>
                              <p className="text-xs text-gray-400 mb-1">File Hashes:</p>
                              <div className="flex flex-wrap gap-1">
                                {threat.indicators.fileHashes.map(hash => (
                                  <span key={hash} className="px-2 py-1 bg-yellow-900/20 text-yellow-500 rounded text-xs font-mono">
                                    {hash}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Mitigation Steps */}
                      <div>
                        <h4 className="text-sm font-bold mb-3">Recommended Mitigation</h4>
                        <div className="space-y-1">
                          {threat.mitigation.map((step, idx) => (
                            <div key={idx} className="flex items-center space-x-2 text-sm">
                              <span className="w-4 h-4 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs">
                                {idx + 1}
                              </span>
                              <span>{step}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="mt-4 flex items-center space-x-3">
                      <button className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm">
                        Block Indicators
                      </button>
                      <button className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-sm">
                        Investigate
                      </button>
                      <button className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm">
                        Mark Mitigated
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

        {viewMode === 'patterns' && (
          <>
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-3">Threat Patterns</h3>
              <p className="text-sm text-gray-400">Behavioral and technical patterns associated with threats</p>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              {patterns.map(pattern => (
                <div key={pattern.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <span className={`w-3 h-3 rounded-full ${
                        pattern.riskLevel === 'high' ? 'bg-red-500' :
                        pattern.riskLevel === 'medium' ? 'bg-yellow-500' :
                        'bg-blue-500'
                      }`} />
                      <span className="text-xs text-gray-400 uppercase">{pattern.category}</span>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded ${
                      pattern.riskLevel === 'high' ? 'bg-red-900/20 text-red-500' :
                      pattern.riskLevel === 'medium' ? 'bg-yellow-900/20 text-yellow-500' :
                      'bg-blue-900/20 text-blue-500'
                    }`}>
                      {pattern.riskLevel.toUpperCase()}
                    </span>
                  </div>
                  
                  <h4 className="text-sm font-bold mb-2">{pattern.pattern}</h4>
                  
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <p className="text-gray-400">Frequency</p>
                      <p className="text-lg font-bold">{pattern.frequency.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Last Seen</p>
                      <p className="text-sm">{pattern.lastSeen}</p>
                    </div>
                  </div>
                  
                  {pattern.associatedThreats.length > 0 && (
                    <div className="mt-3">
                      <p className="text-xs text-gray-400 mb-1">Associated Threats:</p>
                      <div className="flex flex-wrap gap-1">
                        {pattern.associatedThreats.map(threatId => (
                          <span key={threatId} className="px-2 py-1 bg-gray-800 text-gray-400 rounded text-xs">
                            {threatId}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'intelligence' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="text-center py-12">
              <Globe className="w-12 h-12 text-gray-500 mx-auto mb-4" />
              <h3 className="text-lg font-bold mb-2">Threat Intelligence Hub</h3>
              <p className="text-sm text-gray-400 mb-6">
                External threat intelligence feeds and correlation would be displayed here
              </p>
              <div className="flex justify-center space-x-3">
                <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                  Sync Intel Feeds
                </button>
                <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm">
                  Configure Sources
                </button>
              </div>
            </div>
          </div>
        )}

        {viewMode === 'hunting' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="text-center py-12">
              <Eye className="w-12 h-12 text-gray-500 mx-auto mb-4" />
              <h3 className="text-lg font-bold mb-2">Threat Hunting Console</h3>
              <p className="text-sm text-gray-400 mb-6">
                Proactive threat hunting tools and query interface would be implemented here
              </p>
              <div className="flex justify-center space-x-3">
                <button 
                  onClick={() => setThreatHuntingActive(true)}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm flex items-center space-x-2"
                >
                  <Play className="w-4 h-4" />
                  <span>Start Hunt</span>
                </button>
                <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm">
                  Configure Hunts
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}