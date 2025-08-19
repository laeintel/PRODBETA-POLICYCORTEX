'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import { Line, Bar, Doughnut, Radar, Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  RadialLinearScale,
  Filler
} from 'chart.js';
import AuthGuard from '../../../components/AuthGuard';
import { api } from '../../../lib/api-client';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  RadialLinearScale,
  Filler
);

// Add animation styles
const fadeInStyles = `
  @keyframes fade-in {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  .animate-fade-in {
    animation: fade-in 0.5s ease-out;
  }
`;

if (typeof document !== 'undefined') {
  const styleSheet = document.createElement('style');
  styleSheet.textContent = fadeInStyles;
  document.head.appendChild(styleSheet);
}

interface SecurityData {
  threatLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  activeThreats: Array<{
    id: string;
    type: string;
    source: string;
    target: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    status: 'active' | 'mitigating' | 'resolved';
    detected: string;
    description: string;
    impact: string;
    lastActivity: string;
    location: string;
    protocol: string;
    port: number;
    attackVector: string;
    confidence: number;
  }>;
  securityScore: number;
  vulnerabilities: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
  recentActivity: Array<{
    timestamp: string;
    event: string;
    severity: string;
    source: string;
    category: string;
    details: string;
  }>;
  blockedAttempts: number;
  suspiciousActivities: number;
  securityPolicies: {
    active: number;
    pending: number;
    violations: number;
  };
  compliance: {
    score: number;
    frameworks: Array<{
      name: string;
      score: number;
      status: 'compliant' | 'non-compliant' | 'partial';
    }>;
  };
  incidents: Array<{
    id: string;
    title: string;
    severity: string;
    status: string;
    assignee: string;
    created: string;
    resolved?: string;
    category: string;
    priority: number;
  }>;
  metrics: {
    timeline: Array<{
      timestamp: string;
      threats: number;
      blocked: number;
      incidents: number;
    }>;
    geographical: Array<{
      country: string;
      threats: number;
      blocked: number;
    }>;
    categoryBreakdown: Array<{
      category: string;
      count: number;
      percentage: number;
    }>;
  };
  alerts: Array<{
    id: string;
    message: string;
    severity: string;
    timestamp: string;
    acknowledged: boolean;
  }>;
}

export default function SecurityDashboard() {
  return (
    <AuthGuard requireAuth={true}>
      <SecurityDashboardContent />
    </AuthGuard>
  );
}

function SecurityDashboardContent() {
  const [data, setData] = useState<SecurityData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedThreat, setSelectedThreat] = useState<string | null>(null);
  const [streamingLogs, setStreamingLogs] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('severity');
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [selectedIncident, setSelectedIncident] = useState<any>(null);
  const [dateRange, setDateRange] = useState('24h');
  
  const triggerAction = async (actionType: string) => {
    try {
      const resp = await api.createAction('global', actionType)
      if (resp.error || resp.status >= 400) {
        console.error('Action failed', actionType, resp.error)
        return
      }
      const id = resp.data?.action_id || resp.data?.id
      if (id) {
        const stop = api.streamActionEvents(String(id), (m) => console.log('[security-action]', id, m))
        setTimeout(stop, 60000)
      }
    } catch (e) {
      console.error('Trigger action error', actionType, e)
    }
  }

  useEffect(() => {
    fetchSecurityData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchSecurityData, refreshInterval);
      const logInterval = setInterval(() => {
        const newLog = `[${new Date().toISOString()}] ${generateRandomLog()}`;
        setStreamingLogs(prev => [newLog, ...prev.slice(0, 199)]);
      }, 2000);
      
      return () => {
        clearInterval(interval);
        clearInterval(logInterval);
      };
    }
  }, [autoRefresh, refreshInterval]);

  const fetchSecurityData = async () => {
    try {
      const resp = await api.getSecurityThreats()
      if (resp.error) setData(getMockSecurityData()); else setData(resp.data as any)
    } catch (error) {
      setData(getMockSecurityData());
    } finally {
      setLoading(false);
    }
  };

  const getMockSecurityData = (): SecurityData => ({
    threatLevel: 'MEDIUM',
    activeThreats: [
      {
        id: 't1',
        type: 'Brute Force Attack',
        source: '185.220.101.45',
        target: 'vm-prod-web-01',
        severity: 'high',
        status: 'active',
        detected: '12 min ago',
        description: 'Multiple failed SSH login attempts detected',
        impact: 'Authentication bypass risk',
        lastActivity: '2 min ago',
        location: 'Russia',
        protocol: 'SSH',
        port: 22,
        attackVector: 'Dictionary Attack',
        confidence: 95
      },
      {
        id: 't2',
        type: 'SQL Injection Attempt',
        source: '45.142.214.112',
        target: 'sql-prod-01',
        severity: 'critical',
        status: 'mitigating',
        detected: '1 hour ago',
        description: 'Malicious SQL query patterns detected in application logs',
        impact: 'Data exfiltration risk',
        lastActivity: '15 min ago',
        location: 'China',
        protocol: 'HTTP',
        port: 443,
        attackVector: 'Union-based SQLi',
        confidence: 88
      },
      {
        id: 't3',
        type: 'Unusual Data Transfer',
        source: 'storage-prod-02',
        target: 'External IP',
        severity: 'medium',
        status: 'active',
        detected: '3 hours ago',
        description: 'Large volume of data transfer to unrecognized IP address',
        impact: 'Potential data loss',
        lastActivity: '1 hour ago',
        location: 'Unknown',
        protocol: 'FTP',
        port: 21,
        attackVector: 'Data Exfiltration',
        confidence: 72
      },
      {
        id: 't4',
        type: 'DDoS Attack',
        source: '91.234.56.78',
        target: 'web-prod-lb',
        severity: 'high',
        status: 'mitigating',
        detected: '45 min ago',
        description: 'High volume traffic from multiple IPs targeting load balancer',
        impact: 'Service availability',
        lastActivity: '5 min ago',
        location: 'Ukraine',
        protocol: 'HTTP',
        port: 80,
        attackVector: 'Volumetric Attack',
        confidence: 91
      },
      {
        id: 't5',
        type: 'Privilege Escalation',
        source: '10.0.1.150',
        target: 'ad-controller-01',
        severity: 'critical',
        status: 'active',
        detected: '30 min ago',
        description: 'Suspicious privilege escalation attempt detected on domain controller',
        impact: 'Full domain compromise',
        lastActivity: '10 min ago',
        location: 'Internal',
        protocol: 'LDAP',
        port: 389,
        attackVector: 'Lateral Movement',
        confidence: 97
      }
    ],
    securityScore: 87,
    vulnerabilities: {
      critical: 12,
      high: 34,
      medium: 89,
      low: 156
    },
    recentActivity: [
      { timestamp: '10:42:15', event: 'Failed login attempt blocked', severity: 'medium', source: '192.168.1.45', category: 'Authentication', details: 'User: admin, Method: SSH' },
      { timestamp: '10:38:22', event: 'Firewall rule updated', severity: 'low', source: 'System', category: 'Network', details: 'Rule ID: FW-001, Action: Allow HTTP' },
      { timestamp: '10:31:48', event: 'DDoS protection activated', severity: 'high', source: 'CDN', category: 'Protection', details: 'Rate limit: 1000 req/min' },
      { timestamp: '10:27:03', event: 'SSL certificate renewed', severity: 'info', source: 'System', category: 'Cryptography', details: 'Domain: *.contoso.com' },
      { timestamp: '10:23:41', event: 'Malware detected and quarantined', severity: 'high', source: 'AV-Engine', category: 'Malware', details: 'File: suspicious.exe, Hash: abc123' },
      { timestamp: '10:19:15', event: 'Unauthorized API access blocked', severity: 'medium', source: 'API-Gateway', category: 'Authentication', details: 'Endpoint: /api/users, IP: 203.45.67.89' }
    ],
    blockedAttempts: 1847,
    suspiciousActivities: 23,
    securityPolicies: {
      active: 156,
      pending: 8,
      violations: 12
    },
    compliance: {
      score: 89,
      frameworks: [
        { name: 'SOC 2 Type II', score: 94, status: 'compliant' },
        { name: 'ISO 27001', score: 91, status: 'compliant' },
        { name: 'GDPR', score: 87, status: 'partial' },
        { name: 'HIPAA', score: 82, status: 'partial' },
        { name: 'PCI DSS', score: 96, status: 'compliant' }
      ]
    },
    incidents: [
      {
        id: 'INC-001',
        title: 'Suspected Data Breach Investigation',
        severity: 'critical',
        status: 'investigating',
        assignee: 'Sarah Chen',
        created: '2024-01-15T08:30:00Z',
        category: 'Data Security',
        priority: 1
      },
      {
        id: 'INC-002',
        title: 'Malware Outbreak - Finance Department',
        severity: 'high',
        status: 'resolved',
        assignee: 'Mike Johnson',
        created: '2024-01-14T14:20:00Z',
        resolved: '2024-01-14T18:45:00Z',
        category: 'Malware',
        priority: 2
      },
      {
        id: 'INC-003',
        title: 'Phishing Campaign Targeting Executives',
        severity: 'high',
        status: 'mitigating',
        assignee: 'Alex Rodriguez',
        created: '2024-01-13T11:15:00Z',
        category: 'Social Engineering',
        priority: 2
      }
    ],
    metrics: {
      timeline: Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(Date.now() - (23 - i) * 60 * 60 * 1000).toISOString(),
        threats: Math.floor(Math.random() * 50) + 10,
        blocked: Math.floor(Math.random() * 200) + 50,
        incidents: Math.floor(Math.random() * 10)
      })),
      geographical: [
        { country: 'China', threats: 245, blocked: 1890 },
        { country: 'Russia', threats: 189, blocked: 1456 },
        { country: 'North Korea', threats: 134, blocked: 987 },
        { country: 'Iran', threats: 98, blocked: 745 },
        { country: 'Brazil', threats: 67, blocked: 456 },
        { country: 'Nigeria', threats: 45, blocked: 234 }
      ],
      categoryBreakdown: [
        { category: 'Malware', count: 456, percentage: 32 },
        { category: 'Phishing', count: 289, percentage: 20 },
        { category: 'Brute Force', count: 234, percentage: 16 },
        { category: 'SQL Injection', count: 178, percentage: 12 },
        { category: 'DDoS', count: 145, percentage: 10 },
        { category: 'Other', count: 143, percentage: 10 }
      ]
    },
    alerts: [
      {
        id: 'ALR-001',
        message: 'Critical vulnerability detected in Apache server',
        severity: 'critical',
        timestamp: '2024-01-15T09:15:00Z',
        acknowledged: false
      },
      {
        id: 'ALR-002',
        message: 'Unusual login pattern detected for admin user',
        severity: 'high',
        timestamp: '2024-01-15T08:45:00Z',
        acknowledged: true
      }
    ]
  });

  const generateRandomLog = () => {
    const logs = [
      'Authentication attempt from unknown IP',
      'Firewall rule triggered - Port 22 blocked',
      'Port scan detected from 192.168.1.100',
      'SSL handshake completed successfully',
      'Security policy updated - Policy ID: SEC-001',
      'Intrusion detection alert - Signature match',
      'Access granted to resource /api/secure',
      'Security scan completed - 0 threats found',
      'Failed login attempt blocked - User: admin',
      'Malware signature updated',
      'VPN connection established',
      'Certificate validation failed',
      'Rate limit exceeded for API endpoint',
      'Suspicious file upload detected',
      'Password policy violation detected'
    ];
    return logs[Math.floor(Math.random() * logs.length)];
  };
  
  const filteredThreats = useMemo(() => {
    if (!data?.activeThreats) return [];
    
    return data.activeThreats
      .filter(threat => {
        const matchesSearch = threat.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
                             threat.source.toLowerCase().includes(searchTerm.toLowerCase()) ||
                             threat.target.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesSeverity = filterSeverity === 'all' || threat.severity === filterSeverity;
        return matchesSearch && matchesSeverity;
      })
      .sort((a, b) => {
        switch (sortBy) {
          case 'severity':
            const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
            return severityOrder[b.severity as keyof typeof severityOrder] - severityOrder[a.severity as keyof typeof severityOrder];
          case 'detected':
            return new Date(b.detected).getTime() - new Date(a.detected).getTime();
          case 'confidence':
            return b.confidence - a.confidence;
          default:
            return 0;
        }
      });
  }, [data?.activeThreats, searchTerm, sortBy, filterSeverity]);
  
  const exportData = (format: 'csv' | 'json') => {
    if (!data) return;
    
    let content: string;
    let filename: string;
    let mimeType: string;
    
    if (format === 'csv') {
      const headers = ['ID', 'Type', 'Source', 'Target', 'Severity', 'Status', 'Detected'];
      const rows = data.activeThreats.map(threat => [
        threat.id,
        threat.type,
        threat.source,
        threat.target,
        threat.severity,
        threat.status,
        threat.detected
      ]);
      content = [headers, ...rows].map(row => row.join(',')).join('\n');
      filename = 'security-threats.csv';
      mimeType = 'text/csv';
    } else {
      content = JSON.stringify(data, null, 2);
      filename = 'security-data.json';
      mimeType = 'application/json';
    }
    
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const mitigateThreat = async (threatId: string) => {
    try {
      const resp = await api.mitigateThreat(threatId)
      if (resp.error) console.error('Mitigation failed', resp.error)
      fetchSecurityData();
    } catch (error) {
      console.error('Failed to mitigate threat:', error);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="w-20 h-20 border-4 border-red-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <div className="w-16 h-16 border-4 border-orange-500 border-t-transparent rounded-full animate-spin absolute top-2 left-2 opacity-60" />
            <div className="w-12 h-12 border-4 border-yellow-500 border-t-transparent rounded-full animate-spin absolute top-4 left-4 opacity-40" />
          </div>
          <p className="text-lg text-red-500 font-bold animate-pulse">INITIALIZING SECURITY SYSTEMS</p>
          <div className="flex justify-center space-x-1 mt-2">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <div className="w-2 h-2 bg-yellow-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
        </div>
      </div>
    );
  }
  
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        labels: {
          color: '#9CA3AF'
        }
      }
    },
    scales: {
      x: {
        ticks: {
          color: '#9CA3AF'
        },
        grid: {
          color: '#374151'
        }
      },
      y: {
        ticks: {
          color: '#9CA3AF'
        },
        grid: {
          color: '#374151'
        }
      }
    }
  };
  
  const threatTimelineData = {
    labels: data?.metrics.timeline.map(t => new Date(t.timestamp).toLocaleTimeString()) || [],
    datasets: [
      {
        label: 'Threats Detected',
        data: data?.metrics.timeline.map(t => t.threats) || [],
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        fill: true,
        tension: 0.4
      },
      {
        label: 'Blocked Attempts',
        data: data?.metrics.timeline.map(t => t.blocked) || [],
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        fill: true,
        tension: 0.4
      }
    ]
  };
  
  const vulnerabilityData = {
    labels: ['Critical', 'High', 'Medium', 'Low'],
    datasets: [
      {
        data: [
          data?.vulnerabilities.critical || 0,
          data?.vulnerabilities.high || 0,
          data?.vulnerabilities.medium || 0,
          data?.vulnerabilities.low || 0
        ],
        backgroundColor: [
          'rgba(239, 68, 68, 0.8)',
          'rgba(249, 115, 22, 0.8)',
          'rgba(251, 191, 36, 0.8)',
          'rgba(156, 163, 175, 0.8)'
        ],
        borderColor: [
          'rgb(239, 68, 68)',
          'rgb(249, 115, 22)',
          'rgb(251, 191, 36)',
          'rgb(156, 163, 175)'
        ],
        borderWidth: 2
      }
    ]
  };
  
  const geographicalData = {
    labels: data?.metrics.geographical.map(g => g.country) || [],
    datasets: [
      {
        label: 'Threats by Country',
        data: data?.metrics.geographical.map(g => g.threats) || [],
        backgroundColor: 'rgba(239, 68, 68, 0.7)',
        borderColor: 'rgb(239, 68, 68)',
        borderWidth: 1
      }
    ]
  };

  return (
    <div className="min-h-screen bg-black text-gray-100">
      {/* Header */}
      <header className="bg-gray-900 border-b border-red-900/30 backdrop-blur-sm">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/tactical" className="text-gray-400 hover:text-red-400 transition-colors">
                ← TACTICAL
              </Link>
              <div className="h-6 w-px bg-red-800" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-red-500 to-orange-500 bg-clip-text text-transparent">
                SECURITY DASHBOARD
              </h1>
              <div className={`px-3 py-1 rounded-lg text-xs font-bold border ${
                data?.threatLevel === 'CRITICAL' ? 'bg-red-900/30 text-red-400 border-red-600 animate-pulse' :
                data?.threatLevel === 'HIGH' ? 'bg-orange-900/30 text-orange-400 border-orange-600' :
                data?.threatLevel === 'MEDIUM' ? 'bg-yellow-900/30 text-yellow-400 border-yellow-600' :
                'bg-green-900/30 text-green-400 border-green-600'
              }`}>
                DEFCON {data?.threatLevel === 'CRITICAL' ? '1' : data?.threatLevel === 'HIGH' ? '2' : data?.threatLevel === 'MEDIUM' ? '3' : '4'}
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-400">Auto Refresh:</label>
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                    autoRefresh ? 'bg-green-900/30 text-green-400 border border-green-600' : 'bg-gray-800 text-gray-400 border border-gray-600'
                  }`}
                >
                  {autoRefresh ? 'ON' : 'OFF'}
                </button>
              </div>
              <button onClick={() => exportData('csv')} className="px-4 py-2 bg-blue-900/30 hover:bg-blue-900/50 border border-blue-600 text-blue-400 text-sm font-medium rounded-lg transition-all">
                EXPORT CSV
              </button>
              <button onClick={() => triggerAction('emergency_lockdown')} className="px-4 py-2 bg-red-900/30 hover:bg-red-900/50 border border-red-600 text-red-400 text-sm font-medium rounded-lg transition-all animate-pulse">
                EMERGENCY LOCKDOWN
              </button>
              <button onClick={() => triggerAction('security_scan')} className="px-4 py-2 bg-purple-900/30 hover:bg-purple-900/50 border border-purple-600 text-purple-400 text-sm font-medium rounded-lg transition-all">
                FULL SCAN
              </button>
            </div>
          </div>
        </div>
      </header>
      
      {/* Navigation Tabs */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6">
        <div className="flex space-x-8">
          {[
            { id: 'overview', label: 'THREAT OVERVIEW', icon: '🛡️' },
            { id: 'incidents', label: 'INCIDENT RESPONSE', icon: '🚨' },
            { id: 'analytics', label: 'THREAT ANALYTICS', icon: '📊' },
            { id: 'compliance', label: 'COMPLIANCE', icon: '✅' },
            { id: 'policies', label: 'SECURITY POLICIES', icon: '📋' },
            { id: 'monitoring', label: 'LIVE MONITORING', icon: '👁️' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-red-500 text-red-400'
                  : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-600'
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="p-6">
        {/* Enhanced Metrics Row */}
        <div className="grid grid-cols-6 gap-4 mb-6">
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-red-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-red-400">Security Score</p>
            <p className="text-3xl font-bold font-mono text-white">{data?.securityScore}%</p>
            <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-green-500 to-blue-500 rounded-full transition-all duration-1000 ease-out"
                style={{ width: `${data?.securityScore}%` }}
              />
            </div>
            <p className="text-xs text-green-400 mt-1">+2.3% this week</p>
          </div>
          
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-red-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-red-400">Active Threats</p>
            <p className="text-3xl font-bold font-mono text-red-400">
              {data?.activeThreats.filter(t => t.status === 'active').length}
            </p>
            <p className="text-xs text-yellow-400 mt-1">
              {data?.activeThreats.filter(t => t.status === 'mitigating').length} mitigating
            </p>
            <div className="mt-1 flex space-x-1">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className={`w-1 h-4 rounded-full ${
                  i < (data?.activeThreats.filter(t => t.status === 'active').length || 0) ? 'bg-red-500' : 'bg-gray-700'
                }`} />
              ))}
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-green-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-green-400">Blocked Today</p>
            <p className="text-3xl font-bold font-mono text-green-400">{data?.blockedAttempts.toLocaleString()}</p>
            <p className="text-xs text-gray-400 mt-1">attempts</p>
            <div className="mt-1 text-xs text-green-400">↑ 15.2% vs yesterday</div>
          </div>
          
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-orange-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-orange-400">Vulnerabilities</p>
            <div className="flex items-center space-x-3 mt-2">
              <div className="text-center">
                <div className="text-lg font-bold text-red-400">{data?.vulnerabilities.critical}</div>
                <div className="text-xs text-gray-500">CRIT</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-orange-400">{data?.vulnerabilities.high}</div>
                <div className="text-xs text-gray-500">HIGH</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-yellow-400">{data?.vulnerabilities.medium}</div>
                <div className="text-xs text-gray-500">MED</div>
              </div>
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-yellow-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-yellow-400">Suspicious</p>
            <p className="text-3xl font-bold font-mono text-yellow-400">{data?.suspiciousActivities}</p>
            <p className="text-xs text-gray-400 mt-1">activities</p>
            <div className="mt-1 w-full bg-gray-700 rounded-full h-1">
              <div className="bg-yellow-400 h-1 rounded-full" style={{ width: '30%' }} />
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-blue-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-blue-400">Compliance</p>
            <p className="text-3xl font-bold font-mono text-blue-400">{data?.compliance.score}%</p>
            <p className="text-xs text-gray-400 mt-1">overall</p>
            <div className="mt-1 flex space-x-1">
              {data?.compliance.frameworks.map((fw, i) => (
                <div key={i} className={`w-2 h-2 rounded-full ${
                  fw.status === 'compliant' ? 'bg-green-400' :
                  fw.status === 'partial' ? 'bg-yellow-400' : 'bg-red-400'
                }`} />
              ))}
            </div>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Search and Filters */}
            <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-gray-200">THREAT INTELLIGENCE</h3>
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Search threats..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-sm text-gray-200 placeholder-gray-400 focus:border-red-500 focus:outline-none"
                    />
                    <div className="absolute right-3 top-2.5">
                      <div className="w-4 h-4 text-gray-400">🔍</div>
                    </div>
                  </div>
                  <select
                    value={filterSeverity}
                    onChange={(e) => setFilterSeverity(e.target.value)}
                    className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-200"
                  >
                    <option value="all">All Severities</option>
                    <option value="critical">Critical</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                    className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-200"
                  >
                    <option value="severity">Sort by Severity</option>
                    <option value="detected">Sort by Detection Time</option>
                    <option value="confidence">Sort by Confidence</option>
                  </select>
                </div>
              </div>
              
              {/* Threat Grid */}
              <div className="grid grid-cols-1 gap-4">
                {filteredThreats.map((threat) => (
                  <div
                    key={threat.id}
                    className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 hover:bg-gray-800/70 transition-all cursor-pointer group"
                    onClick={() => setSelectedThreat(threat.id)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <div className={`w-3 h-3 rounded-full ${
                            threat.severity === 'critical' ? 'bg-red-500 animate-pulse shadow-red-500/50 shadow-lg' :
                            threat.severity === 'high' ? 'bg-orange-500 shadow-orange-500/50 shadow-md' :
                            threat.severity === 'medium' ? 'bg-yellow-500 shadow-yellow-500/50 shadow-sm' :
                            'bg-gray-500'
                          }`} />
                          <h4 className="font-bold text-gray-200 group-hover:text-red-400 transition-colors">{threat.type}</h4>
                          <span className={`text-xs px-3 py-1 rounded-full font-medium ${
                            threat.status === 'active' ? 'bg-red-900/30 text-red-400 border border-red-600/30' :
                            threat.status === 'mitigating' ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-600/30' :
                            'bg-green-900/30 text-green-400 border border-green-600/30'
                          }`}>
                            {threat.status.toUpperCase()}
                          </span>
                          <div className="ml-auto text-xs text-gray-400 bg-gray-700/50 px-2 py-1 rounded">
                            Confidence: {threat.confidence}%
                          </div>
                        </div>
                        <p className="text-sm text-gray-400 mb-3">{threat.description}</p>
                        <div className="grid grid-cols-4 gap-4 text-xs">
                          <div>
                            <span className="text-gray-500">Source:</span>
                            <div className="text-gray-300 font-mono mt-1">{threat.source}</div>
                            <div className="text-gray-500">{threat.location}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">Target:</span>
                            <div className="text-gray-300 font-mono mt-1">{threat.target}</div>
                            <div className="text-gray-500">{threat.protocol}:{threat.port}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">Attack Vector:</span>
                            <div className="text-gray-300 mt-1">{threat.attackVector}</div>
                            <div className="text-gray-500">Detected: {threat.detected}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">Impact:</span>
                            <div className="text-orange-300 mt-1">{threat.impact}</div>
                            <div className="text-gray-500">Last: {threat.lastActivity}</div>
                          </div>
                        </div>
                      </div>
                      <div className="flex flex-col space-y-2 ml-4">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            mitigateThreat(threat.id);
                          }}
                          className="px-4 py-2 bg-yellow-900/30 hover:bg-yellow-900/50 border border-yellow-600 rounded-lg text-yellow-400 text-xs font-medium transition-all hover:shadow-yellow-500/20 hover:shadow-lg"
                        >
                          MITIGATE
                        </button>
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedThreat(threat.id);
                            setShowDetailsModal(true);
                          }}
                          className="px-4 py-2 bg-blue-900/30 hover:bg-blue-900/50 border border-blue-600 rounded-lg text-blue-400 text-xs font-medium transition-all hover:shadow-blue-500/20 hover:shadow-lg"
                        >
                          ANALYZE
                        </button>
                        <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-lg text-gray-400 text-xs font-medium transition-all">
                          QUARANTINE
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Live Security Stream */}
            <div className="grid grid-cols-3 gap-6">
              <div className="col-span-2 bg-gray-900/50 border border-gray-700 rounded-xl">
                <div className="p-4 border-b border-gray-700 flex items-center justify-between">
                  <h3 className="text-lg font-bold text-gray-200">REAL-TIME THREAT STREAM</h3>
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                      <span className="text-xs text-green-400 font-medium">LIVE</span>
                    </div>
                    <button className="text-xs bg-gray-800 hover:bg-gray-700 border border-gray-600 px-3 py-1 rounded-lg text-gray-400 transition-colors">
                      PAUSE
                    </button>
                  </div>
                </div>
                <div className="p-4 h-96 overflow-y-auto">
                  <div className="space-y-2 font-mono text-xs">
                    {streamingLogs.map((log, i) => (
                      <div
                        key={i}
                        className={`p-2 rounded border-l-2 animate-fade-in ${
                          log.includes('blocked') || log.includes('Failed') ? 'bg-red-900/20 border-red-500 text-red-300' :
                          log.includes('alert') || log.includes('detected') ? 'bg-yellow-900/20 border-yellow-500 text-yellow-300' :
                          log.includes('success') || log.includes('completed') ? 'bg-green-900/20 border-green-500 text-green-300' :
                          'bg-gray-800/30 border-gray-600 text-gray-400'
                        }`}
                        style={{ animationDelay: `${i * 50}ms` }}
                      >
                        {log}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              {/* Quick Actions */}
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl">
                <div className="p-4 border-b border-gray-700">
                  <h3 className="text-lg font-bold text-gray-200">QUICK ACTIONS</h3>
                </div>
                <div className="p-4 space-y-3">
                  <button 
                    onClick={() => triggerAction('full_scan')}
                    className="w-full p-3 bg-purple-900/30 hover:bg-purple-900/50 border border-purple-600 rounded-lg text-purple-400 font-medium transition-all hover:shadow-purple-500/20 hover:shadow-lg"
                  >
                    🔍 FULL SYSTEM SCAN
                  </button>
                  <button 
                    onClick={() => triggerAction('update_signatures')}
                    className="w-full p-3 bg-blue-900/30 hover:bg-blue-900/50 border border-blue-600 rounded-lg text-blue-400 font-medium transition-all hover:shadow-blue-500/20 hover:shadow-lg"
                  >
                    🔄 UPDATE SIGNATURES
                  </button>
                  <button 
                    onClick={() => triggerAction('isolate_threats')}
                    className="w-full p-3 bg-orange-900/30 hover:bg-orange-900/50 border border-orange-600 rounded-lg text-orange-400 font-medium transition-all hover:shadow-orange-500/20 hover:shadow-lg"
                  >
                    🚫 ISOLATE THREATS
                  </button>
                  <button 
                    onClick={() => triggerAction('backup_config')}
                    className="w-full p-3 bg-green-900/30 hover:bg-green-900/50 border border-green-600 rounded-lg text-green-400 font-medium transition-all hover:shadow-green-500/20 hover:shadow-lg"
                  >
                    💾 BACKUP CONFIG
                  </button>
                  <button 
                    onClick={() => triggerAction('generate_report')}
                    className="w-full p-3 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-lg text-gray-400 font-medium transition-all"
                  >
                    📊 GENERATE REPORT
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'analytics' && (
          <div className="space-y-6">
            {/* Charts Row */}
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-200 mb-4">THREAT TIMELINE (24H)</h3>
                <div className="h-80">
                  <Line data={threatTimelineData} options={chartOptions} />
                </div>
              </div>
              
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-200 mb-4">VULNERABILITY BREAKDOWN</h3>
                <div className="h-80">
                  <Doughnut data={vulnerabilityData} options={chartOptions} />
                </div>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-200 mb-4">THREATS BY GEOGRAPHY</h3>
                <div className="h-80">
                  <Bar data={geographicalData} options={chartOptions} />
                </div>
              </div>
              
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-200 mb-4">ATTACK CATEGORIES</h3>
                <div className="space-y-4">
                  {data?.metrics.categoryBreakdown.map((category, i) => (
                    <div key={i} className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${
                          i === 0 ? 'bg-red-500' :
                          i === 1 ? 'bg-orange-500' :
                          i === 2 ? 'bg-yellow-500' :
                          i === 3 ? 'bg-green-500' :
                          i === 4 ? 'bg-blue-500' : 'bg-purple-500'
                        }`} />
                        <span className="text-gray-200">{category.category}</span>
                      </div>
                      <div className="flex items-center space-x-4">
                        <div className="w-32 bg-gray-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              i === 0 ? 'bg-red-500' :
                              i === 1 ? 'bg-orange-500' :
                              i === 2 ? 'bg-yellow-500' :
                              i === 3 ? 'bg-green-500' :
                              i === 4 ? 'bg-blue-500' : 'bg-purple-500'
                            }`}
                            style={{ width: `${category.percentage}%` }}
                          />
                        </div>
                        <span className="text-gray-400 font-mono text-sm">{category.count}</span>
                        <span className="text-gray-500 text-sm">{category.percentage}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'compliance' && (
          <div className="space-y-6">
            <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
              <h3 className="text-lg font-bold text-gray-200 mb-6">COMPLIANCE FRAMEWORKS</h3>
              <div className="grid grid-cols-1 gap-4">
                {data?.compliance.frameworks.map((framework, i) => (
                  <div key={i} className="bg-gray-800/50 border border-gray-600 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-lg font-medium text-gray-200">{framework.name}</h4>
                      <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                        framework.status === 'compliant' ? 'bg-green-900/30 text-green-400 border border-green-600' :
                        framework.status === 'partial' ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-600' :
                        'bg-red-900/30 text-red-400 border border-red-600'
                      }`}>
                        {framework.status.toUpperCase()}
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="flex-1 bg-gray-700 rounded-full h-3">
                        <div
                          className={`h-3 rounded-full transition-all duration-1000 ${
                            framework.status === 'compliant' ? 'bg-green-500' :
                            framework.status === 'partial' ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${framework.score}%` }}
                        />
                      </div>
                      <span className="text-2xl font-bold text-gray-200">{framework.score}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'incidents' && (
          <div className="space-y-6">
            <div className="bg-gray-900/50 border border-gray-700 rounded-xl">
              <div className="p-4 border-b border-gray-700 flex items-center justify-between">
                <h3 className="text-lg font-bold text-gray-200">SECURITY INCIDENTS</h3>
                <button 
                  onClick={() => setShowCreateModal(true)}
                  className="px-4 py-2 bg-red-900/30 hover:bg-red-900/50 border border-red-600 text-red-400 text-sm font-medium rounded-lg transition-all"
                >
                  + CREATE INCIDENT
                </button>
              </div>
              <div className="divide-y divide-gray-700">
                {data?.incidents.map((incident) => (
                  <div key={incident.id} className="p-4 hover:bg-gray-800/30 transition-colors cursor-pointer" onClick={() => setSelectedIncident(incident)}>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <span className={`w-3 h-3 rounded-full ${
                            incident.severity === 'critical' ? 'bg-red-500 animate-pulse' :
                            incident.severity === 'high' ? 'bg-orange-500' :
                            'bg-yellow-500'
                          }`} />
                          <h4 className="font-medium text-gray-200">{incident.title}</h4>
                          <span className={`text-xs px-2 py-1 rounded-full ${
                            incident.status === 'investigating' ? 'bg-blue-900/30 text-blue-400 border border-blue-600/30' :
                            incident.status === 'mitigating' ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-600/30' :
                            'bg-green-900/30 text-green-400 border border-green-600/30'
                          }`}>
                            {incident.status.toUpperCase()}
                          </span>
                        </div>
                        <div className="grid grid-cols-4 gap-4 text-sm text-gray-400">
                          <div>
                            <span className="text-gray-500">ID:</span>
                            <div className="font-mono">{incident.id}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">Assignee:</span>
                            <div>{incident.assignee}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">Created:</span>
                            <div>{new Date(incident.created).toLocaleDateString()}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">Category:</span>
                            <div>{incident.category}</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'monitoring' && (
          <div className="space-y-6">
            {/* Recent Activity */}
            <div className="bg-gray-900/50 border border-gray-700 rounded-xl">
              <div className="p-4 border-b border-gray-700">
                <h3 className="text-lg font-bold text-gray-200">SECURITY EVENT STREAM</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="border-b border-gray-700 bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Timestamp</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Event</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Category</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Severity</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Source</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Details</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-700">
                    {data?.recentActivity.map((activity, i) => (
                      <tr key={i} className="hover:bg-gray-800/30 transition-colors">
                        <td className="px-4 py-3 text-sm font-mono text-gray-300">{activity.timestamp}</td>
                        <td className="px-4 py-3 text-sm text-gray-200">{activity.event}</td>
                        <td className="px-4 py-3 text-sm text-gray-400">{activity.category}</td>
                        <td className="px-4 py-3">
                          <span className={`text-xs px-3 py-1 rounded-full font-medium ${
                            activity.severity === 'high' ? 'bg-red-900/30 text-red-400 border border-red-600/30' :
                            activity.severity === 'medium' ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-600/30' :
                            activity.severity === 'low' ? 'bg-green-900/30 text-green-400 border border-green-600/30' :
                            'bg-gray-800 text-gray-400 border border-gray-600'
                          }`}>
                            {activity.severity.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm font-mono text-gray-400">{activity.source}</td>
                        <td className="px-4 py-3 text-sm text-gray-500">{activity.details}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Details Modal */}
      {showDetailsModal && selectedThreat && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-4xl w-full m-4 max-h-screen overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-200">THREAT ANALYSIS</h2>
              <button 
                onClick={() => setShowDetailsModal(false)}
                className="text-gray-400 hover:text-gray-200 text-xl"
              >
                ✕
              </button>
            </div>
            
            {(() => {
              const threat = data?.activeThreats.find(t => t.id === selectedThreat);
              if (!threat) return null;
              
              return (
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="bg-gray-800/50 border border-gray-600 rounded-lg p-4">
                      <h3 className="text-lg font-medium text-gray-200 mb-3">Threat Details</h3>
                      <div className="space-y-3 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Type:</span>
                          <span className="text-gray-200">{threat.type}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Severity:</span>
                          <span className={`font-medium ${
                            threat.severity === 'critical' ? 'text-red-400' :
                            threat.severity === 'high' ? 'text-orange-400' :
                            threat.severity === 'medium' ? 'text-yellow-400' :
                            'text-gray-400'
                          }`}>
                            {threat.severity.toUpperCase()}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Confidence:</span>
                          <span className="text-gray-200">{threat.confidence}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Attack Vector:</span>
                          <span className="text-gray-200">{threat.attackVector}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-gray-800/50 border border-gray-600 rounded-lg p-4">
                      <h3 className="text-lg font-medium text-gray-200 mb-3">Network Information</h3>
                      <div className="space-y-3 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Source IP:</span>
                          <span className="text-gray-200 font-mono">{threat.source}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Target:</span>
                          <span className="text-gray-200 font-mono">{threat.target}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Protocol:</span>
                          <span className="text-gray-200">{threat.protocol}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Port:</span>
                          <span className="text-gray-200">{threat.port}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Location:</span>
                          <span className="text-gray-200">{threat.location}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="bg-gray-800/50 border border-gray-600 rounded-lg p-4">
                      <h3 className="text-lg font-medium text-gray-200 mb-3">Timeline</h3>
                      <div className="space-y-3 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">First Detected:</span>
                          <span className="text-gray-200">{threat.detected}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Last Activity:</span>
                          <span className="text-gray-200">{threat.lastActivity}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Status:</span>
                          <span className={`font-medium ${
                            threat.status === 'active' ? 'text-red-400' :
                            threat.status === 'mitigating' ? 'text-yellow-400' :
                            'text-green-400'
                          }`}>
                            {threat.status.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-gray-800/50 border border-gray-600 rounded-lg p-4">
                      <h3 className="text-lg font-medium text-gray-200 mb-3">Impact Assessment</h3>
                      <p className="text-sm text-gray-400">{threat.impact}</p>
                    </div>
                    
                    <div className="bg-gray-800/50 border border-gray-600 rounded-lg p-4">
                      <h3 className="text-lg font-medium text-gray-200 mb-3">Description</h3>
                      <p className="text-sm text-gray-400">{threat.description}</p>
                    </div>
                  </div>
                </div>
              );
            })()}
            
            <div className="flex justify-end space-x-4 mt-6">
              <button 
                onClick={() => setShowDetailsModal(false)}
                className="px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-600 text-gray-400 rounded-lg transition-colors"
              >
                CLOSE
              </button>
              <button 
                onClick={() => {
                  if (selectedThreat) mitigateThreat(selectedThreat);
                  setShowDetailsModal(false);
                }}
                className="px-4 py-2 bg-yellow-900/30 hover:bg-yellow-900/50 border border-yellow-600 text-yellow-400 rounded-lg transition-all"
              >
                MITIGATE THREAT
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}