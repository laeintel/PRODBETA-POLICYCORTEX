'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import AuthGuard from '../../../components/AuthGuard';

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
  }>;
  blockedAttempts: number;
  suspiciousActivities: number;
}

export default function SecurityOperationsCenter() {
  return (
    <AuthGuard requireAuth={true}>
      <SecurityOperationsCenterContent />
    </AuthGuard>
  );
}

function SecurityOperationsCenterContent() {
  const [data, setData] = useState<SecurityData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedThreat, setSelectedThreat] = useState<string | null>(null);
  const [streamingLogs, setStreamingLogs] = useState<string[]>([]);

  useEffect(() => {
    fetchSecurityData();
    const interval = setInterval(fetchSecurityData, 10000);
    
    // Simulate streaming security logs
    const logInterval = setInterval(() => {
      const newLog = `[${new Date().toISOString()}] ${generateRandomLog()}`;
      setStreamingLogs(prev => [newLog, ...prev.slice(0, 99)]);
    }, 3000);
    
    return () => {
      clearInterval(interval);
      clearInterval(logInterval);
    };
  }, []);

  const fetchSecurityData = async () => {
    try {
      const response = await fetch('/api/v1/security/threats');
      if (response.ok) {
        const data = await response.json();
        setData(data);
      } else {
        setData(getMockSecurityData());
      }
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
        description: 'Multiple failed SSH login attempts detected'
      },
      {
        id: 't2',
        type: 'SQL Injection Attempt',
        source: '45.142.214.112',
        target: 'sql-prod-01',
        severity: 'critical',
        status: 'mitigating',
        detected: '1 hour ago',
        description: 'Malicious SQL query patterns detected in application logs'
      },
      {
        id: 't3',
        type: 'Unusual Data Transfer',
        source: 'storage-prod-02',
        target: 'External IP',
        severity: 'medium',
        status: 'active',
        detected: '3 hours ago',
        description: 'Large volume of data transfer to unrecognized IP address'
      }
    ],
    securityScore: 87,
    vulnerabilities: {
      critical: 2,
      high: 8,
      medium: 23,
      low: 47
    },
    recentActivity: [
      { timestamp: '10:42:15', event: 'Failed login attempt blocked', severity: 'medium', source: '192.168.1.45' },
      { timestamp: '10:38:22', event: 'Firewall rule updated', severity: 'low', source: 'System' },
      { timestamp: '10:31:48', event: 'DDoS protection activated', severity: 'high', source: 'CDN' },
      { timestamp: '10:27:03', event: 'SSL certificate renewed', severity: 'info', source: 'System' },
    ],
    blockedAttempts: 1847,
    suspiciousActivities: 23
  });

  const generateRandomLog = () => {
    const logs = [
      'Authentication attempt from unknown IP',
      'Firewall rule triggered',
      'Port scan detected',
      'SSL handshake completed',
      'Security policy updated',
      'Intrusion detection alert',
      'Access granted to resource',
      'Security scan completed'
    ];
    return logs[Math.floor(Math.random() * logs.length)];
  };

  const mitigateThreat = async (threatId: string) => {
    try {
      await fetch(`/api/v1/security/threats/${threatId}/mitigate`, { method: 'POST' });
      fetchSecurityData();
    } catch (error) {
      console.error('Failed to mitigate threat:', error);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-red-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-gray-400">INITIALIZING SECURITY SYSTEMS...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/tactical" className="text-gray-400 hover:text-gray-200">
                ← BACK
              </Link>
              <div className="h-6 w-px bg-gray-700" />
              <h1 className="text-xl font-bold">SECURITY OPERATIONS CENTER</h1>
              <div className={`px-3 py-1 rounded text-xs font-bold ${
                data?.threatLevel === 'CRITICAL' ? 'bg-red-900/30 text-red-500 animate-pulse' :
                data?.threatLevel === 'HIGH' ? 'bg-orange-900/30 text-orange-500' :
                data?.threatLevel === 'MEDIUM' ? 'bg-yellow-900/30 text-yellow-500' :
                'bg-green-900/30 text-green-500'
              }`}>
                THREAT LEVEL: {data?.threatLevel}
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded transition-colors">
                EMERGENCY LOCKDOWN
              </button>
              <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors">
                RUN SECURITY SCAN
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="p-6">
        {/* Metrics Row */}
        <div className="grid grid-cols-5 gap-4 mb-6">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Security Score</p>
            <p className="text-3xl font-bold font-mono">{data?.securityScore}%</p>
            <div className="mt-2 h-1 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full"
                style={{ width: `${data?.securityScore}%` }}
              />
            </div>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Active Threats</p>
            <p className="text-3xl font-bold font-mono text-red-500">
              {data?.activeThreats.filter(t => t.status === 'active').length}
            </p>
            <p className="text-xs text-yellow-500 mt-1">
              {data?.activeThreats.filter(t => t.status === 'mitigating').length} mitigating
            </p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Blocked Today</p>
            <p className="text-3xl font-bold font-mono text-green-500">{data?.blockedAttempts}</p>
            <p className="text-xs text-gray-500 mt-1">attempts</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Vulnerabilities</p>
            <div className="flex items-center space-x-2 mt-1">
              <span className="text-red-500 font-bold">{data?.vulnerabilities.critical}C</span>
              <span className="text-orange-500 font-bold">{data?.vulnerabilities.high}H</span>
              <span className="text-yellow-500 font-bold">{data?.vulnerabilities.medium}M</span>
              <span className="text-gray-500">{data?.vulnerabilities.low}L</span>
            </div>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Suspicious</p>
            <p className="text-3xl font-bold font-mono text-yellow-500">{data?.suspiciousActivities}</p>
            <p className="text-xs text-gray-500 mt-1">activities</p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-6">
          {/* Active Threats */}
          <div className="col-span-2 bg-gray-900 border border-gray-800 rounded-lg">
            <div className="p-4 border-b border-gray-800">
              <h3 className="text-sm font-bold text-gray-400 uppercase">ACTIVE THREAT MONITOR</h3>
            </div>
            <div className="divide-y divide-gray-800">
              {data?.activeThreats.map((threat) => (
                <div
                  key={threat.id}
                  className={`p-4 hover:bg-gray-800/50 transition-colors cursor-pointer ${
                    selectedThreat === threat.id ? 'bg-gray-800/30' : ''
                  }`}
                  onClick={() => setSelectedThreat(threat.id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <span className={`w-2 h-2 rounded-full ${
                          threat.severity === 'critical' ? 'bg-red-500 animate-pulse' :
                          threat.severity === 'high' ? 'bg-orange-500' :
                          threat.severity === 'medium' ? 'bg-yellow-500' :
                          'bg-gray-500'
                        }`} />
                        <h4 className="font-medium">{threat.type}</h4>
                        <span className={`text-xs px-2 py-0.5 rounded ${
                          threat.status === 'active' ? 'bg-red-900/30 text-red-500' :
                          threat.status === 'mitigating' ? 'bg-yellow-900/30 text-yellow-500' :
                          'bg-green-900/30 text-green-500'
                        }`}>
                          {threat.status.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-sm text-gray-500 mt-1">{threat.description}</p>
                      <div className="flex items-center space-x-4 mt-2 text-xs text-gray-600">
                        <span>Source: <span className="text-gray-400 font-mono">{threat.source}</span></span>
                        <span>Target: <span className="text-gray-400 font-mono">{threat.target}</span></span>
                        <span>Detected: <span className="text-gray-400">{threat.detected}</span></span>
                      </div>
                    </div>
                    <div className="flex space-x-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          mitigateThreat(threat.id);
                        }}
                        className="px-3 py-1 bg-yellow-900/30 hover:bg-yellow-900/50 border border-yellow-800 rounded text-yellow-500 text-xs transition-colors"
                      >
                        MITIGATE
                      </button>
                      <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-gray-400 text-xs transition-colors">
                        DETAILS
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Security Log Stream */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg">
            <div className="p-4 border-b border-gray-800 flex items-center justify-between">
              <h3 className="text-sm font-bold text-gray-400 uppercase">SECURITY LOG STREAM</h3>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-xs text-gray-500">LIVE</span>
              </div>
            </div>
            <div className="p-4 h-96 overflow-y-auto font-mono text-xs">
              {streamingLogs.map((log, i) => (
                <div
                  key={i}
                  className={`py-1 ${
                    log.includes('blocked') || log.includes('Failed') ? 'text-red-400' :
                    log.includes('alert') || log.includes('detected') ? 'text-yellow-400' :
                    'text-gray-500'
                  }`}
                >
                  {log}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="mt-6 bg-gray-900 border border-gray-800 rounded-lg">
          <div className="p-4 border-b border-gray-800">
            <h3 className="text-sm font-bold text-gray-400 uppercase">RECENT SECURITY EVENTS</h3>
          </div>
          <table className="w-full">
            <thead className="border-b border-gray-800">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Event</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Severity</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Source</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {data?.recentActivity.map((activity, i) => (
                <tr key={i} className="hover:bg-gray-800/50 transition-colors">
                  <td className="px-4 py-3 text-sm font-mono text-gray-500">{activity.timestamp}</td>
                  <td className="px-4 py-3 text-sm">{activity.event}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      activity.severity === 'high' ? 'bg-red-900/30 text-red-500' :
                      activity.severity === 'medium' ? 'bg-yellow-900/30 text-yellow-500' :
                      activity.severity === 'low' ? 'bg-green-900/30 text-green-500' :
                      'bg-gray-800 text-gray-500'
                    }`}>
                      {activity.severity.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm font-mono text-gray-500">{activity.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}