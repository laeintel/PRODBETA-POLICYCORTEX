'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import AuthGuard from '../../../components/AuthGuard';

interface ComplianceData {
  overallScore: number;
  policies: Array<{
    id: string;
    name: string;
    status: 'compliant' | 'non-compliant' | 'warning';
    resources: number;
    lastChecked: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
  }>;
  trends: Array<{ date: string; score: number }>;
  violations: Array<{
    id: string;
    resource: string;
    policy: string;
    severity: string;
    detected: string;
    status: 'open' | 'investigating' | 'resolved';
  }>;
}

export default function ComplianceControl() {
  return (
    <AuthGuard requireAuth={true}>
      <ComplianceControlContent />
    </AuthGuard>
  );
}

function ComplianceControlContent() {
  const [data, setData] = useState<ComplianceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedView, setSelectedView] = useState<'overview' | 'policies' | 'violations'>('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    fetchComplianceData();
    const interval = autoRefresh ? setInterval(fetchComplianceData, 30000) : null;
    return () => { if (interval) clearInterval(interval); };
  }, [autoRefresh]);

  const fetchComplianceData = async () => {
    try {
      // In production, this would call the real API
      const response = await fetch('/api/v1/compliance');
      if (response.ok) {
        const data = await response.json();
        setData(data);
      } else {
        // Mock data for demo
        setData(getMockComplianceData());
      }
    } catch (error) {
      // Use mock data if API is not available
      setData(getMockComplianceData());
    } finally {
      setLoading(false);
    }
  };

  const getMockComplianceData = (): ComplianceData => ({
    overallScore: 98.7,
    policies: [
      { id: '1', name: 'Data Encryption at Rest', status: 'compliant', resources: 342, lastChecked: '2 min ago', severity: 'critical' },
      { id: '2', name: 'Network Security Groups', status: 'compliant', resources: 89, lastChecked: '5 min ago', severity: 'high' },
      { id: '3', name: 'Key Vault Access Policy', status: 'warning', resources: 12, lastChecked: '1 min ago', severity: 'high' },
      { id: '4', name: 'SQL Auditing Enabled', status: 'non-compliant', resources: 3, lastChecked: '3 min ago', severity: 'critical' },
      { id: '5', name: 'Storage Account HTTPS', status: 'compliant', resources: 156, lastChecked: '8 min ago', severity: 'medium' },
      { id: '6', name: 'VM Backup Policy', status: 'compliant', resources: 78, lastChecked: '12 min ago', severity: 'medium' },
    ],
    trends: [
      { date: '00:00', score: 96.2 },
      { date: '04:00', score: 97.1 },
      { date: '08:00', score: 97.8 },
      { date: '12:00', score: 98.3 },
      { date: '16:00', score: 98.5 },
      { date: '20:00', score: 98.7 },
    ],
    violations: [
      { id: 'v1', resource: 'sql-prod-01', policy: 'SQL Auditing Enabled', severity: 'critical', detected: '15 min ago', status: 'open' },
      { id: 'v2', resource: 'kv-app-02', policy: 'Key Vault Access Policy', severity: 'high', detected: '1 hour ago', status: 'investigating' },
      { id: 'v3', resource: 'sql-dev-03', policy: 'SQL Auditing Enabled', severity: 'critical', detected: '2 hours ago', status: 'open' },
    ],
  });

  const runComplianceScan = async () => {
    setLoading(true);
    // Call backend API to trigger scan
    try {
      await fetch('/api/v1/compliance/scan', { method: 'POST' });
    } catch (error) {
      console.error('Scan failed:', error);
    }
    await fetchComplianceData();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-gray-400">LOADING COMPLIANCE DATA...</p>
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
              <h1 className="text-xl font-bold">COMPLIANCE CONTROL CENTER</h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="autoRefresh"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="rounded border-gray-600 bg-gray-800 text-blue-600 focus:ring-blue-500"
                />
                <label htmlFor="autoRefresh" className="text-xs text-gray-400">
                  AUTO-REFRESH
                </label>
              </div>
              <button
                onClick={runComplianceScan}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors"
              >
                RUN FULL SCAN
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Score Overview */}
      <div className="px-6 py-6">
        <div className="grid grid-cols-4 gap-6 mb-6">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <p className="text-sm text-gray-500 uppercase mb-2">Overall Compliance</p>
            <div className="flex items-end justify-between">
              <p className="text-5xl font-bold font-mono text-green-500">
                {data?.overallScore.toFixed(1)}%
              </p>
              <div className="text-xs text-green-500">↑ 0.2%</div>
            </div>
            <div className="mt-4 h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-green-600 to-green-400 rounded-full"
                style={{ width: `${data?.overallScore}%` }}
              />
            </div>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <p className="text-sm text-gray-500 uppercase mb-2">Compliant Resources</p>
            <p className="text-3xl font-bold font-mono">2,789</p>
            <p className="text-xs text-gray-400 mt-2">OF 2,847 TOTAL</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <p className="text-sm text-gray-500 uppercase mb-2">Active Violations</p>
            <p className="text-3xl font-bold font-mono text-red-500">
              {data?.violations.filter(v => v.status === 'open').length}
            </p>
            <p className="text-xs text-yellow-500 mt-2">
              {data?.violations.filter(v => v.status === 'investigating').length} INVESTIGATING
            </p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <p className="text-sm text-gray-500 uppercase mb-2">Policy Coverage</p>
            <p className="text-3xl font-bold font-mono">412</p>
            <p className="text-xs text-blue-500 mt-2">14 NEW THIS WEEK</p>
          </div>
        </div>

        {/* View Tabs */}
        <div className="flex space-x-1 mb-6 bg-gray-900 p-1 rounded-lg w-fit">
          {(['overview', 'policies', 'violations'] as const).map((view) => (
            <button
              key={view}
              onClick={() => setSelectedView(view)}
              className={`px-4 py-2 text-sm font-medium rounded transition-colors ${
                selectedView === view
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
              }`}
            >
              {view.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Content Area */}
        {selectedView === 'overview' && (
          <div className="grid grid-cols-2 gap-6">
            {/* Compliance Trend */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">24-HOUR TREND</h3>
              <div className="h-48 flex items-end justify-between">
                {data?.trends.map((point, i) => (
                  <div key={i} className="flex-1 flex flex-col items-center">
                    <div
                      className="w-full mx-0.5 bg-blue-600 rounded-t"
                      style={{ height: `${(point.score - 95) * 40}%` }}
                    />
                    <p className="text-xs text-gray-600 mt-2">{point.date}</p>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Recent Violations */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">RECENT VIOLATIONS</h3>
              <div className="space-y-3">
                {data?.violations.slice(0, 3).map((violation) => (
                  <div key={violation.id} className="flex items-center justify-between py-2 border-b border-gray-800">
                    <div>
                      <p className="text-sm font-medium">{violation.resource}</p>
                      <p className="text-xs text-gray-500">{violation.policy}</p>
                    </div>
                    <div className="text-right">
                      <span className={`text-xs px-2 py-1 rounded ${
                        violation.severity === 'critical'
                          ? 'bg-red-900/30 text-red-500'
                          : 'bg-yellow-900/30 text-yellow-500'
                      }`}>
                        {violation.severity.toUpperCase()}
                      </span>
                      <p className="text-xs text-gray-600 mt-1">{violation.detected}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {selectedView === 'policies' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg">
            <table className="w-full">
              <thead className="border-b border-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Policy Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Resources</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Severity</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Last Checked</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {data?.policies.map((policy) => (
                  <tr key={policy.id} className="hover:bg-gray-800/50 transition-colors">
                    <td className="px-6 py-4 text-sm">{policy.name}</td>
                    <td className="px-6 py-4">
                      <span className={`text-xs px-2 py-1 rounded font-medium ${
                        policy.status === 'compliant'
                          ? 'bg-green-900/30 text-green-500'
                          : policy.status === 'warning'
                          ? 'bg-yellow-900/30 text-yellow-500'
                          : 'bg-red-900/30 text-red-500'
                      }`}>
                        {policy.status.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm font-mono">{policy.resources}</td>
                    <td className="px-6 py-4">
                      <span className={`text-xs ${
                        policy.severity === 'critical' ? 'text-red-500' :
                        policy.severity === 'high' ? 'text-orange-500' :
                        policy.severity === 'medium' ? 'text-yellow-500' :
                        'text-gray-500'
                      }`}>
                        {policy.severity.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">{policy.lastChecked}</td>
                    <td className="px-6 py-4">
                      <button className="text-xs text-blue-500 hover:text-blue-400">
                        VIEW DETAILS
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {selectedView === 'violations' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg">
            <div className="p-4 border-b border-gray-800">
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-400">
                  {data?.violations.length} ACTIVE VIOLATIONS
                </p>
                <button className="text-xs px-3 py-1 bg-red-900/30 hover:bg-red-900/50 border border-red-800 rounded text-red-400 transition-colors">
                  REMEDIATE ALL
                </button>
              </div>
            </div>
            <div className="divide-y divide-gray-800">
              {data?.violations.map((violation) => (
                <div key={violation.id} className="p-4 hover:bg-gray-800/50 transition-colors">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="font-medium">{violation.resource}</p>
                      <p className="text-sm text-gray-500 mt-1">{violation.policy}</p>
                      <p className="text-xs text-gray-600 mt-2">Detected: {violation.detected}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`text-xs px-2 py-1 rounded ${
                        violation.status === 'open'
                          ? 'bg-red-900/30 text-red-500'
                          : violation.status === 'investigating'
                          ? 'bg-yellow-900/30 text-yellow-500'
                          : 'bg-green-900/30 text-green-500'
                      }`}>
                        {violation.status.toUpperCase()}
                      </span>
                      <button className="text-xs px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-gray-300 transition-colors">
                        INVESTIGATE
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}