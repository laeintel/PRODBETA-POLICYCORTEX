'use client';

import { Card } from '@/components/ui/card';
import { Shield, CheckCircle, XCircle, AlertTriangle, TrendingUp, FileText, GitBranch } from 'lucide-react';
import { useState, useEffect } from 'react';

interface Policy {
  id: string;
  name: string;
  category: string;
  status: 'compliant' | 'non-compliant' | 'warning';
  resourcesAffected: number;
  lastScanned: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export default function GovernancePage() {
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [stats, setStats] = useState({
    totalPolicies: 0,
    compliant: 0,
    nonCompliant: 0,
    warnings: 0
  });

  useEffect(() => {
    // Mock data
    const mockPolicies: Policy[] = [
      {
        id: 'pol-001',
        name: 'Require HTTPS for all web apps',
        category: 'Security',
        status: 'compliant',
        resourcesAffected: 45,
        lastScanned: '2 hours ago',
        severity: 'high'
      },
      {
        id: 'pol-002',
        name: 'Enable backup for databases',
        category: 'Reliability',
        status: 'non-compliant',
        resourcesAffected: 12,
        lastScanned: '1 hour ago',
        severity: 'critical'
      },
      {
        id: 'pol-003',
        name: 'Tag all resources with cost center',
        category: 'Cost Management',
        status: 'warning',
        resourcesAffected: 156,
        lastScanned: '3 hours ago',
        severity: 'medium'
      },
      {
        id: 'pol-004',
        name: 'Enable diagnostic logs',
        category: 'Operations',
        status: 'compliant',
        resourcesAffected: 89,
        lastScanned: '30 minutes ago',
        severity: 'medium'
      },
      {
        id: 'pol-005',
        name: 'Encrypt data at rest',
        category: 'Security',
        status: 'compliant',
        resourcesAffected: 67,
        lastScanned: '1 hour ago',
        severity: 'critical'
      },
      {
        id: 'pol-006',
        name: 'Restrict public network access',
        category: 'Security',
        status: 'non-compliant',
        resourcesAffected: 8,
        lastScanned: '45 minutes ago',
        severity: 'critical'
      }
    ];

    setPolicies(mockPolicies);
    
    // Calculate stats
    setStats({
      totalPolicies: mockPolicies.length,
      compliant: mockPolicies.filter(p => p.status === 'compliant').length,
      nonCompliant: mockPolicies.filter(p => p.status === 'non-compliant').length,
      warnings: mockPolicies.filter(p => p.status === 'warning').length
    });
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'compliant':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'non-compliant':
        return <XCircle className="h-5 w-5 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      default:
        return null;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-600 bg-red-50 dark:bg-red-950';
      case 'high':
        return 'text-orange-600 bg-orange-50 dark:bg-orange-950';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-950';
      case 'low':
        return 'text-green-600 bg-green-50 dark:bg-green-950';
      default:
        return 'text-gray-600 bg-gray-50 dark:bg-gray-950';
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Shield className="h-8 w-8" />
          Governance & Compliance
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Monitor and enforce governance policies across your cloud infrastructure
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Policies</p>
              <p className="text-2xl font-bold">{stats.totalPolicies}</p>
            </div>
            <FileText className="h-8 w-8 text-gray-400" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Compliant</p>
              <p className="text-2xl font-bold text-green-600">{stats.compliant}</p>
            </div>
            <CheckCircle className="h-8 w-8 text-green-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Non-Compliant</p>
              <p className="text-2xl font-bold text-red-600">{stats.nonCompliant}</p>
            </div>
            <XCircle className="h-8 w-8 text-red-500" />
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Warnings</p>
              <p className="text-2xl font-bold text-yellow-600">{stats.warnings}</p>
            </div>
            <AlertTriangle className="h-8 w-8 text-yellow-500" />
          </div>
        </Card>
      </div>

      {/* Compliance Score */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Overall Compliance Score</h2>
          <TrendingUp className="h-5 w-5 text-green-500" />
        </div>
        <div className="flex items-center gap-4">
          <div className="text-4xl font-bold">
            {Math.round((stats.compliant / stats.totalPolicies) * 100)}%
          </div>
          <div className="flex-1">
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
              <div 
                className="bg-gradient-to-r from-blue-500 to-green-500 h-4 rounded-full"
                style={{ width: `${(stats.compliant / stats.totalPolicies) * 100}%` }}
              />
            </div>
          </div>
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
          Based on {stats.totalPolicies} active policies across all resources
        </p>
      </Card>

      {/* Policies List */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold">Active Policies</h2>
          <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm">
            Create Policy
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="border-b border-gray-200 dark:border-gray-700">
              <tr>
                <th className="text-left py-3 px-4">Policy Name</th>
                <th className="text-left py-3 px-4">Category</th>
                <th className="text-left py-3 px-4">Status</th>
                <th className="text-left py-3 px-4">Severity</th>
                <th className="text-left py-3 px-4">Resources</th>
                <th className="text-left py-3 px-4">Last Scan</th>
                <th className="text-left py-3 px-4">Actions</th>
              </tr>
            </thead>
            <tbody>
              {policies.map(policy => (
                <tr key={policy.id} className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900">
                  <td className="py-3 px-4">
                    <div className="font-medium">{policy.name}</div>
                  </td>
                  <td className="py-3 px-4">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {policy.category}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(policy.status)}
                      <span className="text-sm capitalize">{policy.status.replace('-', ' ')}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(policy.severity)}`}>
                      {policy.severity.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <span className="text-sm">{policy.resourcesAffected}</span>
                  </td>
                  <td className="py-3 px-4">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {policy.lastScanned}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <button className="text-blue-600 hover:underline text-sm">View</button>
                      {policy.status === 'non-compliant' && (
                        <button className="text-green-600 hover:underline text-sm flex items-center gap-1">
                          <GitBranch className="h-3 w-3" />
                          Fix
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}