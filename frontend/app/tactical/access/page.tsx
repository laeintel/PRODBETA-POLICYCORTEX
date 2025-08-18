'use client';

import React, { useState, useEffect } from 'react';
import { 
  Lock, Users, UserCheck, Shield, Key, AlertTriangle, CheckCircle,
  XCircle, Clock, Eye, EyeOff, Search, Filter, RefreshCw, Download,
  Settings, Plus, Edit, Trash2, UserX, UserPlus, Activity, Globe,
  Database, Server, FileText, ChevronRight, ArrowRight, Info
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface AccessPolicy {
  id: string;
  name: string;
  type: 'role' | 'custom' | 'system';
  permissions: string[];
  description: string;
  assignedUsers: number;
  assignedGroups: number;
  createdDate: string;
  lastModified: string;
  status: 'active' | 'inactive' | 'review';
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  compliance: boolean;
}

interface RoleAssignment {
  id: string;
  principalId: string;
  principalName: string;
  principalType: 'user' | 'group' | 'service_principal';
  roleName: string;
  roleId: string;
  scope: string;
  scopeType: 'management_group' | 'subscription' | 'resource_group' | 'resource';
  assignedDate: string;
  assignedBy: string;
  expiryDate?: string;
  justification?: string;
  lastUsed?: string;
  status: 'active' | 'expired' | 'pending' | 'revoked';
  privilegeLevel: 'read' | 'write' | 'owner' | 'contributor';
}

interface AccessRequest {
  id: string;
  requesterId: string;
  requesterName: string;
  requestType: 'new_access' | 'privilege_elevation' | 'temporary_access';
  resource: string;
  requestedRole: string;
  justification: string;
  duration?: number;
  status: 'pending' | 'approved' | 'denied' | 'expired';
  approver?: string;
  requestDate: string;
  reviewDate?: string;
  riskScore: number;
}

interface PrivilegedAccount {
  id: string;
  accountName: string;
  accountType: 'admin' | 'service' | 'emergency';
  lastUsed: string;
  sessionsToday: number;
  criticalActions: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  mfaEnabled: boolean;
  justInTimeAccess: boolean;
}

export default function AccessControl() {
  const [policies, setPolicies] = useState<AccessPolicy[]>([]);
  const [assignments, setAssignments] = useState<RoleAssignment[]>([]);
  const [requests, setRequests] = useState<AccessRequest[]>([]);
  const [privilegedAccounts, setPrivilegedAccounts] = useState<PrivilegedAccount[]>([]);
  const [selectedPolicy, setSelectedPolicy] = useState<AccessPolicy | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'policies' | 'assignments' | 'requests' | 'privileged'>('policies');
  const [filterStatus, setFilterStatus] = useState('all');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    // Initialize with comprehensive mock data
    setPolicies([
      {
        id: 'POL-001',
        name: 'Owner Role',
        type: 'role',
        permissions: ['read', 'write', 'delete', 'manage_access', 'manage_billing'],
        description: 'Full control over all resources including access management',
        assignedUsers: 3,
        assignedGroups: 1,
        createdDate: '2023-01-15',
        lastModified: '2024-01-20',
        status: 'active',
        riskLevel: 'critical',
        compliance: true
      },
      {
        id: 'POL-002',
        name: 'Contributor Role',
        type: 'role',
        permissions: ['read', 'write', 'delete'],
        description: 'Can manage resources but not access control',
        assignedUsers: 45,
        assignedGroups: 8,
        createdDate: '2023-01-15',
        lastModified: '2024-01-18',
        status: 'active',
        riskLevel: 'medium',
        compliance: true
      },
      {
        id: 'POL-003',
        name: 'Reader Role',
        type: 'role',
        permissions: ['read'],
        description: 'Read-only access to resources',
        assignedUsers: 127,
        assignedGroups: 15,
        createdDate: '2023-01-15',
        lastModified: '2024-01-10',
        status: 'active',
        riskLevel: 'low',
        compliance: true
      },
      {
        id: 'POL-004',
        name: 'Security Administrator',
        type: 'system',
        permissions: ['manage_security', 'view_audit_logs', 'manage_policies', 'incident_response'],
        description: 'Manage security policies and incident response',
        assignedUsers: 5,
        assignedGroups: 2,
        createdDate: '2023-02-01',
        lastModified: '2024-01-22',
        status: 'active',
        riskLevel: 'high',
        compliance: true
      },
      {
        id: 'POL-005',
        name: 'Custom Database Admin',
        type: 'custom',
        permissions: ['database_read', 'database_write', 'backup_restore', 'performance_tuning'],
        description: 'Custom role for database administration',
        assignedUsers: 8,
        assignedGroups: 1,
        createdDate: '2023-06-15',
        lastModified: '2024-01-15',
        status: 'review',
        riskLevel: 'high',
        compliance: false
      }
    ]);

    setAssignments([
      {
        id: 'ASSIGN-001',
        principalId: 'user-001',
        principalName: 'john.admin@company.com',
        principalType: 'user',
        roleName: 'Owner',
        roleId: 'POL-001',
        scope: '/subscriptions/sub-001',
        scopeType: 'subscription',
        assignedDate: '2024-01-10',
        assignedBy: 'system.admin@company.com',
        lastUsed: '2 hours ago',
        status: 'active',
        privilegeLevel: 'owner'
      },
      {
        id: 'ASSIGN-002',
        principalId: 'svc-001',
        principalName: 'backup-service',
        principalType: 'service_principal',
        roleName: 'Contributor',
        roleId: 'POL-002',
        scope: '/subscriptions/sub-001/resourceGroups/rg-backup',
        scopeType: 'resource_group',
        assignedDate: '2023-12-01',
        assignedBy: 'ops.team@company.com',
        lastUsed: '5 minutes ago',
        status: 'active',
        privilegeLevel: 'contributor'
      },
      {
        id: 'ASSIGN-003',
        principalId: 'user-045',
        principalName: 'alice.developer@company.com',
        principalType: 'user',
        roleName: 'Reader',
        roleId: 'POL-003',
        scope: '/subscriptions/sub-001/resourceGroups/rg-dev',
        scopeType: 'resource_group',
        assignedDate: '2024-01-20',
        assignedBy: 'team.lead@company.com',
        expiryDate: '2024-02-20',
        justification: 'Temporary access for project review',
        lastUsed: '1 day ago',
        status: 'active',
        privilegeLevel: 'read'
      },
      {
        id: 'ASSIGN-004',
        principalId: 'group-001',
        principalName: 'Security Team',
        principalType: 'group',
        roleName: 'Security Administrator',
        roleId: 'POL-004',
        scope: '/subscriptions/sub-001',
        scopeType: 'subscription',
        assignedDate: '2023-02-15',
        assignedBy: 'ciso@company.com',
        lastUsed: '30 minutes ago',
        status: 'active',
        privilegeLevel: 'owner'
      },
      {
        id: 'ASSIGN-005',
        principalId: 'user-089',
        principalName: 'bob.contractor@external.com',
        principalType: 'user',
        roleName: 'Custom Database Admin',
        roleId: 'POL-005',
        scope: '/subscriptions/sub-001/resourceGroups/rg-data/providers/Microsoft.Sql/servers/sql-prod',
        scopeType: 'resource',
        assignedDate: '2024-01-15',
        assignedBy: 'db.admin@company.com',
        expiryDate: '2024-01-25',
        justification: 'Database migration project',
        status: 'pending',
        privilegeLevel: 'write'
      }
    ]);

    setRequests([
      {
        id: 'REQ-001',
        requesterId: 'user-156',
        requesterName: 'sarah.analyst@company.com',
        requestType: 'new_access',
        resource: '/subscriptions/sub-001/resourceGroups/rg-analytics',
        requestedRole: 'Contributor',
        justification: 'Need write access for analytics pipeline development',
        status: 'pending',
        requestDate: '2024-01-22T10:30:00Z',
        riskScore: 6.5
      },
      {
        id: 'REQ-002',
        requesterId: 'user-203',
        requesterName: 'mike.support@company.com',
        requestType: 'privilege_elevation',
        resource: '/subscriptions/sub-001/resourceGroups/rg-prod',
        requestedRole: 'Owner',
        justification: 'Emergency production issue requires admin access',
        duration: 4,
        status: 'approved',
        approver: 'oncall.lead@company.com',
        requestDate: '2024-01-22T08:15:00Z',
        reviewDate: '2024-01-22T08:20:00Z',
        riskScore: 8.2
      },
      {
        id: 'REQ-003',
        requesterId: 'user-089',
        requesterName: 'lisa.vendor@partner.com',
        requestType: 'temporary_access',
        resource: '/subscriptions/sub-001/resourceGroups/rg-test',
        requestedRole: 'Reader',
        justification: 'Security audit review',
        duration: 24,
        status: 'denied',
        approver: 'security.team@company.com',
        requestDate: '2024-01-21T14:00:00Z',
        reviewDate: '2024-01-21T16:30:00Z',
        riskScore: 4.1
      }
    ]);

    setPrivilegedAccounts([
      {
        id: 'PRIV-001',
        accountName: 'admin@company.com',
        accountType: 'admin',
        lastUsed: '10 minutes ago',
        sessionsToday: 5,
        criticalActions: 2,
        riskLevel: 'medium',
        mfaEnabled: true,
        justInTimeAccess: true
      },
      {
        id: 'PRIV-002',
        accountName: 'emergency-break-glass',
        accountType: 'emergency',
        lastUsed: '45 days ago',
        sessionsToday: 0,
        criticalActions: 0,
        riskLevel: 'low',
        mfaEnabled: true,
        justInTimeAccess: false
      },
      {
        id: 'PRIV-003',
        accountName: 'backup-service-principal',
        accountType: 'service',
        lastUsed: '1 hour ago',
        sessionsToday: 24,
        criticalActions: 156,
        riskLevel: 'high',
        mfaEnabled: false,
        justInTimeAccess: false
      }
    ]);

    // Simulate real-time updates
    if (autoRefresh) {
      const interval = setInterval(() => {
        setAssignments(prev => 
          prev.map(assignment => ({
            ...assignment,
            lastUsed: assignment.status === 'active' ? 
              `${Math.floor(Math.random() * 60)} minutes ago` : assignment.lastUsed
          }))
        );
      }, 10000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getRiskColor = (level: string) => {
    switch(level) {
      case 'critical': return 'text-red-500 bg-red-900/20';
      case 'high': return 'text-orange-500 bg-orange-900/20';
      case 'medium': return 'text-yellow-500 bg-yellow-900/20';
      case 'low': return 'text-green-500 bg-green-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': return 'text-green-500 bg-green-900/20';
      case 'inactive': return 'text-gray-500 bg-gray-900/20';
      case 'review': return 'text-yellow-500 bg-yellow-900/20';
      case 'pending': return 'text-blue-500 bg-blue-900/20';
      case 'approved': return 'text-green-500 bg-green-900/20';
      case 'denied': return 'text-red-500 bg-red-900/20';
      case 'expired': return 'text-gray-500 bg-gray-900/20';
      case 'revoked': return 'text-red-500 bg-red-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getPrincipalIcon = (type: string) => {
    switch(type) {
      case 'user': return <Users className="w-4 h-4 text-blue-500" />;
      case 'group': return <Users className="w-4 h-4 text-green-500" />;
      case 'service_principal': return <Server className="w-4 h-4 text-purple-500" />;
      default: return <Shield className="w-4 h-4 text-gray-500" />;
    }
  };

  const activeAssignments = assignments.filter(a => a.status === 'active').length;
  const pendingRequests = requests.filter(r => r.status === 'pending').length;
  const highRiskPolicies = policies.filter(p => p.riskLevel === 'high' || p.riskLevel === 'critical').length;
  const nonCompliantPolicies = policies.filter(p => !p.compliance).length;

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Access Control</h1>
            <p className="text-sm text-gray-400 mt-1">Role-based access control and privilege management</p>
          </div>
          
          <div className="flex items-center space-x-3">
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
                viewMode === 'policies' ? 'assignments' : 
                viewMode === 'assignments' ? 'requests' : 
                viewMode === 'requests' ? 'privileged' : 'policies'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
            >
              {viewMode === 'policies' ? 'Assignments' : 
               viewMode === 'assignments' ? 'Requests' : 
               viewMode === 'requests' ? 'Privileged' : 'Policies'}
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Plus className="w-4 h-4" />
              <span>New Policy</span>
            </button>
          </div>
        </div>
      </header>

      {/* Metrics Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <Shield className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Total Policies</p>
              <p className="text-xl font-bold">{policies.length}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <UserCheck className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Active Assignments</p>
              <p className="text-xl font-bold">{activeAssignments}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Clock className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Pending Requests</p>
              <p className="text-xl font-bold text-yellow-500">{pendingRequests}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-5 h-5 text-orange-500" />
            <div>
              <p className="text-xs text-gray-400">High Risk</p>
              <p className="text-xl font-bold text-orange-500">{highRiskPolicies}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XCircle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Non-Compliant</p>
              <p className="text-xl font-bold text-red-500">{nonCompliantPolicies}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Key className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Privileged</p>
              <p className="text-xl font-bold">{privilegedAccounts.length}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {viewMode === 'policies' && (
          <>
            <div className="mb-6">
              <div className="flex items-center space-x-3">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search policies..."
                    className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                  />
                </div>
                
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="inactive">Inactive</option>
                  <option value="review">Under Review</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {policies.map(policy => (
                <div 
                  key={policy.id} 
                  className={`bg-gray-900 border border-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-800/50 ${
                    selectedPolicy?.id === policy.id ? 'ring-2 ring-blue-500' : ''
                  }`}
                  onClick={() => setSelectedPolicy(selectedPolicy?.id === policy.id ? null : policy)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center space-x-2 mb-1">
                        <Shield className="w-4 h-4 text-blue-500" />
                        <h3 className="text-sm font-bold">{policy.name}</h3>
                        <span className={`px-2 py-1 text-xs rounded ${getRiskColor(policy.riskLevel)}`}>
                          {policy.riskLevel.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-xs text-gray-400">{policy.description}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 text-xs rounded ${getStatusColor(policy.status)}`}>
                        {policy.status.toUpperCase()}
                      </span>
                      {!policy.compliance && (
                        <AlertTriangle className="w-4 h-4 text-red-500" />
                      )}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3 text-xs mb-3">
                    <div>
                      <p className="text-gray-400">Assigned Users</p>
                      <p className="text-lg font-bold">{policy.assignedUsers}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Assigned Groups</p>
                      <p className="text-lg font-bold">{policy.assignedGroups}</p>
                    </div>
                  </div>
                  
                  {selectedPolicy?.id === policy.id && (
                    <div className="mt-3 pt-3 border-t border-gray-700">
                      <div className="mb-2">
                        <p className="text-xs text-gray-400 mb-1">Permissions:</p>
                        <div className="flex flex-wrap gap-1">
                          {policy.permissions.map(perm => (
                            <span key={perm} className="px-2 py-1 bg-gray-800 text-gray-400 rounded text-xs">
                              {perm}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>Created: {policy.createdDate}</span>
                        <span>Modified: {policy.lastModified}</span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'assignments' && (
          <div className="space-y-3">
            {assignments.map(assignment => (
              <div key={assignment.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getPrincipalIcon(assignment.principalType)}
                    <div>
                      <h4 className="text-sm font-bold">{assignment.principalName}</h4>
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>Role: {assignment.roleName}</span>
                        <span>Scope: {assignment.scopeType}</span>
                        <span>Assigned: {assignment.assignedDate}</span>
                        {assignment.lastUsed && <span>Last used: {assignment.lastUsed}</span>}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 text-xs rounded ${getStatusColor(assignment.status)}`}>
                      {assignment.status.toUpperCase()}
                    </span>
                    <button className="p-1 hover:bg-gray-800 rounded">
                      <Edit className="w-4 h-4 text-gray-500" />
                    </button>
                    <button className="p-1 hover:bg-gray-800 rounded">
                      <Trash2 className="w-4 h-4 text-red-500" />
                    </button>
                  </div>
                </div>
                
                {assignment.justification && (
                  <div className="mt-2 p-2 bg-gray-800 rounded text-xs">
                    <span className="text-gray-400">Justification: </span>
                    <span>{assignment.justification}</span>
                  </div>
                )}
                
                {assignment.expiryDate && (
                  <div className="mt-2 text-xs text-yellow-500">
                    <Clock className="w-3 h-3 inline mr-1" />
                    Expires: {assignment.expiryDate}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {viewMode === 'requests' && (
          <div className="space-y-3">
            {requests.map(request => (
              <div key={request.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <UserPlus className="w-4 h-4 text-blue-500" />
                      <h4 className="text-sm font-bold">{request.requesterName}</h4>
                      <span className={`px-2 py-1 text-xs rounded ${getStatusColor(request.status)}`}>
                        {request.status.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-sm text-gray-400 mb-2">{request.justification}</p>
                    <div className="grid grid-cols-4 gap-4 text-xs text-gray-500">
                      <div>
                        <span className="text-gray-400">Type: </span>
                        <span>{request.requestType.replace('_', ' ')}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Role: </span>
                        <span>{request.requestedRole}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Risk Score: </span>
                        <span className={request.riskScore > 7 ? 'text-red-500' : request.riskScore > 4 ? 'text-yellow-500' : 'text-green-500'}>
                          {request.riskScore.toFixed(1)}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">Duration: </span>
                        <span>{request.duration ? `${request.duration}h` : 'Permanent'}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {request.status === 'pending' && (
                      <>
                        <button className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs">
                          Approve
                        </button>
                        <button className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-xs">
                          Deny
                        </button>
                      </>
                    )}
                  </div>
                </div>
                
                {request.approver && (
                  <div className="mt-2 text-xs text-gray-500">
                    Reviewed by {request.approver} at {request.reviewDate}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {viewMode === 'privileged' && (
          <>
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-3">Privileged Account Management</h3>
              <p className="text-sm text-gray-400">Monitor and manage high-privilege accounts</p>
            </div>
            
            <div className="grid grid-cols-3 gap-4">
              {privilegedAccounts.map(account => (
                <div key={account.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <Key className="w-4 h-4 text-purple-500" />
                      <h4 className="text-sm font-bold">{account.accountName}</h4>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded ${getRiskColor(account.riskLevel)}`}>
                      {account.riskLevel.toUpperCase()}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3 text-xs mb-3">
                    <div>
                      <p className="text-gray-400">Sessions Today</p>
                      <p className="text-lg font-bold">{account.sessionsToday}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Critical Actions</p>
                      <p className="text-lg font-bold text-orange-500">{account.criticalActions}</p>
                    </div>
                  </div>
                  
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Last Used:</span>
                      <span>{account.lastUsed}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">MFA Enabled:</span>
                      {account.mfaEnabled ? 
                        <CheckCircle className="w-3 h-3 text-green-500" /> : 
                        <XCircle className="w-3 h-3 text-red-500" />
                      }
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">JIT Access:</span>
                      {account.justInTimeAccess ? 
                        <CheckCircle className="w-3 h-3 text-green-500" /> : 
                        <XCircle className="w-3 h-3 text-red-500" />
                      }
                    </div>
                  </div>
                  
                  <div className="mt-3 pt-3 border-t border-gray-700">
                    <button className="w-full px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                      View Activity
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </>
  );
}