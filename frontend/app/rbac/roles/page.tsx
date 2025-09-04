'use client';

import React, { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  Shield, Users, Key, Lock, Plus, Edit, Trash2, Copy,
  Search, Filter, Download, Upload, ChevronRight, ChevronDown,
  AlertCircle, CheckCircle, Clock, Settings, Building,
  UserCheck, FileText, Database, Cloud, Code, Globe,
  Package, Terminal, Layers, Award, ShieldCheck, Info,
  Eye, EyeOff, RefreshCw, Save, X, ArrowLeft
} from 'lucide-react';
import ResponsiveGrid, { ResponsiveContainer } from '@/components/ResponsiveGrid';
import { toast } from '@/hooks/useToast';

interface Role {
  id: string;
  name: string;
  displayName: string;
  description: string;
  type: 'BuiltInRole' | 'CustomRole';
  assignableScopes: string[];
  permissions: Permission[];
  assignedUsers: number;
  assignedGroups: number;
  assignedServicePrincipals: number;
  createdOn: string;
  updatedOn: string;
  createdBy: string;
  isDeprecated: boolean;
  riskScore: number;
}

interface Permission {
  id: string;
  actions: string[];
  notActions: string[];
  dataActions?: string[];
  notDataActions?: string[];
  condition?: string;
}

interface Assignment {
  id: string;
  principalId: string;
  principalName: string;
  principalType: 'User' | 'Group' | 'ServicePrincipal';
  roleDefinitionId: string;
  scope: string;
  createdOn: string;
  updatedOn: string;
}

export default function RolesPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const roleId = searchParams.get('id');
  const action = searchParams.get('action');

  const [roles, setRoles] = useState<Role[]>([]);
  const [selectedRole, setSelectedRole] = useState<Role | null>(null);
  const [assignments, setAssignments] = useState<Assignment[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'builtin' | 'custom'>('all');
  const [showCreateModal, setShowCreateModal] = useState(action === 'create');
  const [expandedPermissions, setExpandedPermissions] = useState<Set<string>>(new Set());
  const [isEditMode, setIsEditMode] = useState(false);

  useEffect(() => {
    // Mock data for roles
    const mockRoles: Role[] = [
      {
        id: 'role-001',
        name: 'Contributor',
        displayName: 'Contributor',
        description: 'Grants full access to manage all resources, but does not allow you to assign roles in Azure RBAC, manage assignments in Azure Blueprints, or share image galleries.',
        type: 'BuiltInRole',
        assignableScopes: ['/'],
        permissions: [
          {
            id: 'perm-001',
            actions: ['*'],
            notActions: [
              'Microsoft.Authorization/*/Delete',
              'Microsoft.Authorization/*/Write',
              'Microsoft.Authorization/elevateAccess/Action',
              'Microsoft.Blueprint/blueprintAssignments/write',
              'Microsoft.Blueprint/blueprintAssignments/delete'
            ]
          }
        ],
        assignedUsers: 342,
        assignedGroups: 28,
        assignedServicePrincipals: 15,
        createdOn: '2015-10-01T00:00:00Z',
        updatedOn: '2024-11-01T00:00:00Z',
        createdBy: 'System',
        isDeprecated: false,
        riskScore: 35
      },
      {
        id: 'role-002',
        name: 'Owner',
        displayName: 'Owner',
        description: 'Grants full access to manage all resources, including the ability to assign roles in Azure RBAC.',
        type: 'BuiltInRole',
        assignableScopes: ['/'],
        permissions: [
          {
            id: 'perm-002',
            actions: ['*'],
            notActions: []
          }
        ],
        assignedUsers: 12,
        assignedGroups: 3,
        assignedServicePrincipals: 2,
        createdOn: '2015-10-01T00:00:00Z',
        updatedOn: '2024-11-01T00:00:00Z',
        createdBy: 'System',
        isDeprecated: false,
        riskScore: 95
      },
      {
        id: 'role-003',
        name: 'Reader',
        displayName: 'Reader',
        description: 'View all resources, but does not allow you to make any changes.',
        type: 'BuiltInRole',
        assignableScopes: ['/'],
        permissions: [
          {
            id: 'perm-003',
            actions: ['*/read'],
            notActions: []
          }
        ],
        assignedUsers: 892,
        assignedGroups: 67,
        assignedServicePrincipals: 34,
        createdOn: '2015-10-01T00:00:00Z',
        updatedOn: '2024-11-01T00:00:00Z',
        createdBy: 'System',
        isDeprecated: false,
        riskScore: 10
      },
      {
        id: 'role-004',
        name: 'CustomDevOpsEngineer',
        displayName: 'DevOps Engineer',
        description: 'Custom role for DevOps team with specific permissions for CI/CD pipelines and infrastructure management.',
        type: 'CustomRole',
        assignableScopes: [
          '/subscriptions/12345678-1234-1234-1234-123456789012',
          '/subscriptions/87654321-4321-4321-4321-210987654321'
        ],
        permissions: [
          {
            id: 'perm-004',
            actions: [
              'Microsoft.Compute/*',
              'Microsoft.Storage/*',
              'Microsoft.Network/*',
              'Microsoft.DevTestLab/*',
              'Microsoft.ContainerRegistry/*'
            ],
            notActions: [
              'Microsoft.Compute/virtualMachines/delete',
              'Microsoft.Storage/storageAccounts/delete'
            ],
            dataActions: [
              'Microsoft.Storage/storageAccounts/blobServices/containers/blobs/*'
            ],
            notDataActions: []
          }
        ],
        assignedUsers: 45,
        assignedGroups: 5,
        assignedServicePrincipals: 8,
        createdOn: '2024-06-15T10:30:00Z',
        updatedOn: '2024-11-28T14:22:00Z',
        createdBy: 'admin@company.com',
        isDeprecated: false,
        riskScore: 42
      },
      {
        id: 'role-005',
        name: 'SecurityAuditor',
        displayName: 'Security Auditor',
        description: 'Custom role for security team to audit and review security configurations.',
        type: 'CustomRole',
        assignableScopes: ['/subscriptions/12345678-1234-1234-1234-123456789012'],
        permissions: [
          {
            id: 'perm-005',
            actions: [
              '*/read',
              'Microsoft.Security/*',
              'Microsoft.Authorization/*/read',
              'Microsoft.PolicyInsights/*',
              'Microsoft.Advisor/*'
            ],
            notActions: [],
            dataActions: [],
            notDataActions: []
          }
        ],
        assignedUsers: 23,
        assignedGroups: 4,
        assignedServicePrincipals: 3,
        createdOn: '2024-03-20T09:15:00Z',
        updatedOn: '2024-11-25T16:45:00Z',
        createdBy: 'security@company.com',
        isDeprecated: false,
        riskScore: 25
      }
    ];

    // Mock assignments
    const mockAssignments: Assignment[] = [
      {
        id: 'assign-001',
        principalId: 'user-001',
        principalName: 'john.doe@company.com',
        principalType: 'User',
        roleDefinitionId: 'role-001',
        scope: '/subscriptions/12345678-1234-1234-1234-123456789012',
        createdOn: '2024-11-20T10:00:00Z',
        updatedOn: '2024-11-20T10:00:00Z'
      },
      {
        id: 'assign-002',
        principalId: 'group-001',
        principalName: 'DevOps Team',
        principalType: 'Group',
        roleDefinitionId: 'role-004',
        scope: '/subscriptions/12345678-1234-1234-1234-123456789012/resourceGroups/prod-rg',
        createdOn: '2024-11-15T14:30:00Z',
        updatedOn: '2024-11-15T14:30:00Z'
      },
      {
        id: 'assign-003',
        principalId: 'sp-001',
        principalName: 'AKS Service Principal',
        principalType: 'ServicePrincipal',
        roleDefinitionId: 'role-001',
        scope: '/subscriptions/12345678-1234-1234-1234-123456789012/resourceGroups/aks-rg',
        createdOn: '2024-10-01T09:00:00Z',
        updatedOn: '2024-10-01T09:00:00Z'
      }
    ];

    setRoles(mockRoles);
    setAssignments(mockAssignments);

    if (roleId) {
      const role = mockRoles.find(r => r.id === roleId);
      if (role) {
        setSelectedRole(role);
      }
    }
  }, [roleId]);

  const getRiskColor = (score: number) => {
    if (score >= 70) return 'text-red-500 bg-red-500/10';
    if (score >= 40) return 'text-yellow-500 bg-yellow-500/10';
    return 'text-green-500 bg-green-500/10';
  };

  const formatScope = (scope: string) => {
    if (scope === '/') return 'Root Management Group';
    const parts = scope.split('/');
    if (parts.length >= 3) {
      return `${parts[parts.length - 2]}: ${parts[parts.length - 1]}`;
    }
    return scope;
  };

  const togglePermissionExpansion = (permId: string) => {
    const newExpanded = new Set(expandedPermissions);
    if (newExpanded.has(permId)) {
      newExpanded.delete(permId);
    } else {
      newExpanded.add(permId);
    }
    setExpandedPermissions(newExpanded);
  };

  const handleCreateRole = () => {
    toast({ title: 'Create Role', description: 'Opening role creation wizard...' });
    setShowCreateModal(true);
  };

  const handleEditRole = (role: Role) => {
    setSelectedRole(role);
    setIsEditMode(true);
    toast({ title: 'Edit Mode', description: `Editing ${role.displayName}` });
  };

  const handleDeleteRole = (role: Role) => {
    if (role.type === 'BuiltInRole') {
      toast({ 
        title: 'Cannot Delete', 
        description: 'Built-in roles cannot be deleted',
        variant: 'destructive'
      });
      return;
    }
    toast({ title: 'Delete Role', description: `Deleting ${role.displayName}...` });
  };

  const handleCloneRole = (role: Role) => {
    toast({ title: 'Clone Role', description: `Creating copy of ${role.displayName}...` });
    setShowCreateModal(true);
  };

  const filteredRoles = roles.filter(role => {
    const matchesSearch = role.displayName.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         role.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filterType === 'all' ||
                         (filterType === 'builtin' && role.type === 'BuiltInRole') ||
                         (filterType === 'custom' && role.type === 'CustomRole');
    return matchesSearch && matchesFilter;
  });

  if (selectedRole && !showCreateModal) {
    // Role Detail View
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
        <ResponsiveContainer className="py-6">
          {/* Header */}
          <div className="mb-8">
            <button
              onClick={() => {
                setSelectedRole(null);
                setIsEditMode(false);
              }}
              className="mb-4 px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white flex items-center gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Roles
            </button>

            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
                  <Shield className="w-8 h-8 text-purple-500" />
                  {selectedRole.displayName}
                </h1>
                <p className="text-gray-600 dark:text-gray-400">{selectedRole.description}</p>
                <div className="flex items-center gap-3 mt-3">
                  <span className={`px-3 py-1 text-sm rounded-full ${
                    selectedRole.type === 'BuiltInRole' 
                      ? 'bg-blue-500/10 text-blue-500' 
                      : 'bg-purple-500/10 text-purple-500'
                  }`}>
                    {selectedRole.type === 'BuiltInRole' ? 'Built-in Role' : 'Custom Role'}
                  </span>
                  <span className={`px-3 py-1 text-sm rounded-full ${getRiskColor(selectedRole.riskScore)}`}>
                    Risk Score: {selectedRole.riskScore}
                  </span>
                  {selectedRole.isDeprecated && (
                    <span className="px-3 py-1 text-sm rounded-full bg-red-500/10 text-red-500">
                      Deprecated
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-3">
                {!isEditMode ? (
                  <>
                    <button
                      onClick={() => handleCloneRole(selectedRole)}
                      className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center gap-2"
                    >
                      <Copy className="w-4 h-4" />
                      Clone
                    </button>
                    {selectedRole.type === 'CustomRole' && (
                      <>
                        <button
                          onClick={() => handleEditRole(selectedRole)}
                          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2"
                        >
                          <Edit className="w-4 h-4" />
                          Edit
                        </button>
                        <button
                          onClick={() => handleDeleteRole(selectedRole)}
                          className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors flex items-center gap-2"
                        >
                          <Trash2 className="w-4 h-4" />
                          Delete
                        </button>
                      </>
                    )}
                  </>
                ) : (
                  <>
                    <button
                      onClick={() => {
                        setIsEditMode(false);
                        toast({ title: 'Changes Saved', description: 'Role updated successfully' });
                      }}
                      className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center gap-2"
                    >
                      <Save className="w-4 h-4" />
                      Save Changes
                    </button>
                    <button
                      onClick={() => setIsEditMode(false)}
                      className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center gap-2"
                    >
                      <X className="w-4 h-4" />
                      Cancel
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Role Details Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column - Permissions and Scopes */}
            <div className="lg:col-span-2 space-y-6">
              {/* Permissions */}
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Key className="w-5 h-5 text-purple-500" />
                    Permissions
                  </h2>
                </div>
                <div className="p-6">
                  {selectedRole.permissions.map((perm) => (
                    <div key={perm.id} className="mb-4">
                      <button
                        onClick={() => togglePermissionExpansion(perm.id)}
                        className="w-full flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                      >
                        <span className="font-medium">Permission Set {perm.id}</span>
                        <ChevronDown className={`w-4 h-4 transition-transform ${
                          expandedPermissions.has(perm.id) ? 'rotate-180' : ''
                        }`} />
                      </button>
                      
                      {expandedPermissions.has(perm.id) && (
                        <div className="mt-4 space-y-4 pl-4">
                          {perm.actions.length > 0 && (
                            <div>
                              <h4 className="text-sm font-medium text-green-600 dark:text-green-400 mb-2">
                                Allowed Actions ({perm.actions.length})
                              </h4>
                              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
                                <ul className="space-y-1">
                                  {perm.actions.slice(0, 5).map((action, idx) => (
                                    <li key={idx} className="text-sm font-mono text-gray-700 dark:text-gray-300">
                                      {action}
                                    </li>
                                  ))}
                                  {perm.actions.length > 5 && (
                                    <li className="text-sm text-gray-500">
                                      ... and {perm.actions.length - 5} more
                                    </li>
                                  )}
                                </ul>
                              </div>
                            </div>
                          )}
                          
                          {perm.notActions.length > 0 && (
                            <div>
                              <h4 className="text-sm font-medium text-red-600 dark:text-red-400 mb-2">
                                Denied Actions ({perm.notActions.length})
                              </h4>
                              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
                                <ul className="space-y-1">
                                  {perm.notActions.slice(0, 5).map((action, idx) => (
                                    <li key={idx} className="text-sm font-mono text-gray-700 dark:text-gray-300">
                                      {action}
                                    </li>
                                  ))}
                                  {perm.notActions.length > 5 && (
                                    <li className="text-sm text-gray-500">
                                      ... and {perm.notActions.length - 5} more
                                    </li>
                                  )}
                                </ul>
                              </div>
                            </div>
                          )}

                          {perm.dataActions && perm.dataActions.length > 0 && (
                            <div>
                              <h4 className="text-sm font-medium text-blue-600 dark:text-blue-400 mb-2">
                                Data Actions ({perm.dataActions.length})
                              </h4>
                              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                                <ul className="space-y-1">
                                  {perm.dataActions.map((action, idx) => (
                                    <li key={idx} className="text-sm font-mono text-gray-700 dark:text-gray-300">
                                      {action}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}

                  {isEditMode && (
                    <button className="mt-4 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center gap-2">
                      <Plus className="w-4 h-4" />
                      Add Permission
                    </button>
                  )}
                </div>
              </div>

              {/* Assignable Scopes */}
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Globe className="w-5 h-5 text-blue-500" />
                    Assignable Scopes
                  </h2>
                </div>
                <div className="p-6">
                  <ul className="space-y-2">
                    {selectedRole.assignableScopes.map((scope, idx) => (
                      <li key={idx} className="flex items-center gap-2 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                        <Database className="w-4 h-4 text-gray-400" />
                        <code className="text-sm flex-1">{formatScope(scope)}</code>
                        {isEditMode && (
                          <button className="text-red-500 hover:text-red-600">
                            <X className="w-4 h-4" />
                          </button>
                        )}
                      </li>
                    ))}
                  </ul>
                  {isEditMode && (
                    <button className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2">
                      <Plus className="w-4 h-4" />
                      Add Scope
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Right Column - Metadata and Assignments */}
            <div className="space-y-6">
              {/* Role Information */}
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Info className="w-5 h-5 text-blue-500" />
                  Role Information
                </h2>
                <div className="space-y-3">
                  <div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">Role ID</span>
                    <p className="font-mono text-sm">{selectedRole.id}</p>
                  </div>
                  <div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">Created On</span>
                    <p className="text-sm">{new Date(selectedRole.createdOn).toLocaleDateString()}</p>
                  </div>
                  <div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">Last Updated</span>
                    <p className="text-sm">{new Date(selectedRole.updatedOn).toLocaleDateString()}</p>
                  </div>
                  <div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">Created By</span>
                    <p className="text-sm">{selectedRole.createdBy}</p>
                  </div>
                </div>
              </div>

              {/* Assignment Statistics */}
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Users className="w-5 h-5 text-green-500" />
                  Current Assignments
                </h2>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <UserCheck className="w-4 h-4 text-gray-400" />
                      <span className="text-sm">Users</span>
                    </div>
                    <span className="font-bold">{selectedRole.assignedUsers}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Building className="w-4 h-4 text-gray-400" />
                      <span className="text-sm">Groups</span>
                    </div>
                    <span className="font-bold">{selectedRole.assignedGroups}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Terminal className="w-4 h-4 text-gray-400" />
                      <span className="text-sm">Service Principals</span>
                    </div>
                    <span className="font-bold">{selectedRole.assignedServicePrincipals}</span>
                  </div>
                  <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Total</span>
                      <span className="text-lg font-bold text-purple-500">
                        {selectedRole.assignedUsers + selectedRole.assignedGroups + selectedRole.assignedServicePrincipals}
                      </span>
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => router.push('/security/iam')}
                  className="w-full mt-4 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  View Assignments
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </ResponsiveContainer>
      </div>
    );
  }

  // Main Roles List View
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
      <ResponsiveContainer className="py-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
                <Shield className="w-8 h-8 text-purple-500" />
                Role Management
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Manage built-in and custom roles for your organization
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => toast({ title: 'Export', description: 'Exporting roles to CSV...' })}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Export
              </button>
              <button
                onClick={handleCreateRole}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                <Plus className="w-4 h-4" />
                Create Role
              </button>
            </div>
          </div>

          {/* Search and Filters */}
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search roles by name or description..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as any)}
              className="px-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              <option value="all">All Roles</option>
              <option value="builtin">Built-in Only</option>
              <option value="custom">Custom Only</option>
            </select>
            <button className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center gap-2">
              <Filter className="w-4 h-4" />
              More Filters
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Total Roles</span>
              <Shield className="w-4 h-4 text-purple-500" />
            </div>
            <div className="text-2xl font-bold">{roles.length}</div>
            <p className="text-xs text-gray-500 mt-1">Active in system</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Built-in Roles</span>
              <Package className="w-4 h-4 text-blue-500" />
            </div>
            <div className="text-2xl font-bold">{roles.filter(r => r.type === 'BuiltInRole').length}</div>
            <p className="text-xs text-gray-500 mt-1">Azure default</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Custom Roles</span>
              <Settings className="w-4 h-4 text-green-500" />
            </div>
            <div className="text-2xl font-bold">{roles.filter(r => r.type === 'CustomRole').length}</div>
            <p className="text-xs text-gray-500 mt-1">Organization-specific</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Total Assignments</span>
              <Users className="w-4 h-4 text-orange-500" />
            </div>
            <div className="text-2xl font-bold">
              {roles.reduce((acc, r) => acc + r.assignedUsers + r.assignedGroups + r.assignedServicePrincipals, 0)}
            </div>
            <p className="text-xs text-gray-500 mt-1">Active assignments</p>
          </div>
        </div>

        {/* Roles Table */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Role Name
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Assignments
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Risk Score
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Last Modified
                  </th>
                  <th className="px-6 py-4 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {filteredRoles.map((role) => (
                  <tr
                    key={role.id}
                    className="hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer transition-colors"
                    onClick={() => setSelectedRole(role)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div>
                        <div className="text-sm font-medium">{role.displayName}</div>
                        <div className="text-xs text-gray-500 max-w-xs truncate">
                          {role.description}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        role.type === 'BuiltInRole' 
                          ? 'bg-blue-500/10 text-blue-500' 
                          : 'bg-purple-500/10 text-purple-500'
                      }`}>
                        {role.type === 'BuiltInRole' ? 'Built-in' : 'Custom'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center gap-4 text-sm">
                        <div className="flex items-center gap-1">
                          <Users className="w-3 h-3 text-gray-400" />
                          <span>{role.assignedUsers}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Building className="w-3 h-3 text-gray-400" />
                          <span>{role.assignedGroups}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Terminal className="w-3 h-3 text-gray-400" />
                          <span>{role.assignedServicePrincipals}</span>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-3 py-1 text-sm rounded-lg font-medium ${getRiskColor(role.riskScore)}`}>
                        {role.riskScore}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(role.updatedOn).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCloneRole(role);
                          }}
                          className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded transition-colors"
                          title="Clone Role"
                        >
                          <Copy className="w-4 h-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
                        </button>
                        {role.type === 'CustomRole' && (
                          <>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleEditRole(role);
                              }}
                              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded transition-colors"
                              title="Edit Role"
                            >
                              <Edit className="w-4 h-4 text-blue-500 hover:text-blue-600" />
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteRole(role);
                              }}
                              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded transition-colors"
                              title="Delete Role"
                            >
                              <Trash2 className="w-4 h-4 text-red-500 hover:text-red-600" />
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </ResponsiveContainer>

      {/* Create/Edit Role Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">Create New Role</h2>
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>
            <div className="p-6">
              <form className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Role Name</label>
                  <input
                    type="text"
                    placeholder="e.g., Custom Developer Role"
                    className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Description</label>
                  <textarea
                    rows={3}
                    placeholder="Describe the purpose and permissions of this role..."
                    className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Base Role Template</label>
                  <select className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500">
                    <option>Start from scratch</option>
                    <option>Clone from Reader</option>
                    <option>Clone from Contributor</option>
                    <option>Clone from Owner</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Assignable Scopes</label>
                  <div className="space-y-2">
                    <div className="flex gap-2">
                      <input
                        type="text"
                        placeholder="/subscriptions/..."
                        className="flex-1 px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />
                      <button
                        type="button"
                        className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                      >
                        Add
                      </button>
                    </div>
                  </div>
                </div>
                <div className="pt-4 border-t border-gray-200 dark:border-gray-700 flex justify-end gap-3">
                  <button
                    type="button"
                    onClick={() => setShowCreateModal(false)}
                    className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      toast({ title: 'Role Created', description: 'New role created successfully' });
                      setShowCreateModal(false);
                    }}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                  >
                    Create Role
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}