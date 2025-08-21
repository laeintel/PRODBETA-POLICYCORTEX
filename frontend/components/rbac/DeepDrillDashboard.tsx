'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { 
  User, Shield, AlertTriangle, Clock, TrendingDown, 
  Activity, Users, Lock, Unlock, ChevronRight,
  Download, RefreshCw, Search, Filter, BarChart3,
  Eye, Settings, AlertCircle, CheckCircle, XCircle,
  ArrowUp, ArrowDown, Minus, MoreVertical
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { format, formatDistanceToNow, subDays } from 'date-fns';

interface UserPermissionDetail {
  userId: string;
  displayName: string;
  email: string;
  department?: string;
  jobTitle?: string;
  permissions: PermissionDetail[];
  roles: RoleDetail[];
  groups: GroupMembership[];
  riskScore: number;
  overProvisioningScore: number;
  lastSignIn?: string;
  accountEnabled: boolean;
  recommendations: PermissionRecommendation[];
  accessPatterns: AccessPatternAnalysis;
  complianceStatus: ComplianceStatus;
}

interface PermissionDetail {
  permissionId: string;
  permissionName: string;
  resourceType: string;
  resourceId: string;
  resourceName: string;
  scope: string;
  actions: string[];
  assignedDate: string;
  assignedBy: string;
  assignmentType: 'direct' | 'role_based' | 'group_inherited';
  lastUsed?: string;
  usageCount30d: number;
  usageCount90d: number;
  isHighPrivilege: boolean;
  riskLevel: 'critical' | 'high' | 'medium' | 'low' | 'none';
  usagePattern: 'daily' | 'weekly' | 'monthly' | 'occasional' | 'rare' | 'never';
  similarUsersHaveThis: number;
  removalImpact: {
    severity: 'critical' | 'high' | 'medium' | 'low' | 'none';
    affectedWorkflows: string[];
  };
}

interface RoleDetail {
  roleId: string;
  roleName: string;
  roleType: 'built_in' | 'custom' | 'application_specific' | 'delegated';
  isBuiltin: boolean;
  assignedDate: string;
  assignedBy: string;
  scope: string;
  permissionsCount: number;
  highRiskPermissions: string[];
  lastActivity?: string;
  usageFrequency: 'very_high' | 'high' | 'medium' | 'low' | 'very_low' | 'none';
  justification?: string;
  expiryDate?: string;
  isEligible: boolean;
  isActive: boolean;
}

interface GroupMembership {
  groupId: string;
  groupName: string;
  membershipType: 'direct' | 'dynamic' | 'nested' | 'transitive';
  joinedDate: string;
  addedBy: string;
  permissionsInherited: number;
  nestedGroups: string[];
  isDynamic: boolean;
  dynamicRule?: string;
}

interface AccessPatternAnalysis {
  typicalAccessTimes: TimeRange[];
  typicalLocations: Location[];
  typicalDevices: Device[];
  unusualActivities: UnusualActivity[];
  accessVelocity: number;
  failedAttempts30d: number;
  mfaUsageRate: number;
  conditionalAccessCompliance: number;
  privilegedOperationsCount: number;
  dataAccessVolume: DataVolume;
  serviceUsage: Record<string, ServiceUsage>;
}

interface ComplianceStatus {
  isCompliant: boolean;
  violations: ComplianceViolation[];
  certifications: Certification[];
  lastReviewDate?: string;
  nextReviewDate?: string;
  reviewer?: string;
  attestationStatus: 'attested' | 'pending' | 'expired' | 'rejected';
}

interface PermissionRecommendation {
  recommendationId: string;
  recommendationType: 'remove_permission' | 'downgrade_permission' | 'convert_to_just_in_time' | 'add_conditional_access' | 'enable_mfa' | 'review_group_membership' | 'certify_access';
  title: string;
  description: string;
  impact: 'critical' | 'high' | 'medium' | 'low' | 'none';
  confidence: number;
  affectedPermissions: string[];
  suggestedAction: SuggestedAction;
  estimatedRiskReduction: number;
  similarUsersImplemented: number;
  autoRemediationAvailable: boolean;
  requiresApproval: boolean;
  approvalWorkflowId?: string;
}

interface TimeRange {
  start: string;
  end: string;
}

interface Location {
  country: string;
  region: string;
  city: string;
  ipRange: string;
}

interface Device {
  deviceId: string;
  deviceType: string;
  os: string;
  isCompliant: boolean;
  isManaged: boolean;
}

interface UnusualActivity {
  activityId: string;
  activityType: string;
  timestamp: string;
  riskScore: number;
  description: string;
  affectedResources: string[];
  detectionMethod: string;
  isInvestigated: boolean;
  investigationNotes?: string;
}

interface DataVolume {
  readsGb: number;
  writesGb: number;
  downloadsGb: number;
}

interface ServiceUsage {
  accessCount: number;
  uniqueOperations: number;
  dataVolume: DataVolume;
  peakTimes: string[];
}

interface ComplianceViolation {
  violationId: string;
  policyId: string;
  policyName: string;
  violationType: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  detectedDate: string;
  remediationDeadline?: string;
  remediationSteps: string[];
}

interface Certification {
  certificationId: string;
  name: string;
  issuedDate: string;
  expiryDate?: string;
  certifier: string;
}

interface SuggestedAction {
  actionType: string;
  description: string;
  automated: boolean;
  requiresApproval: boolean;
  implementationSteps: string[];
}

interface DrillPath {
  level: number;
  type: string;
  id: string;
  name: string;
}

export default function RbacDeepDrillDashboard() {
  const [selectedUser, setSelectedUser] = useState<UserPermissionDetail | null>(null);
  const [selectedPermission, setSelectedPermission] = useState<PermissionDetail | null>(null);
  const [drillPath, setDrillPath] = useState<DrillPath[]>([]);
  const [viewMode, setViewMode] = useState<'overview' | 'permissions' | 'usage' | 'recommendations'>('overview');
  const [filterRisk, setFilterRisk] = useState<string>('all');
  const [filterUsage, setFilterUsage] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['permissions']));

  const fetchUserDetails = async (userId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/rbac/users/${userId}/deep-drill`);
      const data = await response.json();
      setSelectedUser(data);
      setDrillPath([
        ...drillPath,
        { level: drillPath.length + 1, type: 'user', id: userId, name: data.displayName }
      ]);
    } catch (error) {
      console.error('Error fetching user details:', error);
    } finally {
      setLoading(false);
    }
  };

  const drillIntoPermission = (permission: PermissionDetail) => {
    setSelectedPermission(permission);
    setDrillPath([
      ...drillPath,
      { 
        level: drillPath.length + 1, 
        type: 'permission', 
        id: permission.permissionId, 
        name: permission.permissionName 
      }
    ]);
  };

  const navigateToDrillLevel = (level: number) => {
    setDrillPath(drillPath.slice(0, level + 1));
    if (level === 0) {
      setSelectedUser(null);
      setSelectedPermission(null);
    } else if (level === 1) {
      setSelectedPermission(null);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getUsageIndicator = (pattern: string) => {
    switch (pattern) {
      case 'daily': return { icon: <ArrowUp className="w-4 h-4" />, color: 'text-green-600' };
      case 'weekly': return { icon: <ArrowUp className="w-4 h-4" />, color: 'text-green-500' };
      case 'monthly': return { icon: <Minus className="w-4 h-4" />, color: 'text-yellow-600' };
      case 'occasional': return { icon: <ArrowDown className="w-4 h-4" />, color: 'text-orange-600' };
      case 'rare': return { icon: <ArrowDown className="w-4 h-4" />, color: 'text-red-500' };
      case 'never': return { icon: <XCircle className="w-4 h-4" />, color: 'text-red-600' };
      default: return { icon: <Minus className="w-4 h-4" />, color: 'text-gray-500' };
    }
  };

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const renderBreadcrumb = () => (
    <div className="flex items-center space-x-2 text-sm mb-6 p-3 bg-gray-50 rounded-lg">
      <button type="button"
        onClick={() => navigateToDrillLevel(-1)}
        className="text-blue-600 hover:text-blue-800 font-medium"
      >
        RBAC Dashboard
      </button>
      {drillPath.map((path, index) => (
        <React.Fragment key={path.id}>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <button type="button"
            onClick={() => navigateToDrillLevel(index)}
            className={`hover:text-blue-800 font-medium ${
              index === drillPath.length - 1 ? 'text-gray-900' : 'text-blue-600'
            }`}
          >
            {path.name}
          </button>
        </React.Fragment>
      ))}
    </div>
  );

  const renderUserOverview = () => {
    if (!selectedUser) return null;

    return (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* User Info Card */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center mb-4">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                <User className="w-6 h-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <h3 className="font-semibold text-gray-900">{selectedUser.displayName}</h3>
                <p className="text-sm text-gray-600">{selectedUser.email}</p>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Department</span>
                <span className="text-sm font-medium">{selectedUser.department || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Job Title</span>
                <span className="text-sm font-medium">{selectedUser.jobTitle || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Last Sign In</span>
                <span className="text-sm font-medium">
                  {selectedUser.lastSignIn 
                    ? formatDistanceToNow(new Date(selectedUser.lastSignIn), { addSuffix: true })
                    : 'Never'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Account Status</span>
                <span className={`text-sm font-medium ${
                  selectedUser.accountEnabled ? 'text-green-600' : 'text-red-600'
                }`}>
                  {selectedUser.accountEnabled ? 'Active' : 'Disabled'}
                </span>
              </div>
            </div>

            {/* Risk Scores */}
            <div className="mt-6 pt-6 border-t border-gray-200">
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm text-gray-600">Risk Score</span>
                    <span className="text-sm font-semibold text-red-600">
                      {(selectedUser.riskScore * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-red-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${selectedUser.riskScore * 100}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm text-gray-600">Over-provisioning</span>
                    <span className="text-sm font-semibold text-orange-600">
                      {(selectedUser.overProvisioningScore * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-orange-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${selectedUser.overProvisioningScore * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Permissions Overview */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-semibold text-gray-900">Permissions Analysis</h3>
              <div className="flex space-x-2">
                <button type="button" className="px-3 py-1 text-sm bg-blue-50 text-blue-600 rounded-md hover:bg-blue-100" onClick={() => console.log('Refresh permissions') }>
                  <RefreshCw className="w-4 h-4 inline mr-1" />
                  Refresh
                </button>
                <button type="button" className="px-3 py-1 text-sm bg-gray-50 text-gray-600 rounded-md hover:bg-gray-100" onClick={() => console.log('Export permissions analysis') }>
                  <Download className="w-4 h-4 inline mr-1" />
                  Export
                </button>
              </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-2xl font-bold text-gray-900">
                  {selectedUser.permissions.length}
                </div>
                <div className="text-xs text-gray-600">Total Permissions</div>
              </div>
              <div className="bg-red-50 rounded-lg p-3">
                <div className="text-2xl font-bold text-red-600">
                  {selectedUser.permissions.filter(p => p.isHighPrivilege).length}
                </div>
                <div className="text-xs text-gray-600">High Privilege</div>
              </div>
              <div className="bg-orange-50 rounded-lg p-3">
                <div className="text-2xl font-bold text-orange-600">
                  {selectedUser.permissions.filter(p => p.usagePattern === 'never').length}
                </div>
                <div className="text-xs text-gray-600">Never Used</div>
              </div>
              <div className="bg-blue-50 rounded-lg p-3">
                <div className="text-2xl font-bold text-blue-600">
                  {selectedUser.roles.length}
                </div>
                <div className="text-xs text-gray-600">Assigned Roles</div>
              </div>
            </div>

            {/* Permissions List */}
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {selectedUser.permissions
                .filter(p => 
                  (filterRisk === 'all' || p.riskLevel === filterRisk) &&
                  (filterUsage === 'all' || p.usagePattern === filterUsage) &&
                  (searchTerm === '' || p.permissionName.toLowerCase().includes(searchTerm.toLowerCase()))
                )
                .map((permission) => (
                <motion.div
                  key={permission.permissionId}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                  onClick={() => drillIntoPermission(permission)}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <Shield className="w-4 h-4 text-gray-400" />
                        <h4 className="font-medium text-gray-900">{permission.permissionName}</h4>
                        {permission.isHighPrivilege && (
                          <span className="px-2 py-0.5 text-xs bg-red-100 text-red-600 rounded-full">
                            High Privilege
                          </span>
                        )}
                        <span className={`px-2 py-0.5 text-xs rounded-full ${getRiskColor(permission.riskLevel)}`}>
                          {permission.riskLevel}
                        </span>
                      </div>
                      
                      <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500">Resource:</span>
                          <span className="ml-1 font-medium">{permission.resourceName}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Type:</span>
                          <span className="ml-1 font-medium">{permission.assignmentType}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Last Used:</span>
                          <span className="ml-1 font-medium">
                            {permission.lastUsed 
                              ? formatDistanceToNow(new Date(permission.lastUsed), { addSuffix: true })
                              : 'Never'}
                          </span>
                        </div>
                        <div className="flex items-center">
                          <span className="text-gray-500">Usage:</span>
                          <span className={`ml-1 flex items-center ${getUsageIndicator(permission.usagePattern).color}`}>
                            {getUsageIndicator(permission.usagePattern).icon}
                            <span className="ml-1">{permission.usagePattern}</span>
                          </span>
                        </div>
                      </div>

                      <div className="mt-3 flex items-center space-x-4">
                        <div className="flex items-center text-xs text-gray-600">
                          <Activity className="w-3 h-3 mr-1" />
                          {permission.usageCount30d} uses (30d)
                        </div>
                        <div className="flex items-center text-xs text-gray-600">
                          <Users className="w-3 h-3 mr-1" />
                          {permission.similarUsersHaveThis}% peers have this
                        </div>
                        {permission.removalImpact.severity !== 'none' && (
                          <div className="flex items-center text-xs text-orange-600">
                            <AlertCircle className="w-3 h-3 mr-1" />
                            Removal impact: {permission.removalImpact.severity}
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <ChevronRight className="w-5 h-5 text-gray-400 mt-1" />
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderPermissionDetail = () => {
    if (!selectedPermission) return null;

    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex justify-between items-start mb-6">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">{selectedPermission.permissionName}</h2>
            <p className="text-sm text-gray-600 mt-1">Permission ID: {selectedPermission.permissionId}</p>
          </div>
          <div className="flex space-x-2">
            <button type="button" 
              className="px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700" 
              onClick={() => {
                if (selectedPermission) {
                  const confirmed = confirm(`Request removal of permission: ${selectedPermission.permissionName}?`);
                  if (confirmed) {
                    alert(`Removal request submitted for ${selectedPermission.permissionName}. You will receive an email confirmation.`);
                  }
                }
              }}>
              Request Removal
            </button>
            <button type="button" 
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700" 
              onClick={() => {
                if (selectedPermission) {
                  const confirmed = confirm(`Convert ${selectedPermission.permissionName} to Just-In-Time access?`);
                  if (confirmed) {
                    alert(`JIT conversion initiated for ${selectedPermission.permissionName}. This will require approval for future access.`);
                  }
                }
              }}>
              Convert to JIT
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Permission Details */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-4">Permission Details</h3>
            <div className="space-y-3">
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Resource Type</span>
                <span className="text-sm font-medium">{selectedPermission.resourceType}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Resource</span>
                <span className="text-sm font-medium">{selectedPermission.resourceName}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Scope</span>
                <span className="text-sm font-medium">{selectedPermission.scope}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Assignment Type</span>
                <span className="text-sm font-medium">{selectedPermission.assignmentType}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Assigned By</span>
                <span className="text-sm font-medium">{selectedPermission.assignedBy}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Assigned Date</span>
                <span className="text-sm font-medium">
                  {format(new Date(selectedPermission.assignedDate), 'MMM dd, yyyy')}
                </span>
              </div>
            </div>
          </div>

          {/* Usage Analytics */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-4">Usage Analytics</h3>
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-600">30-Day Usage</span>
                  <span className="text-2xl font-bold text-gray-900">{selectedPermission.usageCount30d}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${Math.min(selectedPermission.usageCount30d / 100 * 100, 100)}%` }}
                  />
                </div>
              </div>

              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-600">90-Day Usage</span>
                  <span className="text-2xl font-bold text-gray-900">{selectedPermission.usageCount90d}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-600 h-2 rounded-full"
                    style={{ width: `${Math.min(selectedPermission.usageCount90d / 300 * 100, 100)}%` }}
                  />
                </div>
              </div>

              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div className="flex items-start">
                  <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-gray-900">Usage Pattern: {selectedPermission.usagePattern}</h4>
                    <p className="text-sm text-gray-600 mt-1">
                      This permission has been used {selectedPermission.usageCount30d} times in the last 30 days.
                      {selectedPermission.usagePattern === 'never' && 
                        ' Consider removing this permission as it has never been used.'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="mt-6">
          <h3 className="font-semibold text-gray-900 mb-4">Allowed Actions</h3>
          <div className="flex flex-wrap gap-2">
            {selectedPermission.actions.map((action, index) => (
              <span 
                key={index}
                className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-sm"
              >
                {action}
              </span>
            ))}
          </div>
        </div>

        {/* Impact Analysis */}
        {selectedPermission.removalImpact.affectedWorkflows.length > 0 && (
          <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <h3 className="font-semibold text-red-900 mb-2">Removal Impact Analysis</h3>
            <p className="text-sm text-red-700 mb-3">
              Removing this permission will affect the following workflows:
            </p>
            <ul className="list-disc list-inside space-y-1">
              {selectedPermission.removalImpact.affectedWorkflows.map((workflow, index) => (
                <li key={index} className="text-sm text-red-600">{workflow}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900">RBAC Deep-Drill Analysis</h1>
          <p className="text-gray-600">Comprehensive permission and access management insights</p>
        </div>

        {/* Breadcrumb */}
        {renderBreadcrumb()}

        {/* Filters */}
        {selectedUser && !selectedPermission && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
            <div className="flex flex-wrap gap-4">
              <div className="flex-1 min-w-[200px]">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search permissions..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>
              
              <select
                value={filterRisk}
                onChange={(e) => setFilterRisk(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Risk Levels</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
                <option value="none">None</option>
              </select>

              <select
                value={filterUsage}
                onChange={(e) => setFilterUsage(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Usage Patterns</option>
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
                <option value="occasional">Occasional</option>
                <option value="rare">Rare</option>
                <option value="never">Never Used</option>
              </select>

              <button type="button" className="px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 flex items-center">
                <Filter className="w-4 h-4 mr-2" />
                More Filters
              </button>
            </div>
          </div>
        )}

        {/* Main Content */}
        <AnimatePresence mode="wait">
          {loading ? (
            <div className="flex justify-center items-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
          ) : selectedPermission ? (
            renderPermissionDetail()
          ) : selectedUser ? (
            renderUserOverview()
          ) : (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
              <User className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Select a User to Analyze</h3>
              <p className="text-gray-600 mb-6">
                Choose a user from the main RBAC dashboard to begin deep-drill analysis
              </p>
              <button type="button" 
                onClick={() => fetchUserDetails('user-123')}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Load Sample User
              </button>
            </div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}