'use client';

import React, { useState, useEffect } from 'react';
import { 
  Users, UserCheck, UserX, Shield, Key, Lock, Fingerprint, Smartphone,
  Mail, Globe, Building, Calendar, Clock, AlertTriangle, CheckCircle,
  XCircle, Search, Filter, RefreshCw, Download, Settings, Plus, Edit,
  Trash2, ArrowRight, ChevronRight, Activity, TrendingUp, BarChart,
  Eye, EyeOff, Link2, UserPlus, Database, FileText, Hash, Info
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface Identity {
  id: string;
  displayName: string;
  principalName: string;
  objectId: string;
  type: 'user' | 'service_principal' | 'managed_identity' | 'group' | 'guest';
  status: 'active' | 'inactive' | 'suspended' | 'pending' | 'expired';
  department?: string;
  jobTitle?: string;
  manager?: string;
  createdDate: string;
  lastSignIn?: string;
  riskLevel: 'none' | 'low' | 'medium' | 'high' | 'critical';
  mfaEnabled: boolean;
  conditionalAccess: boolean;
  privilegedAccess: boolean;
  licenses: string[];
  groups: string[];
  applications: string[];
}

interface AuthenticationMethod {
  id: string;
  userId: string;
  type: 'password' | 'mfa' | 'fido2' | 'phone' | 'email' | 'app';
  status: 'enabled' | 'disabled' | 'pending';
  lastUsed?: string;
  strength: 'weak' | 'medium' | 'strong';
  configured: boolean;
}

interface SignInActivity {
  id: string;
  userId: string;
  userName: string;
  timestamp: string;
  status: 'success' | 'failure' | 'blocked' | 'risky';
  location: string;
  ipAddress: string;
  device: string;
  application: string;
  riskLevel: 'none' | 'low' | 'medium' | 'high';
  mfaUsed: boolean;
  conditionalAccessApplied: boolean;
}

interface RiskyUser {
  id: string;
  userId: string;
  displayName: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  riskState: 'atRisk' | 'confirmedCompromised' | 'dismissed' | 'remediated';
  riskLastUpdated: string;
  riskDetail: string;
  riskEventTypes: string[];
  recommendedActions: string[];
}

interface IdentityMetrics {
  totalIdentities: number;
  activeUsers: number;
  guestUsers: number;
  servicePrincipals: number;
  mfaAdoption: number;
  riskySignIns: number;
  failedSignIns: number;
  passwordExpiring: number;
}

export default function IdentityManagement() {
  const [identities, setIdentities] = useState<Identity[]>([]);
  const [authMethods, setAuthMethods] = useState<AuthenticationMethod[]>([]);
  const [signInActivity, setSignInActivity] = useState<SignInActivity[]>([]);
  const [riskyUsers, setRiskyUsers] = useState<RiskyUser[]>([]);
  const [metrics, setMetrics] = useState<IdentityMetrics | null>(null);
  const [selectedIdentity, setSelectedIdentity] = useState<Identity | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'identities' | 'authentication' | 'activity' | 'risks'>('identities');
  const [filterType, setFilterType] = useState('all');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    // Initialize with comprehensive mock data
    setIdentities([
      {
        id: 'USR-001',
        displayName: 'John Admin',
        principalName: 'john.admin@company.com',
        objectId: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
        type: 'user',
        status: 'active',
        department: 'IT Operations',
        jobTitle: 'System Administrator',
        manager: 'Jane Director',
        createdDate: '2022-03-15',
        lastSignIn: '2 minutes ago',
        riskLevel: 'low',
        mfaEnabled: true,
        conditionalAccess: true,
        privilegedAccess: true,
        licenses: ['Microsoft 365 E5', 'Azure AD Premium P2'],
        groups: ['Domain Admins', 'Azure Admins', 'Security Team'],
        applications: ['Azure Portal', 'Exchange Admin', 'SharePoint']
      },
      {
        id: 'USR-002',
        displayName: 'Sarah Developer',
        principalName: 'sarah.dev@company.com',
        objectId: 'b2c3d4e5-f6g7-8901-bcde-f23456789012',
        type: 'user',
        status: 'active',
        department: 'Engineering',
        jobTitle: 'Senior Developer',
        manager: 'Mike Manager',
        createdDate: '2023-01-10',
        lastSignIn: '1 hour ago',
        riskLevel: 'none',
        mfaEnabled: true,
        conditionalAccess: false,
        privilegedAccess: false,
        licenses: ['Microsoft 365 E3', 'Visual Studio Enterprise'],
        groups: ['Developers', 'DevOps Team'],
        applications: ['Azure DevOps', 'GitHub', 'VS Code']
      },
      {
        id: 'SPN-001',
        displayName: 'Backup Service Principal',
        principalName: 'backup-service@app.company.com',
        objectId: 'c3d4e5f6-g7h8-9012-cdef-345678901234',
        type: 'service_principal',
        status: 'active',
        createdDate: '2023-06-01',
        lastSignIn: '5 minutes ago',
        riskLevel: 'medium',
        mfaEnabled: false,
        conditionalAccess: true,
        privilegedAccess: true,
        licenses: [],
        groups: ['Service Accounts'],
        applications: ['Storage Account', 'Key Vault']
      },
      {
        id: 'USR-003',
        displayName: 'External Contractor',
        principalName: 'contractor@external.com',
        objectId: 'd4e5f6g7-h8i9-0123-defg-456789012345',
        type: 'guest',
        status: 'active',
        department: 'External',
        jobTitle: 'Consultant',
        createdDate: '2024-01-01',
        lastSignIn: '2 days ago',
        riskLevel: 'high',
        mfaEnabled: false,
        conditionalAccess: true,
        privilegedAccess: false,
        licenses: [],
        groups: ['Guest Users', 'Project Alpha'],
        applications: ['Teams', 'SharePoint']
      },
      {
        id: 'USR-004',
        displayName: 'Inactive User',
        principalName: 'old.employee@company.com',
        objectId: 'e5f6g7h8-i9j0-1234-efgh-567890123456',
        type: 'user',
        status: 'inactive',
        department: 'Sales',
        jobTitle: 'Sales Rep',
        createdDate: '2021-05-20',
        lastSignIn: '90 days ago',
        riskLevel: 'critical',
        mfaEnabled: false,
        conditionalAccess: false,
        privilegedAccess: false,
        licenses: ['Microsoft 365 E3'],
        groups: ['Sales Team'],
        applications: []
      }
    ]);

    setAuthMethods([
      {
        id: 'AUTH-001',
        userId: 'USR-001',
        type: 'mfa',
        status: 'enabled',
        lastUsed: '2 minutes ago',
        strength: 'strong',
        configured: true
      },
      {
        id: 'AUTH-002',
        userId: 'USR-001',
        type: 'fido2',
        status: 'enabled',
        lastUsed: '1 day ago',
        strength: 'strong',
        configured: true
      },
      {
        id: 'AUTH-003',
        userId: 'USR-002',
        type: 'app',
        status: 'enabled',
        lastUsed: '1 hour ago',
        strength: 'strong',
        configured: true
      },
      {
        id: 'AUTH-004',
        userId: 'USR-003',
        type: 'password',
        status: 'enabled',
        lastUsed: '2 days ago',
        strength: 'weak',
        configured: true
      },
      {
        id: 'AUTH-005',
        userId: 'USR-004',
        type: 'password',
        status: 'enabled',
        lastUsed: '90 days ago',
        strength: 'weak',
        configured: false
      }
    ]);

    setSignInActivity([
      {
        id: 'SIGN-001',
        userId: 'USR-001',
        userName: 'John Admin',
        timestamp: '2024-01-22T14:45:00Z',
        status: 'success',
        location: 'Seattle, WA',
        ipAddress: '192.168.1.100',
        device: 'Windows 11',
        application: 'Azure Portal',
        riskLevel: 'none',
        mfaUsed: true,
        conditionalAccessApplied: true
      },
      {
        id: 'SIGN-002',
        userId: 'USR-003',
        userName: 'External Contractor',
        timestamp: '2024-01-22T14:30:00Z',
        status: 'risky',
        location: 'Unknown Location',
        ipAddress: '185.220.101.45',
        device: 'Unknown Device',
        application: 'SharePoint',
        riskLevel: 'high',
        mfaUsed: false,
        conditionalAccessApplied: false
      },
      {
        id: 'SIGN-003',
        userId: 'USR-002',
        userName: 'Sarah Developer',
        timestamp: '2024-01-22T13:45:00Z',
        status: 'success',
        location: 'San Francisco, CA',
        ipAddress: '10.0.1.45',
        device: 'MacBook Pro',
        application: 'Azure DevOps',
        riskLevel: 'none',
        mfaUsed: true,
        conditionalAccessApplied: false
      },
      {
        id: 'SIGN-004',
        userId: 'USR-004',
        userName: 'Inactive User',
        timestamp: '2024-01-22T12:00:00Z',
        status: 'blocked',
        location: 'Moscow, Russia',
        ipAddress: '45.142.214.112',
        device: 'Linux',
        application: 'Exchange Online',
        riskLevel: 'high',
        mfaUsed: false,
        conditionalAccessApplied: true
      },
      {
        id: 'SIGN-005',
        userId: 'SPN-001',
        userName: 'Backup Service Principal',
        timestamp: '2024-01-22T14:40:00Z',
        status: 'success',
        location: 'Azure Datacenter',
        ipAddress: '10.0.0.1',
        device: 'Service',
        application: 'Storage Account',
        riskLevel: 'low',
        mfaUsed: false,
        conditionalAccessApplied: true
      }
    ]);

    setRiskyUsers([
      {
        id: 'RISK-001',
        userId: 'USR-003',
        displayName: 'External Contractor',
        riskLevel: 'high',
        riskState: 'atRisk',
        riskLastUpdated: '1 hour ago',
        riskDetail: 'Sign-ins from anonymous IP addresses',
        riskEventTypes: ['anonymousIPAddress', 'unfamiliarLocation'],
        recommendedActions: ['Enable MFA', 'Review access permissions', 'Reset password']
      },
      {
        id: 'RISK-002',
        userId: 'USR-004',
        displayName: 'Inactive User',
        riskLevel: 'critical',
        riskState: 'confirmedCompromised',
        riskLastUpdated: '2 hours ago',
        riskDetail: 'Account compromise detected',
        riskEventTypes: ['leakedCredentials', 'suspiciousActivity'],
        recommendedActions: ['Disable account', 'Force password reset', 'Review audit logs']
      },
      {
        id: 'RISK-003',
        userId: 'SPN-001',
        displayName: 'Backup Service Principal',
        riskLevel: 'medium',
        riskState: 'atRisk',
        riskLastUpdated: '6 hours ago',
        riskDetail: 'Unusual application access pattern',
        riskEventTypes: ['anomalousToken', 'suspiciousAPIUsage'],
        recommendedActions: ['Rotate credentials', 'Review permissions', 'Enable monitoring']
      }
    ]);

    setMetrics({
      totalIdentities: 1847,
      activeUsers: 1523,
      guestUsers: 89,
      servicePrincipals: 235,
      mfaAdoption: 78.5,
      riskySignIns: 23,
      failedSignIns: 156,
      passwordExpiring: 42
    });

    // Simulate real-time updates
    if (autoRefresh) {
      const interval = setInterval(() => {
        setSignInActivity(prev => {
          const newActivity: SignInActivity = {
            id: `SIGN-${Date.now()}`,
            userId: `USR-00${Math.floor(Math.random() * 4) + 1}`,
            userName: ['John Admin', 'Sarah Developer', 'External Contractor', 'Backup Service'][Math.floor(Math.random() * 4)],
            timestamp: new Date().toISOString(),
            status: ['success', 'failure', 'blocked', 'risky'][Math.floor(Math.random() * 4)] as any,
            location: ['Seattle, WA', 'San Francisco, CA', 'Unknown'][Math.floor(Math.random() * 3)],
            ipAddress: `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
            device: ['Windows', 'Mac', 'Mobile'][Math.floor(Math.random() * 3)],
            application: ['Azure Portal', 'Teams', 'SharePoint'][Math.floor(Math.random() * 3)],
            riskLevel: ['none', 'low', 'medium', 'high'][Math.floor(Math.random() * 4)] as any,
            mfaUsed: Math.random() > 0.3,
            conditionalAccessApplied: Math.random() > 0.5
          };
          return [newActivity, ...prev.slice(0, 9)];
        });
      }, 15000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getTypeIcon = (type: string) => {
    switch(type) {
      case 'user': return <Users className="w-4 h-4 text-blue-500" />;
      case 'service_principal': return <Key className="w-4 h-4 text-purple-500" />;
      case 'managed_identity': return <Shield className="w-4 h-4 text-green-500" />;
      case 'group': return <Users className="w-4 h-4 text-cyan-500" />;
      case 'guest': return <Globe className="w-4 h-4 text-orange-500" />;
      default: return <Users className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': return 'text-green-500 bg-green-900/20';
      case 'inactive': return 'text-gray-500 bg-gray-900/20';
      case 'suspended': return 'text-red-500 bg-red-900/20';
      case 'pending': return 'text-yellow-500 bg-yellow-900/20';
      case 'expired': return 'text-orange-500 bg-orange-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getRiskColor = (level: string) => {
    switch(level) {
      case 'critical': return 'text-red-500 bg-red-900/20';
      case 'high': return 'text-orange-500 bg-orange-900/20';
      case 'medium': return 'text-yellow-500 bg-yellow-900/20';
      case 'low': return 'text-blue-500 bg-blue-900/20';
      case 'none': return 'text-green-500 bg-green-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getAuthMethodIcon = (type: string) => {
    switch(type) {
      case 'password': return <Key className="w-4 h-4 text-gray-500" />;
      case 'mfa': return <Smartphone className="w-4 h-4 text-green-500" />;
      case 'fido2': return <Fingerprint className="w-4 h-4 text-blue-500" />;
      case 'phone': return <Smartphone className="w-4 h-4 text-purple-500" />;
      case 'email': return <Mail className="w-4 h-4 text-orange-500" />;
      case 'app': return <Shield className="w-4 h-4 text-cyan-500" />;
      default: return <Lock className="w-4 h-4 text-gray-500" />;
    }
  };

  const filteredIdentities = identities.filter(identity => {
    if (searchQuery && !identity.displayName.toLowerCase().includes(searchQuery.toLowerCase()) && 
        !identity.principalName.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    if (filterType !== 'all' && identity.type !== filterType) return false;
    return true;
  });

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Identity Management</h1>
            <p className="text-sm text-gray-400 mt-1">Azure Active Directory identity and access management</p>
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
                viewMode === 'identities' ? 'authentication' : 
                viewMode === 'authentication' ? 'activity' : 
                viewMode === 'activity' ? 'risks' : 'identities'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
            >
              {viewMode === 'identities' ? 'Authentication' : 
               viewMode === 'authentication' ? 'Activity' : 
               viewMode === 'activity' ? 'Risks' : 'Identities'}
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <UserPlus className="w-4 h-4" />
              <span>New Identity</span>
            </button>
          </div>
        </div>
      </header>

      {/* Metrics Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-8 gap-4">
          <div className="flex items-center space-x-3">
            <Users className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Total</p>
              <p className="text-xl font-bold">{metrics?.totalIdentities.toLocaleString()}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <UserCheck className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Active</p>
              <p className="text-xl font-bold">{metrics?.activeUsers.toLocaleString()}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Globe className="w-5 h-5 text-orange-500" />
            <div>
              <p className="text-xs text-gray-400">Guests</p>
              <p className="text-xl font-bold">{metrics?.guestUsers}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Key className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Service</p>
              <p className="text-xl font-bold">{metrics?.servicePrincipals}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Shield className="w-5 h-5 text-cyan-500" />
            <div>
              <p className="text-xs text-gray-400">MFA</p>
              <p className="text-xl font-bold">{metrics?.mfaAdoption}%</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Risky</p>
              <p className="text-xl font-bold text-red-500">{metrics?.riskySignIns}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XCircle className="w-5 h-5 text-orange-500" />
            <div>
              <p className="text-xs text-gray-400">Failed</p>
              <p className="text-xl font-bold text-orange-500">{metrics?.failedSignIns}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Clock className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Expiring</p>
              <p className="text-xl font-bold text-yellow-500">{metrics?.passwordExpiring}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {viewMode === 'identities' && (
          <>
            <div className="mb-6">
              <div className="flex items-center space-x-3">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search identities..."
                    className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                  />
                </div>
                
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                >
                  <option value="all">All Types</option>
                  <option value="user">Users</option>
                  <option value="guest">Guests</option>
                  <option value="service_principal">Service Principals</option>
                  <option value="managed_identity">Managed Identities</option>
                  <option value="group">Groups</option>
                </select>
              </div>
            </div>

            <div className="space-y-3">
              {filteredIdentities.map(identity => (
                <div 
                  key={identity.id} 
                  className={`bg-gray-900 border border-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-800/50 ${
                    selectedIdentity?.id === identity.id ? 'ring-2 ring-blue-500' : ''
                  }`}
                  onClick={() => setSelectedIdentity(selectedIdentity?.id === identity.id ? null : identity)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3">
                      {getTypeIcon(identity.type)}
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-1">
                          <h3 className="text-sm font-bold">{identity.displayName}</h3>
                          <span className={`px-2 py-1 text-xs rounded ${getStatusColor(identity.status)}`}>
                            {identity.status.toUpperCase()}
                          </span>
                          <span className={`px-2 py-1 text-xs rounded ${getRiskColor(identity.riskLevel)}`}>
                            {identity.riskLevel.toUpperCase()} RISK
                          </span>
                        </div>
                        <p className="text-xs text-gray-400 mb-2">{identity.principalName}</p>
                        
                        <div className="grid grid-cols-4 gap-4 text-xs">
                          <div>
                            <span className="text-gray-400">Type: </span>
                            <span className="capitalize">{identity.type.replace('_', ' ')}</span>
                          </div>
                          {identity.department && (
                            <div>
                              <span className="text-gray-400">Dept: </span>
                              <span>{identity.department}</span>
                            </div>
                          )}
                          <div>
                            <span className="text-gray-400">Created: </span>
                            <span>{identity.createdDate}</span>
                          </div>
                          {identity.lastSignIn && (
                            <div>
                              <span className="text-gray-400">Last Sign-in: </span>
                              <span>{identity.lastSignIn}</span>
                            </div>
                          )}
                        </div>

                        <div className="flex items-center space-x-4 mt-2">
                          {identity.mfaEnabled ? (
                            <CheckCircle className="w-3 h-3 text-green-500" />
                          ) : (
                            <XCircle className="w-3 h-3 text-red-500" />
                          )}
                          <span className="text-xs text-gray-500">MFA</span>
                          
                          {identity.conditionalAccess ? (
                            <CheckCircle className="w-3 h-3 text-green-500" />
                          ) : (
                            <XCircle className="w-3 h-3 text-gray-500" />
                          )}
                          <span className="text-xs text-gray-500">Conditional Access</span>
                          
                          {identity.privilegedAccess && (
                            <>
                              <Shield className="w-3 h-3 text-yellow-500" />
                              <span className="text-xs text-gray-500">Privileged</span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <button className="p-1 hover:bg-gray-800 rounded">
                        <Edit className="w-4 h-4 text-gray-500" />
                      </button>
                      <button className="p-1 hover:bg-gray-800 rounded">
                        <Trash2 className="w-4 h-4 text-red-500" />
                      </button>
                    </div>
                  </div>

                  {selectedIdentity?.id === identity.id && (
                    <div className="mt-4 pt-4 border-t border-gray-700">
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <p className="text-xs text-gray-400 mb-2">Groups ({identity.groups.length})</p>
                          <div className="space-y-1">
                            {identity.groups.map(group => (
                              <span key={group} className="block text-xs text-gray-300">{group}</span>
                            ))}
                          </div>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400 mb-2">Applications ({identity.applications.length})</p>
                          <div className="space-y-1">
                            {identity.applications.map(app => (
                              <span key={app} className="block text-xs text-gray-300">{app}</span>
                            ))}
                          </div>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400 mb-2">Licenses ({identity.licenses.length})</p>
                          <div className="space-y-1">
                            {identity.licenses.map(license => (
                              <span key={license} className="block text-xs text-gray-300">{license}</span>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div className="mt-4 flex items-center space-x-3">
                        <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                          Reset Password
                        </button>
                        <button className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-xs">
                          Enable MFA
                        </button>
                        <button className="px-3 py-1 bg-gray-600 hover:bg-gray-700 rounded text-xs">
                          View Audit Log
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'authentication' && (
          <>
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-3">Authentication Methods</h3>
              <p className="text-sm text-gray-400">Configured authentication methods for all identities</p>
            </div>
            
            <div className="grid grid-cols-3 gap-4">
              {authMethods.map(method => (
                <div key={method.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    {getAuthMethodIcon(method.type)}
                    <span className={`px-2 py-1 text-xs rounded ${
                      method.status === 'enabled' ? 'bg-green-900/20 text-green-500' : 'bg-gray-900/20 text-gray-500'
                    }`}>
                      {method.status.toUpperCase()}
                    </span>
                  </div>
                  
                  <h4 className="text-sm font-bold mb-1 capitalize">{method.type.replace('_', ' ')}</h4>
                  <p className="text-xs text-gray-400 mb-3">User: {method.userId}</p>
                  
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Strength:</span>
                      <span className={
                        method.strength === 'strong' ? 'text-green-500' :
                        method.strength === 'medium' ? 'text-yellow-500' : 'text-red-500'
                      }>
                        {method.strength.toUpperCase()}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Configured:</span>
                      {method.configured ? 
                        <CheckCircle className="w-3 h-3 text-green-500" /> : 
                        <XCircle className="w-3 h-3 text-red-500" />
                      }
                    </div>
                    {method.lastUsed && (
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Last Used:</span>
                        <span>{method.lastUsed}</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="mt-3 pt-3 border-t border-gray-700">
                    <button className="w-full px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                      Configure
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'activity' && (
          <div className="space-y-3">
            <div className="mb-4">
              <h3 className="text-sm font-bold mb-2">Sign-in Activity</h3>
              <p className="text-sm text-gray-400">Real-time sign-in monitoring and analysis</p>
            </div>
            
            {signInActivity.map(activity => (
              <div key={activity.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`w-2 h-2 rounded-full ${
                      activity.status === 'success' ? 'bg-green-500' :
                      activity.status === 'failure' ? 'bg-red-500' :
                      activity.status === 'blocked' ? 'bg-orange-500' :
                      'bg-yellow-500 animate-pulse'
                    }`} />
                    <div>
                      <div className="flex items-center space-x-3">
                        <h4 className="text-sm font-bold">{activity.userName}</h4>
                        <span className={`px-2 py-1 text-xs rounded ${
                          activity.status === 'success' ? 'bg-green-900/20 text-green-500' :
                          activity.status === 'failure' ? 'bg-red-900/20 text-red-500' :
                          activity.status === 'blocked' ? 'bg-orange-900/20 text-orange-500' :
                          'bg-yellow-900/20 text-yellow-500'
                        }`}>
                          {activity.status.toUpperCase()}
                        </span>
                        {activity.riskLevel !== 'none' && (
                          <span className={`px-2 py-1 text-xs rounded ${getRiskColor(activity.riskLevel)}`}>
                            {activity.riskLevel.toUpperCase()} RISK
                          </span>
                        )}
                      </div>
                      <div className="flex items-center space-x-4 mt-1 text-xs text-gray-500">
                        <span>{new Date(activity.timestamp).toLocaleString()}</span>
                        <span>{activity.location}</span>
                        <span className="font-mono">{activity.ipAddress}</span>
                        <span>{activity.device}</span>
                        <span>{activity.application}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    {activity.mfaUsed && (
                      <Shield className="w-4 h-4 text-green-500" />
                    )}
                    {activity.conditionalAccessApplied && (
                      <Lock className="w-4 h-4 text-blue-500" />
                    )}
                    <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                      Details
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'risks' && (
          <>
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-3">Risky Users</h3>
              <p className="text-sm text-gray-400">Users with elevated risk levels requiring attention</p>
            </div>
            
            <div className="space-y-4">
              {riskyUsers.map(user => (
                <div key={user.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center space-x-3 mb-2">
                        <AlertTriangle className={`w-5 h-5 ${
                          user.riskLevel === 'critical' ? 'text-red-500' :
                          user.riskLevel === 'high' ? 'text-orange-500' :
                          user.riskLevel === 'medium' ? 'text-yellow-500' :
                          'text-blue-500'
                        }`} />
                        <h4 className="text-sm font-bold">{user.displayName}</h4>
                        <span className={`px-2 py-1 text-xs rounded ${getRiskColor(user.riskLevel)}`}>
                          {user.riskLevel.toUpperCase()} RISK
                        </span>
                        <span className={`px-2 py-1 text-xs rounded ${
                          user.riskState === 'confirmedCompromised' ? 'bg-red-900/20 text-red-500' :
                          user.riskState === 'atRisk' ? 'bg-yellow-900/20 text-yellow-500' :
                          user.riskState === 'dismissed' ? 'bg-gray-900/20 text-gray-500' :
                          'bg-green-900/20 text-green-500'
                        }`}>
                          {user.riskState.replace(/([A-Z])/g, ' $1').toUpperCase()}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400">{user.riskDetail}</p>
                    </div>
                    <span className="text-xs text-gray-500">{user.riskLastUpdated}</span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 mb-3">
                    <div>
                      <p className="text-xs text-gray-400 mb-2">Risk Event Types</p>
                      <div className="flex flex-wrap gap-1">
                        {user.riskEventTypes.map(event => (
                          <span key={event} className="px-2 py-1 bg-red-900/20 text-red-500 rounded text-xs">
                            {event}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400 mb-2">Recommended Actions</p>
                      <div className="space-y-1">
                        {user.recommendedActions.map((action, idx) => (
                          <div key={idx} className="flex items-center space-x-2 text-xs">
                            <ChevronRight className="w-3 h-3 text-gray-500" />
                            <span>{action}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <button className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-xs">
                      Remediate
                    </button>
                    <button className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-xs">
                      Investigate
                    </button>
                    <button className="px-3 py-1 bg-gray-600 hover:bg-gray-700 rounded text-xs">
                      Dismiss
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