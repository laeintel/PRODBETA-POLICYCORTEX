'use client';

import React, { useState, useEffect } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { 
  UserCheck, 
  Plus, 
  Search, 
  Filter,
  MoreVertical,
  Edit,
  Trash2,
  Shield,
  Key,
  Crown,
  Users,
  Settings,
  Download,
  Upload,
  RefreshCw,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  Copy,
  Star,
  Award,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Activity,
  BarChart3,
  TrendingUp,
  Globe,
  Database,
  Server,
  Cloud,
  Terminal,
  FileText,
  Mail,
  Calendar
} from 'lucide-react';
import { Button } from '../../../components/ui/button';
import { Input } from '../../../components/ui/input';
import { Badge } from '../../../components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../components/ui/tabs';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '../../../components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../../components/ui/select';
import { Switch } from '../../../components/ui/switch';
import { Label } from '../../../components/ui/label';
import { Textarea } from '../../../components/ui/textarea';
import { Progress } from '../../../components/ui/progress';

interface Role {
  id: string;
  name: string;
  displayName: string;
  description: string;
  type: 'system' | 'custom' | 'inherited';
  permissions: Permission[];
  userCount: number;
  groupCount: number;
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  isActive: boolean;
  category: 'admin' | 'manager' | 'analyst' | 'developer' | 'viewer' | 'custom';
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  inheritance?: {
    parentRole: string;
    overrides: string[];
  };
}

interface Permission {
  id: string;
  name: string;
  displayName: string;
  description: string;
  category: 'system' | 'data' | 'user' | 'policy' | 'infrastructure' | 'api';
  resource: string;
  action: 'create' | 'read' | 'update' | 'delete' | 'execute' | 'admin';
  scope: 'global' | 'tenant' | 'group' | 'self';
  critical: boolean;
  conditions?: string[];
}

interface RoleTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  permissions: string[];
  recommended: boolean;
}

const mockPermissions: Permission[] = [
  {
    id: 'perm-1',
    name: 'system:admin',
    displayName: 'System Administration',
    description: 'Full administrative access to all system functions',
    category: 'system',
    resource: '*',
    action: 'admin',
    scope: 'global',
    critical: true
  },
  {
    id: 'perm-2',
    name: 'policies:read',
    displayName: 'Read Policies',
    description: 'View policy configurations and rules',
    category: 'policy',
    resource: 'policies',
    action: 'read',
    scope: 'tenant',
    critical: false
  },
  {
    id: 'perm-3',
    name: 'policies:write',
    displayName: 'Manage Policies',
    description: 'Create and modify policy configurations',
    category: 'policy',
    resource: 'policies',
    action: 'update',
    scope: 'tenant',
    critical: true
  },
  {
    id: 'perm-4',
    name: 'users:manage',
    displayName: 'User Management',
    description: 'Create, modify, and delete user accounts',
    category: 'user',
    resource: 'users',
    action: 'admin',
    scope: 'tenant',
    critical: true
  },
  {
    id: 'perm-5',
    name: 'data:read',
    displayName: 'Read Data',
    description: 'Access to read application data and reports',
    category: 'data',
    resource: 'data',
    action: 'read',
    scope: 'group',
    critical: false
  },
  {
    id: 'perm-6',
    name: 'api:execute',
    displayName: 'API Access',
    description: 'Execute API calls and integrations',
    category: 'api',
    resource: 'api',
    action: 'execute',
    scope: 'tenant',
    critical: false
  }
];

const mockRoles: Role[] = [
  {
    id: 'role-1',
    name: 'system-admin',
    displayName: 'System Administrator',
    description: 'Full administrative access to all system functions and data',
    type: 'system',
    permissions: mockPermissions,
    userCount: 3,
    groupCount: 1,
    createdAt: '2023-01-15T08:00:00Z',
    updatedAt: '2024-01-15T10:30:00Z',
    createdBy: 'System',
    isActive: true,
    category: 'admin',
    riskLevel: 'critical'
  },
  {
    id: 'role-2',
    name: 'policy-manager',
    displayName: 'Policy Manager',
    description: 'Manage and configure organizational policies and compliance rules',
    type: 'custom',
    permissions: mockPermissions.filter(p => p.category === 'policy' || p.name === 'data:read'),
    userCount: 8,
    groupCount: 2,
    createdAt: '2023-06-20T14:15:00Z',
    updatedAt: '2024-01-10T09:20:00Z',
    createdBy: 'admin@policycortex.com',
    isActive: true,
    category: 'manager',
    riskLevel: 'high'
  },
  {
    id: 'role-3',
    name: 'security-analyst',
    displayName: 'Security Analyst',
    description: 'Analyze security data and generate compliance reports',
    type: 'custom',
    permissions: [mockPermissions[1], mockPermissions[4], mockPermissions[5]],
    userCount: 15,
    groupCount: 3,
    createdAt: '2023-08-10T11:30:00Z',
    updatedAt: '2023-12-05T16:45:00Z',
    createdBy: 'manager@policycortex.com',
    isActive: true,
    category: 'analyst',
    riskLevel: 'medium'
  },
  {
    id: 'role-4',
    name: 'developer',
    displayName: 'Developer',
    description: 'Development and API access for integration purposes',
    type: 'custom',
    permissions: [mockPermissions[5], mockPermissions[4]],
    userCount: 24,
    groupCount: 4,
    createdAt: '2023-09-15T13:20:00Z',
    updatedAt: '2024-01-08T11:10:00Z',
    createdBy: 'lead@policycortex.com',
    isActive: true,
    category: 'developer',
    riskLevel: 'low'
  },
  {
    id: 'role-5',
    name: 'read-only',
    displayName: 'Read-Only Viewer',
    description: 'View-only access to dashboards and reports',
    type: 'system',
    permissions: [mockPermissions[1], mockPermissions[4]],
    userCount: 45,
    groupCount: 8,
    createdAt: '2023-01-15T08:00:00Z',
    updatedAt: '2023-11-20T14:30:00Z',
    createdBy: 'System',
    isActive: true,
    category: 'viewer',
    riskLevel: 'low'
  },
  {
    id: 'role-6',
    name: 'guest-contractor',
    displayName: 'Guest Contractor',
    description: 'Limited access for external contractors and consultants',
    type: 'custom',
    permissions: [mockPermissions[4]],
    userCount: 7,
    groupCount: 1,
    createdAt: '2023-11-05T09:45:00Z',
    updatedAt: '2024-01-12T15:20:00Z',
    createdBy: 'hr@policycortex.com',
    isActive: false,
    category: 'custom',
    riskLevel: 'low'
  }
];

const mockTemplates: RoleTemplate[] = [
  {
    id: 'template-1',
    name: 'Compliance Officer',
    description: 'Standard role for compliance and audit personnel',
    category: 'compliance',
    permissions: ['policies:read', 'data:read', 'reports:generate'],
    recommended: true
  },
  {
    id: 'template-2',
    name: 'Team Lead',
    description: 'Team leadership with user management capabilities',
    category: 'management',
    permissions: ['users:manage', 'data:read', 'policies:read'],
    recommended: true
  },
  {
    id: 'template-3',
    name: 'API Developer',
    description: 'Developer role focused on API integration',
    category: 'technical',
    permissions: ['api:execute', 'data:read', 'docs:read'],
    recommended: false
  }
];

export default function RolesPage() {
  const [roles, setRoles] = useState<Role[]>(mockRoles);
  const [permissions, setPermissions] = useState<Permission[]>(mockPermissions);
  const [templates, setTemplates] = useState<RoleTemplate[]>(mockTemplates);
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [riskFilter, setRiskFilter] = useState<string>('all');
  const [activeTab, setActiveTab] = useState('roles');
  const [selectedRole, setSelectedRole] = useState<Role | null>(null);
  const [isCreatingRole, setIsCreatingRole] = useState(false);
  const [isEditingRole, setIsEditingRole] = useState(false);
  const [viewPermissionDetails, setViewPermissionDetails] = useState(false);

  const [newRole, setNewRole] = useState({
    name: '',
    displayName: '',
    description: '',
    category: 'custom' as const,
    permissions: [] as string[],
    isActive: true
  });

  const filteredRoles = roles.filter(role => {
    const matchesSearch = 
      role.displayName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      role.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      role.description.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesType = typeFilter === 'all' || role.type === typeFilter;
    const matchesCategory = categoryFilter === 'all' || role.category === categoryFilter;
    const matchesRisk = riskFilter === 'all' || role.riskLevel === riskFilter;

    return matchesSearch && matchesType && matchesCategory && matchesRisk;
  });

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'bg-green-500/20 text-green-400 border-green-500/20';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/20';
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/20';
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/20';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/20';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'system': return 'bg-purple-500/20 text-purple-400 border-purple-500/20';
      case 'custom': return 'bg-blue-500/20 text-blue-400 border-blue-500/20';
      case 'inherited': return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/20';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/20';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'admin': return <Crown className="w-4 h-4" />;
      case 'manager': return <Users className="w-4 h-4" />;
      case 'analyst': return <BarChart3 className="w-4 h-4" />;
      case 'developer': return <Terminal className="w-4 h-4" />;
      case 'viewer': return <Eye className="w-4 h-4" />;
      default: return <Shield className="w-4 h-4" />;
    }
  };

  const getPermissionCategoryIcon = (category: string) => {
    switch (category) {
      case 'system': return <Server className="w-4 h-4" />;
      case 'data': return <Database className="w-4 h-4" />;
      case 'user': return <Users className="w-4 h-4" />;
      case 'policy': return <Shield className="w-4 h-4" />;
      case 'infrastructure': return <Cloud className="w-4 h-4" />;
      case 'api': return <Globe className="w-4 h-4" />;
      default: return <Key className="w-4 h-4" />;
    }
  };

  const handleCreateRole = () => {
    const role: Role = {
      id: `role-${Date.now()}`,
      name: newRole.name.toLowerCase().replace(/\s+/g, '-'),
      displayName: newRole.displayName,
      description: newRole.description,
      type: 'custom',
      permissions: permissions.filter(p => newRole.permissions.includes(p.id)),
      userCount: 0,
      groupCount: 0,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      createdBy: 'current-user@policycortex.com',
      isActive: newRole.isActive,
      category: newRole.category,
      riskLevel: newRole.permissions.some(id => permissions.find(p => p.id === id)?.critical) ? 'high' : 'medium'
    };

    setRoles([...roles, role]);
    setNewRole({
      name: '',
      displayName: '',
      description: '',
      category: 'custom',
      permissions: [],
      isActive: true
    });
    setIsCreatingRole(false);
  };

  const handleToggleRoleStatus = (roleId: string) => {
    setRoles(roles.map(role => 
      role.id === roleId 
        ? { ...role, isActive: !role.isActive, updatedAt: new Date().toISOString() }
        : role
    ));
  };

  const totalUsers = roles.reduce((sum, role) => sum + role.userCount, 0);
  const activeRoles = roles.filter(r => r.isActive).length;
  const systemRoles = roles.filter(r => r.type === 'system').length;
  const criticalRoles = roles.filter(r => r.riskLevel === 'critical').length;

  const content = (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Roles</p>
                <p className="text-2xl font-bold text-white">{roles.length}</p>
              </div>
              <Crown className="w-8 h-8 text-yellow-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-yellow-500">
              <TrendingUp className="w-3 h-3 mr-1" />
              <span>{activeRoles} active</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Assignments</p>
                <p className="text-2xl font-bold text-white">{totalUsers}</p>
              </div>
              <Users className="w-8 h-8 text-blue-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-blue-500">
              <Activity className="w-3 h-3 mr-1" />
              <span>Across all roles</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">System Roles</p>
                <p className="text-2xl font-bold text-white">{systemRoles}</p>
              </div>
              <Shield className="w-8 h-8 text-purple-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-purple-500">
              <Lock className="w-3 h-3 mr-1" />
              <span>Protected</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Critical Roles</p>
                <p className="text-2xl font-bold text-white">{criticalRoles}</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-red-500">
              <Star className="w-3 h-3 mr-1" />
              <span>High privilege</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-gray-900 border border-gray-800">
          <TabsTrigger value="roles" className="data-[state=active]:bg-black">Roles</TabsTrigger>
          <TabsTrigger value="permissions" className="data-[state=active]:bg-black">Permissions</TabsTrigger>
          <TabsTrigger value="templates" className="data-[state=active]:bg-black">Templates</TabsTrigger>
          <TabsTrigger value="analytics" className="data-[state=active]:bg-black">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="roles" className="space-y-6">
          {/* Filters and Actions */}
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
            <div className="flex flex-wrap items-center gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search roles..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 w-64 bg-gray-900 border-gray-700 text-white"
                />
              </div>
              
              <Select value={typeFilter} onValueChange={setTypeFilter}>
                <SelectTrigger className="w-32 bg-gray-900 border-gray-700">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="system">System</SelectItem>
                  <SelectItem value="custom">Custom</SelectItem>
                  <SelectItem value="inherited">Inherited</SelectItem>
                </SelectContent>
              </Select>

              <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                <SelectTrigger className="w-40 bg-gray-900 border-gray-700">
                  <SelectValue placeholder="Category" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="all">All Categories</SelectItem>
                  <SelectItem value="admin">Admin</SelectItem>
                  <SelectItem value="manager">Manager</SelectItem>
                  <SelectItem value="analyst">Analyst</SelectItem>
                  <SelectItem value="developer">Developer</SelectItem>
                  <SelectItem value="viewer">Viewer</SelectItem>
                </SelectContent>
              </Select>

              <Select value={riskFilter} onValueChange={setRiskFilter}>
                <SelectTrigger className="w-32 bg-gray-900 border-gray-700">
                  <SelectValue placeholder="Risk" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="all">All Risk</SelectItem>
                  <SelectItem value="low">Low</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                  <SelectItem value="critical">Critical</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-3">
              <Button
                onClick={() => setIsCreatingRole(true)}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                <Plus className="w-4 h-4 mr-2" />
                Create Role
              </Button>
              
              <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
          </div>

          {/* Roles Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {filteredRoles.map((role) => (
              <Card key={role.id} className="bg-black border-gray-800">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      {getCategoryIcon(role.category)}
                      <div>
                        <h3 className="font-semibold text-white">{role.displayName}</h3>
                        <p className="text-sm text-gray-400">{role.name}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge className={getTypeColor(role.type)} variant="outline">
                        {role.type}
                      </Badge>
                      <Badge className={getRiskColor(role.riskLevel)} variant="outline">
                        {role.riskLevel}
                      </Badge>
                    </div>
                  </div>

                  <p className="text-sm text-gray-400 mb-4 line-clamp-2">{role.description}</p>

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="text-center p-3 bg-gray-900 rounded-lg">
                      <p className="text-lg font-bold text-white">{role.userCount}</p>
                      <p className="text-xs text-gray-400">Users</p>
                    </div>
                    <div className="text-center p-3 bg-gray-900 rounded-lg">
                      <p className="text-lg font-bold text-white">{role.permissions.length}</p>
                      <p className="text-xs text-gray-400">Permissions</p>
                    </div>
                  </div>

                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Status:</span>
                      <div className="flex items-center space-x-1">
                        {role.isActive ? (
                          <>
                            <CheckCircle className="w-3 h-3 text-green-500" />
                            <span className="text-green-500">Active</span>
                          </>
                        ) : (
                          <>
                            <XCircle className="w-3 h-3 text-red-500" />
                            <span className="text-red-500">Inactive</span>
                          </>
                        )}
                      </div>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Created:</span>
                      <span className="text-white">{new Date(role.createdAt).toLocaleDateString()}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Updated:</span>
                      <span className="text-white">{new Date(role.updatedAt).toLocaleDateString()}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-gray-800">
                    <div className="flex items-center space-x-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setSelectedRole(role);
                          setViewPermissionDetails(true);
                        }}
                        className="text-blue-400 hover:text-blue-300"
                      >
                        <Eye className="w-3 h-3 mr-1" />
                        Details
                      </Button>
                      {role.type !== 'system' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setSelectedRole(role);
                            setIsEditingRole(true);
                          }}
                          className="text-gray-400 hover:text-white"
                        >
                          <Edit className="w-3 h-3 mr-1" />
                          Edit
                        </Button>
                      )}
                    </div>
                    <div className="flex items-center space-x-1">
                      {role.type !== 'system' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleToggleRoleStatus(role.id)}
                          className="text-gray-400 hover:text-white h-8 w-8 p-0"
                        >
                          {role.isActive ? (
                            <Lock className="w-3 h-3" />
                          ) : (
                            <Unlock className="w-3 h-3" />
                          )}
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-gray-400 hover:text-white h-8 w-8 p-0"
                      >
                        <MoreVertical className="w-3 h-3" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="permissions" className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-white">Permission Registry</h3>
            <Button className="bg-blue-600 hover:bg-blue-700 text-white">
              <Plus className="w-4 h-4 mr-2" />
              Add Permission
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {permissions.map((permission) => (
              <Card key={permission.id} className="bg-black border-gray-800">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      {getPermissionCategoryIcon(permission.category)}
                      <div>
                        <h4 className="font-semibold text-white">{permission.displayName}</h4>
                        <p className="text-sm text-gray-400 font-mono">{permission.name}</p>
                      </div>
                    </div>
                    {permission.critical && (
                      <Badge className="bg-red-500/20 text-red-400 border-red-500/20" variant="outline">
                        Critical
                      </Badge>
                    )}
                  </div>
                  
                  <p className="text-sm text-gray-400 mb-4">{permission.description}</p>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Category:</span>
                      <Badge variant="secondary" className="bg-blue-500/20 text-blue-400 capitalize">
                        {permission.category}
                      </Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Action:</span>
                      <span className="text-white capitalize">{permission.action}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Scope:</span>
                      <span className="text-white capitalize">{permission.scope}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Resource:</span>
                      <span className="text-white font-mono">{permission.resource}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-end pt-4 border-t border-gray-800 mt-4">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-gray-400 hover:text-white"
                    >
                      <Eye className="w-3 h-3 mr-1" />
                      View Usage
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="templates" className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-white">Role Templates</h3>
            <Button className="bg-purple-600 hover:bg-purple-700 text-white">
              <Plus className="w-4 h-4 mr-2" />
              Create Template
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {templates.map((template) => (
              <Card key={template.id} className="bg-black border-gray-800">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <div className="flex items-center space-x-2 mb-1">
                        <h4 className="font-semibold text-white">{template.name}</h4>
                        {template.recommended && (
                          <Badge className="bg-green-500/20 text-green-400 border-green-500/20" variant="outline">
                            <Star className="w-3 h-3 mr-1" />
                            Recommended
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm text-gray-400 capitalize">{template.category}</p>
                    </div>
                  </div>
                  
                  <p className="text-sm text-gray-400 mb-4">{template.description}</p>
                  
                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Permissions:</span>
                      <span className="text-white">{template.permissions.length}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-gray-800">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-purple-400 hover:text-purple-300"
                    >
                      <Eye className="w-3 h-3 mr-1" />
                      Preview
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-green-400 hover:text-green-300"
                    >
                      <Copy className="w-3 h-3 mr-1" />
                      Use Template
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <Card className="bg-black border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Role Analytics</CardTitle>
              <CardDescription>Role usage and permission distribution analytics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12">
                <BarChart3 className="w-16 h-16 text-gray-700 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">Role Analytics</h3>
                <p className="text-gray-400 mb-6">Comprehensive role and permission analytics coming soon</p>
                <div className="flex justify-center space-x-4">
                  <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                    <TrendingUp className="w-4 h-4 mr-2" />
                    Usage Trends
                  </Button>
                  <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                    <Shield className="w-4 h-4 mr-2" />
                    Risk Analysis
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Create Role Dialog */}
      <Dialog open={isCreatingRole} onOpenChange={setIsCreatingRole}>
        <DialogContent className="bg-gray-900 border-gray-800 text-white max-w-3xl">
          <DialogHeader>
            <DialogTitle>Create New Role</DialogTitle>
            <DialogDescription>
              Define a new role with specific permissions and access levels
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-6 py-4 max-h-96 overflow-y-auto">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="role-name" className="text-sm font-medium">
                  Role Name (Internal)
                </Label>
                <Input
                  id="role-name"
                  value={newRole.name}
                  onChange={(e) => setNewRole({...newRole, name: e.target.value})}
                  placeholder="policy-reviewer"
                  className="bg-black border-gray-700 mt-1 font-mono"
                />
              </div>
              <div>
                <Label htmlFor="role-display-name" className="text-sm font-medium">
                  Display Name
                </Label>
                <Input
                  id="role-display-name"
                  value={newRole.displayName}
                  onChange={(e) => setNewRole({...newRole, displayName: e.target.value})}
                  placeholder="Policy Reviewer"
                  className="bg-black border-gray-700 mt-1"
                />
              </div>
            </div>

            <div>
              <Label htmlFor="role-description" className="text-sm font-medium">
                Description
              </Label>
              <Textarea
                id="role-description"
                value={newRole.description}
                onChange={(e) => setNewRole({...newRole, description: e.target.value})}
                placeholder="Describe the role's purpose and responsibilities..."
                rows={3}
                className="bg-black border-gray-700 mt-1"
              />
            </div>

            <div>
              <Label htmlFor="role-category" className="text-sm font-medium">
                Category
              </Label>
              <Select value={newRole.category} onValueChange={(value: any) => setNewRole({...newRole, category: value})}>
                <SelectTrigger className="bg-black border-gray-700 mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="custom">Custom Role</SelectItem>
                  <SelectItem value="admin">Administrator</SelectItem>
                  <SelectItem value="manager">Manager</SelectItem>
                  <SelectItem value="analyst">Analyst</SelectItem>
                  <SelectItem value="developer">Developer</SelectItem>
                  <SelectItem value="viewer">Viewer</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-sm font-medium mb-3 block">
                Permissions
              </Label>
              <div className="grid grid-cols-1 gap-2 max-h-40 overflow-y-auto border border-gray-700 rounded p-3">
                {permissions.map((permission) => (
                  <div key={permission.id} className="flex items-start space-x-3 p-2 hover:bg-gray-800 rounded">
                    <input
                      type="checkbox"
                      id={`perm-${permission.id}`}
                      checked={newRole.permissions.includes(permission.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setNewRole({
                            ...newRole,
                            permissions: [...newRole.permissions, permission.id]
                          });
                        } else {
                          setNewRole({
                            ...newRole,
                            permissions: newRole.permissions.filter(id => id !== permission.id)
                          });
                        }
                      }}
                      className="rounded border-gray-700 bg-black mt-0.5"
                    />
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <Label htmlFor={`perm-${permission.id}`} className="text-sm font-medium">
                          {permission.displayName}
                        </Label>
                        {permission.critical && (
                          <Badge className="bg-red-500/20 text-red-400 border-red-500/20 text-xs" variant="outline">
                            Critical
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-gray-400">{permission.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="role-active"
                checked={newRole.isActive}
                onCheckedChange={(checked) => setNewRole({...newRole, isActive: checked})}
                className="data-[state=checked]:bg-green-600"
              />
              <Label htmlFor="role-active" className="text-sm">Active Role</Label>
            </div>
          </div>
          <div className="flex justify-end space-x-3">
            <Button
              variant="outline"
              onClick={() => setIsCreatingRole(false)}
              className="border-gray-700 hover:bg-gray-800"
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreateRole}
              disabled={!newRole.name || !newRole.displayName || !newRole.description}
              className="bg-green-600 hover:bg-green-700"
            >
              <Plus className="w-4 h-4 mr-2" />
              Create Role
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Permission Details Dialog */}
      <Dialog open={viewPermissionDetails} onOpenChange={setViewPermissionDetails}>
        <DialogContent className="bg-gray-900 border-gray-800 text-white max-w-4xl">
          <DialogHeader>
            <DialogTitle>Role Details: {selectedRole?.displayName}</DialogTitle>
            <DialogDescription>
              Complete role information and permission breakdown
            </DialogDescription>
          </DialogHeader>
          {selectedRole && (
            <div className="space-y-6 py-4 max-h-96 overflow-y-auto">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-400 mb-2">Basic Information</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Name:</span>
                        <span className="text-white font-mono">{selectedRole.name}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Type:</span>
                        <Badge className={getTypeColor(selectedRole.type)} variant="outline">
                          {selectedRole.type}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Risk Level:</span>
                        <Badge className={getRiskColor(selectedRole.riskLevel)} variant="outline">
                          {selectedRole.riskLevel}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Created By:</span>
                        <span className="text-white">{selectedRole.createdBy}</span>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-400 mb-2">Usage Statistics</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Assigned Users:</span>
                        <span className="text-white">{selectedRole.userCount}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Assigned Groups:</span>
                        <span className="text-white">{selectedRole.groupCount}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Total Permissions:</span>
                        <span className="text-white">{selectedRole.permissions.length}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Critical Permissions:</span>
                        <span className="text-red-400">
                          {selectedRole.permissions.filter(p => p.critical).length}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-400 mb-3">Permissions ({selectedRole.permissions.length})</h4>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {selectedRole.permissions.map((permission) => (
                    <div key={permission.id} className="flex items-center justify-between p-3 bg-gray-800 rounded">
                      <div className="flex items-center space-x-3">
                        {getPermissionCategoryIcon(permission.category)}
                        <div>
                          <p className="text-sm font-medium text-white">{permission.displayName}</p>
                          <p className="text-xs text-gray-400">{permission.name}</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="secondary" className="bg-blue-500/20 text-blue-400 text-xs">
                          {permission.category}
                        </Badge>
                        {permission.critical && (
                          <Badge className="bg-red-500/20 text-red-400 border-red-500/20 text-xs" variant="outline">
                            Critical
                          </Badge>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
          <div className="flex justify-end">
            <Button
              variant="outline"
              onClick={() => setViewPermissionDetails(false)}
              className="border-gray-700 hover:bg-gray-800"
            >
              Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );

  return (
    <TacticalPageTemplate 
      title="Role Management" 
      subtitle="Role Management Operations Center" 
      icon={UserCheck}
    >
      {content}
    </TacticalPageTemplate>
  );
}