'use client';

import React, { useState, useEffect } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { 
  Users, 
  Plus, 
  Search, 
  Filter,
  MoreVertical,
  Edit,
  Trash2,
  Shield,
  Key,
  Mail,
  Phone,
  Calendar,
  Clock,
  Activity,
  CheckCircle,
  XCircle,
  AlertTriangle,
  UserPlus,
  UserMinus,
  UserCheck,
  Settings,
  Download,
  Upload,
  RefreshCw,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  Crown,
  User,
  Globe,
  Building,
  MapPin,
  Briefcase,
  GraduationCap,
  Award,
  Star,
  TrendingUp,
  BarChart3
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
import { Avatar } from '../../../components/ui/avatar';
import { Progress } from '../../../components/ui/progress';

interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  displayName: string;
  avatar?: string;
  status: 'active' | 'inactive' | 'suspended' | 'pending';
  roles: string[];
  permissions: string[];
  department: string;
  title: string;
  manager?: string;
  location: string;
  timezone: string;
  phone?: string;
  lastLogin: string;
  createdAt: string;
  lastActivity: string;
  loginCount: number;
  mfaEnabled: boolean;
  ssoProvider?: string;
  attributes: Record<string, any>;
  groups: string[];
  preferences: {
    theme: 'dark' | 'light' | 'system';
    language: string;
    notifications: boolean;
    emailUpdates: boolean;
  };
}

interface Role {
  id: string;
  name: string;
  description: string;
  permissions: string[];
  userCount: number;
  system: boolean;
}

interface Permission {
  id: string;
  name: string;
  description: string;
  category: string;
  critical: boolean;
}

interface UserGroup {
  id: string;
  name: string;
  description: string;
  userCount: number;
  permissions: string[];
}

const mockUsers: User[] = [
  {
    id: 'user-1',
    email: 'sarah.chen@policycortex.com',
    firstName: 'Sarah',
    lastName: 'Chen',
    displayName: 'Sarah Chen',
    avatar: '/avatars/sarah.jpg',
    status: 'active',
    roles: ['admin', 'policy-manager'],
    permissions: ['*'],
    department: 'Engineering',
    title: 'Senior DevOps Engineer',
    manager: 'user-2',
    location: 'San Francisco, CA',
    timezone: 'America/Los_Angeles',
    phone: '+1 (555) 123-4567',
    lastLogin: '2024-01-20T10:15:00Z',
    createdAt: '2023-06-15T08:00:00Z',
    lastActivity: '2024-01-20T10:25:00Z',
    loginCount: 847,
    mfaEnabled: true,
    ssoProvider: 'okta',
    attributes: { clearanceLevel: 'secret', project: 'quantum' },
    groups: ['engineering', 'admins', 'on-call'],
    preferences: {
      theme: 'dark',
      language: 'en-US',
      notifications: true,
      emailUpdates: true
    }
  },
  {
    id: 'user-2',
    email: 'mike.rodriguez@policycortex.com',
    firstName: 'Mike',
    lastName: 'Rodriguez',
    displayName: 'Mike Rodriguez',
    status: 'active',
    roles: ['manager', 'policy-reviewer'],
    permissions: ['read:all', 'write:policies', 'manage:team'],
    department: 'Operations',
    title: 'Operations Manager',
    location: 'Austin, TX',
    timezone: 'America/Chicago',
    phone: '+1 (555) 234-5678',
    lastLogin: '2024-01-20T09:45:00Z',
    createdAt: '2023-03-22T09:30:00Z',
    lastActivity: '2024-01-20T09:50:00Z',
    loginCount: 1243,
    mfaEnabled: true,
    ssoProvider: 'azure-ad',
    attributes: { clearanceLevel: 'confidential', department: 'operations' },
    groups: ['operations', 'managers'],
    preferences: {
      theme: 'light',
      language: 'en-US',
      notifications: true,
      emailUpdates: false
    }
  },
  {
    id: 'user-3',
    email: 'alex.kim@policycortex.com',
    firstName: 'Alex',
    lastName: 'Kim',
    displayName: 'Alex Kim',
    status: 'active',
    roles: ['analyst', 'compliance-officer'],
    permissions: ['read:policies', 'write:reports', 'read:compliance'],
    department: 'Compliance',
    title: 'Security Analyst',
    manager: 'user-2',
    location: 'Remote',
    timezone: 'America/New_York',
    phone: '+1 (555) 345-6789',
    lastLogin: '2024-01-19T16:30:00Z',
    createdAt: '2023-09-10T10:00:00Z',
    lastActivity: '2024-01-19T17:15:00Z',
    loginCount: 432,
    mfaEnabled: false,
    attributes: { clearanceLevel: 'public', specialization: 'azure-security' },
    groups: ['compliance', 'analysts'],
    preferences: {
      theme: 'system',
      language: 'en-US',
      notifications: false,
      emailUpdates: true
    }
  },
  {
    id: 'user-4',
    email: 'emma.taylor@external.com',
    firstName: 'Emma',
    lastName: 'Taylor',
    displayName: 'Emma Taylor',
    status: 'pending',
    roles: ['guest'],
    permissions: ['read:basic'],
    department: 'External',
    title: 'Consultant',
    location: 'London, UK',
    timezone: 'Europe/London',
    lastLogin: 'never',
    createdAt: '2024-01-18T14:20:00Z',
    lastActivity: 'never',
    loginCount: 0,
    mfaEnabled: false,
    attributes: { contractor: true, endDate: '2024-06-30' },
    groups: ['external'],
    preferences: {
      theme: 'dark',
      language: 'en-GB',
      notifications: true,
      emailUpdates: true
    }
  },
  {
    id: 'user-5',
    email: 'john.suspended@policycortex.com',
    firstName: 'John',
    lastName: 'Suspended',
    displayName: 'John Suspended',
    status: 'suspended',
    roles: ['developer'],
    permissions: [],
    department: 'Engineering',
    title: 'Junior Developer',
    manager: 'user-1',
    location: 'Seattle, WA',
    timezone: 'America/Los_Angeles',
    lastLogin: '2024-01-15T13:22:00Z',
    createdAt: '2023-11-08T16:45:00Z',
    lastActivity: '2024-01-15T13:30:00Z',
    loginCount: 156,
    mfaEnabled: true,
    attributes: { suspended: true, reason: 'policy-violation' },
    groups: ['engineering'],
    preferences: {
      theme: 'dark',
      language: 'en-US',
      notifications: false,
      emailUpdates: false
    }
  }
];

const mockRoles: Role[] = [
  {
    id: 'admin',
    name: 'System Administrator',
    description: 'Full system access with all permissions',
    permissions: ['*'],
    userCount: 3,
    system: true
  },
  {
    id: 'manager',
    name: 'Manager',
    description: 'Team management and policy oversight',
    permissions: ['read:all', 'write:policies', 'manage:team'],
    userCount: 8,
    system: false
  },
  {
    id: 'policy-manager',
    name: 'Policy Manager',
    description: 'Policy creation and management',
    permissions: ['read:all', 'write:policies', 'delete:policies'],
    userCount: 12,
    system: false
  },
  {
    id: 'analyst',
    name: 'Security Analyst',
    description: 'Read access to security data and reports',
    permissions: ['read:policies', 'read:reports', 'write:reports'],
    userCount: 24,
    system: false
  },
  {
    id: 'developer',
    name: 'Developer',
    description: 'Development access to APIs and resources',
    permissions: ['read:api', 'write:api', 'read:docs'],
    userCount: 45,
    system: false
  }
];

const mockGroups: UserGroup[] = [
  {
    id: 'engineering',
    name: 'Engineering',
    description: 'Software engineering team',
    userCount: 34,
    permissions: ['read:api', 'write:api', 'read:docs']
  },
  {
    id: 'operations',
    name: 'Operations',
    description: 'IT operations and infrastructure team',
    userCount: 18,
    permissions: ['read:infrastructure', 'write:configs', 'read:monitoring']
  },
  {
    id: 'compliance',
    name: 'Compliance',
    description: 'Compliance and security team',
    userCount: 12,
    permissions: ['read:compliance', 'write:reports', 'read:audits']
  }
];

export default function UsersPage() {
  const [users, setUsers] = useState<User[]>(mockUsers);
  const [roles, setRoles] = useState<Role[]>(mockRoles);
  const [groups, setGroups] = useState<UserGroup[]>(mockGroups);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [roleFilter, setRoleFilter] = useState<string>('all');
  const [departmentFilter, setDepartmentFilter] = useState<string>('all');
  const [activeTab, setActiveTab] = useState('users');
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [isCreatingUser, setIsCreatingUser] = useState(false);
  const [isEditingUser, setIsEditingUser] = useState(false);
  const [bulkSelection, setBulkSelection] = useState<string[]>([]);

  const [newUser, setNewUser] = useState({
    email: '',
    firstName: '',
    lastName: '',
    department: '',
    title: '',
    location: '',
    phone: '',
    roles: [] as string[],
    groups: [] as string[],
    mfaEnabled: true,
    sendInvite: true
  });

  const filteredUsers = users.filter(user => {
    const matchesSearch = 
      user.displayName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.department.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.title.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesStatus = statusFilter === 'all' || user.status === statusFilter;
    const matchesRole = roleFilter === 'all' || user.roles.some(role => role === roleFilter);
    const matchesDepartment = departmentFilter === 'all' || user.department === departmentFilter;

    return matchesSearch && matchesStatus && matchesRole && matchesDepartment;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500/20 text-green-400 border-green-500/20';
      case 'inactive': return 'bg-gray-500/20 text-gray-400 border-gray-500/20';
      case 'suspended': return 'bg-red-500/20 text-red-400 border-red-500/20';
      case 'pending': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/20';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle className="w-4 h-4" />;
      case 'inactive': return <XCircle className="w-4 h-4" />;
      case 'suspended': return <AlertTriangle className="w-4 h-4" />;
      case 'pending': return <Clock className="w-4 h-4" />;
      default: return <User className="w-4 h-4" />;
    }
  };

  const handleCreateUser = () => {
    const user: User = {
      id: `user-${Date.now()}`,
      email: newUser.email,
      firstName: newUser.firstName,
      lastName: newUser.lastName,
      displayName: `${newUser.firstName} ${newUser.lastName}`,
      status: 'pending',
      roles: newUser.roles,
      permissions: [],
      department: newUser.department,
      title: newUser.title,
      location: newUser.location,
      timezone: 'America/Los_Angeles',
      phone: newUser.phone,
      lastLogin: 'never',
      createdAt: new Date().toISOString(),
      lastActivity: 'never',
      loginCount: 0,
      mfaEnabled: newUser.mfaEnabled,
      attributes: {},
      groups: newUser.groups,
      preferences: {
        theme: 'dark',
        language: 'en-US',
        notifications: true,
        emailUpdates: true
      }
    };

    setUsers([...users, user]);
    setNewUser({
      email: '',
      firstName: '',
      lastName: '',
      department: '',
      title: '',
      location: '',
      phone: '',
      roles: [],
      groups: [],
      mfaEnabled: true,
      sendInvite: true
    });
    setIsCreatingUser(false);
  };

  const handleToggleUserStatus = (userId: string) => {
    setUsers(users.map(user => 
      user.id === userId 
        ? { ...user, status: user.status === 'active' ? 'suspended' : 'active' }
        : user
    ));
  };

  const handleBulkAction = (action: string) => {
    switch (action) {
      case 'activate':
        setUsers(users.map(user => 
          bulkSelection.includes(user.id) ? { ...user, status: 'active' } : user
        ));
        break;
      case 'suspend':
        setUsers(users.map(user => 
          bulkSelection.includes(user.id) ? { ...user, status: 'suspended' } : user
        ));
        break;
      case 'delete':
        setUsers(users.filter(user => !bulkSelection.includes(user.id)));
        break;
    }
    setBulkSelection([]);
  };

  const departments = [...new Set(users.map(user => user.department))];
  const activeUsers = users.filter(u => u.status === 'active').length;
  const pendingUsers = users.filter(u => u.status === 'pending').length;
  const suspendedUsers = users.filter(u => u.status === 'suspended').length;

  const content = (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Users</p>
                <p className="text-2xl font-bold text-white">{users.length}</p>
              </div>
              <Users className="w-8 h-8 text-blue-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-blue-500">
              <TrendingUp className="w-3 h-3 mr-1" />
              <span>+{Math.round(users.length * 0.08)} this month</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Active Users</p>
                <p className="text-2xl font-bold text-white">{activeUsers}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-green-500">
              <Activity className="w-3 h-3 mr-1" />
              <span>{Math.round((activeUsers / users.length) * 100)}% of total</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Pending Invites</p>
                <p className="text-2xl font-bold text-white">{pendingUsers}</p>
              </div>
              <Clock className="w-8 h-8 text-yellow-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-yellow-500">
              <Mail className="w-3 h-3 mr-1" />
              <span>Awaiting activation</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Suspended</p>
                <p className="text-2xl font-bold text-white">{suspendedUsers}</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-red-500">
              <Shield className="w-3 h-3 mr-1" />
              <span>Require attention</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-gray-900 border border-gray-800">
          <TabsTrigger value="users" className="data-[state=active]:bg-black">Users</TabsTrigger>
          <TabsTrigger value="roles" className="data-[state=active]:bg-black">Roles</TabsTrigger>
          <TabsTrigger value="groups" className="data-[state=active]:bg-black">Groups</TabsTrigger>
          <TabsTrigger value="analytics" className="data-[state=active]:bg-black">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="users" className="space-y-6">
          {/* Filters and Actions */}
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
            <div className="flex flex-wrap items-center gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search users..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 w-64 bg-gray-900 border-gray-700 text-white"
                />
              </div>
              
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-32 bg-gray-900 border-gray-700">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="active">Active</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                  <SelectItem value="suspended">Suspended</SelectItem>
                  <SelectItem value="inactive">Inactive</SelectItem>
                </SelectContent>
              </Select>

              <Select value={roleFilter} onValueChange={setRoleFilter}>
                <SelectTrigger className="w-40 bg-gray-900 border-gray-700">
                  <SelectValue placeholder="Role" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="all">All Roles</SelectItem>
                  {roles.map(role => (
                    <SelectItem key={role.id} value={role.id}>{role.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select value={departmentFilter} onValueChange={setDepartmentFilter}>
                <SelectTrigger className="w-40 bg-gray-900 border-gray-700">
                  <SelectValue placeholder="Department" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700">
                  <SelectItem value="all">All Departments</SelectItem>
                  {departments.map(dept => (
                    <SelectItem key={dept} value={dept}>{dept}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-3">
              {bulkSelection.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-400">{bulkSelection.length} selected</span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleBulkAction('activate')}
                    className="border-green-600 text-green-400 hover:bg-green-600/10"
                  >
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Activate
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleBulkAction('suspend')}
                    className="border-yellow-600 text-yellow-400 hover:bg-yellow-600/10"
                  >
                    <Lock className="w-3 h-3 mr-1" />
                    Suspend
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleBulkAction('delete')}
                    className="border-red-600 text-red-400 hover:bg-red-600/10"
                  >
                    <Trash2 className="w-3 h-3 mr-1" />
                    Delete
                  </Button>
                </div>
              )}
              
              <Button
                onClick={() => setIsCreatingUser(true)}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                <UserPlus className="w-4 h-4 mr-2" />
                Add User
              </Button>
              
              <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
          </div>

          {/* User List */}
          <Card className="bg-black border-gray-800">
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="border-b border-gray-800">
                    <tr className="text-left">
                      <th className="p-4 w-12">
                        <input
                          type="checkbox"
                          checked={bulkSelection.length === filteredUsers.length && filteredUsers.length > 0}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setBulkSelection(filteredUsers.map(u => u.id));
                            } else {
                              setBulkSelection([]);
                            }
                          }}
                          className="rounded border-gray-700 bg-black"
                        />
                      </th>
                      <th className="p-4 text-gray-400 font-medium">User</th>
                      <th className="p-4 text-gray-400 font-medium">Status</th>
                      <th className="p-4 text-gray-400 font-medium">Role</th>
                      <th className="p-4 text-gray-400 font-medium">Department</th>
                      <th className="p-4 text-gray-400 font-medium">Last Login</th>
                      <th className="p-4 text-gray-400 font-medium">MFA</th>
                      <th className="p-4 text-gray-400 font-medium w-16"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredUsers.map((user) => (
                      <tr key={user.id} className="border-b border-gray-800 hover:bg-gray-900/50">
                        <td className="p-4">
                          <input
                            type="checkbox"
                            checked={bulkSelection.includes(user.id)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setBulkSelection([...bulkSelection, user.id]);
                              } else {
                                setBulkSelection(bulkSelection.filter(id => id !== user.id));
                              }
                            }}
                            className="rounded border-gray-700 bg-black"
                          />
                        </td>
                        <td className="p-4">
                          <div className="flex items-center space-x-3">
                            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center text-white text-sm font-semibold">
                              {user.firstName.charAt(0)}{user.lastName.charAt(0)}
                            </div>
                            <div>
                              <p className="font-medium text-white">{user.displayName}</p>
                              <p className="text-sm text-gray-400">{user.email}</p>
                            </div>
                          </div>
                        </td>
                        <td className="p-4">
                          <Badge className={getStatusColor(user.status)} variant="outline">
                            {getStatusIcon(user.status)}
                            <span className="ml-1 capitalize">{user.status}</span>
                          </Badge>
                        </td>
                        <td className="p-4">
                          <div className="flex flex-wrap gap-1">
                            {user.roles.slice(0, 2).map(role => (
                              <Badge key={role} variant="secondary" className="bg-blue-500/20 text-blue-400 text-xs">
                                {role}
                              </Badge>
                            ))}
                            {user.roles.length > 2 && (
                              <Badge variant="secondary" className="bg-gray-700 text-gray-300 text-xs">
                                +{user.roles.length - 2}
                              </Badge>
                            )}
                          </div>
                        </td>
                        <td className="p-4">
                          <div>
                            <p className="text-sm text-white">{user.department}</p>
                            <p className="text-xs text-gray-400">{user.title}</p>
                          </div>
                        </td>
                        <td className="p-4">
                          <p className="text-sm text-white">
                            {user.lastLogin === 'never' ? 'Never' : new Date(user.lastLogin).toLocaleDateString()}
                          </p>
                          <p className="text-xs text-gray-400">
                            {user.loginCount} logins
                          </p>
                        </td>
                        <td className="p-4">
                          <div className="flex items-center space-x-1">
                            {user.mfaEnabled ? (
                              <Shield className="w-4 h-4 text-green-500" />
                            ) : (
                              <Shield className="w-4 h-4 text-gray-500" />
                            )}
                            {user.ssoProvider && (
                              <span title={`SSO: ${user.ssoProvider}`}>
                                <Globe className="w-4 h-4 text-blue-500" />
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="p-4">
                          <div className="flex items-center space-x-1">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setSelectedUser(user);
                                setIsEditingUser(true);
                              }}
                              className="h-8 w-8 p-0 text-gray-400 hover:text-white"
                            >
                              <Edit className="w-3 h-3" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleToggleUserStatus(user.id)}
                              className="h-8 w-8 p-0 text-gray-400 hover:text-white"
                            >
                              {user.status === 'active' ? (
                                <Lock className="w-3 h-3" />
                              ) : (
                                <Unlock className="w-3 h-3" />
                              )}
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 w-8 p-0 text-gray-400 hover:text-white"
                            >
                              <MoreVertical className="w-3 h-3" />
                            </Button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="roles" className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-white">Role Management</h3>
            <Button className="bg-blue-600 hover:bg-blue-700 text-white">
              <Plus className="w-4 h-4 mr-2" />
              Create Role
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {roles.map((role) => (
              <Card key={role.id} className="bg-black border-gray-800">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      <Crown className="w-5 h-5 text-yellow-500" />
                      <div>
                        <h4 className="font-semibold text-white">{role.name}</h4>
                        {role.system && (
                          <Badge variant="secondary" className="bg-blue-500/20 text-blue-300 text-xs">
                            System Role
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <p className="text-sm text-gray-400 mb-4">{role.description}</p>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Users:</span>
                      <span className="text-white font-semibold">{role.userCount}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Permissions:</span>
                      <span className="text-white font-semibold">{role.permissions.length}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-gray-800 mt-4">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-blue-400 hover:text-blue-300"
                    >
                      <Eye className="w-3 h-3 mr-1" />
                      View Details
                    </Button>
                    {!role.system && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-gray-400 hover:text-white"
                      >
                        <Edit className="w-3 h-3" />
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="groups" className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-white">User Groups</h3>
            <Button className="bg-blue-600 hover:bg-blue-700 text-white">
              <Plus className="w-4 h-4 mr-2" />
              Create Group
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {groups.map((group) => (
              <Card key={group.id} className="bg-black border-gray-800">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      <Users className="w-5 h-5 text-blue-500" />
                      <h4 className="font-semibold text-white">{group.name}</h4>
                    </div>
                  </div>
                  
                  <p className="text-sm text-gray-400 mb-4">{group.description}</p>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Members:</span>
                      <span className="text-white font-semibold">{group.userCount}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Permissions:</span>
                      <span className="text-white font-semibold">{group.permissions.length}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-gray-800 mt-4">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-blue-400 hover:text-blue-300"
                    >
                      <Users className="w-3 h-3 mr-1" />
                      Manage Members
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-gray-400 hover:text-white"
                    >
                      <Edit className="w-3 h-3" />
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
              <CardTitle className="text-white">User Analytics</CardTitle>
              <CardDescription>User activity and engagement metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12">
                <BarChart3 className="w-16 h-16 text-gray-700 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">Analytics Dashboard</h3>
                <p className="text-gray-400 mb-6">Comprehensive user analytics and insights coming soon</p>
                <div className="flex justify-center space-x-4">
                  <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                    <TrendingUp className="w-4 h-4 mr-2" />
                    Login Trends
                  </Button>
                  <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                    <Award className="w-4 h-4 mr-2" />
                    User Engagement
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Create User Dialog */}
      <Dialog open={isCreatingUser} onOpenChange={setIsCreatingUser}>
        <DialogContent className="bg-gray-900 border-gray-800 text-white max-w-3xl">
          <DialogHeader>
            <DialogTitle>Add New User</DialogTitle>
            <DialogDescription>
              Create a new user account and assign roles and permissions
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-6 py-4 max-h-96 overflow-y-auto">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="first-name" className="text-sm font-medium">
                  First Name
                </Label>
                <Input
                  id="first-name"
                  value={newUser.firstName}
                  onChange={(e) => setNewUser({...newUser, firstName: e.target.value})}
                  placeholder="John"
                  className="bg-black border-gray-700 mt-1"
                />
              </div>
              <div>
                <Label htmlFor="last-name" className="text-sm font-medium">
                  Last Name
                </Label>
                <Input
                  id="last-name"
                  value={newUser.lastName}
                  onChange={(e) => setNewUser({...newUser, lastName: e.target.value})}
                  placeholder="Doe"
                  className="bg-black border-gray-700 mt-1"
                />
              </div>
            </div>

            <div>
              <Label htmlFor="email" className="text-sm font-medium">
                Email Address
              </Label>
              <Input
                id="email"
                type="email"
                value={newUser.email}
                onChange={(e) => setNewUser({...newUser, email: e.target.value})}
                placeholder="john.doe@company.com"
                className="bg-black border-gray-700 mt-1"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="department" className="text-sm font-medium">
                  Department
                </Label>
                <Select value={newUser.department} onValueChange={(value) => setNewUser({...newUser, department: value})}>
                  <SelectTrigger className="bg-black border-gray-700 mt-1">
                    <SelectValue placeholder="Select department" />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-900 border-gray-700">
                    {departments.map(dept => (
                      <SelectItem key={dept} value={dept}>{dept}</SelectItem>
                    ))}
                    <SelectItem value="Marketing">Marketing</SelectItem>
                    <SelectItem value="Sales">Sales</SelectItem>
                    <SelectItem value="HR">Human Resources</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label htmlFor="title" className="text-sm font-medium">
                  Job Title
                </Label>
                <Input
                  id="title"
                  value={newUser.title}
                  onChange={(e) => setNewUser({...newUser, title: e.target.value})}
                  placeholder="Software Engineer"
                  className="bg-black border-gray-700 mt-1"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="location" className="text-sm font-medium">
                  Location
                </Label>
                <Input
                  id="location"
                  value={newUser.location}
                  onChange={(e) => setNewUser({...newUser, location: e.target.value})}
                  placeholder="San Francisco, CA"
                  className="bg-black border-gray-700 mt-1"
                />
              </div>
              <div>
                <Label htmlFor="phone" className="text-sm font-medium">
                  Phone Number
                </Label>
                <Input
                  id="phone"
                  value={newUser.phone}
                  onChange={(e) => setNewUser({...newUser, phone: e.target.value})}
                  placeholder="+1 (555) 123-4567"
                  className="bg-black border-gray-700 mt-1"
                />
              </div>
            </div>

            <div>
              <Label className="text-sm font-medium mb-3 block">
                Roles
              </Label>
              <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto">
                {roles.map((role) => (
                  <div key={role.id} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id={`role-${role.id}`}
                      checked={newUser.roles.includes(role.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setNewUser({
                            ...newUser,
                            roles: [...newUser.roles, role.id]
                          });
                        } else {
                          setNewUser({
                            ...newUser,
                            roles: newUser.roles.filter(id => id !== role.id)
                          });
                        }
                      }}
                      className="rounded border-gray-700 bg-black"
                    />
                    <Label htmlFor={`role-${role.id}`} className="text-sm">
                      {role.name}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <Label className="text-sm font-medium mb-3 block">
                User Groups
              </Label>
              <div className="grid grid-cols-2 gap-2">
                {groups.map((group) => (
                  <div key={group.id} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id={`group-${group.id}`}
                      checked={newUser.groups.includes(group.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setNewUser({
                            ...newUser,
                            groups: [...newUser.groups, group.id]
                          });
                        } else {
                          setNewUser({
                            ...newUser,
                            groups: newUser.groups.filter(id => id !== group.id)
                          });
                        }
                      }}
                      className="rounded border-gray-700 bg-black"
                    />
                    <Label htmlFor={`group-${group.id}`} className="text-sm">
                      {group.name}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <Switch
                  id="mfa-enabled"
                  checked={newUser.mfaEnabled}
                  onCheckedChange={(checked) => setNewUser({...newUser, mfaEnabled: checked})}
                  className="data-[state=checked]:bg-green-600"
                />
                <Label htmlFor="mfa-enabled" className="text-sm">Require MFA Setup</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="send-invite"
                  checked={newUser.sendInvite}
                  onCheckedChange={(checked) => setNewUser({...newUser, sendInvite: checked})}
                  className="data-[state=checked]:bg-blue-600"
                />
                <Label htmlFor="send-invite" className="text-sm">Send Welcome Email</Label>
              </div>
            </div>
          </div>
          <div className="flex justify-end space-x-3">
            <Button
              variant="outline"
              onClick={() => setIsCreatingUser(false)}
              className="border-gray-700 hover:bg-gray-800"
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreateUser}
              disabled={!newUser.email || !newUser.firstName || !newUser.lastName}
              className="bg-green-600 hover:bg-green-700"
            >
              <UserPlus className="w-4 h-4 mr-2" />
              Create User
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );

  return (
    <TacticalPageTemplate 
      title="User Management" 
      subtitle="User Management Operations Center" 
      icon={Users}
    >
      {content}
    </TacticalPageTemplate>
  );
}