'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Users,
  UserPlus,
  UserMinus,
  UserCheck,
  UserX,
  Shield,
  Lock,
  Key,
  Settings,
  Eye,
  Edit,
  Trash2,
  Plus,
  Search,
  Filter,
  ChevronRight,
  ChevronDown,
  AlertTriangle,
  CheckCircle,
  Info,
  Clock,
  Calendar,
  Activity,
  FileText,
  FolderOpen,
  Database,
  Cloud,
  Server,
  Network,
  HardDrive,
  Mail,
  RefreshCw,
  Download,
  MoreVertical,
  Crown,
  UserCog,
  ShieldCheck
} from 'lucide-react';

interface User {
  id: string;
  name: string;
  email: string;
  department: string;
  role: Role;
  groups: string[];
  status: 'active' | 'inactive' | 'suspended';
  lastActive: Date;
  created: Date;
  mfaEnabled: boolean;
  privilegedAccess: boolean;
}

interface Role {
  id: string;
  name: string;
  description: string;
  permissions: Permission[];
  users: number;
  priority: number;
  isBuiltIn: boolean;
  color: string;
}

interface Permission {
  id: string;
  resource: string;
  actions: string[];
  scope: 'read' | 'write' | 'delete' | 'admin';
}

interface Group {
  id: string;
  name: string;
  description: string;
  members: number;
  roles: string[];
  created: Date;
  modified: Date;
}

interface AccessRequest {
  id: string;
  requester: string;
  requestType: 'role' | 'permission' | 'group';
  requestedItem: string;
  reason: string;
  status: 'pending' | 'approved' | 'rejected';
  requestedAt: Date;
  reviewedBy?: string;
  reviewedAt?: Date;
}

interface AuditLog {
  id: string;
  user: string;
  action: string;
  resource: string;
  timestamp: Date;
  result: 'success' | 'failure';
  details: string;
}

export default function RBACManagement() {
  const [loading, setLoading] = useState(true);
  const [selectedView, setSelectedView] = useState<'users' | 'roles' | 'groups' | 'requests' | 'audit'>('users');
  const [users, setUsers] = useState<User[]>([]);
  const [roles, setRoles] = useState<Role[]>([]);
  const [groups, setGroups] = useState<Group[]>([]);
  const [accessRequests, setAccessRequests] = useState<AccessRequest[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedRole, setSelectedRole] = useState<Role | null>(null);
  const [expandedPermissions, setExpandedPermissions] = useState<string[]>([]);

  useEffect(() => {
    // Simulate loading data
    setTimeout(() => {
      // Mock roles
      const mockRoles: Role[] = [
        {
          id: 'role-admin',
          name: 'Administrator',
          description: 'Full system access with all permissions',
          permissions: [
            { id: 'p1', resource: 'All Resources', actions: ['create', 'read', 'update', 'delete'], scope: 'admin' }
          ],
          users: 3,
          priority: 1,
          isBuiltIn: true,
          color: '#EF4444'
        },
        {
          id: 'role-developer',
          name: 'Developer',
          description: 'Access to development resources and deployment',
          permissions: [
            { id: 'p2', resource: 'Development Resources', actions: ['create', 'read', 'update'], scope: 'write' },
            { id: 'p3', resource: 'CI/CD Pipelines', actions: ['read', 'execute'], scope: 'write' },
            { id: 'p4', resource: 'Code Repositories', actions: ['read', 'write', 'merge'], scope: 'write' }
          ],
          users: 15,
          priority: 2,
          isBuiltIn: false,
          color: '#3B82F6'
        },
        {
          id: 'role-analyst',
          name: 'Business Analyst',
          description: 'Read access to analytics and reporting',
          permissions: [
            { id: 'p5', resource: 'Analytics Dashboard', actions: ['read'], scope: 'read' },
            { id: 'p6', resource: 'Reports', actions: ['read', 'export'], scope: 'read' },
            { id: 'p7', resource: 'Data Warehouse', actions: ['query'], scope: 'read' }
          ],
          users: 8,
          priority: 3,
          isBuiltIn: false,
          color: '#10B981'
        },
        {
          id: 'role-auditor',
          name: 'Compliance Auditor',
          description: 'Read-only access for compliance and audit purposes',
          permissions: [
            { id: 'p8', resource: 'Audit Logs', actions: ['read'], scope: 'read' },
            { id: 'p9', resource: 'Compliance Reports', actions: ['read', 'export'], scope: 'read' },
            { id: 'p10', resource: 'Security Policies', actions: ['read'], scope: 'read' }
          ],
          users: 4,
          priority: 4,
          isBuiltIn: true,
          color: '#F59E0B'
        },
        {
          id: 'role-operator',
          name: 'Operations',
          description: 'Manage infrastructure and monitoring',
          permissions: [
            { id: 'p11', resource: 'Virtual Machines', actions: ['start', 'stop', 'restart'], scope: 'write' },
            { id: 'p12', resource: 'Monitoring', actions: ['read', 'configure'], scope: 'write' },
            { id: 'p13', resource: 'Backups', actions: ['create', 'restore'], scope: 'write' }
          ],
          users: 6,
          priority: 5,
          isBuiltIn: false,
          color: '#8B5CF6'
        }
      ];

      // Mock users
      setUsers([
        {
          id: 'user-001',
          name: 'John Smith',
          email: 'john.smith@company.com',
          department: 'Engineering',
          role: mockRoles[0],
          groups: ['Administrators', 'Security Team'],
          status: 'active',
          lastActive: new Date('2024-01-09T10:30:00'),
          created: new Date('2023-01-15'),
          mfaEnabled: true,
          privilegedAccess: true
        },
        {
          id: 'user-002',
          name: 'Sarah Johnson',
          email: 'sarah.johnson@company.com',
          department: 'Development',
          role: mockRoles[1],
          groups: ['Developers', 'DevOps'],
          status: 'active',
          lastActive: new Date('2024-01-09T09:45:00'),
          created: new Date('2023-03-20'),
          mfaEnabled: true,
          privilegedAccess: false
        },
        {
          id: 'user-003',
          name: 'Mike Davis',
          email: 'mike.davis@company.com',
          department: 'Analytics',
          role: mockRoles[2],
          groups: ['Analytics Team'],
          status: 'active',
          lastActive: new Date('2024-01-08T16:20:00'),
          created: new Date('2023-06-10'),
          mfaEnabled: false,
          privilegedAccess: false
        },
        {
          id: 'user-004',
          name: 'Emily Chen',
          email: 'emily.chen@company.com',
          department: 'Compliance',
          role: mockRoles[3],
          groups: ['Compliance', 'Audit Team'],
          status: 'active',
          lastActive: new Date('2024-01-09T08:00:00'),
          created: new Date('2023-02-28'),
          mfaEnabled: true,
          privilegedAccess: false
        },
        {
          id: 'user-005',
          name: 'David Wilson',
          email: 'david.wilson@company.com',
          department: 'Operations',
          role: mockRoles[4],
          groups: ['Operations', 'Infrastructure'],
          status: 'inactive',
          lastActive: new Date('2024-01-05T14:30:00'),
          created: new Date('2023-04-15'),
          mfaEnabled: false,
          privilegedAccess: false
        },
        {
          id: 'user-006',
          name: 'Lisa Anderson',
          email: 'lisa.anderson@company.com',
          department: 'Engineering',
          role: mockRoles[1],
          groups: ['Developers'],
          status: 'suspended',
          lastActive: new Date('2023-12-20T10:00:00'),
          created: new Date('2023-07-01'),
          mfaEnabled: true,
          privilegedAccess: false
        }
      ]);

      setRoles(mockRoles);

      // Mock groups
      setGroups([
        {
          id: 'group-001',
          name: 'Administrators',
          description: 'System administrators with full access',
          members: 3,
          roles: ['Administrator'],
          created: new Date('2023-01-01'),
          modified: new Date('2024-01-05')
        },
        {
          id: 'group-002',
          name: 'Developers',
          description: 'Development team members',
          members: 15,
          roles: ['Developer'],
          created: new Date('2023-01-01'),
          modified: new Date('2024-01-08')
        },
        {
          id: 'group-003',
          name: 'Analytics Team',
          description: 'Business analytics and reporting team',
          members: 8,
          roles: ['Business Analyst'],
          created: new Date('2023-01-15'),
          modified: new Date('2024-01-03')
        },
        {
          id: 'group-004',
          name: 'Security Team',
          description: 'Security and compliance team',
          members: 6,
          roles: ['Administrator', 'Compliance Auditor'],
          created: new Date('2023-01-10'),
          modified: new Date('2024-01-07')
        }
      ]);

      // Mock access requests
      setAccessRequests([
        {
          id: 'req-001',
          requester: 'Alex Thompson',
          requestType: 'role',
          requestedItem: 'Developer',
          reason: 'Need access to deploy new features',
          status: 'pending',
          requestedAt: new Date('2024-01-09T08:00:00')
        },
        {
          id: 'req-002',
          requester: 'Rachel Green',
          requestType: 'group',
          requestedItem: 'Analytics Team',
          reason: 'Joining the analytics department',
          status: 'approved',
          requestedAt: new Date('2024-01-08T14:30:00'),
          reviewedBy: 'John Smith',
          reviewedAt: new Date('2024-01-08T16:00:00')
        },
        {
          id: 'req-003',
          requester: 'Tom Baker',
          requestType: 'permission',
          requestedItem: 'Database Write Access',
          reason: 'Required for data migration project',
          status: 'rejected',
          requestedAt: new Date('2024-01-07T10:00:00'),
          reviewedBy: 'Emily Chen',
          reviewedAt: new Date('2024-01-07T15:00:00')
        }
      ]);

      // Mock audit logs
      setAuditLogs([
        {
          id: 'audit-001',
          user: 'john.smith@company.com',
          action: 'Role Assignment',
          resource: 'user:sarah.johnson',
          timestamp: new Date('2024-01-09T10:15:00'),
          result: 'success',
          details: 'Assigned Developer role to Sarah Johnson'
        },
        {
          id: 'audit-002',
          user: 'emily.chen@company.com',
          action: 'Permission Revoke',
          resource: 'user:david.wilson',
          timestamp: new Date('2024-01-09T09:30:00'),
          result: 'success',
          details: 'Revoked write access to production database'
        },
        {
          id: 'audit-003',
          user: 'system',
          action: 'Automatic Suspension',
          resource: 'user:lisa.anderson',
          timestamp: new Date('2024-01-08T00:00:00'),
          result: 'success',
          details: 'User suspended due to 30 days of inactivity'
        }
      ]);

      setLoading(false);
    }, 1000);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'inactive':
        return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
      case 'suspended':
        return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      case 'pending':
        return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'approved':
        return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'rejected':
        return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      default:
        return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
    }
  };

  const togglePermissionExpand = (roleId: string) => {
    setExpandedPermissions(prev =>
      prev.includes(roleId)
        ? prev.filter(id => id !== roleId)
        : [...prev, roleId]
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading RBAC data...</p>
        </div>
      </div>
    );
  }

  const totalUsers = users.length;
  const activeUsers = users.filter(u => u.status === 'active').length;
  const mfaEnabledUsers = users.filter(u => u.mfaEnabled).length;
  const privilegedUsers = users.filter(u => u.privilegedAccess).length;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 pb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
              <Shield className="h-8 w-8 text-blue-600" />
              Role-Based Access Control
            </h1>
            <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
              Manage users, roles, permissions, and access requests
            </p>
          </div>
          <div className="flex gap-2">
            <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
              <UserPlus className="h-4 w-4" />
              Add User
            </button>
            <button className="px-4 py-2 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center gap-2">
              <Download className="h-4 w-4" />
              Export
            </button>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Total Users
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalUsers}</div>
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {activeUsers} active, {totalUsers - activeUsers} inactive
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Active Roles
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{roles.length}</div>
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {roles.filter(r => r.isBuiltIn).length} built-in, {roles.filter(r => !r.isBuiltIn).length} custom
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
              MFA Enabled
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {Math.round((mfaEnabledUsers / totalUsers) * 100)}%
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {mfaEnabledUsers} of {totalUsers} users
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Privileged Access
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">
              {privilegedUsers}
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Users with admin rights
            </div>
          </CardContent>
        </Card>
      </div>

      {/* View Selector */}
      <div className="flex gap-2 p-1 bg-gray-100 dark:bg-gray-800 rounded-lg w-fit">
        {(['users', 'roles', 'groups', 'requests', 'audit'] as const).map((view) => (
          <button
            key={view}
            onClick={() => setSelectedView(view)}
            className={`px-4 py-2 rounded-md font-medium transition-colors capitalize ${
              selectedView === view
                ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            {view === 'requests' ? 'Access Requests' : view}
          </button>
        ))}
      </div>

      {/* Search Bar */}
      <div className="flex gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder={`Search ${selectedView}...`}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <button className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 flex items-center gap-2">
          <Filter className="h-4 w-4" />
          Filters
        </button>
      </div>

      {/* Content based on selected view */}
      {selectedView === 'users' && (
        <Card>
          <CardHeader>
            <CardTitle>Users</CardTitle>
            <CardDescription>Manage user accounts and their access levels</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="border-b border-gray-200 dark:border-gray-700">
                  <tr>
                    <th className="text-left py-3 px-4">User</th>
                    <th className="text-left py-3 px-4">Role</th>
                    <th className="text-left py-3 px-4">Department</th>
                    <th className="text-center py-3 px-4">Status</th>
                    <th className="text-center py-3 px-4">MFA</th>
                    <th className="text-left py-3 px-4">Last Active</th>
                    <th className="text-center py-3 px-4">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((user) => (
                    <tr key={user.id} className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900/50">
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 bg-gray-200 dark:bg-gray-700 rounded-full flex items-center justify-center">
                            {user.privilegedAccess ? (
                              <Crown className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                            ) : (
                              <span className="text-sm font-medium">{user.name.split(' ').map(n => n[0]).join('')}</span>
                            )}
                          </div>
                          <div>
                            <div className="font-medium">{user.name}</div>
                            <div className="text-sm text-gray-600 dark:text-gray-400">{user.email}</div>
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium"
                              style={{ backgroundColor: `${user.role.color}20`, color: user.role.color }}>
                          {user.role.name}
                        </span>
                      </td>
                      <td className="py-3 px-4">{user.department}</td>
                      <td className="text-center py-3 px-4">
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(user.status)}`}>
                          {user.status}
                        </span>
                      </td>
                      <td className="text-center py-3 px-4">
                        {user.mfaEnabled ? (
                          <CheckCircle className="h-5 w-5 text-green-500 mx-auto" />
                        ) : (
                          <XCircle className="h-5 w-5 text-gray-400 mx-auto" />
                        )}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400">
                        {user.lastActive.toLocaleDateString()}
                      </td>
                      <td className="text-center py-3 px-4">
                        <div className="flex items-center justify-center gap-1">
                          <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded">
                            <Eye className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                          </button>
                          <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded">
                            <Edit className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                          </button>
                          <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded">
                            <MoreVertical className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {selectedView === 'roles' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {roles.map((role) => (
            <Card key={role.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: role.color }} />
                      {role.name}
                      {role.isBuiltIn && (
                        <span className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-800 rounded-full">
                          Built-in
                        </span>
                      )}
                    </CardTitle>
                    <CardDescription className="mt-1">{role.description}</CardDescription>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold">{role.users}</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">users</div>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Permissions ({role.permissions.length})</span>
                    <button
                      onClick={() => togglePermissionExpand(role.id)}
                      className="text-sm text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-1"
                    >
                      {expandedPermissions.includes(role.id) ? (
                        <>Hide <ChevronDown className="h-3 w-3" /></>
                      ) : (
                        <>Show <ChevronRight className="h-3 w-3" /></>
                      )}
                    </button>
                  </div>
                  {expandedPermissions.includes(role.id) && (
                    <div className="space-y-2 pl-4 border-l-2 border-gray-200 dark:border-gray-700">
                      {role.permissions.map((permission) => (
                        <div key={permission.id} className="text-sm">
                          <div className="font-medium">{permission.resource}</div>
                          <div className="text-gray-600 dark:text-gray-400">
                            {permission.actions.join(', ')} ({permission.scope})
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="flex gap-2 pt-2">
                    <button className="flex-1 px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700">
                      Manage
                    </button>
                    {!role.isBuiltIn && (
                      <button className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-800">
                        Delete
                      </button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {selectedView === 'groups' && (
        <Card>
          <CardHeader>
            <CardTitle>Groups</CardTitle>
            <CardDescription>Manage user groups and their role assignments</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {groups.map((group) => (
                <div key={group.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h4 className="font-semibold text-lg flex items-center gap-2">
                        <Users className="h-5 w-5 text-gray-600 dark:text-gray-400" />
                        {group.name}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {group.description}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold">{group.members}</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">members</div>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex gap-2">
                      {group.roles.map((roleName) => (
                        <span key={roleName} className="px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-full">
                          {roleName}
                        </span>
                      ))}
                    </div>
                    <div className="flex gap-2">
                      <button className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700">
                        View Members
                      </button>
                      <button className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-800">
                        Edit
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {selectedView === 'requests' && (
        <Card>
          <CardHeader>
            <CardTitle>Access Requests</CardTitle>
            <CardDescription>Review and approve pending access requests</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {accessRequests.map((request) => (
                <div key={request.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-semibold">{request.requester}</h4>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(request.status)}`}>
                          {request.status}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Requesting {request.requestType}: <strong>{request.requestedItem}</strong>
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        Reason: {request.reason}
                      </p>
                    </div>
                    <div className="text-right text-sm text-gray-600 dark:text-gray-400">
                      <div>{request.requestedAt.toLocaleDateString()}</div>
                      <div>{request.requestedAt.toLocaleTimeString()}</div>
                    </div>
                  </div>
                  {request.reviewedBy && (
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      Reviewed by {request.reviewedBy} on {request.reviewedAt?.toLocaleString()}
                    </div>
                  )}
                  {request.status === 'pending' && (
                    <div className="flex gap-2">
                      <button className="px-3 py-1 text-sm bg-green-600 text-white rounded hover:bg-green-700">
                        Approve
                      </button>
                      <button className="px-3 py-1 text-sm bg-red-600 text-white rounded hover:bg-red-700">
                        Reject
                      </button>
                      <button className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-800">
                        Request Info
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {selectedView === 'audit' && (
        <Card>
          <CardHeader>
            <CardTitle>Audit Logs</CardTitle>
            <CardDescription>Recent access control changes and activities</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="border-b border-gray-200 dark:border-gray-700">
                  <tr>
                    <th className="text-left py-3 px-4">Timestamp</th>
                    <th className="text-left py-3 px-4">User</th>
                    <th className="text-left py-3 px-4">Action</th>
                    <th className="text-left py-3 px-4">Resource</th>
                    <th className="text-center py-3 px-4">Result</th>
                    <th className="text-left py-3 px-4">Details</th>
                  </tr>
                </thead>
                <tbody>
                  {auditLogs.map((log) => (
                    <tr key={log.id} className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-3 px-4 text-sm">
                        {log.timestamp.toLocaleString()}
                      </td>
                      <td className="py-3 px-4 text-sm">{log.user}</td>
                      <td className="py-3 px-4 text-sm font-medium">{log.action}</td>
                      <td className="py-3 px-4 text-sm">{log.resource}</td>
                      <td className="text-center py-3 px-4">
                        {log.result === 'success' ? (
                          <CheckCircle className="h-4 w-4 text-green-500 mx-auto" />
                        ) : (
                          <XCircle className="h-4 w-4 text-red-500 mx-auto" />
                        )}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400">
                        {log.details}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}