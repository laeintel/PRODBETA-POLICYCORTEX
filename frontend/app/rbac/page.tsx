'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Shield, Users, Lock, Key, UserCheck, AlertCircle,
  CheckCircle, Clock, TrendingUp, Settings, Building,
  FileCheck, UserPlus, UserMinus, RefreshCw, Search,
  Filter, Download, Upload, Eye, EyeOff, ChevronRight,
  ShieldCheck, Activity, Database, Award, Target,
  BarChart3, PieChart, Calendar, AlertTriangle, Info
} from 'lucide-react';
import ResponsiveGrid, { ResponsiveContainer, ResponsiveText } from '@/components/ResponsiveGrid';
import { toast } from '@/hooks/useToast';

interface RoleStats {
  totalRoles: number;
  customRoles: number;
  builtInRoles: number;
  activeAssignments: number;
  pendingReviews: number;
  complianceScore: number;
  riskLevel: 'low' | 'medium' | 'high';
  lastAudit: string;
}

interface AccessMetric {
  label: string;
  value: number | string;
  change?: number;
  trend?: 'up' | 'down' | 'stable';
  status?: 'healthy' | 'warning' | 'critical';
}

interface RoleOverview {
  id: string;
  name: string;
  type: 'built-in' | 'custom';
  description: string;
  assignedUsers: number;
  assignedGroups: number;
  permissions: number;
  riskScore: number;
  lastModified: string;
  status: 'active' | 'review' | 'deprecated';
}

interface AccessReview {
  id: string;
  name: string;
  scope: string;
  reviewers: number;
  progress: number;
  dueDate: string;
  status: 'not-started' | 'in-progress' | 'completed' | 'overdue';
  priority: 'low' | 'medium' | 'high' | 'critical';
}

export default function RBACPage() {
  const router = useRouter();
  const [roleStats, setRoleStats] = useState<RoleStats>({
    totalRoles: 89,
    customRoles: 34,
    builtInRoles: 55,
    activeAssignments: 2451,
    pendingReviews: 12,
    complianceScore: 94,
    riskLevel: 'low',
    lastAudit: '2 days ago'
  });

  const [selectedTab, setSelectedTab] = useState<'overview' | 'roles' | 'reviews' | 'analytics'>('overview');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'built-in' | 'custom'>('all');
  const [topRoles, setTopRoles] = useState<RoleOverview[]>([]);
  const [activeReviews, setActiveReviews] = useState<AccessReview[]>([]);

  useEffect(() => {
    // Mock data for top roles
    const mockRoles: RoleOverview[] = [
      {
        id: 'role-001',
        name: 'Contributor',
        type: 'built-in',
        description: 'Can create and manage all types of Azure resources',
        assignedUsers: 342,
        assignedGroups: 28,
        permissions: 156,
        riskScore: 35,
        lastModified: '5 hours ago',
        status: 'active'
      },
      {
        id: 'role-002',
        name: 'Security Administrator',
        type: 'built-in',
        description: 'Security Admin with full access to Security Center',
        assignedUsers: 12,
        assignedGroups: 3,
        permissions: 245,
        riskScore: 85,
        lastModified: '2 days ago',
        status: 'review'
      },
      {
        id: 'role-003',
        name: 'Custom DevOps Engineer',
        type: 'custom',
        description: 'Custom role for DevOps team with specific permissions',
        assignedUsers: 45,
        assignedGroups: 5,
        permissions: 89,
        riskScore: 42,
        lastModified: '1 week ago',
        status: 'active'
      },
      {
        id: 'role-004',
        name: 'Billing Reader',
        type: 'built-in',
        description: 'Can view billing information and download invoices',
        assignedUsers: 156,
        assignedGroups: 12,
        permissions: 23,
        riskScore: 15,
        lastModified: '3 days ago',
        status: 'active'
      },
      {
        id: 'role-005',
        name: 'Custom Data Analyst',
        type: 'custom',
        description: 'Read-only access to data resources and analytics',
        assignedUsers: 78,
        assignedGroups: 8,
        permissions: 45,
        riskScore: 20,
        lastModified: '12 hours ago',
        status: 'active'
      }
    ];

    // Mock data for active reviews
    const mockReviews: AccessReview[] = [
      {
        id: 'review-001',
        name: 'Q4 2024 Privileged Access Review',
        scope: 'All privileged roles',
        reviewers: 5,
        progress: 67,
        dueDate: '2024-12-15',
        status: 'in-progress',
        priority: 'high'
      },
      {
        id: 'review-002',
        name: 'Monthly Guest User Review',
        scope: 'External guest users',
        reviewers: 3,
        progress: 100,
        dueDate: '2024-11-30',
        status: 'completed',
        priority: 'medium'
      },
      {
        id: 'review-003',
        name: 'Service Principal Access Audit',
        scope: 'Service principals and apps',
        reviewers: 4,
        progress: 25,
        dueDate: '2024-12-20',
        status: 'in-progress',
        priority: 'critical'
      },
      {
        id: 'review-004',
        name: 'Contractor Access Review',
        scope: 'Contractor accounts',
        reviewers: 2,
        progress: 0,
        dueDate: '2024-12-10',
        status: 'not-started',
        priority: 'high'
      }
    ];

    setTopRoles(mockRoles);
    setActiveReviews(mockReviews);
  }, []);

  const accessMetrics: AccessMetric[] = [
    { label: 'Total Users', value: '2,451', change: 3.2, trend: 'up', status: 'healthy' },
    { label: 'Active Roles', value: 89, change: -2.1, trend: 'down', status: 'healthy' },
    { label: 'Privileged Users', value: 45, change: 0, trend: 'stable', status: 'warning' },
    { label: 'Guest Users', value: 234, change: 12.5, trend: 'up', status: 'warning' },
    { label: 'Service Principals', value: 156, change: 5.3, trend: 'up', status: 'healthy' },
    { label: 'Compliance Score', value: '94%', change: 2.0, trend: 'up', status: 'healthy' }
  ];

  const getRiskColor = (score: number) => {
    if (score >= 70) return 'text-red-500 bg-red-500/10';
    if (score >= 40) return 'text-yellow-500 bg-yellow-500/10';
    return 'text-green-500 bg-green-500/10';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
      case 'completed':
        return 'text-green-500 bg-green-500/10';
      case 'review':
      case 'in-progress':
        return 'text-yellow-500 bg-yellow-500/10';
      case 'deprecated':
      case 'overdue':
        return 'text-red-500 bg-red-500/10';
      case 'not-started':
        return 'text-gray-500 bg-gray-500/10';
      default:
        return 'text-gray-500 bg-gray-500/10';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'text-red-500';
      case 'high': return 'text-orange-500';
      case 'medium': return 'text-yellow-500';
      case 'low': return 'text-blue-500';
      default: return 'text-gray-500';
    }
  };

  const handleQuickAction = (action: string) => {
    switch (action) {
      case 'new-role':
        toast({ title: 'Create Role', description: 'Opening role creation wizard...' });
        router.push('/rbac/roles?action=create');
        break;
      case 'start-review':
        toast({ title: 'Access Review', description: 'Starting new access review...' });
        router.push('/rbac/reviews?action=create');
        break;
      case 'audit-log':
        toast({ title: 'Audit Log', description: 'Opening RBAC audit log...' });
        router.push('/audit?filter=rbac');
        break;
      case 'export-report':
        toast({ title: 'Export Started', description: 'Generating RBAC compliance report...' });
        break;
      default:
        break;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
      <ResponsiveContainer className="py-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
                <div className="p-2 bg-purple-500/10 rounded-lg">
                  <Shield className="w-8 h-8 text-purple-500" />
                </div>
                <span>Role-Based Access Control (RBAC)</span>
              </h1>
              <p className="text-gray-600 dark:text-gray-400 ml-14">
                Manage roles, permissions, and access reviews for your organization
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => handleQuickAction('export-report')}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Export Report
              </button>
              <button
                onClick={() => router.push('/security')}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors flex items-center gap-2"
              >
                <Settings className="w-4 h-4" />
                Security Settings
              </button>
            </div>
          </div>

          {/* Alert Banner */}
          {roleStats.pendingReviews > 0 && (
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <AlertCircle className="w-5 h-5 text-yellow-500" />
                  <div>
                    <p className="font-medium text-yellow-500">
                      {roleStats.pendingReviews} Access Reviews Pending
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Complete reviews to maintain compliance and security posture
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => router.push('/rbac/reviews')}
                  className="px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 transition-colors flex items-center gap-2"
                >
                  Review Now
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Key Metrics */}
        <ResponsiveGrid variant="metrics" className="mb-8">
          {accessMetrics.map((metric, idx) => (
            <div
              key={idx}
              className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">{metric.label}</span>
                {metric.trend && (
                  <div className={`flex items-center gap-1 text-xs ${
                    metric.trend === 'up' ? 'text-green-500' : 
                    metric.trend === 'down' ? 'text-red-500' : 
                    'text-gray-500'
                  }`}>
                    {metric.change !== undefined && (
                      <span>{metric.change > 0 ? '+' : ''}{metric.change}%</span>
                    )}
                    {metric.trend === 'up' && <TrendingUp className="w-3 h-3" />}
                    {metric.trend === 'down' && <TrendingUp className="w-3 h-3 rotate-180" />}
                    {metric.trend === 'stable' && <Activity className="w-3 h-3" />}
                  </div>
                )}
              </div>
              <div className="text-2xl font-bold">{metric.value}</div>
              {metric.status && (
                <div className="mt-2 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      metric.status === 'healthy' ? 'bg-green-500' :
                      metric.status === 'warning' ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`}
                    style={{ width: `${metric.status === 'healthy' ? 100 : metric.status === 'warning' ? 60 : 30}%` }}
                  />
                </div>
              )}
            </div>
          ))}
        </ResponsiveGrid>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-8">
          <button
            onClick={() => handleQuickAction('new-role')}
            className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-all group"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-500/10 rounded-lg group-hover:bg-blue-500/20 transition-colors">
                  <UserPlus className="w-5 h-5 text-blue-500" />
                </div>
                <div className="text-left">
                  <p className="font-medium">Create New Role</p>
                  <p className="text-xs text-gray-500">Define custom permissions</p>
                </div>
              </div>
              <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300" />
            </div>
          </button>

          <button
            onClick={() => handleQuickAction('start-review')}
            className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-all group"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-green-500/10 rounded-lg group-hover:bg-green-500/20 transition-colors">
                  <FileCheck className="w-5 h-5 text-green-500" />
                </div>
                <div className="text-left">
                  <p className="font-medium">Start Access Review</p>
                  <p className="text-xs text-gray-500">Review user permissions</p>
                </div>
              </div>
              <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300" />
            </div>
          </button>

          <button
            onClick={() => router.push('/security/pim')}
            className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-all group"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-500/10 rounded-lg group-hover:bg-purple-500/20 transition-colors">
                  <Key className="w-5 h-5 text-purple-500" />
                </div>
                <div className="text-left">
                  <p className="font-medium">Privileged Access</p>
                  <p className="text-xs text-gray-500">Manage PIM requests</p>
                </div>
              </div>
              <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300" />
            </div>
          </button>

          <button
            onClick={() => handleQuickAction('audit-log')}
            className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-all group"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-orange-500/10 rounded-lg group-hover:bg-orange-500/20 transition-colors">
                  <Activity className="w-5 h-5 text-orange-500" />
                </div>
                <div className="text-left">
                  <p className="font-medium">View Audit Log</p>
                  <p className="text-xs text-gray-500">Track RBAC changes</p>
                </div>
              </div>
              <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300" />
            </div>
          </button>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Roles Overview */}
          <div className="lg:col-span-2 space-y-6">
            {/* Top Roles */}
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Users className="w-5 h-5 text-purple-500" />
                    Role Assignments
                  </h2>
                  <button
                    onClick={() => router.push('/rbac/roles')}
                    className="text-sm text-blue-500 hover:text-blue-600 flex items-center gap-1"
                  >
                    View All
                    <ChevronRight className="w-3 h-3" />
                  </button>
                </div>

                {/* Search and Filter */}
                <div className="flex gap-3">
                  <div className="flex-1 relative">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search roles..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <select
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value as any)}
                    className="px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="all">All Roles</option>
                    <option value="built-in">Built-in</option>
                    <option value="custom">Custom</option>
                  </select>
                </div>
              </div>

              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {topRoles
                  .filter(role => 
                    (filterType === 'all' || role.type === filterType) &&
                    role.name.toLowerCase().includes(searchQuery.toLowerCase())
                  )
                  .map((role) => (
                    <div
                      key={role.id}
                      className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer transition-colors"
                      onClick={() => router.push(`/rbac/roles?id=${role.id}`)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="font-medium">{role.name}</h3>
                            <span className={`px-2 py-1 text-xs rounded-full ${
                              role.type === 'built-in' ? 'bg-blue-500/10 text-blue-500' : 'bg-purple-500/10 text-purple-500'
                            }`}>
                              {role.type === 'built-in' ? 'Built-in' : 'Custom'}
                            </span>
                            <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(role.status)}`}>
                              {role.status}
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                            {role.description}
                          </p>
                          <div className="flex items-center gap-6 text-sm">
                            <div className="flex items-center gap-1">
                              <Users className="w-4 h-4 text-gray-400" />
                              <span>{role.assignedUsers} users</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Building className="w-4 h-4 text-gray-400" />
                              <span>{role.assignedGroups} groups</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Key className="w-4 h-4 text-gray-400" />
                              <span>{role.permissions} permissions</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Clock className="w-4 h-4 text-gray-400" />
                              <span>{role.lastModified}</span>
                            </div>
                          </div>
                        </div>
                        <div className="ml-4 text-center">
                          <div className={`px-3 py-1 rounded-lg ${getRiskColor(role.riskScore)}`}>
                            <div className="text-2xl font-bold">{role.riskScore}</div>
                            <div className="text-xs">Risk Score</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>

            {/* Access Reviews */}
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <FileCheck className="w-5 h-5 text-green-500" />
                    Active Access Reviews
                  </h2>
                  <button
                    onClick={() => router.push('/rbac/reviews')}
                    className="text-sm text-blue-500 hover:text-blue-600 flex items-center gap-1"
                  >
                    View All
                    <ChevronRight className="w-3 h-3" />
                  </button>
                </div>
              </div>

              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {activeReviews.map((review) => (
                  <div
                    key={review.id}
                    className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer transition-colors"
                    onClick={() => router.push(`/rbac/reviews?id=${review.id}`)}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h3 className="font-medium flex items-center gap-2">
                          {review.name}
                          <span className={`text-xs ${getPriorityColor(review.priority)}`}>
                            {review.priority.toUpperCase()}
                          </span>
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          {review.scope} • {review.reviewers} reviewers
                        </p>
                      </div>
                      <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(review.status)}`}>
                        {review.status.replace('-', ' ')}
                      </span>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">Progress</span>
                        <span className="font-medium">{review.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all ${
                            review.progress === 100 ? 'bg-green-500' :
                            review.progress >= 50 ? 'bg-blue-500' :
                            'bg-yellow-500'
                          }`}
                          style={{ width: `${review.progress}%` }}
                        />
                      </div>
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>Due: {new Date(review.dueDate).toLocaleDateString()}</span>
                        {review.status === 'overdue' && (
                          <span className="text-red-500 flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3" />
                            Overdue
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column - Stats and Info */}
          <div className="space-y-6">
            {/* Compliance Overview */}
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-green-500" />
                Compliance Status
              </h2>
              
              <div className="flex items-center justify-center mb-6">
                <div className="relative w-32 h-32">
                  <svg className="w-32 h-32 transform -rotate-90">
                    <circle
                      cx="64"
                      cy="64"
                      r="56"
                      stroke="currentColor"
                      strokeWidth="12"
                      fill="none"
                      className="text-gray-200 dark:text-gray-700"
                    />
                    <circle
                      cx="64"
                      cy="64"
                      r="56"
                      stroke="currentColor"
                      strokeWidth="12"
                      fill="none"
                      strokeDasharray={`${2 * Math.PI * 56}`}
                      strokeDashoffset={`${2 * Math.PI * 56 * (1 - roleStats.complianceScore / 100)}`}
                      className="text-green-500 transition-all duration-1000"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-3xl font-bold">{roleStats.complianceScore}%</div>
                      <div className="text-xs text-gray-500">Compliant</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Risk Level</span>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    roleStats.riskLevel === 'low' ? 'bg-green-500/10 text-green-500' :
                    roleStats.riskLevel === 'medium' ? 'bg-yellow-500/10 text-yellow-500' :
                    'bg-red-500/10 text-red-500'
                  }`}>
                    {roleStats.riskLevel.toUpperCase()}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Last Audit</span>
                  <span className="text-sm font-medium">{roleStats.lastAudit}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Total Roles</span>
                  <span className="text-sm font-medium">{roleStats.totalRoles}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Custom Roles</span>
                  <span className="text-sm font-medium">{roleStats.customRoles}</span>
                </div>
              </div>

              <button
                onClick={() => router.push('/governance/compliance')}
                className="w-full mt-4 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                View Compliance Report
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {/* Role Distribution */}
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <PieChart className="w-5 h-5 text-purple-500" />
                Role Distribution
              </h2>

              <div className="space-y-4">
                {[
                  { name: 'Readers', count: 892, percentage: 36, color: 'bg-blue-500' },
                  { name: 'Contributors', count: 645, percentage: 26, color: 'bg-green-500' },
                  { name: 'Owners', count: 234, percentage: 10, color: 'bg-purple-500' },
                  { name: 'Custom Roles', count: 456, percentage: 19, color: 'bg-orange-500' },
                  { name: 'Service Principals', count: 224, percentage: 9, color: 'bg-pink-500' }
                ].map((item, idx) => (
                  <div key={idx}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600 dark:text-gray-400">{item.name}</span>
                      <span className="font-medium">{item.count} ({item.percentage}%)</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${item.color} transition-all`}
                        style={{ width: `${item.percentage}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Changes */}
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-orange-500" />
                Recent RBAC Changes
              </h2>

              <div className="space-y-3">
                {[
                  { action: 'Role assigned', user: 'john.doe@company.com', time: '5 min ago', icon: UserPlus },
                  { action: 'Permission removed', user: 'Security Admin Role', time: '2 hours ago', icon: UserMinus },
                  { action: 'Review completed', user: 'Q4 Guest Review', time: '3 hours ago', icon: CheckCircle },
                  { action: 'Role created', user: 'Custom Analytics Role', time: '1 day ago', icon: Shield }
                ].map((change, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-3 p-2 rounded hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer transition-colors"
                    onClick={() => router.push('/audit')}
                  >
                    <change.icon className="w-4 h-4 text-gray-400 mt-1" />
                    <div className="flex-1">
                      <p className="text-sm">{change.action}</p>
                      <p className="text-xs text-gray-500">{change.user}</p>
                    </div>
                    <span className="text-xs text-gray-400">{change.time}</span>
                  </div>
                ))}
              </div>

              <button
                onClick={() => router.push('/audit?filter=rbac')}
                className="w-full mt-4 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-sm"
              >
                View All Changes
              </button>
            </div>
          </div>
        </div>
      </ResponsiveContainer>
    </div>
  );
}