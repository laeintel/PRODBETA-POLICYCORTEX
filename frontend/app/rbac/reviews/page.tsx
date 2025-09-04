'use client';

import React, { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  FileCheck, Users, Clock, AlertCircle, CheckCircle,
  Play, Pause, RefreshCw, Download, Calendar, Filter,
  Search, ChevronRight, TrendingUp, Target, Award,
  UserCheck, UserX, Shield, Building, Terminal, 
  AlertTriangle, Info, ArrowRight, BarChart3, PieChart,
  Eye, Mail, Bell, Settings, Plus, X, ArrowLeft, Check
} from 'lucide-react';
import ResponsiveGrid, { ResponsiveContainer } from '@/components/ResponsiveGrid';
import { toast } from '@/hooks/useToast';

interface Review {
  id: string;
  name: string;
  description: string;
  type: 'user-access' | 'guest-access' | 'privileged-access' | 'app-permissions' | 'group-membership';
  scope: string;
  frequency: 'one-time' | 'weekly' | 'monthly' | 'quarterly' | 'annual';
  startDate: string;
  endDate: string;
  status: 'not-started' | 'in-progress' | 'completed' | 'overdue' | 'cancelled';
  priority: 'low' | 'medium' | 'high' | 'critical';
  reviewers: Reviewer[];
  progress: number;
  totalItems: number;
  reviewedItems: number;
  approvedItems: number;
  deniedItems: number;
  pendingItems: number;
  createdBy: string;
  createdOn: string;
  lastModified: string;
  notifications: boolean;
  autoComplete: boolean;
  requireJustification: boolean;
}

interface Reviewer {
  id: string;
  name: string;
  email: string;
  role: string;
  assignedItems: number;
  completedItems: number;
  status: 'not-started' | 'in-progress' | 'completed';
  lastActivity: string;
}

interface ReviewItem {
  id: string;
  principalName: string;
  principalType: 'User' | 'Group' | 'ServicePrincipal' | 'Guest';
  resourceName: string;
  resourceType: string;
  accessLevel: string;
  lastUsed: string;
  riskScore: number;
  recommendation: 'approve' | 'deny' | 'review';
  decision?: 'approved' | 'denied' | 'pending';
  reviewer?: string;
  justification?: string;
  reviewedOn?: string;
}

interface ReviewStats {
  totalReviews: number;
  activeReviews: number;
  completedReviews: number;
  overdueReviews: number;
  complianceRate: number;
  averageCompletionTime: number;
  itemsReviewedToday: number;
  itemsPendingReview: number;
}

export default function ReviewsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const reviewId = searchParams.get('id');
  const action = searchParams.get('action');

  const [reviews, setReviews] = useState<Review[]>([]);
  const [selectedReview, setSelectedReview] = useState<Review | null>(null);
  const [reviewItems, setReviewItems] = useState<ReviewItem[]>([]);
  const [stats, setStats] = useState<ReviewStats>({
    totalReviews: 24,
    activeReviews: 8,
    completedReviews: 14,
    overdueReviews: 2,
    complianceRate: 92,
    averageCompletionTime: 4.5,
    itemsReviewedToday: 47,
    itemsPendingReview: 234
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState<'all' | 'active' | 'completed' | 'overdue'>('all');
  const [showCreateModal, setShowCreateModal] = useState(action === 'create');
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());

  useEffect(() => {
    // Mock data for reviews
    const mockReviews: Review[] = [
      {
        id: 'review-001',
        name: 'Q4 2024 Privileged Access Review',
        description: 'Quarterly review of all privileged role assignments',
        type: 'privileged-access',
        scope: 'All subscriptions',
        frequency: 'quarterly',
        startDate: '2024-12-01',
        endDate: '2024-12-15',
        status: 'in-progress',
        priority: 'high',
        reviewers: [
          {
            id: 'reviewer-001',
            name: 'Sarah Johnson',
            email: 'sarah.johnson@company.com',
            role: 'Security Manager',
            assignedItems: 45,
            completedItems: 30,
            status: 'in-progress',
            lastActivity: '2 hours ago'
          },
          {
            id: 'reviewer-002',
            name: 'Mike Chen',
            email: 'mike.chen@company.com',
            role: 'Compliance Officer',
            assignedItems: 38,
            completedItems: 38,
            status: 'completed',
            lastActivity: '1 day ago'
          }
        ],
        progress: 67,
        totalItems: 83,
        reviewedItems: 56,
        approvedItems: 48,
        deniedItems: 8,
        pendingItems: 27,
        createdBy: 'admin@company.com',
        createdOn: '2024-11-25',
        lastModified: '2024-12-02',
        notifications: true,
        autoComplete: false,
        requireJustification: true
      },
      {
        id: 'review-002',
        name: 'Monthly Guest User Access Review',
        description: 'Review external guest user access to resources',
        type: 'guest-access',
        scope: 'Production environment',
        frequency: 'monthly',
        startDate: '2024-11-15',
        endDate: '2024-11-30',
        status: 'completed',
        priority: 'medium',
        reviewers: [
          {
            id: 'reviewer-003',
            name: 'John Doe',
            email: 'john.doe@company.com',
            role: 'IT Manager',
            assignedItems: 156,
            completedItems: 156,
            status: 'completed',
            lastActivity: '3 days ago'
          }
        ],
        progress: 100,
        totalItems: 156,
        reviewedItems: 156,
        approvedItems: 123,
        deniedItems: 33,
        pendingItems: 0,
        createdBy: 'security@company.com',
        createdOn: '2024-11-10',
        lastModified: '2024-11-30',
        notifications: true,
        autoComplete: true,
        requireJustification: false
      },
      {
        id: 'review-003',
        name: 'Service Principal Permissions Audit',
        description: 'Audit all service principal and application permissions',
        type: 'app-permissions',
        scope: 'All applications',
        frequency: 'quarterly',
        startDate: '2024-12-05',
        endDate: '2024-12-20',
        status: 'not-started',
        priority: 'critical',
        reviewers: [
          {
            id: 'reviewer-004',
            name: 'Emily Davis',
            email: 'emily.davis@company.com',
            role: 'DevOps Lead',
            assignedItems: 89,
            completedItems: 0,
            status: 'not-started',
            lastActivity: 'Not started'
          }
        ],
        progress: 0,
        totalItems: 89,
        reviewedItems: 0,
        approvedItems: 0,
        deniedItems: 0,
        pendingItems: 89,
        createdBy: 'admin@company.com',
        createdOn: '2024-12-01',
        lastModified: '2024-12-01',
        notifications: true,
        autoComplete: false,
        requireJustification: true
      },
      {
        id: 'review-004',
        name: 'Contractor Access Review',
        description: 'Review all contractor accounts and permissions',
        type: 'user-access',
        scope: 'Contractor accounts',
        frequency: 'monthly',
        startDate: '2024-11-20',
        endDate: '2024-12-10',
        status: 'overdue',
        priority: 'high',
        reviewers: [
          {
            id: 'reviewer-005',
            name: 'Robert Smith',
            email: 'robert.smith@company.com',
            role: 'HR Manager',
            assignedItems: 67,
            completedItems: 12,
            status: 'in-progress',
            lastActivity: '5 days ago'
          }
        ],
        progress: 18,
        totalItems: 67,
        reviewedItems: 12,
        approvedItems: 10,
        deniedItems: 2,
        pendingItems: 55,
        createdBy: 'hr@company.com',
        createdOn: '2024-11-15',
        lastModified: '2024-11-25',
        notifications: true,
        autoComplete: false,
        requireJustification: true
      }
    ];

    // Mock review items
    const mockReviewItems: ReviewItem[] = [
      {
        id: 'item-001',
        principalName: 'john.contractor@external.com',
        principalType: 'Guest',
        resourceName: 'Production Database',
        resourceType: 'SQL Database',
        accessLevel: 'Contributor',
        lastUsed: '5 days ago',
        riskScore: 75,
        recommendation: 'review',
        decision: 'pending'
      },
      {
        id: 'item-002',
        principalName: 'DevOps Service Principal',
        principalType: 'ServicePrincipal',
        resourceName: 'Key Vault - Prod',
        resourceType: 'Key Vault',
        accessLevel: 'Key Vault Administrator',
        lastUsed: '1 hour ago',
        riskScore: 45,
        recommendation: 'approve',
        decision: 'approved',
        reviewer: 'sarah.johnson@company.com',
        justification: 'Required for CI/CD pipeline',
        reviewedOn: '2024-12-01'
      },
      {
        id: 'item-003',
        principalName: 'Analytics Team',
        principalType: 'Group',
        resourceName: 'Data Lake Storage',
        resourceType: 'Storage Account',
        accessLevel: 'Storage Blob Data Reader',
        lastUsed: '2 days ago',
        riskScore: 25,
        recommendation: 'approve',
        decision: 'approved',
        reviewer: 'mike.chen@company.com',
        reviewedOn: '2024-11-30'
      },
      {
        id: 'item-004',
        principalName: 'legacy.app@company.com',
        principalType: 'ServicePrincipal',
        resourceName: 'Legacy API',
        resourceType: 'App Service',
        accessLevel: 'Owner',
        lastUsed: '3 months ago',
        riskScore: 90,
        recommendation: 'deny',
        decision: 'denied',
        reviewer: 'sarah.johnson@company.com',
        justification: 'Deprecated application, no longer in use',
        reviewedOn: '2024-12-02'
      },
      {
        id: 'item-005',
        principalName: 'temp.user@company.com',
        principalType: 'User',
        resourceName: 'Test Environment',
        resourceType: 'Resource Group',
        accessLevel: 'Contributor',
        lastUsed: '1 month ago',
        riskScore: 60,
        recommendation: 'review',
        decision: 'pending'
      }
    ];

    setReviews(mockReviews);
    setReviewItems(mockReviewItems);

    if (reviewId) {
      const review = mockReviews.find(r => r.id === reviewId);
      if (review) {
        setSelectedReview(review);
      }
    }
  }, [reviewId]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-500 bg-green-500/10';
      case 'in-progress': return 'text-yellow-500 bg-yellow-500/10';
      case 'not-started': return 'text-gray-500 bg-gray-500/10';
      case 'overdue': return 'text-red-500 bg-red-500/10';
      case 'cancelled': return 'text-gray-500 bg-gray-500/10';
      default: return 'text-gray-500 bg-gray-500/10';
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

  const getRiskColor = (score: number) => {
    if (score >= 70) return 'text-red-500 bg-red-500/10';
    if (score >= 40) return 'text-yellow-500 bg-yellow-500/10';
    return 'text-green-500 bg-green-500/10';
  };

  const getDecisionIcon = (decision?: string) => {
    switch (decision) {
      case 'approved': return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'denied': return <X className="w-5 h-5 text-red-500" />;
      default: return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const handleBulkAction = (action: 'approve' | 'deny') => {
    const count = selectedItems.size;
    if (count === 0) {
      toast({ 
        title: 'No items selected', 
        description: 'Please select items to review',
        variant: 'destructive'
      });
      return;
    }
    toast({ 
      title: `Bulk ${action}`, 
      description: `${count} items ${action}d successfully` 
    });
    setSelectedItems(new Set());
  };

  const handleStartReview = (review: Review) => {
    setSelectedReview(review);
    toast({ title: 'Review Started', description: `Starting ${review.name}` });
  };

  const handleCompleteReview = () => {
    toast({ title: 'Review Completed', description: 'Access review completed successfully' });
    setSelectedReview(null);
  };

  const filteredReviews = reviews.filter(review => {
    const matchesSearch = review.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         review.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filterStatus === 'all' ||
                         (filterStatus === 'active' && review.status === 'in-progress') ||
                         (filterStatus === 'completed' && review.status === 'completed') ||
                         (filterStatus === 'overdue' && review.status === 'overdue');
    return matchesSearch && matchesFilter;
  });

  if (selectedReview && !showCreateModal) {
    // Review Detail View
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
        <ResponsiveContainer className="py-6">
          {/* Header */}
          <div className="mb-8">
            <button
              onClick={() => setSelectedReview(null)}
              className="mb-4 px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white flex items-center gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Reviews
            </button>

            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold mb-2">{selectedReview.name}</h1>
                <p className="text-gray-600 dark:text-gray-400">{selectedReview.description}</p>
                <div className="flex items-center gap-3 mt-3">
                  <span className={`px-3 py-1 text-sm rounded-full ${getStatusColor(selectedReview.status)}`}>
                    {selectedReview.status.replace('-', ' ')}
                  </span>
                  <span className={`text-sm ${getPriorityColor(selectedReview.priority)}`}>
                    {selectedReview.priority.toUpperCase()} Priority
                  </span>
                  <span className="text-sm text-gray-500">
                    Due: {new Date(selectedReview.endDate).toLocaleDateString()}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <button className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center gap-2">
                  <Download className="w-4 h-4" />
                  Export
                </button>
                {selectedReview.status === 'in-progress' && (
                  <>
                    <button className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center gap-2">
                      <Pause className="w-4 h-4" />
                      Pause
                    </button>
                    <button
                      onClick={handleCompleteReview}
                      className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center gap-2"
                    >
                      <CheckCircle className="w-4 h-4" />
                      Complete Review
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Progress Overview */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 mb-6">
            <h2 className="text-lg font-semibold mb-4">Review Progress</h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Overall Progress</span>
                  <span className="font-bold">{selectedReview.progress}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <div
                    className="h-3 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all"
                    style={{ width: `${selectedReview.progress}%` }}
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold">{selectedReview.totalItems}</div>
                  <div className="text-xs text-gray-500">Total Items</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-500">{selectedReview.reviewedItems}</div>
                  <div className="text-xs text-gray-500">Reviewed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-500">{selectedReview.approvedItems}</div>
                  <div className="text-xs text-gray-500">Approved</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-500">{selectedReview.deniedItems}</div>
                  <div className="text-xs text-gray-500">Denied</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-yellow-500">{selectedReview.pendingItems}</div>
                  <div className="text-xs text-gray-500">Pending</div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            {/* Reviewers */}
            <div className="lg:col-span-2">
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Users className="w-5 h-5 text-purple-500" />
                    Reviewers
                  </h2>
                </div>
                <div className="divide-y divide-gray-200 dark:divide-gray-700">
                  {selectedReview.reviewers.map((reviewer) => (
                    <div key={reviewer.id} className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">{reviewer.name}</div>
                          <div className="text-sm text-gray-500">{reviewer.email} • {reviewer.role}</div>
                          <div className="text-xs text-gray-400 mt-1">Last active: {reviewer.lastActivity}</div>
                        </div>
                        <div className="text-right">
                          <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(reviewer.status)}`}>
                            {reviewer.status.replace('-', ' ')}
                          </span>
                          <div className="mt-2 text-sm">
                            <span className="font-bold">{reviewer.completedItems}</span>
                            <span className="text-gray-500"> / {reviewer.assignedItems}</span>
                          </div>
                        </div>
                      </div>
                      <div className="mt-3">
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className="h-2 rounded-full bg-purple-500"
                            style={{ width: `${(reviewer.completedItems / reviewer.assignedItems) * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Review Settings */}
            <div>
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Settings className="w-5 h-5 text-gray-500" />
                  Review Settings
                </h2>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Frequency</span>
                    <span className="text-sm font-medium capitalize">{selectedReview.frequency}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Notifications</span>
                    {selectedReview.notifications ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <X className="w-4 h-4 text-gray-400" />
                    )}
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Auto-complete</span>
                    {selectedReview.autoComplete ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <X className="w-4 h-4 text-gray-400" />
                    )}
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Require Justification</span>
                    {selectedReview.requireJustification ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <X className="w-4 h-4 text-gray-400" />
                    )}
                  </div>
                  <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                    <div className="text-xs text-gray-500">
                      <div>Created: {new Date(selectedReview.createdOn).toLocaleDateString()}</div>
                      <div>Modified: {new Date(selectedReview.lastModified).toLocaleDateString()}</div>
                      <div>By: {selectedReview.createdBy}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Review Items */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="p-6 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Items to Review</h2>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => handleBulkAction('approve')}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center gap-2"
                    disabled={selectedItems.size === 0}
                  >
                    <CheckCircle className="w-4 h-4" />
                    Approve Selected ({selectedItems.size})
                  </button>
                  <button
                    onClick={() => handleBulkAction('deny')}
                    className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors flex items-center gap-2"
                    disabled={selectedItems.size === 0}
                  >
                    <X className="w-4 h-4" />
                    Deny Selected ({selectedItems.size})
                  </button>
                </div>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="px-6 py-4 text-left">
                      <input
                        type="checkbox"
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedItems(new Set(reviewItems.filter(i => i.decision === 'pending').map(i => i.id)));
                          } else {
                            setSelectedItems(new Set());
                          }
                        }}
                        className="rounded border-gray-300 dark:border-gray-600"
                      />
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Principal
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Resource
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Access Level
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Last Used
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Risk
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Recommendation
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Decision
                    </th>
                    <th className="px-6 py-4 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {reviewItems.map((item) => (
                    <tr key={item.id} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                      <td className="px-6 py-4">
                        <input
                          type="checkbox"
                          checked={selectedItems.has(item.id)}
                          onChange={(e) => {
                            const newSelected = new Set(selectedItems);
                            if (e.target.checked) {
                              newSelected.add(item.id);
                            } else {
                              newSelected.delete(item.id);
                            }
                            setSelectedItems(newSelected);
                          }}
                          disabled={item.decision !== 'pending'}
                          className="rounded border-gray-300 dark:border-gray-600"
                        />
                      </td>
                      <td className="px-6 py-4">
                        <div>
                          <div className="text-sm font-medium">{item.principalName}</div>
                          <div className="text-xs text-gray-500">
                            {item.principalType === 'ServicePrincipal' && <Terminal className="w-3 h-3 inline mr-1" />}
                            {item.principalType === 'Group' && <Building className="w-3 h-3 inline mr-1" />}
                            {item.principalType === 'Guest' && <Users className="w-3 h-3 inline mr-1" />}
                            {item.principalType}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div>
                          <div className="text-sm">{item.resourceName}</div>
                          <div className="text-xs text-gray-500">{item.resourceType}</div>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm">{item.accessLevel}</td>
                      <td className="px-6 py-4 text-sm text-gray-500">{item.lastUsed}</td>
                      <td className="px-6 py-4">
                        <span className={`px-2 py-1 text-xs rounded-lg font-medium ${getRiskColor(item.riskScore)}`}>
                          {item.riskScore}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`text-sm ${
                          item.recommendation === 'approve' ? 'text-green-500' :
                          item.recommendation === 'deny' ? 'text-red-500' :
                          'text-yellow-500'
                        }`}>
                          {item.recommendation}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          {getDecisionIcon(item.decision)}
                          {item.decision && (
                            <div>
                              <div className="text-sm capitalize">{item.decision}</div>
                              {item.reviewer && (
                                <div className="text-xs text-gray-500">{item.reviewedOn}</div>
                              )}
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right">
                        {item.decision === 'pending' ? (
                          <div className="flex items-center justify-end gap-2">
                            <button
                              onClick={() => {
                                toast({ title: 'Approved', description: `Access approved for ${item.principalName}` });
                              }}
                              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded"
                              title="Approve"
                            >
                              <CheckCircle className="w-4 h-4 text-green-500" />
                            </button>
                            <button
                              onClick={() => {
                                toast({ title: 'Denied', description: `Access denied for ${item.principalName}` });
                              }}
                              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded"
                              title="Deny"
                            >
                              <X className="w-4 h-4 text-red-500" />
                            </button>
                            <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded" title="View Details">
                              <Eye className="w-4 h-4 text-gray-400" />
                            </button>
                          </div>
                        ) : (
                          <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded" title="View Details">
                            <Eye className="w-4 h-4 text-gray-400" />
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </ResponsiveContainer>
      </div>
    );
  }

  // Main Reviews List View
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
      <ResponsiveContainer className="py-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
                <FileCheck className="w-8 h-8 text-green-500" />
                Access Reviews
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Manage and track access reviews for compliance and security
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center gap-2">
                <Calendar className="w-4 h-4" />
                Schedule
              </button>
              <button
                onClick={() => setShowCreateModal(true)}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                <Plus className="w-4 h-4" />
                Create Review
              </button>
            </div>
          </div>

          {/* Search and Filters */}
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search reviews..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value as any)}
              className="px-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
            >
              <option value="all">All Reviews</option>
              <option value="active">Active</option>
              <option value="completed">Completed</option>
              <option value="overdue">Overdue</option>
            </select>
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Active Reviews</span>
              <Activity className="w-4 h-4 text-green-500" />
            </div>
            <div className="text-2xl font-bold">{stats.activeReviews}</div>
            <div className="text-xs text-gray-500 mt-1">In progress</div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Items Pending</span>
              <Clock className="w-4 h-4 text-yellow-500" />
            </div>
            <div className="text-2xl font-bold">{stats.itemsPendingReview}</div>
            <div className="text-xs text-gray-500 mt-1">Awaiting review</div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Compliance Rate</span>
              <Award className="w-4 h-4 text-purple-500" />
            </div>
            <div className="text-2xl font-bold">{stats.complianceRate}%</div>
            <div className="text-xs text-gray-500 mt-1">On schedule</div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Overdue</span>
              <AlertTriangle className="w-4 h-4 text-red-500" />
            </div>
            <div className="text-2xl font-bold text-red-500">{stats.overdueReviews}</div>
            <div className="text-xs text-gray-500 mt-1">Need attention</div>
          </div>
        </div>

        {/* Reviews List */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {filteredReviews.map((review) => (
              <div
                key={review.id}
                className="p-6 hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer transition-colors"
                onClick={() => setSelectedReview(review)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-lg font-medium">{review.name}</h3>
                      <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(review.status)}`}>
                        {review.status.replace('-', ' ')}
                      </span>
                      <span className={`text-xs ${getPriorityColor(review.priority)}`}>
                        {review.priority.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      {review.description}
                    </p>
                    <div className="flex items-center gap-6 text-sm text-gray-500">
                      <div className="flex items-center gap-1">
                        <Users className="w-4 h-4" />
                        <span>{review.reviewers.length} reviewers</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <FileCheck className="w-4 h-4" />
                        <span>{review.totalItems} items</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Calendar className="w-4 h-4" />
                        <span>Due {new Date(review.endDate).toLocaleDateString()}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <RefreshCw className="w-4 h-4" />
                        <span>{review.frequency}</span>
                      </div>
                    </div>
                  </div>
                  <div className="ml-6">
                    <div className="text-center mb-2">
                      <div className="text-3xl font-bold">{review.progress}%</div>
                      <div className="text-xs text-gray-500">Complete</div>
                    </div>
                    <div className="w-32">
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            review.progress === 100 ? 'bg-green-500' :
                            review.progress >= 50 ? 'bg-blue-500' :
                            'bg-yellow-500'
                          }`}
                          style={{ width: `${review.progress}%` }}
                        />
                      </div>
                    </div>
                    {review.status === 'not-started' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleStartReview(review);
                        }}
                        className="mt-3 w-full px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors flex items-center justify-center gap-1"
                      >
                        <Play className="w-3 h-3" />
                        Start
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </ResponsiveContainer>

      {/* Create Review Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">Create Access Review</h2>
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
                  <label className="block text-sm font-medium mb-2">Review Name</label>
                  <input
                    type="text"
                    placeholder="e.g., Q1 2025 Privileged Access Review"
                    className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Description</label>
                  <textarea
                    rows={3}
                    placeholder="Describe the purpose and scope of this review..."
                    className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Review Type</label>
                    <select className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500">
                      <option value="user-access">User Access</option>
                      <option value="guest-access">Guest Access</option>
                      <option value="privileged-access">Privileged Access</option>
                      <option value="app-permissions">App Permissions</option>
                      <option value="group-membership">Group Membership</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Frequency</label>
                    <select className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500">
                      <option value="one-time">One Time</option>
                      <option value="weekly">Weekly</option>
                      <option value="monthly">Monthly</option>
                      <option value="quarterly">Quarterly</option>
                      <option value="annual">Annual</option>
                    </select>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Start Date</label>
                    <input
                      type="date"
                      className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">End Date</label>
                    <input
                      type="date"
                      className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Settings</label>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="rounded border-gray-300 dark:border-gray-600" defaultChecked />
                      <span className="text-sm">Send email notifications</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="rounded border-gray-300 dark:border-gray-600" defaultChecked />
                      <span className="text-sm">Require justification for decisions</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="rounded border-gray-300 dark:border-gray-600" />
                      <span className="text-sm">Auto-complete when all items reviewed</span>
                    </label>
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
                      toast({ title: 'Review Created', description: 'Access review created successfully' });
                      setShowCreateModal(false);
                    }}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                  >
                    Create Review
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