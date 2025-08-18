'use client';

import React, { useState, useEffect } from 'react';
import { 
  GitBranch, Calendar, Clock, User, Users, CheckCircle, XCircle,
  AlertTriangle, Info, FileText, Shield, Zap, Server, Database,
  Package, Settings, Filter, Search, Plus, Edit, Eye, ChevronRight,
  BarChart, TrendingUp, Timer, Activity, Bell, Tag, ArrowRight
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface ChangeRequest {
  id: string;
  title: string;
  description: string;
  type: 'standard' | 'normal' | 'emergency' | 'routine';
  status: 'pending' | 'approved' | 'in_progress' | 'completed' | 'failed' | 'cancelled';
  priority: 'critical' | 'high' | 'medium' | 'low';
  category: 'infrastructure' | 'application' | 'database' | 'network' | 'security' | 'other';
  requestedBy: string;
  assignee: string;
  approvers: string[];
  approvalStatus: { name: string; status: 'approved' | 'pending' | 'rejected' }[];
  scheduledDate: string;
  scheduledTime: string;
  duration: string;
  impactLevel: 'high' | 'medium' | 'low';
  affectedServices: string[];
  rollbackPlan: string;
  testingRequired: boolean;
  createdAt: string;
  updatedAt: string;
  completedAt?: string;
  changeWindow: { start: string; end: string };
  risk: {
    level: 'high' | 'medium' | 'low';
    description: string;
    mitigations: string[];
  };
}

export default function ChangeManagement() {
  const [changes, setChanges] = useState<ChangeRequest[]>([]);
  const [selectedChange, setSelectedChange] = useState<ChangeRequest | null>(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterType, setFilterType] = useState('all');
  const [filterPriority, setFilterPriority] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'calendar' | 'list'>('list');
  const [showNewChange, setShowNewChange] = useState(false);

  useEffect(() => {
    // Initialize with mock data
    setChanges([
      {
        id: 'CHG-2024-001',
        title: 'Database Migration to PostgreSQL 15',
        description: 'Upgrade production database from PostgreSQL 13 to 15 for performance improvements',
        type: 'normal',
        status: 'approved',
        priority: 'high',
        category: 'database',
        requestedBy: 'John Smith',
        assignee: 'Database Team',
        approvers: ['CTO', 'Database Lead', 'Security Lead'],
        approvalStatus: [
          { name: 'CTO', status: 'approved' },
          { name: 'Database Lead', status: 'approved' },
          { name: 'Security Lead', status: 'pending' }
        ],
        scheduledDate: 'Tomorrow',
        scheduledTime: '2:00 AM - 6:00 AM',
        duration: '4 hours',
        impactLevel: 'high',
        affectedServices: ['User Service', 'Order Service', 'Analytics'],
        rollbackPlan: 'Restore from snapshot, revert configuration changes',
        testingRequired: true,
        createdAt: '2 days ago',
        updatedAt: '1 hour ago',
        changeWindow: { start: '2:00 AM', end: '6:00 AM' },
        risk: {
          level: 'medium',
          description: 'Potential data inconsistency during migration',
          mitigations: ['Full backup before migration', 'Staged rollout', 'Rollback procedure tested']
        }
      },
      {
        id: 'CHG-2024-002',
        title: 'Emergency Security Patch Deployment',
        description: 'Critical security vulnerability patch for authentication service',
        type: 'emergency',
        status: 'in_progress',
        priority: 'critical',
        category: 'security',
        requestedBy: 'Security Team',
        assignee: 'DevOps Team',
        approvers: ['Security Lead'],
        approvalStatus: [
          { name: 'Security Lead', status: 'approved' }
        ],
        scheduledDate: 'Today',
        scheduledTime: 'Immediate',
        duration: '30 minutes',
        impactLevel: 'low',
        affectedServices: ['Authentication Service'],
        rollbackPlan: 'Revert to previous version via blue-green deployment',
        testingRequired: false,
        createdAt: '2 hours ago',
        updatedAt: '5 minutes ago',
        changeWindow: { start: 'Now', end: '30 min' },
        risk: {
          level: 'low',
          description: 'Minimal risk with blue-green deployment',
          mitigations: ['Blue-green deployment', 'Automated rollback on failure']
        }
      },
      {
        id: 'CHG-2024-003',
        title: 'Network Load Balancer Configuration Update',
        description: 'Update load balancer rules for improved traffic distribution',
        type: 'standard',
        status: 'pending',
        priority: 'medium',
        category: 'network',
        requestedBy: 'Network Team',
        assignee: 'Network Engineers',
        approvers: ['Network Lead', 'Operations Manager'],
        approvalStatus: [
          { name: 'Network Lead', status: 'pending' },
          { name: 'Operations Manager', status: 'pending' }
        ],
        scheduledDate: 'Next Week',
        scheduledTime: '10:00 PM - 11:00 PM',
        duration: '1 hour',
        impactLevel: 'medium',
        affectedServices: ['All public-facing services'],
        rollbackPlan: 'Restore previous configuration from backup',
        testingRequired: true,
        createdAt: '1 week ago',
        updatedAt: '3 days ago',
        changeWindow: { start: '10:00 PM', end: '11:00 PM' },
        risk: {
          level: 'medium',
          description: 'Potential brief service interruption',
          mitigations: ['Gradual rollout', 'Health checks', 'Monitoring alerts']
        }
      },
      {
        id: 'CHG-2024-004',
        title: 'Application Server Memory Upgrade',
        description: 'Increase memory allocation for application servers to handle load',
        type: 'normal',
        status: 'completed',
        priority: 'medium',
        category: 'infrastructure',
        requestedBy: 'Operations Team',
        assignee: 'Infrastructure Team',
        approvers: ['Infrastructure Lead', 'Finance'],
        approvalStatus: [
          { name: 'Infrastructure Lead', status: 'approved' },
          { name: 'Finance', status: 'approved' }
        ],
        scheduledDate: 'Yesterday',
        scheduledTime: '3:00 AM - 4:00 AM',
        duration: '1 hour',
        impactLevel: 'low',
        affectedServices: ['Application Servers'],
        rollbackPlan: 'Reduce memory allocation to previous levels',
        testingRequired: false,
        createdAt: '5 days ago',
        updatedAt: 'Yesterday',
        completedAt: 'Yesterday',
        changeWindow: { start: '3:00 AM', end: '4:00 AM' },
        risk: {
          level: 'low',
          description: 'Minimal risk with rolling update',
          mitigations: ['Rolling update', 'Resource monitoring']
        }
      },
      {
        id: 'CHG-2024-005',
        title: 'SSL Certificate Renewal',
        description: 'Renew and deploy SSL certificates for all domains',
        type: 'routine',
        status: 'approved',
        priority: 'high',
        category: 'security',
        requestedBy: 'Security Team',
        assignee: 'DevOps Team',
        approvers: ['Security Lead'],
        approvalStatus: [
          { name: 'Security Lead', status: 'approved' }
        ],
        scheduledDate: 'In 3 days',
        scheduledTime: '11:00 PM - 12:00 AM',
        duration: '1 hour',
        impactLevel: 'low',
        affectedServices: ['All HTTPS services'],
        rollbackPlan: 'Revert to current certificates',
        testingRequired: true,
        createdAt: '1 week ago',
        updatedAt: '2 days ago',
        changeWindow: { start: '11:00 PM', end: '12:00 AM' },
        risk: {
          level: 'low',
          description: 'Standard procedure with minimal risk',
          mitigations: ['Staged deployment', 'Certificate validation']
        }
      }
    ]);
  }, []);

  const filteredChanges = changes.filter(change => {
    if (filterStatus !== 'all' && change.status !== filterStatus) return false;
    if (filterType !== 'all' && change.type !== filterType) return false;
    if (filterPriority !== 'all' && change.priority !== filterPriority) return false;
    if (searchQuery && !change.title.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !change.description.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const stats = {
    pending: changes.filter(c => c.status === 'pending').length,
    approved: changes.filter(c => c.status === 'approved').length,
    inProgress: changes.filter(c => c.status === 'in_progress').length,
    completed: changes.filter(c => c.status === 'completed').length,
    emergency: changes.filter(c => c.type === 'emergency').length,
    thisWeek: changes.filter(c => c.scheduledDate.includes('Today') || c.scheduledDate.includes('Tomorrow')).length
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'pending': return 'text-yellow-500 bg-yellow-900/20';
      case 'approved': return 'text-blue-500 bg-blue-900/20';
      case 'in_progress': return 'text-orange-500 bg-orange-900/20';
      case 'completed': return 'text-green-500 bg-green-900/20';
      case 'failed': return 'text-red-500 bg-red-900/20';
      case 'cancelled': return 'text-gray-500 bg-gray-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch(priority) {
      case 'critical': return 'text-red-500';
      case 'high': return 'text-orange-500';
      case 'medium': return 'text-yellow-500';
      case 'low': return 'text-blue-500';
      default: return 'text-gray-500';
    }
  };

  const getTypeIcon = (type: string) => {
    switch(type) {
      case 'emergency': return <Zap className="w-4 h-4 text-red-500" />;
      case 'standard': return <FileText className="w-4 h-4 text-blue-500" />;
      case 'normal': return <Settings className="w-4 h-4 text-green-500" />;
      case 'routine': return <Clock className="w-4 h-4 text-gray-500" />;
      default: return null;
    }
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Change Management Center</h1>
            <p className="text-sm text-gray-400 mt-1">Track and manage all system changes</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setViewMode(viewMode === 'list' ? 'calendar' : 'list')}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2"
            >
              {viewMode === 'list' ? <Calendar className="w-4 h-4" /> : <BarChart className="w-4 h-4" />}
              <span>{viewMode === 'list' ? 'Calendar View' : 'List View'}</span>
            </button>
            
            <button
              onClick={() => setShowNewChange(true)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2"
            >
              <Plus className="w-4 h-4" />
              <span>New Change Request</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <Clock className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Pending</p>
              <p className="text-xl font-bold">{stats.pending}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Approved</p>
              <p className="text-xl font-bold">{stats.approved}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Activity className="w-5 h-5 text-orange-500" />
            <div>
              <p className="text-xs text-gray-400">In Progress</p>
              <p className="text-xl font-bold">{stats.inProgress}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Completed</p>
              <p className="text-xl font-bold">{stats.completed}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Zap className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Emergency</p>
              <p className="text-xl font-bold text-red-500">{stats.emergency}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Calendar className="w-5 h-5 text-cyan-500" />
            <div>
              <p className="text-xs text-gray-400">This Week</p>
              <p className="text-xl font-bold">{stats.thisWeek}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Filters */}
        <div className="flex items-center space-x-3 mb-6">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search changes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
            />
          </div>
          
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
            <option value="in_progress">In Progress</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
          
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Types</option>
            <option value="emergency">Emergency</option>
            <option value="standard">Standard</option>
            <option value="normal">Normal</option>
            <option value="routine">Routine</option>
          </select>
          
          <select
            value={filterPriority}
            onChange={(e) => setFilterPriority(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Priorities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>

        {/* Changes List */}
        <div className="space-y-4">
          {filteredChanges.map(change => (
            <div
              key={change.id}
              className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:bg-gray-850 cursor-pointer"
              onClick={() => setSelectedChange(change)}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    {getTypeIcon(change.type)}
                    <span className="text-xs text-gray-500 font-mono">{change.id}</span>
                    <span className={`px-2 py-1 text-xs rounded ${getStatusColor(change.status)}`}>
                      {change.status.toUpperCase().replace('_', ' ')}
                    </span>
                    <span className={`text-xs ${getPriorityColor(change.priority)}`}>
                      {change.priority.toUpperCase()} PRIORITY
                    </span>
                  </div>
                  <h3 className="text-sm font-bold mb-1">{change.title}</h3>
                  <p className="text-xs text-gray-400 mb-2">{change.description}</p>
                  
                  <div className="flex items-center space-x-6 text-xs text-gray-500">
                    <span className="flex items-center space-x-1">
                      <User className="w-3 h-3" />
                      <span>{change.requestedBy}</span>
                    </span>
                    <span className="flex items-center space-x-1">
                      <Users className="w-3 h-3" />
                      <span>{change.assignee}</span>
                    </span>
                    <span className="flex items-center space-x-1">
                      <Calendar className="w-3 h-3" />
                      <span>{change.scheduledDate}</span>
                    </span>
                    <span className="flex items-center space-x-1">
                      <Clock className="w-3 h-3" />
                      <span>{change.scheduledTime}</span>
                    </span>
                  </div>
                </div>
                
                <ChevronRight className="w-5 h-5 text-gray-500" />
              </div>
              
              <div className="flex items-center justify-between pt-3 border-t border-gray-800">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-400">Impact:</span>
                    <span className={`text-xs font-medium ${
                      change.impactLevel === 'high' ? 'text-red-500' :
                      change.impactLevel === 'medium' ? 'text-yellow-500' :
                      'text-green-500'
                    }`}>
                      {change.impactLevel.toUpperCase()}
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-400">Risk:</span>
                    <span className={`text-xs font-medium ${
                      change.risk.level === 'high' ? 'text-red-500' :
                      change.risk.level === 'medium' ? 'text-yellow-500' :
                      'text-green-500'
                    }`}>
                      {change.risk.level.toUpperCase()}
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-400">Affected:</span>
                    <div className="flex flex-wrap gap-1">
                      {change.affectedServices.slice(0, 2).map(service => (
                        <span key={service} className="px-2 py-0.5 bg-gray-800 rounded text-xs">
                          {service}
                        </span>
                      ))}
                      {change.affectedServices.length > 2 && (
                        <span className="text-xs text-gray-500">
                          +{change.affectedServices.length - 2} more
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  {/* Approval Status */}
                  <div className="flex items-center space-x-1">
                    {change.approvalStatus.map((approval, idx) => (
                      <div
                        key={idx}
                        className={`w-6 h-6 rounded-full flex items-center justify-center ${
                          approval.status === 'approved' ? 'bg-green-900/30' :
                          approval.status === 'rejected' ? 'bg-red-900/30' :
                          'bg-gray-900/30'
                        }`}
                        title={`${approval.name}: ${approval.status}`}
                      >
                        {approval.status === 'approved' ? (
                          <CheckCircle className="w-3 h-3 text-green-500" />
                        ) : approval.status === 'rejected' ? (
                          <XCircle className="w-3 h-3 text-red-500" />
                        ) : (
                          <Clock className="w-3 h-3 text-gray-500" />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}