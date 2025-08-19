'use client';

import React, { useState } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { 
  Bell, 
  Plus, 
  Search, 
  Filter, 
  Download, 
  Settings, 
  Check, 
  X, 
  AlertTriangle, 
  Info, 
  CheckCircle, 
  Clock, 
  Play,
  Pause,
  Edit3,
  Trash2,
  Eye,
  BarChart3,
  Target
} from 'lucide-react';

interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'warning' | 'error' | 'success';
  priority: 'low' | 'medium' | 'high' | 'critical';
  channels: string[];
  status: 'active' | 'paused' | 'draft';
  triggers: string[];
  recipients: number;
  sentCount: number;
  openRate: number;
  createdAt: Date;
  lastSent?: Date;
}

const mockNotifications: Notification[] = [
  {
    id: '1',
    title: 'Critical Security Alert',
    message: 'Unauthorized access attempt detected on production environment',
    type: 'error',
    priority: 'critical',
    channels: ['email', 'sms', 'slack', 'webhook'],
    status: 'active',
    triggers: ['security.unauthorized_access', 'security.brute_force'],
    recipients: 25,
    sentCount: 147,
    openRate: 98.5,
    createdAt: new Date('2024-01-15'),
    lastSent: new Date('2024-01-20T10:30:00')
  },
  {
    id: '2',
    title: 'Policy Compliance Drift',
    message: 'Azure resources have drifted from compliance policies',
    type: 'warning',
    priority: 'high',
    channels: ['email', 'slack'],
    status: 'active',
    triggers: ['compliance.drift_detected', 'governance.policy_violation'],
    recipients: 12,
    sentCount: 89,
    openRate: 87.2,
    createdAt: new Date('2024-01-10'),
    lastSent: new Date('2024-01-19T14:15:00')
  },
  {
    id: '3',
    title: 'Cost Threshold Exceeded',
    message: 'Monthly Azure spending has exceeded 85% of budget allocation',
    type: 'warning',
    priority: 'medium',
    channels: ['email'],
    status: 'active',
    triggers: ['cost.threshold_exceeded', 'budget.warning'],
    recipients: 8,
    sentCount: 34,
    openRate: 92.1,
    createdAt: new Date('2024-01-05'),
    lastSent: new Date('2024-01-18T09:00:00')
  },
  {
    id: '4',
    title: 'Backup Completion Report',
    message: 'Daily backup operations completed successfully',
    type: 'success',
    priority: 'low',
    channels: ['email'],
    status: 'active',
    triggers: ['backup.completed', 'operations.daily_report'],
    recipients: 15,
    sentCount: 456,
    openRate: 76.8,
    createdAt: new Date('2024-01-01'),
    lastSent: new Date('2024-01-20T06:00:00')
  },
  {
    id: '5',
    title: 'New Resource Provisioned',
    message: 'Azure resource provisioning workflow has been triggered',
    type: 'info',
    priority: 'low',
    channels: ['slack'],
    status: 'paused',
    triggers: ['resource.provisioned', 'infrastructure.change'],
    recipients: 6,
    sentCount: 78,
    openRate: 65.4,
    createdAt: new Date('2023-12-20'),
    lastSent: new Date('2024-01-15T16:45:00')
  }
];

export default function Page() {
  const [notifications, setNotifications] = useState<Notification[]>(mockNotifications);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [priorityFilter, setPriorityFilter] = useState<string>('all');
  const [showCreateModal, setShowCreateModal] = useState(false);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'error': return <AlertTriangle className="w-4 h-4 text-red-400" />;
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      case 'success': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'info': return <Info className="w-4 h-4 text-blue-400" />;
      default: return <Bell className="w-4 h-4 text-gray-400" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'text-red-400 bg-red-400/10';
      case 'high': return 'text-orange-400 bg-orange-400/10';
      case 'medium': return 'text-yellow-400 bg-yellow-400/10';
      case 'low': return 'text-green-400 bg-green-400/10';
      default: return 'text-gray-400 bg-gray-400/10';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400 bg-green-400/10';
      case 'paused': return 'text-yellow-400 bg-yellow-400/10';
      case 'draft': return 'text-gray-400 bg-gray-400/10';
      default: return 'text-gray-400 bg-gray-400/10';
    }
  };

  const filteredNotifications = notifications.filter(notification => {
    const matchesSearch = notification.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         notification.message.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || notification.status === statusFilter;
    const matchesPriority = priorityFilter === 'all' || notification.priority === priorityFilter;
    
    return matchesSearch && matchesStatus && matchesPriority;
  });

  const toggleNotificationStatus = (id: string) => {
    setNotifications(prev => prev.map(notification =>
      notification.id === id
        ? { ...notification, status: notification.status === 'active' ? 'paused' : 'active' }
        : notification
    ));
  };

  const deleteNotification = (id: string) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
  };

  return (
    <TacticalPageTemplate title="Notifications" subtitle="Real-time Notification Management Center" icon={Bell}>
      <div className="space-y-6">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Notifications</p>
                <p className="text-2xl font-bold text-white">{notifications.length}</p>
              </div>
              <Bell className="w-8 h-8 text-blue-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400 flex items-center">
                <CheckCircle className="w-4 h-4 mr-1" />
                {notifications.filter(n => n.status === 'active').length} Active
              </span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Messages Sent</p>
                <p className="text-2xl font-bold text-white">
                  {notifications.reduce((sum, n) => sum + n.sentCount, 0).toLocaleString()}
                </p>
              </div>
              <Target className="w-8 h-8 text-green-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400">+23.5% this month</span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Avg Open Rate</p>
                <p className="text-2xl font-bold text-white">
                  {(notifications.reduce((sum, n) => sum + n.openRate, 0) / notifications.length).toFixed(1)}%
                </p>
              </div>
              <BarChart3 className="w-8 h-8 text-purple-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400">Above industry avg</span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Critical Alerts</p>
                <p className="text-2xl font-bold text-white">
                  {notifications.filter(n => n.priority === 'critical').length}
                </p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-red-400">Requires immediate attention</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search notifications..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg pl-10 pr-4 py-2 w-full sm:w-80 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              
              <div className="flex gap-2">
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="paused">Paused</option>
                  <option value="draft">Draft</option>
                </select>
                
                <select
                  value={priorityFilter}
                  onChange={(e) => setPriorityFilter(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Priority</option>
                  <option value="critical">Critical</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
              </div>
            </div>
            
            <div className="flex gap-2">
              <button className="flex items-center space-x-2 bg-gray-800 hover:bg-gray-700 text-gray-300 px-4 py-2 rounded-lg border border-gray-700 transition-colors">
                <Download className="w-4 h-4" />
                <span>Export</span>
              </button>
              <button className="flex items-center space-x-2 bg-gray-800 hover:bg-gray-700 text-gray-300 px-4 py-2 rounded-lg border border-gray-700 transition-colors">
                <Settings className="w-4 h-4" />
                <span>Settings</span>
              </button>
              <button 
                onClick={() => setShowCreateModal(true)}
                className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
              >
                <Plus className="w-4 h-4" />
                <span>Create Notification</span>
              </button>
            </div>
          </div>
        </div>

        {/* Notifications Table */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800 bg-gray-850">
                  <th className="text-left py-4 px-6 font-medium text-gray-300">Notification</th>
                  <th className="text-left py-4 px-6 font-medium text-gray-300">Type</th>
                  <th className="text-left py-4 px-6 font-medium text-gray-300">Priority</th>
                  <th className="text-left py-4 px-6 font-medium text-gray-300">Status</th>
                  <th className="text-left py-4 px-6 font-medium text-gray-300">Channels</th>
                  <th className="text-left py-4 px-6 font-medium text-gray-300">Recipients</th>
                  <th className="text-left py-4 px-6 font-medium text-gray-300">Open Rate</th>
                  <th className="text-left py-4 px-6 font-medium text-gray-300">Last Sent</th>
                  <th className="text-right py-4 px-6 font-medium text-gray-300">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredNotifications.map((notification, index) => (
                  <tr key={notification.id} className={`border-b border-gray-800 hover:bg-gray-800/50 ${index % 2 === 0 ? 'bg-gray-900' : 'bg-gray-850'}`}>
                    <td className="py-4 px-6">
                      <div className="flex items-start space-x-3">
                        {getTypeIcon(notification.type)}
                        <div className="min-w-0 flex-1">
                          <p className="font-medium text-white truncate">{notification.title}</p>
                          <p className="text-sm text-gray-400 truncate">{notification.message}</p>
                          <div className="flex flex-wrap gap-1 mt-2">
                            {notification.triggers.slice(0, 2).map((trigger, idx) => (
                              <span key={idx} className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-700 text-gray-300">
                                {trigger}
                              </span>
                            ))}
                            {notification.triggers.length > 2 && (
                              <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-700 text-gray-300">
                                +{notification.triggers.length - 2} more
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="py-4 px-6">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${getPriorityColor(notification.type)}`}>
                        {notification.type}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${getPriorityColor(notification.priority)}`}>
                        {notification.priority}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${getStatusColor(notification.status)}`}>
                        {notification.status}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <div className="flex flex-wrap gap-1">
                        {notification.channels.map((channel, idx) => (
                          <span key={idx} className="inline-flex items-center px-2 py-1 rounded text-xs bg-blue-900/30 text-blue-300 border border-blue-800">
                            {channel}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="py-4 px-6">
                      <div className="text-sm">
                        <div className="text-white font-medium">{notification.recipients}</div>
                        <div className="text-gray-400 text-xs">{notification.sentCount} sent</div>
                      </div>
                    </td>
                    <td className="py-4 px-6">
                      <div className="flex items-center space-x-2">
                        <div className="text-sm font-medium text-white">{notification.openRate}%</div>
                        <div className="w-16 bg-gray-700 rounded-full h-2">
                          <div 
                            className="bg-green-400 h-2 rounded-full transition-all duration-300" 
                            style={{ width: `${notification.openRate}%` }}
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td className="py-4 px-6">
                      <div className="text-sm text-gray-300">
                        {notification.lastSent ? (
                          <div>
                            <div>{notification.lastSent.toLocaleDateString()}</div>
                            <div className="text-xs text-gray-400">{notification.lastSent.toLocaleTimeString()}</div>
                          </div>
                        ) : (
                          <span className="text-gray-500">Never</span>
                        )}
                      </div>
                    </td>
                    <td className="py-4 px-6">
                      <div className="flex items-center justify-end space-x-2">
                        <button className="text-gray-400 hover:text-blue-400 transition-colors">
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="text-gray-400 hover:text-yellow-400 transition-colors">
                          <Edit3 className="w-4 h-4" />
                        </button>
                        <button 
                          onClick={() => toggleNotificationStatus(notification.id)}
                          className="text-gray-400 hover:text-green-400 transition-colors"
                        >
                          {notification.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                        </button>
                        <button 
                          onClick={() => deleteNotification(notification.id)}
                          className="text-gray-400 hover:text-red-400 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {filteredNotifications.length === 0 && (
            <div className="text-center py-12">
              <Bell className="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-300 mb-2">No notifications found</h3>
              <p className="text-gray-500">Try adjusting your search terms or filters.</p>
            </div>
          )}
        </div>

        {/* Real-time Activity Feed */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg">
          <div className="px-6 py-4 border-b border-gray-800">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">Recent Activity</h3>
              <div className="flex items-center space-x-2 text-sm text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Live</span>
              </div>
            </div>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-red-400 rounded-full mt-2 flex-shrink-0"></div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white">
                    <span className="font-medium">Critical Security Alert</span> sent to 25 recipients
                  </p>
                  <p className="text-xs text-gray-400 mt-1">2 minutes ago • 98.5% open rate</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-yellow-400 rounded-full mt-2 flex-shrink-0"></div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white">
                    <span className="font-medium">Policy Compliance Drift</span> notification triggered
                  </p>
                  <p className="text-xs text-gray-400 mt-1">5 minutes ago • Slack + Email</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white">
                    <span className="font-medium">Backup Completion Report</span> sent successfully
                  </p>
                  <p className="text-xs text-gray-400 mt-1">1 hour ago • 15 recipients</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </TacticalPageTemplate>
  );
}