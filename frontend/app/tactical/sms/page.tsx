'use client';

import React, { useState } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { 
  Smartphone, 
  Plus, 
  Search, 
  Filter, 
  Download, 
  Settings, 
  Edit3, 
  Trash2, 
  Eye, 
  Send,
  MessageSquare,
  Clock,
  Users,
  CheckCircle,
  AlertTriangle,
  Calendar,
  BarChart3,
  Activity,
  Zap,
  Globe,
  Shield,
  Phone
} from 'lucide-react';

interface SMSMessage {
  id: string;
  recipient: string;
  message: string;
  type: 'alert' | 'notification' | 'reminder' | 'verification';
  priority: 'low' | 'medium' | 'high' | 'critical';
  status: 'sent' | 'delivered' | 'failed' | 'pending' | 'scheduled';
  scheduledFor?: Date;
  sentAt?: Date;
  deliveredAt?: Date;
  cost: number;
  template?: string;
  campaign?: string;
}

interface SMSCampaign {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'paused' | 'completed' | 'draft';
  type: string;
  recipientCount: number;
  sentCount: number;
  deliveredCount: number;
  failedCount: number;
  deliveryRate: number;
  totalCost: number;
  createdAt: Date;
  lastSent?: Date;
}

const mockMessages: SMSMessage[] = [
  {
    id: '1',
    recipient: '+1-555-0123',
    message: 'URGENT: Critical security incident detected in production environment. Immediate action required.',
    type: 'alert',
    priority: 'critical',
    status: 'delivered',
    sentAt: new Date('2024-01-20T10:30:00'),
    deliveredAt: new Date('2024-01-20T10:30:15'),
    cost: 0.05,
    template: 'Security Alert',
    campaign: 'Critical Alerts'
  },
  {
    id: '2',
    recipient: '+1-555-0456',
    message: 'Azure compliance drift detected. Review your governance dashboard for details.',
    type: 'notification',
    priority: 'high',
    status: 'delivered',
    sentAt: new Date('2024-01-20T09:15:00'),
    deliveredAt: new Date('2024-01-20T09:15:08'),
    cost: 0.04,
    template: 'Compliance Alert',
    campaign: 'Governance Updates'
  },
  {
    id: '3',
    recipient: '+1-555-0789',
    message: 'Monthly Azure spending exceeded 85% of budget. Cost optimization recommendations available.',
    type: 'notification',
    priority: 'medium',
    status: 'delivered',
    sentAt: new Date('2024-01-20T08:00:00'),
    deliveredAt: new Date('2024-01-20T08:00:12'),
    cost: 0.04,
    template: 'Cost Alert',
    campaign: 'Budget Notifications'
  },
  {
    id: '4',
    recipient: '+1-555-0321',
    message: 'Verification code: 847392. Valid for 5 minutes.',
    type: 'verification',
    priority: 'high',
    status: 'delivered',
    sentAt: new Date('2024-01-20T07:45:00'),
    deliveredAt: new Date('2024-01-20T07:45:03'),
    cost: 0.03,
    template: 'Verification Code'
  },
  {
    id: '5',
    recipient: '+1-555-0654',
    message: 'Scheduled maintenance window begins in 30 minutes. Systems may be briefly unavailable.',
    type: 'reminder',
    priority: 'low',
    status: 'scheduled',
    scheduledFor: new Date('2024-01-20T14:30:00'),
    cost: 0.04,
    template: 'Maintenance Reminder',
    campaign: 'Operations Updates'
  }
];

const mockCampaigns: SMSCampaign[] = [
  {
    id: '1',
    name: 'Critical Security Alerts',
    description: 'High-priority security incident notifications',
    status: 'active',
    type: 'Security',
    recipientCount: 25,
    sentCount: 147,
    deliveredCount: 144,
    failedCount: 3,
    deliveryRate: 97.9,
    totalCost: 7.35,
    createdAt: new Date('2024-01-01'),
    lastSent: new Date('2024-01-20T10:30:00')
  },
  {
    id: '2',
    name: 'Governance Updates',
    description: 'Policy compliance and governance notifications',
    status: 'active',
    type: 'Compliance',
    recipientCount: 12,
    sentCount: 89,
    deliveredCount: 87,
    failedCount: 2,
    deliveryRate: 97.8,
    totalCost: 3.56,
    createdAt: new Date('2024-01-05'),
    lastSent: new Date('2024-01-20T09:15:00')
  },
  {
    id: '3',
    name: 'Budget Notifications',
    description: 'Cost management and budget alerts',
    status: 'active',
    type: 'Cost Management',
    recipientCount: 8,
    sentCount: 34,
    deliveredCount: 34,
    failedCount: 0,
    deliveryRate: 100.0,
    totalCost: 1.36,
    createdAt: new Date('2024-01-10'),
    lastSent: new Date('2024-01-20T08:00:00')
  }
];

export default function Page() {
  const [messages, setMessages] = useState<SMSMessage[]>(mockMessages);
  const [campaigns, setCampaigns] = useState<SMSCampaign[]>(mockCampaigns);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [priorityFilter, setPriorityFilter] = useState<string>('all');
  const [activeTab, setActiveTab] = useState<'messages' | 'campaigns' | 'analytics'>('messages');

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'delivered': return 'text-green-400 bg-green-400/10';
      case 'sent': return 'text-blue-400 bg-blue-400/10';
      case 'pending': return 'text-yellow-400 bg-yellow-400/10';
      case 'scheduled': return 'text-purple-400 bg-purple-400/10';
      case 'failed': return 'text-red-400 bg-red-400/10';
      default: return 'text-gray-400 bg-gray-400/10';
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

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'alert': return <AlertTriangle className="w-4 h-4 text-red-400" />;
      case 'notification': return <MessageSquare className="w-4 h-4 text-blue-400" />;
      case 'reminder': return <Clock className="w-4 h-4 text-yellow-400" />;
      case 'verification': return <Shield className="w-4 h-4 text-green-400" />;
      default: return <MessageSquare className="w-4 h-4 text-gray-400" />;
    }
  };

  const filteredMessages = messages.filter(message => {
    const matchesSearch = message.recipient.includes(searchTerm) ||
                         message.message.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || message.status === statusFilter;
    const matchesPriority = priorityFilter === 'all' || message.priority === priorityFilter;
    
    return matchesSearch && matchesStatus && matchesPriority;
  });

  const totalMessages = messages.length;
  const deliveredMessages = messages.filter(m => m.status === 'delivered').length;
  const totalCost = messages.reduce((sum, m) => sum + m.cost, 0);
  const avgDeliveryRate = totalMessages > 0 ? (deliveredMessages / totalMessages) * 100 : 0;

  return (
    <TacticalPageTemplate title="SMS Alerts" subtitle="SMS Communication & Delivery Management" icon={Smartphone}>
      <div className="space-y-6">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Messages</p>
                <p className="text-2xl font-bold text-white">{totalMessages}</p>
              </div>
              <MessageSquare className="w-8 h-8 text-blue-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400 flex items-center">
                <CheckCircle className="w-4 h-4 mr-1" />
                {deliveredMessages} Delivered
              </span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Delivery Rate</p>
                <p className="text-2xl font-bold text-white">{avgDeliveryRate.toFixed(1)}%</p>
              </div>
              <BarChart3 className="w-8 h-8 text-green-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400">Industry leading</span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Cost</p>
                <p className="text-2xl font-bold text-white">${totalCost.toFixed(2)}</p>
              </div>
              <Activity className="w-8 h-8 text-purple-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-green-400">-12.3% vs last month</span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Active Campaigns</p>
                <p className="text-2xl font-bold text-white">{campaigns.filter(c => c.status === 'active').length}</p>
              </div>
              <Zap className="w-8 h-8 text-orange-400" />
            </div>
            <div className="mt-4 flex items-center text-sm">
              <span className="text-blue-400">Running continuously</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg">
          <div className="border-b border-gray-800">
            <nav className="flex space-x-8 px-6">
              <button
                onClick={() => setActiveTab('messages')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'messages'
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300'
                }`}
              >
                Messages
              </button>
              <button
                onClick={() => setActiveTab('campaigns')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'campaigns'
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300'
                }`}
              >
                Campaigns
              </button>
              <button
                onClick={() => setActiveTab('analytics')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'analytics'
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300'
                }`}
              >
                Analytics
              </button>
            </nav>
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
                  placeholder="Search messages..."
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
                  <option value="delivered">Delivered</option>
                  <option value="sent">Sent</option>
                  <option value="pending">Pending</option>
                  <option value="scheduled">Scheduled</option>
                  <option value="failed">Failed</option>
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
              <button className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors">
                <Plus className="w-4 h-4" />
                <span>Send SMS</span>
              </button>
            </div>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'messages' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-800 bg-gray-850">
                    <th className="text-left py-4 px-6 font-medium text-gray-300">Message</th>
                    <th className="text-left py-4 px-6 font-medium text-gray-300">Recipient</th>
                    <th className="text-left py-4 px-6 font-medium text-gray-300">Type</th>
                    <th className="text-left py-4 px-6 font-medium text-gray-300">Priority</th>
                    <th className="text-left py-4 px-6 font-medium text-gray-300">Status</th>
                    <th className="text-left py-4 px-6 font-medium text-gray-300">Cost</th>
                    <th className="text-left py-4 px-6 font-medium text-gray-300">Timestamp</th>
                    <th className="text-right py-4 px-6 font-medium text-gray-300">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredMessages.map((message, index) => (
                    <tr key={message.id} className={`border-b border-gray-800 hover:bg-gray-800/50 ${index % 2 === 0 ? 'bg-gray-900' : 'bg-gray-850'}`}>
                      <td className="py-4 px-6">
                        <div className="flex items-start space-x-3">
                          {getTypeIcon(message.type)}
                          <div className="min-w-0 flex-1">
                            <p className="text-sm text-white line-clamp-2">{message.message}</p>
                            {message.template && (
                              <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-700 text-gray-300 mt-1">
                                {message.template}
                              </span>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="py-4 px-6">
                        <div className="flex items-center space-x-2">
                          <Phone className="w-4 h-4 text-gray-400" />
                          <span className="text-sm text-white font-mono">{message.recipient}</span>
                        </div>
                      </td>
                      <td className="py-4 px-6">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize text-blue-400 bg-blue-400/10">
                          {message.type}
                        </span>
                      </td>
                      <td className="py-4 px-6">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${getPriorityColor(message.priority)}`}>
                          {message.priority}
                        </span>
                      </td>
                      <td className="py-4 px-6">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${getStatusColor(message.status)}`}>
                          {message.status}
                        </span>
                      </td>
                      <td className="py-4 px-6">
                        <span className="text-sm text-white">${message.cost.toFixed(3)}</span>
                      </td>
                      <td className="py-4 px-6">
                        <div className="text-sm text-gray-300">
                          {message.status === 'scheduled' && message.scheduledFor ? (
                            <div>
                              <div>Scheduled</div>
                              <div className="text-xs text-gray-400">{message.scheduledFor.toLocaleString()}</div>
                            </div>
                          ) : message.deliveredAt ? (
                            <div>
                              <div>Delivered</div>
                              <div className="text-xs text-gray-400">{message.deliveredAt.toLocaleString()}</div>
                            </div>
                          ) : message.sentAt ? (
                            <div>
                              <div>Sent</div>
                              <div className="text-xs text-gray-400">{message.sentAt.toLocaleString()}</div>
                            </div>
                          ) : (
                            <span className="text-gray-500">Pending</span>
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
                          <button className="text-gray-400 hover:text-red-400 transition-colors">
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {filteredMessages.length === 0 && (
              <div className="text-center py-12">
                <Smartphone className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-300 mb-2">No messages found</h3>
                <p className="text-gray-500">Try adjusting your search terms or filters.</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'campaigns' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg">
            <div className="px-6 py-4 border-b border-gray-800">
              <h3 className="text-lg font-semibold text-white">SMS Campaigns</h3>
              <p className="text-sm text-gray-400 mt-1">Manage automated SMS notification campaigns</p>
            </div>
            <div className="divide-y divide-gray-800">
              {campaigns.map((campaign) => (
                <div key={campaign.id} className="p-6 hover:bg-gray-800/50">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-3">
                        <h4 className="text-lg font-medium text-white">{campaign.name}</h4>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${getStatusColor(campaign.status)}`}>
                          {campaign.status}
                        </span>
                        <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-700 text-gray-300">
                          {campaign.type}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400 mt-1">{campaign.description}</p>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-4">
                        <div>
                          <div className="text-sm font-medium text-gray-300">Recipients</div>
                          <div className="text-lg font-bold text-white">{campaign.recipientCount}</div>
                        </div>
                        <div>
                          <div className="text-sm font-medium text-gray-300">Sent</div>
                          <div className="text-lg font-bold text-white">{campaign.sentCount}</div>
                        </div>
                        <div>
                          <div className="text-sm font-medium text-gray-300">Delivered</div>
                          <div className="text-lg font-bold text-green-400">{campaign.deliveredCount}</div>
                        </div>
                        <div>
                          <div className="text-sm font-medium text-gray-300">Delivery Rate</div>
                          <div className="text-lg font-bold text-blue-400">{campaign.deliveryRate.toFixed(1)}%</div>
                        </div>
                        <div>
                          <div className="text-sm font-medium text-gray-300">Cost</div>
                          <div className="text-lg font-bold text-white">${campaign.totalCost.toFixed(2)}</div>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button className="text-gray-400 hover:text-blue-400 transition-colors">
                        <Eye className="w-4 h-4" />
                      </button>
                      <button className="text-gray-400 hover:text-yellow-400 transition-colors">
                        <Edit3 className="w-4 h-4" />
                      </button>
                      <button className="text-gray-400 hover:text-red-400 transition-colors">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  {campaign.lastSent && (
                    <div className="mt-4 text-sm text-gray-400">
                      Last sent: {campaign.lastSent.toLocaleString()}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
                <h4 className="text-lg font-semibold text-white mb-4">Delivery Performance</h4>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Average Delivery Rate</span>
                    <span className="text-lg font-bold text-green-400">{avgDeliveryRate.toFixed(1)}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Failed Messages</span>
                    <span className="text-lg font-bold text-red-400">{messages.filter(m => m.status === 'failed').length}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Avg Delivery Time</span>
                    <span className="text-lg font-bold text-blue-400">8.3s</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
                <h4 className="text-lg font-semibold text-white mb-4">Cost Analysis</h4>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Total Cost (Month)</span>
                    <span className="text-lg font-bold text-white">${totalCost.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Avg Cost per SMS</span>
                    <span className="text-lg font-bold text-purple-400">${(totalCost / totalMessages).toFixed(3)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Cost Trend</span>
                    <span className="text-lg font-bold text-green-400">-12.3%</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
                <h4 className="text-lg font-semibold text-white mb-4">Usage Patterns</h4>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Peak Hour</span>
                    <span className="text-lg font-bold text-orange-400">10:00 AM</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Most Used Type</span>
                    <span className="text-lg font-bold text-yellow-400">Alerts</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Critical Messages</span>
                    <span className="text-lg font-bold text-red-400">{messages.filter(m => m.priority === 'critical').length}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="px-6 py-4 border-b border-gray-800">
                <h3 className="text-lg font-semibold text-white">Recent Activity</h3>
                <div className="flex items-center space-x-2 text-sm text-green-400 mt-1">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span>Live updates</span>
                </div>
              </div>
              <div className="p-6">
                <div className="space-y-4">
                  {messages.slice(0, 5).map((message) => (
                    <div key={message.id} className="flex items-start space-x-3">
                      <div className={`w-2 h-2 rounded-full mt-2 flex-shrink-0 ${
                        message.status === 'delivered' ? 'bg-green-400' :
                        message.status === 'failed' ? 'bg-red-400' :
                        message.status === 'scheduled' ? 'bg-purple-400' : 'bg-yellow-400'
                      }`}></div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-white">
                          <span className="font-medium">{message.type}</span> message to {message.recipient}
                        </p>
                        <p className="text-xs text-gray-400 mt-1">
                          {message.deliveredAt?.toLocaleTimeString() || 
                           message.sentAt?.toLocaleTimeString() || 
                           'Pending'} • ${message.cost.toFixed(3)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </TacticalPageTemplate>
  );
}