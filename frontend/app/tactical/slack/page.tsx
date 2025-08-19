'use client';

import React, { useState, useEffect } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { 
  MessageCircle, 
  Plus, 
  Settings, 
  Users, 
  Hash, 
  Bell, 
  Zap, 
  Shield, 
  Activity, 
  Search,
  Filter,
  MoreVertical,
  Edit,
  Trash2,
  ExternalLink,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Info,
  Download,
  Upload,
  Copy,
  Eye,
  EyeOff,
  Calendar,
  Clock,
  User,
  Globe
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

interface SlackWorkspace {
  id: string;
  name: string;
  domain: string;
  teamId: string;
  status: 'connected' | 'disconnected' | 'error' | 'configuring';
  botToken: string;
  userToken?: string;
  channels: number;
  members: number;
  lastSync: string;
  createdAt: string;
  permissions: string[];
  notifications: {
    alerts: boolean;
    compliance: boolean;
    incidents: boolean;
    reports: boolean;
  };
}

interface SlackChannel {
  id: string;
  name: string;
  workspaceId: string;
  isPrivate: boolean;
  members: number;
  purpose?: string;
  topic?: string;
  notifications: boolean;
  lastActivity: string;
  messageCount: number;
}

interface SlackNotification {
  id: string;
  type: 'alert' | 'compliance' | 'incident' | 'report';
  title: string;
  message: string;
  channel: string;
  workspace: string;
  status: 'sent' | 'pending' | 'failed';
  timestamp: string;
  retries: number;
}

const mockWorkspaces: SlackWorkspace[] = [
  {
    id: 'ws-1',
    name: 'PolicyCortex Engineering',
    domain: 'policycortex-eng',
    teamId: 'T1234567890',
    status: 'connected',
    botToken: 'xoxb-****-****-****',
    userToken: 'xoxp-****-****-****',
    channels: 45,
    members: 127,
    lastSync: '2024-01-20T10:30:00Z',
    createdAt: '2024-01-15T08:00:00Z',
    permissions: ['chat:write', 'channels:read', 'users:read', 'files:write'],
    notifications: {
      alerts: true,
      compliance: true,
      incidents: true,
      reports: false
    }
  },
  {
    id: 'ws-2',
    name: 'PolicyCortex Operations',
    domain: 'policycortex-ops',
    teamId: 'T0987654321',
    status: 'connected',
    botToken: 'xoxb-****-****-****',
    channels: 23,
    members: 78,
    lastSync: '2024-01-20T10:25:00Z',
    createdAt: '2024-01-10T09:15:00Z',
    permissions: ['chat:write', 'channels:read', 'users:read'],
    notifications: {
      alerts: true,
      compliance: false,
      incidents: true,
      reports: true
    }
  }
];

const mockChannels: SlackChannel[] = [
  {
    id: 'ch-1',
    name: 'alerts',
    workspaceId: 'ws-1',
    isPrivate: false,
    members: 89,
    purpose: 'Real-time system alerts and notifications',
    notifications: true,
    lastActivity: '2024-01-20T10:28:00Z',
    messageCount: 1247
  },
  {
    id: 'ch-2',
    name: 'compliance-updates',
    workspaceId: 'ws-1',
    isPrivate: true,
    members: 15,
    purpose: 'Compliance status updates and reports',
    notifications: true,
    lastActivity: '2024-01-20T09:45:00Z',
    messageCount: 334
  },
  {
    id: 'ch-3',
    name: 'incidents',
    workspaceId: 'ws-2',
    isPrivate: false,
    members: 45,
    purpose: 'Incident response coordination',
    notifications: true,
    lastActivity: '2024-01-20T10:15:00Z',
    messageCount: 892
  }
];

const mockNotifications: SlackNotification[] = [
  {
    id: 'n-1',
    type: 'alert',
    title: 'High CPU Usage Detected',
    message: 'Azure VM cpu-usage exceeded 90% threshold',
    channel: 'alerts',
    workspace: 'PolicyCortex Engineering',
    status: 'sent',
    timestamp: '2024-01-20T10:28:00Z',
    retries: 0
  },
  {
    id: 'n-2',
    type: 'compliance',
    title: 'Policy Drift Detected',
    message: '3 resources found non-compliant with security policies',
    channel: 'compliance-updates',
    workspace: 'PolicyCortex Engineering',
    status: 'sent',
    timestamp: '2024-01-20T10:15:00Z',
    retries: 0
  },
  {
    id: 'n-3',
    type: 'incident',
    title: 'Service Degradation',
    message: 'API response times increased by 200%',
    channel: 'incidents',
    workspace: 'PolicyCortex Operations',
    status: 'failed',
    timestamp: '2024-01-20T10:00:00Z',
    retries: 2
  }
];

export default function SlackIntegrationPage() {
  const [workspaces, setWorkspaces] = useState<SlackWorkspace[]>(mockWorkspaces);
  const [channels, setChannels] = useState<SlackChannel[]>(mockChannels);
  const [notifications, setNotifications] = useState<SlackNotification[]>(mockNotifications);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedWorkspace, setSelectedWorkspace] = useState<string>('all');
  const [activeTab, setActiveTab] = useState('overview');
  const [isAddingWorkspace, setIsAddingWorkspace] = useState(false);
  const [isConfiguring, setIsConfiguring] = useState(false);

  const [newWorkspace, setNewWorkspace] = useState({
    name: '',
    domain: '',
    botToken: '',
    userToken: '',
    notifications: {
      alerts: true,
      compliance: true,
      incidents: true,
      reports: false
    }
  });

  const filteredWorkspaces = workspaces.filter(workspace =>
    workspace.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    workspace.domain.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'bg-green-500/20 text-green-400 border-green-500/20';
      case 'disconnected': return 'bg-gray-500/20 text-gray-400 border-gray-500/20';
      case 'error': return 'bg-red-500/20 text-red-400 border-red-500/20';
      case 'configuring': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/20';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return <CheckCircle className="w-4 h-4" />;
      case 'disconnected': return <XCircle className="w-4 h-4" />;
      case 'error': return <AlertTriangle className="w-4 h-4" />;
      case 'configuring': return <RefreshCw className="w-4 h-4 animate-spin" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  const handleAddWorkspace = () => {
    const workspace: SlackWorkspace = {
      id: `ws-${Date.now()}`,
      name: newWorkspace.name,
      domain: newWorkspace.domain,
      teamId: `T${Math.random().toString(36).substr(2, 9).toUpperCase()}`,
      status: 'configuring',
      botToken: newWorkspace.botToken,
      userToken: newWorkspace.userToken,
      channels: 0,
      members: 0,
      lastSync: new Date().toISOString(),
      createdAt: new Date().toISOString(),
      permissions: ['chat:write', 'channels:read'],
      notifications: newWorkspace.notifications
    };

    setWorkspaces([...workspaces, workspace]);
    setNewWorkspace({
      name: '',
      domain: '',
      botToken: '',
      userToken: '',
      notifications: {
        alerts: true,
        compliance: true,
        incidents: true,
        reports: false
      }
    });
    setIsAddingWorkspace(false);
  };

  const handleSyncWorkspace = (workspaceId: string) => {
    setWorkspaces(workspaces.map(ws => 
      ws.id === workspaceId 
        ? { ...ws, status: 'connected', lastSync: new Date().toISOString() }
        : ws
    ));
  };

  const connectedWorkspaces = workspaces.filter(ws => ws.status === 'connected').length;
  const totalChannels = workspaces.reduce((acc, ws) => acc + ws.channels, 0);
  const totalMembers = workspaces.reduce((acc, ws) => acc + ws.members, 0);
  const recentNotifications = notifications.filter(n => 
    new Date(n.timestamp) > new Date(Date.now() - 24 * 60 * 60 * 1000)
  ).length;

  const content = (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Connected Workspaces</p>
                <p className="text-2xl font-bold text-white">{connectedWorkspaces}</p>
              </div>
              <MessageCircle className="w-8 h-8 text-green-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-green-500">
              <CheckCircle className="w-3 h-3 mr-1" />
              <span>{connectedWorkspaces} of {workspaces.length} active</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Channels</p>
                <p className="text-2xl font-bold text-white">{totalChannels}</p>
              </div>
              <Hash className="w-8 h-8 text-blue-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-blue-500">
              <Activity className="w-3 h-3 mr-1" />
              <span>Across all workspaces</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Team Members</p>
                <p className="text-2xl font-bold text-white">{totalMembers}</p>
              </div>
              <Users className="w-8 h-8 text-purple-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-purple-500">
              <User className="w-3 h-3 mr-1" />
              <span>Active users</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Notifications (24h)</p>
                <p className="text-2xl font-bold text-white">{recentNotifications}</p>
              </div>
              <Bell className="w-8 h-8 text-yellow-500" />
            </div>
            <div className="mt-4 flex items-center text-xs text-yellow-500">
              <Clock className="w-3 h-3 mr-1" />
              <span>Last 24 hours</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-gray-900 border border-gray-800">
          <TabsTrigger value="overview" className="data-[state=active]:bg-black">Overview</TabsTrigger>
          <TabsTrigger value="workspaces" className="data-[state=active]:bg-black">Workspaces</TabsTrigger>
          <TabsTrigger value="channels" className="data-[state=active]:bg-black">Channels</TabsTrigger>
          <TabsTrigger value="notifications" className="data-[state=active]:bg-black">Notifications</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Quick Actions */}
          <Card className="bg-black border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Quick Actions</CardTitle>
              <CardDescription>Common Slack integration tasks</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Button
                  onClick={() => setIsAddingWorkspace(true)}
                  className="bg-green-600 hover:bg-green-700 text-white justify-start"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Add Workspace
                </Button>
                <Button
                  onClick={() => setIsConfiguring(true)}
                  className="bg-blue-600 hover:bg-blue-700 text-white justify-start"
                >
                  <Settings className="w-4 h-4 mr-2" />
                  Configure Notifications
                </Button>
                <Button
                  variant="outline"
                  className="border-gray-700 hover:bg-gray-800 justify-start"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Sync All
                </Button>
                <Button
                  variant="outline"
                  className="border-gray-700 hover:bg-gray-800 justify-start"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export Config
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Recent Activity */}
          <Card className="bg-black border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Recent Activity</CardTitle>
              <CardDescription>Latest Slack notifications and events</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {notifications.slice(0, 5).map((notification) => (
                  <div key={notification.id} className="flex items-start space-x-4 p-4 bg-gray-900 rounded-lg">
                    <div className="flex-shrink-0">
                      {notification.type === 'alert' && <AlertTriangle className="w-5 h-5 text-red-500" />}
                      {notification.type === 'compliance' && <Shield className="w-5 h-5 text-blue-500" />}
                      {notification.type === 'incident' && <Zap className="w-5 h-5 text-yellow-500" />}
                      {notification.type === 'report' && <Activity className="w-5 h-5 text-green-500" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-medium text-white">{notification.title}</p>
                        <Badge className={getStatusColor(notification.status)}>
                          {getStatusIcon(notification.status)}
                          <span className="ml-1">{notification.status}</span>
                        </Badge>
                      </div>
                      <p className="text-sm text-gray-400 mt-1">{notification.message}</p>
                      <p className="text-xs text-gray-500 mt-2">
                        #{notification.channel} • {notification.workspace} • {new Date(notification.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="workspaces" className="space-y-6">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div className="flex-1 max-w-sm">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search workspaces..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 bg-gray-900 border-gray-700 text-white"
                />
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Button
                onClick={() => setIsAddingWorkspace(true)}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add Workspace
              </Button>
              <Button variant="outline" className="border-gray-700 hover:bg-gray-800">
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {filteredWorkspaces.map((workspace) => (
              <Card key={workspace.id} className="bg-black border-gray-800">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <MessageCircle className="w-8 h-8 text-blue-500" />
                      <div>
                        <h3 className="font-semibold text-white">{workspace.name}</h3>
                        <p className="text-sm text-gray-400">{workspace.domain}.slack.com</p>
                      </div>
                    </div>
                    <Badge className={getStatusColor(workspace.status)}>
                      {getStatusIcon(workspace.status)}
                      <span className="ml-1 capitalize">{workspace.status}</span>
                    </Badge>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="text-center p-3 bg-gray-900 rounded-lg">
                      <p className="text-2xl font-bold text-white">{workspace.channels}</p>
                      <p className="text-xs text-gray-400">Channels</p>
                    </div>
                    <div className="text-center p-3 bg-gray-900 rounded-lg">
                      <p className="text-2xl font-bold text-white">{workspace.members}</p>
                      <p className="text-xs text-gray-400">Members</p>
                    </div>
                  </div>

                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Team ID:</span>
                      <span className="text-white font-mono">{workspace.teamId}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Last Sync:</span>
                      <span className="text-white">{new Date(workspace.lastSync).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Permissions:</span>
                      <span className="text-white">{workspace.permissions.length} granted</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-gray-800">
                    <div className="flex items-center space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleSyncWorkspace(workspace.id)}
                        className="border-gray-700 hover:bg-gray-800"
                      >
                        <RefreshCw className="w-3 h-3 mr-1" />
                        Sync
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        className="border-gray-700 hover:bg-gray-800"
                      >
                        <Settings className="w-3 h-3 mr-1" />
                        Configure
                      </Button>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-gray-400 hover:text-white"
                    >
                      <MoreVertical className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="channels" className="space-y-6">
          <Card className="bg-black border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Configured Channels</CardTitle>
              <CardDescription>Channels receiving PolicyCortex notifications</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {channels.map((channel) => (
                  <div key={channel.id} className="flex items-center justify-between p-4 bg-gray-900 rounded-lg">
                    <div className="flex items-center space-x-4">
                      <Hash className="w-5 h-5 text-gray-400" />
                      <div>
                        <div className="flex items-center space-x-2">
                          <p className="font-medium text-white">{channel.name}</p>
                          {channel.isPrivate && (
                            <Badge variant="secondary" className="bg-yellow-500/20 text-yellow-400">
                              Private
                            </Badge>
                          )}
                        </div>
                        <p className="text-sm text-gray-400">{channel.purpose}</p>
                        <div className="flex items-center space-x-4 mt-1 text-xs text-gray-500">
                          <span>{channel.members} members</span>
                          <span>{channel.messageCount} messages</span>
                          <span>Last active: {new Date(channel.lastActivity).toLocaleString()}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={channel.notifications}
                        className="data-[state=checked]:bg-green-600"
                      />
                      <Button variant="ghost" size="sm" className="text-gray-400 hover:text-white">
                        <Settings className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-6">
          <Card className="bg-black border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Notification History</CardTitle>
              <CardDescription>Recent notifications sent to Slack</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {notifications.map((notification) => (
                  <div key={notification.id} className="flex items-start space-x-4 p-4 bg-gray-900 rounded-lg">
                    <div className="flex-shrink-0 mt-1">
                      {notification.type === 'alert' && <AlertTriangle className="w-5 h-5 text-red-500" />}
                      {notification.type === 'compliance' && <Shield className="w-5 h-5 text-blue-500" />}
                      {notification.type === 'incident' && <Zap className="w-5 h-5 text-yellow-500" />}
                      {notification.type === 'report' && <Activity className="w-5 h-5 text-green-500" />}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-white">{notification.title}</h4>
                        <Badge className={getStatusColor(notification.status)}>
                          {getStatusIcon(notification.status)}
                          <span className="ml-1 capitalize">{notification.status}</span>
                        </Badge>
                      </div>
                      <p className="text-sm text-gray-400 mb-2">{notification.message}</p>
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <div className="flex items-center space-x-4">
                          <span>#{notification.channel}</span>
                          <span>{notification.workspace}</span>
                          <span>{new Date(notification.timestamp).toLocaleString()}</span>
                        </div>
                        {notification.retries > 0 && (
                          <span className="text-yellow-500">{notification.retries} retries</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Add Workspace Dialog */}
      <Dialog open={isAddingWorkspace} onOpenChange={setIsAddingWorkspace}>
        <DialogContent className="bg-gray-900 border-gray-800 text-white max-w-2xl">
          <DialogHeader>
            <DialogTitle>Add Slack Workspace</DialogTitle>
            <DialogDescription>
              Connect a new Slack workspace to PolicyCortex
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-6 py-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="workspace-name" className="text-sm font-medium">
                  Workspace Name
                </Label>
                <Input
                  id="workspace-name"
                  value={newWorkspace.name}
                  onChange={(e) => setNewWorkspace({...newWorkspace, name: e.target.value})}
                  placeholder="PolicyCortex Team"
                  className="bg-black border-gray-700 mt-1"
                />
              </div>
              <div>
                <Label htmlFor="workspace-domain" className="text-sm font-medium">
                  Domain
                </Label>
                <Input
                  id="workspace-domain"
                  value={newWorkspace.domain}
                  onChange={(e) => setNewWorkspace({...newWorkspace, domain: e.target.value})}
                  placeholder="policycortex-team"
                  className="bg-black border-gray-700 mt-1"
                />
              </div>
            </div>

            <div>
              <Label htmlFor="bot-token" className="text-sm font-medium">
                Bot User OAuth Token
              </Label>
              <Input
                id="bot-token"
                type="password"
                value={newWorkspace.botToken}
                onChange={(e) => setNewWorkspace({...newWorkspace, botToken: e.target.value})}
                placeholder="xoxb-your-bot-token"
                className="bg-black border-gray-700 mt-1"
              />
              <p className="text-xs text-gray-400 mt-1">
                Required for sending messages and accessing public channels
              </p>
            </div>

            <div>
              <Label htmlFor="user-token" className="text-sm font-medium">
                User OAuth Token (Optional)
              </Label>
              <Input
                id="user-token"
                type="password"
                value={newWorkspace.userToken}
                onChange={(e) => setNewWorkspace({...newWorkspace, userToken: e.target.value})}
                placeholder="xoxp-your-user-token"
                className="bg-black border-gray-700 mt-1"
              />
              <p className="text-xs text-gray-400 mt-1">
                Optional: For accessing private channels and enhanced features
              </p>
            </div>

            <div>
              <Label className="text-sm font-medium mb-3 block">
                Notification Types
              </Label>
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center space-x-2">
                  <Switch
                    id="alerts"
                    checked={newWorkspace.notifications.alerts}
                    onCheckedChange={(checked) => setNewWorkspace({
                      ...newWorkspace,
                      notifications: { ...newWorkspace.notifications, alerts: checked }
                    })}
                    className="data-[state=checked]:bg-green-600"
                  />
                  <Label htmlFor="alerts" className="text-sm">System Alerts</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="compliance"
                    checked={newWorkspace.notifications.compliance}
                    onCheckedChange={(checked) => setNewWorkspace({
                      ...newWorkspace,
                      notifications: { ...newWorkspace.notifications, compliance: checked }
                    })}
                    className="data-[state=checked]:bg-green-600"
                  />
                  <Label htmlFor="compliance" className="text-sm">Compliance Updates</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="incidents"
                    checked={newWorkspace.notifications.incidents}
                    onCheckedChange={(checked) => setNewWorkspace({
                      ...newWorkspace,
                      notifications: { ...newWorkspace.notifications, incidents: checked }
                    })}
                    className="data-[state=checked]:bg-green-600"
                  />
                  <Label htmlFor="incidents" className="text-sm">Incidents</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="reports"
                    checked={newWorkspace.notifications.reports}
                    onCheckedChange={(checked) => setNewWorkspace({
                      ...newWorkspace,
                      notifications: { ...newWorkspace.notifications, reports: checked }
                    })}
                    className="data-[state=checked]:bg-green-600"
                  />
                  <Label htmlFor="reports" className="text-sm">Reports</Label>
                </div>
              </div>
            </div>
          </div>
          <div className="flex justify-end space-x-3">
            <Button
              variant="outline"
              onClick={() => setIsAddingWorkspace(false)}
              className="border-gray-700 hover:bg-gray-800"
            >
              Cancel
            </Button>
            <Button
              onClick={handleAddWorkspace}
              disabled={!newWorkspace.name || !newWorkspace.domain || !newWorkspace.botToken}
              className="bg-green-600 hover:bg-green-700"
            >
              Add Workspace
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );

  return (
    <TacticalPageTemplate 
      title="Slack Integration" 
      subtitle="Slack Integration Operations Center" 
      icon={MessageCircle}
    >
      {content}
    </TacticalPageTemplate>
  );
}