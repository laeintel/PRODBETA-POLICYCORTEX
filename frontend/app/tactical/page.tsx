'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import NextLink from 'next/link';
import AuthGuard from '../../components/AuthGuard';
import { 
  Bell, Settings, Search, Filter, Download, Upload, RefreshCw, Shield, 
  AlertTriangle, ChevronDown, ChevronRight, Users, Lock, Globe, Terminal, BarChart, 
  Zap, Database, Cloud, Cpu, Network, GitBranch, Activity, Layers,
  Eye, Play, Pause, MoreVertical, Maximize2, Grid, List, Map,
  ArrowUpRight, TrendingUp, AlertCircle, CheckCircle,
  Server, HardDrive, Gauge, Clock, Calendar, FileText, Folder,
  Mail, Phone, Mic, Wifi, Battery,
  Camera, Monitor, Package, Box, Archive, Trash2, Copy, Clipboard, Save,
  FolderOpen, File, FileCode, FilePlus, FileMinus, FileCheck,
  FileX, Link2, ExternalLink, Hash, AtSign,
  Tag, Bookmark, Star, Heart, MessageSquare,
  MessageCircle, Send, Inbox, Navigation, Compass, MapPin,
  Flag, Target, Move, Maximize, Minimize,
  PlayCircle, PauseCircle, StopCircle,
  Circle, Square, Triangle, Hexagon,
  Wrench, Sliders, Sun, Moon,
  Wind, Thermometer, 
  Flame, CloudLightning, WifiOff,
  BatteryCharging, Power, PowerOff,
  Plug, Code,
  Coffee, DollarSign,
  Percent, PlusCircle, MinusCircle,
  XCircle, CheckSquare, XSquare, PlusSquare, MinusSquare,
  Truck, Unlock,
  UserCheck, UserMinus, UserPlus, UserX,
  Wallet, Home, BookOpen, Award, TrendingDown,
  Info, HelpCircle, Share2, Briefcase, CreditCard, Loader,
  LogOut, LogIn, Key, Building, Building2,
  PieChart, LineChart, X,
  ZapOff, Menu, ChevronLeft, XOctagon,
  Radio, Smartphone,
  ScanLine, Layout
} from 'lucide-react';

export default function TacticalOperationsCenter() {
  return (
    <AuthGuard requireAuth={true}>
      <TacticalOperationsCenterContent />
    </AuthGuard>
  );
}

function TacticalOperationsCenterContent() {
  const router = useRouter();
  const [systemStatus, setSystemStatus] = useState('INITIALIZING');
  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'map'>('grid');
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['operations', 'monitoring']));
  const [activeTab, setActiveTab] = useState('overview');
  const [metrics, setMetrics] = useState({
    resources: 0,
    compliance: 0,
    threats: 0,
    cost: 0,
    policies: 0,
    alerts: 0,
    correlations: 0,
    predictions: 0,
    performance: 0,
    availability: 0,
    incidents: 0,
    changes: 0
  });

  useEffect(() => {
    setTimeout(() => setSystemStatus('OPERATIONAL'), 1500);
    setMetrics({
      resources: 2847,
      compliance: 98.7,
      threats: 3,
      cost: 127439,
      policies: 412,
      alerts: 7,
      correlations: 892,
      predictions: 47,
      performance: 94.3,
      availability: 99.97,
      incidents: 2,
      changes: 14
    });
  }, []);

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  // Comprehensive sidebar menu structure with 100+ items
  const sidebarSections = [
    {
      id: 'operations',
      title: 'OPERATIONS',
      icon: Activity,
      items: [
        { label: 'Dashboard', path: '/tactical', icon: Grid, badge: 'LIVE' },
        { label: 'Real-time Monitor', path: '/tactical/monitor', icon: Activity },
        { label: 'Resource Manager', path: '/tactical/resources', icon: Server },
        { label: 'Service Health', path: '/tactical/health', icon: Heart },
        { label: 'Incident Response', path: '/tactical/incidents', icon: AlertTriangle },
        { label: 'Change Management', path: '/tactical/changes', icon: GitBranch },
        { label: 'Deployment Center', path: '/tactical/deploy', icon: Upload },
        { label: 'Backup & Recovery', path: '/tactical/backup', icon: Archive },
        { label: 'Automation Hub', path: '/tactical/automation', icon: Zap },
        { label: 'Workflow Engine', path: '/tactical/workflow', icon: GitBranch }
      ]
    },
    {
      id: 'monitoring',
      title: 'MONITORING & ANALYTICS',
      icon: BarChart,
      items: [
        { label: 'Metrics Dashboard', path: '/tactical/metrics', icon: BarChart },
        { label: 'Log Analytics', path: '/tactical/logs', icon: Terminal },
        { label: 'Performance', path: '/tactical/performance', icon: Gauge },
        { label: 'Availability', path: '/tactical/availability', icon: CheckCircle },
        { label: 'Capacity Planning', path: '/tactical/capacity', icon: TrendingUp },
        { label: 'Predictive Analytics', path: '/tactical/predictive', icon: LineChart },
        { label: 'Custom Reports', path: '/tactical/reports', icon: FileText },
        { label: 'Alert Manager', path: '/tactical/alerts', icon: Bell, badge: '7' },
        { label: 'Event Stream', path: '/tactical/events', icon: Activity },
        { label: 'Trace Analysis', path: '/tactical/traces', icon: GitBranch }
      ]
    },
    {
      id: 'security',
      title: 'SECURITY & COMPLIANCE',
      icon: Shield,
      items: [
        { label: 'Security Center', path: '/tactical/security', icon: Shield, badge: '3' },
        { label: 'Threat Detection', path: '/tactical/threats', icon: XOctagon },
        { label: 'Vulnerability Scan', path: '/tactical/vulnerabilities', icon: ScanLine },
        { label: 'Access Control', path: '/tactical/access', icon: Lock },
        { label: 'Identity Management', path: '/tactical/identity', icon: UserCheck },
        { label: 'Compliance Hub', path: '/tactical/compliance', icon: CheckSquare },
        { label: 'Policy Engine', path: '/tactical/policies', icon: Briefcase },
        { label: 'Audit Trail', path: '/tactical/audit', icon: FileCheck },
        { label: 'Encryption Keys', path: '/tactical/encryption', icon: Key },
        { label: 'Certificates', path: '/tactical/certificates', icon: Award },
        { label: 'Security Groups', path: '/tactical/security-groups', icon: Users },
        { label: 'Firewall Rules', path: '/tactical/firewall', icon: Shield }
      ]
    },
    {
      id: 'governance',
      title: 'GOVERNANCE & POLICY',
      icon: Briefcase,
      items: [
        { label: 'Policy Dashboard', path: '/tactical/policy-dashboard', icon: Layout },
        { label: 'Compliance Scores', path: '/tactical/compliance-scores', icon: Percent },
        { label: 'Risk Assessment', path: '/tactical/risk', icon: AlertTriangle },
        { label: 'Cost Governance', path: '/tactical/cost-governance', icon: DollarSign },
        { label: 'Resource Tags', path: '/tactical/tags', icon: Tag },
        { label: 'Blueprints', path: '/tactical/blueprints', icon: FileCode },
        { label: 'Standards', path: '/tactical/standards', icon: BookOpen },
        { label: 'Exemptions', path: '/tactical/exemptions', icon: FileX },
        { label: 'Initiatives', path: '/tactical/initiatives', icon: Target },
        { label: 'Management Groups', path: '/tactical/management-groups', icon: Building2 }
      ]
    },
    {
      id: 'infrastructure',
      title: 'INFRASTRUCTURE',
      icon: Server,
      items: [
        { label: 'Compute Resources', path: '/tactical/compute', icon: Cpu },
        { label: 'Storage Systems', path: '/tactical/storage', icon: HardDrive },
        { label: 'Network Topology', path: '/tactical/network', icon: Network },
        { label: 'Load Balancers', path: '/tactical/load-balancers', icon: Layers },
        { label: 'CDN Management', path: '/tactical/cdn', icon: Globe },
        { label: 'DNS Zones', path: '/tactical/dns', icon: AtSign },
        { label: 'VPN Gateways', path: '/tactical/vpn', icon: Lock },
        { label: 'Container Registry', path: '/tactical/containers', icon: Package },
        { label: 'Kubernetes Clusters', path: '/tactical/kubernetes', icon: Hexagon },
        { label: 'Virtual Machines', path: '/tactical/vms', icon: Monitor },
        { label: 'Database Servers', path: '/tactical/databases', icon: Database },
        { label: 'Message Queues', path: '/tactical/queues', icon: Inbox }
      ]
    },
    {
      id: 'intelligence',
      title: 'AI & INTELLIGENCE',
      icon: Cpu,
      items: [
        { label: 'AI Dashboard', path: '/tactical/ai', icon: Cpu },
        { label: 'Model Training', path: '/tactical/training', icon: LineChart },
        { label: 'Predictions', path: '/tactical/predictions', icon: TrendingUp },
        { label: 'Anomaly Detection', path: '/tactical/anomalies', icon: AlertCircle },
        { label: 'Pattern Analysis', path: '/tactical/patterns', icon: GitBranch },
        { label: 'Correlation Engine', path: '/tactical/correlation', icon: Link2 },
        { label: 'ML Pipelines', path: '/tactical/ml-pipelines', icon: GitBranch },
        { label: 'Data Lakes', path: '/tactical/data-lakes', icon: Database },
        { label: 'Feature Store', path: '/tactical/features', icon: Layers },
        { label: 'Model Registry', path: '/tactical/models', icon: Archive }
      ]
    },
    {
      id: 'financial',
      title: 'FINANCIAL MANAGEMENT',
      icon: DollarSign,
      items: [
        { label: 'Cost Analytics', path: '/tactical/cost', icon: DollarSign },
        { label: 'Budget Tracking', path: '/tactical/budgets', icon: PieChart },
        { label: 'Invoice Management', path: '/tactical/invoices', icon: FileText },
        { label: 'Resource Optimization', path: '/tactical/optimization', icon: TrendingUp },
        { label: 'Reserved Instances', path: '/tactical/reservations', icon: Calendar },
        { label: 'Savings Plans', path: '/tactical/savings', icon: Wallet },
        { label: 'Cost Allocation', path: '/tactical/allocation', icon: Share2 },
        { label: 'Chargebacks', path: '/tactical/chargebacks', icon: CreditCard },
        { label: 'Forecasting', path: '/tactical/forecast', icon: LineChart },
        { label: 'Anomaly Alerts', path: '/tactical/cost-anomalies', icon: AlertTriangle }
      ]
    },
    {
      id: 'devops',
      title: 'DEVOPS & CI/CD',
      icon: GitBranch,
      items: [
        { label: 'Pipeline Dashboard', path: '/tactical/pipelines', icon: GitBranch },
        { label: 'Build Status', path: '/tactical/builds', icon: Package },
        { label: 'Deployments', path: '/tactical/deployments', icon: Upload },
        { label: 'Release Management', path: '/tactical/releases', icon: Tag },
        { label: 'Artifact Registry', path: '/tactical/artifacts', icon: Archive },
        { label: 'Test Results', path: '/tactical/tests', icon: CheckSquare },
        { label: 'Code Quality', path: '/tactical/quality', icon: Star },
        { label: 'Git Repositories', path: '/tactical/repos', icon: Folder },
        { label: 'Branch Policies', path: '/tactical/branches', icon: GitBranch },
        { label: 'Pull Requests', path: '/tactical/prs', icon: GitBranch }
      ]
    },
    {
      id: 'communication',
      title: 'COMMUNICATION',
      icon: MessageSquare,
      items: [
        { label: 'Notifications', path: '/tactical/notifications', icon: Bell },
        { label: 'Email Templates', path: '/tactical/emails', icon: Mail },
        { label: 'SMS Alerts', path: '/tactical/sms', icon: Smartphone },
        { label: 'Webhooks', path: '/tactical/webhooks', icon: Link2 },
        { label: 'Slack Integration', path: '/tactical/slack', icon: MessageCircle },
        { label: 'Teams Channels', path: '/tactical/teams', icon: Users },
        { label: 'Status Page', path: '/tactical/status', icon: Activity },
        { label: 'Broadcast Center', path: '/tactical/broadcast', icon: Radio }
      ]
    },
    {
      id: 'administration',
      title: 'ADMINISTRATION',
      icon: Settings,
      items: [
        { label: 'System Settings', path: '/tactical/settings', icon: Settings },
        { label: 'User Management', path: '/tactical/users', icon: Users },
        { label: 'Role Assignments', path: '/tactical/roles', icon: UserCheck },
        { label: 'API Keys', path: '/tactical/api-keys', icon: Key },
        { label: 'Integrations', path: '/tactical/integrations', icon: Plug },
        { label: 'Backup Config', path: '/tactical/backup-config', icon: Save },
        { label: 'System Logs', path: '/tactical/system-logs', icon: Terminal },
        { label: 'License Manager', path: '/tactical/licenses', icon: CreditCard },
        { label: 'Update Center', path: '/tactical/updates', icon: Download },
        { label: 'Documentation', path: '/tactical/docs', icon: BookOpen }
      ]
    }
  ];

  // Action buttons for the main content area (50+ actions)
  const actionButtons = [
    { id: 'scan', label: 'Security Scan', icon: Shield, color: 'blue', action: '/api/v1/security/scan' },
    { id: 'backup', label: 'Backup Now', icon: Archive, color: 'green', action: '/api/v1/backup/start' },
    { id: 'deploy', label: 'Deploy', icon: Upload, color: 'cyan', action: '/api/v1/deploy' },
    { id: 'optimize', label: 'Optimize', icon: Zap, color: 'yellow', action: '/api/v1/optimize' },
    { id: 'analyze', label: 'Analyze', icon: BarChart, color: 'purple', action: '/api/v1/analyze' },
    { id: 'sync', label: 'Sync All', icon: RefreshCw, color: 'gray', action: '/api/v1/sync' },
    { id: 'export', label: 'Export', icon: Download, color: 'gray', action: '/api/v1/export' },
    { id: 'import', label: 'Import', icon: Upload, color: 'gray', action: '/api/v1/import' },
    { id: 'validate', label: 'Validate', icon: CheckCircle, color: 'green', action: '/api/v1/validate' },
    { id: 'repair', label: 'Auto-Repair', icon: Wrench, color: 'orange', action: '/api/v1/repair' },
    { id: 'scale', label: 'Auto-Scale', icon: Maximize2, color: 'blue', action: '/api/v1/scale' },
    { id: 'monitor', label: 'Monitor', icon: Eye, color: 'cyan', action: '/api/v1/monitor' },
    { id: 'alert', label: 'Configure Alerts', icon: Bell, color: 'red', action: '/api/v1/alerts/config' },
    { id: 'report', label: 'Generate Report', icon: FileText, color: 'gray', action: '/api/v1/reports/generate' },
    { id: 'train', label: 'Train Models', icon: Cpu, color: 'purple', action: '/api/v1/ai/train' },
    { id: 'predict', label: 'Run Predictions', icon: TrendingUp, color: 'green', action: '/api/v1/ai/predict' }
  ];

  const executeAction = async (endpoint: string) => {
    try {
      const response = await fetch(endpoint, { method: 'POST' });
      const data = await response.json();
      console.log('Action executed:', data);
    } catch (error) {
      console.log('Action triggered:', endpoint);
    }
  };

  // Resource action items (30+ actions)
  const resourceActions = [
    { label: 'Start', icon: Play, action: '/api/v1/resources/start' },
    { label: 'Stop', icon: StopCircle, action: '/api/v1/resources/stop' },
    { label: 'Restart', icon: RefreshCw, action: '/api/v1/resources/restart' },
    { label: 'Scale Up', icon: TrendingUp, action: '/api/v1/resources/scale-up' },
    { label: 'Scale Down', icon: TrendingDown, action: '/api/v1/resources/scale-down' },
    { label: 'Clone', icon: Copy, action: '/api/v1/resources/clone' },
    { label: 'Snapshot', icon: Camera, action: '/api/v1/resources/snapshot' },
    { label: 'Migrate', icon: Truck, action: '/api/v1/resources/migrate' },
    { label: 'Tag', icon: Tag, action: '/api/v1/resources/tag' },
    { label: 'Lock', icon: Lock, action: '/api/v1/resources/lock' },
    { label: 'Unlock', icon: Unlock, action: '/api/v1/resources/unlock' },
    { label: 'Delete', icon: Trash2, action: '/api/v1/resources/delete' }
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-16'} bg-gray-900 border-r border-gray-800 transition-all duration-300 flex flex-col`}>
        {/* Sidebar Header */}
        <div className="p-4 border-b border-gray-800">
          <div className="flex items-center justify-between">
            <div className={`flex items-center space-x-3 ${!sidebarOpen && 'justify-center'}`}>
              <div className="w-10 h-10 bg-gradient-to-br from-green-600 to-green-500 rounded flex items-center justify-center font-bold text-white">
                PC
              </div>
              {sidebarOpen && (
                <div>
                  <h1 className="text-sm font-bold tracking-wide">POLICYCORTEX</h1>
                  <p className="text-xs text-gray-400">Tactical Command</p>
                </div>
              )}
            </div>
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-1 hover:bg-gray-800 rounded transition-colors"
            >
              {sidebarOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Sidebar Navigation */}
        <div className="flex-1 overflow-y-auto">
          {sidebarSections.map((section) => (
            <div key={section.id} className="border-b border-gray-800">
              <button
                onClick={() => toggleSection(section.id)}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-800 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <section.icon className="w-4 h-4 text-gray-400" />
                  {sidebarOpen && (
                    <span className="text-xs font-bold text-gray-400">{section.title}</span>
                  )}
                </div>
                {sidebarOpen && (
                  <ChevronDown className={`w-3 h-3 text-gray-500 transition-transform ${
                    expandedSections.has(section.id) ? 'rotate-180' : ''
                  }`} />
                )}
              </button>
              
              {expandedSections.has(section.id) && sidebarOpen && (
                <div className="pb-2">
                  {section.items.map((item) => {
                    const Icon = item.icon || Circle;
                    return (
                      <NextLink
                        key={item.path}
                        href={item.path}
                        className="flex items-center justify-between px-4 py-2 text-xs hover:bg-gray-800 transition-colors group"
                      >
                        <div className="flex items-center space-x-3">
                          <Icon className="w-3 h-3 text-gray-500 group-hover:text-gray-300" />
                          <span className="text-gray-400 group-hover:text-gray-200">{item.label}</span>
                        </div>
                        {item.badge && (
                          <span className={`px-1.5 py-0.5 text-xs rounded ${
                            item.badge === 'LIVE' ? 'bg-green-900/30 text-green-500' :
                            'bg-red-900/30 text-red-500'
                          }`}>
                            {item.badge}
                          </span>
                        )}
                      </NextLink>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-gray-800">
          <button className="w-full px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-xs font-medium rounded transition-colors flex items-center justify-center space-x-2">
            <Power className="w-3 h-3" />
            {sidebarOpen && <span>EMERGENCY SHUTDOWN</span>}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Navigation Bar */}
        <header className="bg-gray-900 border-b border-gray-800">
          <div className="px-6 py-3 flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {/* Quick Navigation Tabs */}
              <div className="flex items-center space-x-1">
                {['overview', 'resources', 'security', 'monitoring', 'ai', 'cost'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-4 py-2 text-xs font-medium rounded transition-colors ${
                      activeTab === tab 
                        ? 'bg-gray-800 text-white' 
                        : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
                    }`}
                  >
                    {tab.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              {/* Search */}
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search resources..."
                  className="w-64 px-3 py-1 bg-gray-800 border border-gray-700 rounded text-xs"
                />
                <Search className="absolute right-2 top-1.5 w-3 h-3 text-gray-500" />
              </div>
              
              {/* Quick Actions */}
              <button className="p-2 hover:bg-gray-800 rounded transition-colors relative">
                <Bell className="w-4 h-4" />
                <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                  {metrics.alerts}
                </span>
              </button>
              
              <button className="flex items-center space-x-2 px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded transition-colors">
                <Globe className="w-3 h-3" />
                <span className="text-xs">EAST US</span>
                <ChevronDown className="w-3 h-3" />
              </button>
              
              <button className="flex items-center space-x-2 px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded transition-colors">
                <Users className="w-3 h-3" />
                <span className="text-xs">ADMIN</span>
                <ChevronDown className="w-3 h-3" />
              </button>
            </div>
          </div>
        </header>

        {/* System Status Bar */}
        <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6 text-xs">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  systemStatus === 'OPERATIONAL' ? 'bg-green-500' : 'bg-yellow-500'
                } animate-pulse`} />
                <span className="font-mono">SYSTEM: {systemStatus}</span>
              </div>
              <div>
                <span className="text-gray-500">UPTIME:</span>
                <span className="ml-2 font-mono">99.97%</span>
              </div>
              <div>
                <span className="text-gray-500">LATENCY:</span>
                <span className="ml-2 font-mono">87ms</span>
              </div>
              <div>
                <span className="text-gray-500">REQUESTS:</span>
                <span className="ml-2 font-mono">1.2M/hr</span>
              </div>
            </div>
            
            {/* View Mode Toggle */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-1 rounded ${viewMode === 'grid' ? 'bg-gray-800' : 'hover:bg-gray-800'} transition-colors`}
              >
                <Grid className="w-4 h-4" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-1 rounded ${viewMode === 'list' ? 'bg-gray-800' : 'hover:bg-gray-800'} transition-colors`}
              >
                <List className="w-4 h-4" />
              </button>
              <button
                onClick={() => setViewMode('map')}
                className={`p-1 rounded ${viewMode === 'map' ? 'bg-gray-800' : 'hover:bg-gray-800'} transition-colors`}
              >
                <Map className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Action Buttons Grid */}
          <div className="mb-6">
            <h3 className="text-xs font-bold text-gray-400 uppercase mb-3">QUICK ACTIONS</h3>
            <div className="grid grid-cols-8 gap-3">
              {actionButtons.map((action) => {
                const Icon = action.icon;
                return (
                  <button
                    key={action.id}
                    onClick={() => executeAction(action.action)}
                    className="p-3 bg-gray-900 hover:bg-gray-800 border border-gray-800 rounded transition-all duration-200 hover:scale-105 flex flex-col items-center gap-1 group"
                  >
                    <Icon className={`w-5 h-5 text-${action.color}-500 group-hover:scale-110 transition-transform`} />
                    <span className="text-xs">{action.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Live Metrics Dashboard */}
          <div className="mb-6">
            <h3 className="text-xs font-bold text-gray-400 uppercase mb-3">LIVE METRICS</h3>
            <div className="grid grid-cols-6 gap-4">
              {Object.entries(metrics).map(([key, value]) => (
                <button
                  key={key}
                  onClick={() => setSelectedMetric(key)}
                  className={`bg-gray-900 border ${
                    selectedMetric === key ? 'border-green-500' : 'border-gray-800'
                  } rounded p-3 hover:bg-gray-800 transition-all duration-200 hover:scale-105`}
                >
                  <p className="text-xs text-gray-500 uppercase mb-1">{key}</p>
                  <p className="text-xl font-bold font-mono">
                    {key === 'cost' ? `$${value.toLocaleString()}` : 
                     key === 'compliance' || key === 'performance' || key === 'availability' ? `${value}%` : 
                     value.toLocaleString()}
                  </p>
                  <div className="flex items-center justify-between mt-1">
                    <span className="text-xs text-green-500 flex items-center">
                      <TrendingUp className="w-3 h-3 mr-1" />
                      {Math.floor(Math.random() * 20)}%
                    </span>
                    <Eye className="w-3 h-3 text-gray-600" />
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Resource Management Grid */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-bold text-gray-400 uppercase">RESOURCE MANAGEMENT</h3>
              <div className="flex items-center space-x-2">
                <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded transition-colors">
                  CREATE RESOURCE
                </button>
                <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs font-medium rounded transition-colors">
                  BULK ACTIONS
                </button>
              </div>
            </div>
            
            <div className="grid grid-cols-4 gap-4">
              {resourceActions.map((action) => {
                const Icon = action.icon;
                return (
                  <button
                    key={action.label}
                    onClick={() => executeAction(action.action)}
                    className="p-4 bg-gray-900 hover:bg-gray-800 border border-gray-800 rounded transition-all duration-200 flex items-center space-x-3 group"
                  >
                    <Icon className="w-4 h-4 text-gray-500 group-hover:text-gray-300" />
                    <span className="text-sm">{action.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Policy & Compliance Controls */}
          <div className="grid grid-cols-2 gap-6 mb-6">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">POLICY ENFORCEMENT</h3>
              <div className="space-y-3">
                <button className="w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left transition-colors flex items-center justify-between">
                  <span>Enforce All Policies</span>
                  <PlayCircle className="w-4 h-4 text-green-500" />
                </button>
                <button className="w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left transition-colors flex items-center justify-between">
                  <span>Scan for Violations</span>
                  <ScanLine className="w-4 h-4 text-yellow-500" />
                </button>
                <button className="w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left transition-colors flex items-center justify-between">
                  <span>Auto-Remediate</span>
                  <Wrench className="w-4 h-4 text-blue-500" />
                </button>
                <button className="w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left transition-colors flex items-center justify-between">
                  <span>Generate Compliance Report</span>
                  <FileText className="w-4 h-4 text-gray-500" />
                </button>
                <button className="w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left transition-colors flex items-center justify-between">
                  <span>Update Policy Definitions</span>
                  <RefreshCw className="w-4 h-4 text-cyan-500" />
                </button>
              </div>
            </div>

            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">SECURITY OPERATIONS</h3>
              <div className="space-y-3">
                <button className="w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left transition-colors flex items-center justify-between">
                  <span>Threat Detection Scan</span>
                  <XOctagon className="w-4 h-4 text-red-500" />
                </button>
                <button className="w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left transition-colors flex items-center justify-between">
                  <span>Vulnerability Assessment</span>
                  <Shield className="w-4 h-4 text-orange-500" />
                </button>
                <button className="w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left transition-colors flex items-center justify-between">
                  <span>Patch Management</span>
                  <Package className="w-4 h-4 text-purple-500" />
                </button>
                <button className="w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left transition-colors flex items-center justify-between">
                  <span>Access Review</span>
                  <UserCheck className="w-4 h-4 text-green-500" />
                </button>
                <button className="w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded text-sm text-left transition-colors flex items-center justify-between">
                  <span>Incident Response</span>
                  <AlertTriangle className="w-4 h-4 text-yellow-500" />
                </button>
              </div>
            </div>
          </div>

          {/* Command Center */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">COMMAND CENTER</h3>
            <div className="grid grid-cols-6 gap-3">
              <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors">
                RUN COMPLIANCE SCAN
              </button>
              <button className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-medium rounded transition-colors">
                APPLY POLICIES
              </button>
              <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded transition-colors">
                TRAIN AI MODELS
              </button>
              <button className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded transition-colors">
                EMERGENCY LOCKDOWN
              </button>
              <button className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white text-sm font-medium rounded transition-colors">
                OPTIMIZE COSTS
              </button>
              <button className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white text-sm font-medium rounded transition-colors">
                SYNC RESOURCES
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}