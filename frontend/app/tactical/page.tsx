'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  AlertTriangle, Shield, Activity, Zap, Terminal, Users,
  AlertCircle, CheckCircle, Clock, TrendingUp, Server,
  Database, Cloud, Lock, Bell, Play, Pause, RefreshCw,
  ChevronRight, Cpu, HardDrive, Wifi, Target, Radio,
  GitBranch, Settings, History, DollarSign, FileCheck,
  ShieldAlert, UserCheck, Key, Package, Brain, MessageSquare,
  BarChart3, Layers, Building, Workflow, Container, Rocket,
  Monitor, BellRing, Bot, TrendingDown, ExternalLink, 
  ArrowRight, ChevronDown, LayoutGrid, LineChart
} from 'lucide-react';
import ResponsiveGrid, { ResponsiveContainer, ResponsiveText } from '@/components/ResponsiveGrid';
import { toast } from '@/hooks/useToast';
import ViewToggle from '@/components/ViewToggle';
import ChartContainer from '@/components/ChartContainer';

interface NavigationCard {
  id: string;
  title: string;
  description: string;
  icon: any;
  href: string;
  stats?: {
    label: string;
    value: string | number;
    trend?: 'up' | 'down' | 'stable';
  }[];
  color: string;
  badge?: string;
  subItems?: {
    label: string;
    href: string;
    count?: number;
  }[];
}

interface Alert {
  id: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  time: string;
  source: string;
  status: 'active' | 'acknowledged' | 'resolved';
}

interface SystemMetric {
  name: string;
  value: number;
  unit: string;
  status: 'healthy' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  onClick?: () => void;
}

export default function TacticalOperationsPage() {
  const router = useRouter();
  const [viewMode, setViewMode] = useState<'cards' | 'visualizations'>('visualizations');
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [commandInput, setCommandInput] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [isWarRoomActive, setIsWarRoomActive] = useState(false);
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());

  useEffect(() => {
    // Simulate real-time alerts
    const mockAlerts: Alert[] = [
      {
        id: 'ALT-001',
        severity: 'critical',
        title: 'Database Connection Failure',
        description: 'Primary database cluster experiencing connectivity issues',
        time: '2 min ago',
        source: 'Azure SQL',
        status: 'active'
      },
      {
        id: 'ALT-002',
        severity: 'high',
        title: 'Unusual Login Activity Detected',
        description: 'Multiple failed login attempts from unknown IP addresses',
        time: '5 min ago',
        source: 'Azure AD',
        status: 'acknowledged'
      },
      {
        id: 'ALT-003',
        severity: 'medium',
        title: 'Cost Threshold Exceeded',
        description: 'Monthly spending has exceeded 80% of budget',
        time: '15 min ago',
        source: 'Cost Management',
        status: 'active'
      },
      {
        id: 'ALT-004',
        severity: 'low',
        title: 'Certificate Expiry Warning',
        description: 'SSL certificate expires in 30 days',
        time: '1 hour ago',
        source: 'Key Vault',
        status: 'resolved'
      }
    ];
    setAlerts(mockAlerts);
  }, []);

  // Main navigation cards with comprehensive routing
  const navigationCards: NavigationCard[] = [
    {
      id: 'governance',
      title: 'Governance & Compliance',
      description: 'Policies, compliance tracking, risk management, and cost optimization',
      icon: Shield,
      href: '/governance',
      color: 'blue',
      stats: [
        { label: 'Compliance', value: '94%', trend: 'up' },
        { label: 'Active Policies', value: 127 },
        { label: 'Risk Score', value: 'Low', trend: 'down' }
      ],
      subItems: [
        { label: 'Policies & Compliance', href: '/governance/compliance', count: 45 },
        { label: 'Risk Management', href: '/governance/risk', count: 3 },
        { label: 'Cost Optimization', href: '/governance/cost', count: 12 }
      ]
    },
    {
      id: 'security',
      title: 'Security & Access Management',
      description: 'Identity management, RBAC, PIM, and zero-trust security',
      icon: Lock,
      href: '/security',
      color: 'red',
      badge: 'CRITICAL',
      stats: [
        { label: 'Active Users', value: '2,451' },
        { label: 'PIM Requests', value: 7 },
        { label: 'Security Score', value: '82%', trend: 'up' }
      ],
      subItems: [
        { label: 'Identity & Access (IAM)', href: '/security/iam', count: 234 },
        { label: 'Role Management (RBAC)', href: '/security/rbac', count: 89 },
        { label: 'Privileged Identity (PIM)', href: '/security/pim', count: 7 },
        { label: 'Conditional Access', href: '/security/conditional-access', count: 23 },
        { label: 'Zero Trust Policies', href: '/security/zero-trust', count: 15 },
        { label: 'Entitlement Management', href: '/security/entitlements', count: 45 },
        { label: 'Access Reviews', href: '/security/access-reviews', count: 12 }
      ]
    },
    {
      id: 'operations',
      title: 'Operations & Monitoring',
      description: 'Resource management, monitoring, automation, and alerts',
      icon: Activity,
      href: '/operations',
      color: 'green',
      stats: [
        { label: 'Resources', value: 342 },
        { label: 'Active Alerts', value: 8, trend: 'down' },
        { label: 'Automation', value: '89%' }
      ],
      subItems: [
        { label: 'Resources', href: '/operations/resources', count: 342 },
        { label: 'Monitoring', href: '/operations/monitoring', count: 45 },
        { label: 'Automation', href: '/operations/automation', count: 23 },
        { label: 'Notifications', href: '/operations/notifications', count: 156 },
        { label: 'Alerts', href: '/operations/alerts', count: 8 }
      ]
    },
    {
      id: 'devops',
      title: 'DevOps',
      description: 'Complete DevSecOps platform with security-first CI/CD',
      icon: GitBranch,
      href: '/devops',
      color: 'purple',
      stats: [
        { label: 'Pipelines', value: 42 },
        { label: 'Security Gates', value: '100%', trend: 'up' },
        { label: 'Deployments', value: '12/day' }
      ],
      subItems: [
        { label: 'Pipelines', href: '/devops/pipelines', count: 42 },
        { label: 'Releases', href: '/devops/releases', count: 18 },
        { label: 'Deployments', href: '/devops/deployments', count: 67 },
        { label: 'Security Gates', href: '/devsecops/gates', count: 15 },
        { label: 'Policy-as-Code', href: '/devsecops/policy-as-code', count: 23 },
        { label: 'Artifacts', href: '/devops/artifacts', count: 234 },
        { label: 'Build Status', href: '/devops/builds', count: 12 },
        { label: 'Repositories', href: '/devops/repos', count: 28 }
      ]
    },
    {
      id: 'ai',
      title: 'AI Intelligence Hub',
      description: 'Patented AI features for predictive compliance and analysis',
      icon: Brain,
      href: '/ai',
      color: 'pink',
      badge: 'PATENTED',
      stats: [
        { label: 'Predictions', value: '1,234' },
        { label: 'Accuracy', value: '99.2%', trend: 'stable' },
        { label: 'AI Insights', value: 45 }
      ],
      subItems: [
        { label: 'Predictive Compliance', href: '/ai/predictive', count: 234 },
        { label: 'Cross-Domain Analysis', href: '/ai/correlations', count: 56 },
        { label: 'Conversational AI', href: '/ai/chat', count: 89 },
        { label: 'Unified Platform', href: '/ai/unified', count: 12 }
      ]
    },
    {
      id: 'audit',
      title: 'Audit Trail & History',
      description: 'Complete activity history and audit logs',
      icon: History,
      href: '/audit',
      color: 'yellow',
      stats: [
        { label: 'Events Today', value: '3,451' },
        { label: 'Users Active', value: 234 },
        { label: 'Compliance', value: '100%' }
      ]
    }
  ];

  const systemMetrics: SystemMetric[] = [
    { 
      name: 'CPU Usage', 
      value: 78, 
      unit: '%', 
      status: 'warning', 
      trend: 'up',
      onClick: () => router.push('/operations/monitoring')
    },
    { 
      name: 'Memory', 
      value: 62, 
      unit: '%', 
      status: 'healthy', 
      trend: 'stable',
      onClick: () => router.push('/operations/monitoring')
    },
    { 
      name: 'Network', 
      value: 245, 
      unit: 'Mbps', 
      status: 'healthy', 
      trend: 'up',
      onClick: () => router.push('/operations/monitoring')
    },
    { 
      name: 'Storage', 
      value: 89, 
      unit: '%', 
      status: 'critical', 
      trend: 'up',
      onClick: () => router.push('/operations/resources')
    },
    { 
      name: 'API Latency', 
      value: 142, 
      unit: 'ms', 
      status: 'healthy', 
      trend: 'down',
      onClick: () => router.push('/operations/monitoring')
    },
    { 
      name: 'Error Rate', 
      value: 0.3, 
      unit: '%', 
      status: 'healthy', 
      trend: 'down',
      onClick: () => router.push('/operations/alerts')
    }
  ];

  const quickActions = [
    { icon: Shield, label: 'Security Scan', color: 'blue', href: '/security' },
    { icon: RefreshCw, label: 'Restart Services', color: 'green', href: '/operations/automation' },
    { icon: Lock, label: 'Lock Resources', color: 'red', href: '/security/rbac' },
    { icon: Database, label: 'Backup DB', color: 'purple', href: '/operations/automation' },
    { icon: Users, label: 'Alert Team', color: 'yellow', href: '/operations/notifications' },
    { icon: Terminal, label: 'Run Playbook', color: 'pink', href: '/operations/automation' }
  ];

  const handleCommand = (e: React.FormEvent) => {
    e.preventDefault();
    if (commandInput.trim()) {
      setCommandHistory([...commandHistory, `> ${commandInput}`, 'Command executed successfully']);
      setCommandInput('');
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-500 bg-red-500/10 border-red-500/20';
      case 'high': return 'text-orange-500 bg-orange-500/10 border-orange-500/20';
      case 'medium': return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/20';
      case 'low': return 'text-blue-500 bg-blue-500/10 border-blue-500/20';
      default: return 'text-gray-500 bg-gray-500/10 border-gray-500/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-500';
      case 'warning': return 'text-yellow-500';
      case 'critical': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const toggleCardExpansion = (cardId: string) => {
    const newExpanded = new Set(expandedCards);
    if (newExpanded.has(cardId)) {
      newExpanded.delete(cardId);
    } else {
      newExpanded.add(cardId);
    }
    setExpandedCards(newExpanded);
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white py-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center space-x-3">
              <Activity className="w-8 h-8 text-blue-500" />
              <span>PolicyCortex Command Center</span>
            </h1>
            <p className="text-gray-400">Executive dashboard with complete system overview</p>
          </div>
          <div className="flex items-center space-x-4">
            <ViewToggle view={viewMode} onViewChange={setViewMode} />
            <button type="button"
              onClick={() => setIsWarRoomActive(!isWarRoomActive)}
              className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center space-x-2 ${
                isWarRoomActive 
                  ? 'bg-red-600 hover:bg-red-700 animate-pulse' 
                  : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              <Radio className="w-5 h-5" />
              <span>{isWarRoomActive ? 'War Room Active' : 'Activate War Room'}</span>
            </button>
            <button
              type="button"
              className="px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
              onClick={() => router.push('/settings')}
            >
              <Settings className="w-4 h-4" />
              <span>Settings</span>
            </button>
          </div>
        </div>

        {/* Critical Alert Banner */}
        {alerts.filter(a => a.severity === 'critical' && a.status === 'active').length > 0 && (
          <div 
            className="bg-red-900/20 border border-red-500 rounded-lg p-4 mb-4 animate-pulse cursor-pointer hover:bg-red-900/30 transition-colors"
            onClick={() => router.push('/operations/alerts')}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <AlertTriangle className="w-6 h-6 text-red-500" />
                <div>
                  <p className="font-semibold text-red-500">CRITICAL ALERTS ACTIVE</p>
                  <p className="text-sm text-gray-300">Immediate action required - {alerts.filter(a => a.severity === 'critical').length} critical issues detected</p>
                </div>
              </div>
              <ChevronRight className="w-5 h-5 text-red-500" />
            </div>
          </div>
        )}
      </div>

      {/* System Metrics Grid - Clickable */}
      <ResponsiveGrid variant="metrics" className="mb-6">
        {systemMetrics.map((metric, index) => (
          <div 
            key={index} 
            className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 cursor-pointer hover:shadow-lg hover:border-blue-500 transition-all"
            onClick={metric.onClick}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-gray-400">{metric.name}</span>
              {metric.trend === 'up' && <TrendingUp className="w-3 h-3 text-green-500" />}
              {metric.trend === 'down' && <TrendingUp className="w-3 h-3 text-red-500 rotate-180" />}
              {metric.trend === 'stable' && <div className="w-3 h-3 bg-gray-400 dark:bg-gray-500 rounded-full" />}
            </div>
            <div className={`text-2xl font-bold ${getStatusColor(metric.status)}`}>
              {metric.value}{metric.unit}
            </div>
            <div className="mt-2 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div 
                className={`h-full ${
                  metric.status === 'critical' ? 'bg-red-500' :
                  metric.status === 'warning' ? 'bg-yellow-500' :
                  'bg-green-500'
                }`}
                style={{ width: `${metric.value}%` }}
              />
            </div>
          </div>
        ))}
      </ResponsiveGrid>

      {/* Main Content - Switch between Card and Visualization Views */}
      {viewMode === 'cards' ? (
        // Card View
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {navigationCards.map((card) => {
          const Icon = card.icon;
          const isExpanded = expandedCards.has(card.id);
          
          return (
            <div 
              key={card.id}
              className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-xl transition-all"
            >
              {/* Card Header - Clickable */}
              <div 
                className="p-6 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                onClick={() => router.push(card.href)}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`p-3 rounded-lg bg-${card.color}-500/10`}>
                      <Icon className={`w-6 h-6 text-${card.color}-500`} />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold">{card.title}</h3>
                      {card.badge && (
                        <span className={`inline-block mt-1 text-xs px-2 py-1 rounded-full bg-${card.color}-500/20 text-${card.color}-500 font-medium`}>
                          {card.badge}
                        </span>
                      )}
                    </div>
                  </div>
                  <ExternalLink className="w-4 h-4 text-gray-400" />
                </div>
                
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                  {card.description}
                </p>

                {/* Stats */}
                {card.stats && (
                  <div className="grid grid-cols-3 gap-3">
                    {card.stats.map((stat, idx) => (
                      <div key={idx} className="text-center">
                        <div className="text-lg font-bold text-gray-900 dark:text-white flex items-center justify-center space-x-1">
                          <span>{stat.value}</span>
                          {stat.trend === 'up' && <TrendingUp className="w-3 h-3 text-green-500" />}
                          {stat.trend === 'down' && <TrendingDown className="w-3 h-3 text-red-500" />}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">{stat.label}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Sub Items - Expandable */}
              {card.subItems && (
                <>
                  <div className="border-t border-gray-200 dark:border-gray-700">
                    <button
                      type="button"
                      className="w-full px-6 py-3 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleCardExpansion(card.id);
                      }}
                    >
                      <span className="text-sm font-medium">Quick Access</span>
                      <ChevronDown className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                    </button>
                  </div>
                  
                  {isExpanded && (
                    <div className="border-t border-gray-200 dark:border-gray-700 p-3">
                      <div className="space-y-1">
                        {card.subItems.map((item, idx) => (
                          <button
                            key={idx}
                            type="button"
                            className="w-full text-left px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center justify-between group"
                            onClick={(e) => {
                              e.stopPropagation();
                              router.push(item.href);
                            }}
                          >
                            <span className="text-sm">{item.label}</span>
                            <div className="flex items-center space-x-2">
                              {item.count !== undefined && (
                                <span className="text-xs bg-gray-200 dark:bg-gray-600 px-2 py-1 rounded">
                                  {item.count}
                                </span>
                              )}
                              <ArrowRight className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>
      ) : (
        // Visualization View
        <div className="space-y-6 mb-8">
          {/* System Performance Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartContainer title="System Performance">
              <div className="h-64 flex items-end justify-around gap-4 p-4">
                {[
                  { name: 'CPU', value: 78, color: 'bg-blue-500' },
                  { name: 'Memory', value: 62, color: 'bg-green-500' },
                  { name: 'Storage', value: 89, color: 'bg-red-500' },
                  { name: 'Network', value: 45, color: 'bg-purple-500' }
                ].map((item) => (
                  <div key={item.name} className="flex-1 flex flex-col items-center">
                    <div 
                      className={`w-full ${item.color} rounded-t transition-all hover:opacity-80`}
                      style={{ height: `${item.value}%` }}
                    />
                    <span className="text-xs mt-2">{item.name}</span>
                    <span className="text-sm font-bold">{item.value}%</span>
                  </div>
                ))}
              </div>
            </ChartContainer>
            <ChartContainer title="Alert Trends">
              <div className="h-64 p-4">
                <div className="relative h-full">
                  {/* Simple line chart visualization */}
                  <div className="absolute inset-0 flex items-end justify-between">
                    {[
                      { time: '00:00', total: 18 },
                      { time: '04:00', total: 16 },
                      { time: '08:00', total: 23 },
                      { time: '12:00', total: 18 },
                      { time: '16:00', total: 28 },
                      { time: '20:00', total: 11 }
                    ].map((point, i) => (
                      <div key={i} className="flex flex-col items-center flex-1">
                        <div className="w-2 bg-blue-500 rounded-t" style={{ height: `${(point.total / 30) * 100}%` }} />
                        <span className="text-xs mt-2">{point.time}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </ChartContainer>
          </div>

          {/* Compliance & Security Metrics */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <ChartContainer title="Compliance Status">
              <div className="h-64 flex items-center justify-center p-4">
                <div className="relative w-48 h-48">
                  <div className="absolute inset-0 rounded-full bg-green-500"></div>
                  <div className="absolute inset-0 rounded-full bg-red-500" style={{ clipPath: 'polygon(50% 50%, 50% 0, 60% 0, 50% 50%)' }}></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-3xl font-bold">94%</div>
                      <div className="text-sm text-gray-500">Compliant</div>
                    </div>
                  </div>
                </div>
              </div>
            </ChartContainer>
            <ChartContainer title="Security Score Trend">
              <div className="h-64 p-4 flex items-end justify-between">
                {[
                  { day: 'Mon', score: 78 },
                  { day: 'Tue', score: 82 },
                  { day: 'Wed', score: 79 },
                  { day: 'Thu', score: 85 },
                  { day: 'Fri', score: 82 },
                  { day: 'Sat', score: 87 },
                  { day: 'Sun', score: 89 }
                ].map((item) => (
                  <div key={item.day} className="flex-1 flex flex-col items-center mx-1">
                    <div className="w-full bg-gradient-to-t from-blue-500 to-blue-300 rounded-t" style={{ height: `${item.score}%` }} />
                    <span className="text-xs mt-1">{item.day}</span>
                  </div>
                ))}
              </div>
            </ChartContainer>
            <ChartContainer title="Resource Distribution">
              <div className="h-64 p-4">
                <div className="space-y-3">
                  {[
                    { name: 'Compute', value: 45, color: 'bg-blue-500' },
                    { name: 'Storage', value: 30, color: 'bg-green-500' },
                    { name: 'Network', value: 15, color: 'bg-yellow-500' },
                    { name: 'Database', value: 10, color: 'bg-purple-500' }
                  ].map((item) => (
                    <div key={item.name}>
                      <div className="flex justify-between text-sm mb-1">
                        <span>{item.name}</span>
                        <span className="font-bold">{item.value}%</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div className={`h-2 rounded-full ${item.color}`} style={{ width: `${item.value}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </ChartContainer>
          </div>

          {/* Cost & DevOps Metrics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartContainer title="Monthly Cost Trend">
              <div className="h-64 p-4 flex items-end justify-between">
                {[
                  { month: 'Jan', cost: 42, budget: 50 },
                  { month: 'Feb', cost: 38, budget: 50 },
                  { month: 'Mar', cost: 45, budget: 50 },
                  { month: 'Apr', cost: 47, budget: 50 },
                  { month: 'May', cost: 43, budget: 50 },
                  { month: 'Jun', cost: 42, budget: 50 }
                ].map((item) => (
                  <div key={item.month} className="flex-1 flex flex-col items-center mx-1">
                    <div className="w-full flex flex-col items-center">
                      <div className="w-8 bg-blue-500 rounded-t" style={{ height: `${(item.cost / 50) * 200}px` }} />
                      <div className="w-8 border-2 border-dashed border-gray-400 absolute" style={{ height: '200px', bottom: '24px' }} />
                    </div>
                    <span className="text-xs mt-1">{item.month}</span>
                  </div>
                ))}
              </div>
            </ChartContainer>
            <ChartContainer title="DevOps Pipeline Performance">
              <div className="h-64 p-4">
                <div className="space-y-4">
                  {[
                    { stage: 'Build', success: 89, failed: 11 },
                    { stage: 'Test', success: 92, failed: 8 },
                    { stage: 'Deploy', success: 95, failed: 5 },
                    { stage: 'Release', success: 97, failed: 3 }
                  ].map((item) => (
                    <div key={item.stage}>
                      <div className="flex justify-between text-sm mb-1">
                        <span>{item.stage}</span>
                        <span className="text-green-500">{item.success}% success</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 flex">
                        <div className="bg-green-500 rounded-l-full" style={{ width: `${item.success}%` }} />
                        <div className="bg-red-500 rounded-r-full" style={{ width: `${item.failed}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </ChartContainer>
          </div>
        </div>
      )}

      {/* Main Grid - Alerts and Command Center */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Alert Feed - Clickable */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <div 
              className="p-4 border-b border-gray-700 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
              onClick={() => router.push('/operations/alerts')}
            >
              <h2 className="text-lg font-semibold flex items-center space-x-2">
                <AlertCircle className="w-5 h-5 text-red-500" />
                <span>Active Alerts</span>
                <span className="ml-auto text-sm bg-red-500 text-white px-2 py-1 rounded">
                  {alerts.filter(a => a.status === 'active').length}
                </span>
                <ChevronRight className="w-4 h-4 text-gray-400" />
              </h2>
            </div>
            <div className="divide-y divide-gray-700 max-h-96 overflow-y-auto">
              {alerts.map((alert) => (
                <div
                  key={alert.id}
                  className="p-4 hover:bg-gray-100 dark:hover:bg-gray-700/50 cursor-pointer transition-colors"
                  onClick={() => setSelectedAlert(alert)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <span className={`px-2 py-1 text-xs rounded border ${getSeverityColor(alert.severity)}`}>
                          {alert.severity.toUpperCase()}
                        </span>
                        <span className="text-xs text-gray-400">{alert.source}</span>
                        <span className="text-xs text-gray-400">{alert.time}</span>
                      </div>
                      <h3 className="font-medium mb-1">{alert.title}</h3>
                      <p className="text-sm text-gray-400">{alert.description}</p>
                    </div>
                    <div className="ml-4">
                      {alert.status === 'active' && <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />}
                      {alert.status === 'acknowledged' && <div className="w-2 h-2 bg-yellow-500 rounded-full" />}
                      {alert.status === 'resolved' && <CheckCircle className="w-4 h-4 text-green-500" />}
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div 
              className="p-3 border-t border-gray-700 text-center cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
              onClick={() => router.push('/operations/alerts')}
            >
              <span className="text-sm text-blue-500 hover:text-blue-400">View All Alerts →</span>
            </div>
          </div>

          {/* Command Center */}
          <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="p-4 border-b border-gray-700">
              <h2 className="text-lg font-semibold flex items-center space-x-2">
                <Terminal className="w-5 h-5 text-green-500" />
                <span>Command Center</span>
              </h2>
            </div>
            <div className="p-4">
              <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm mb-4 h-32 overflow-y-auto">
                {commandHistory.length === 0 ? (
                  <div className="text-gray-500">Ready for commands...</div>
                ) : (
                  commandHistory.map((line, index) => (
                    <div key={index} className={line.startsWith('>') ? 'text-green-400' : 'text-gray-300'}>
                      {line}
                    </div>
                  ))
                )}
              </div>
              <form onSubmit={handleCommand} className="flex space-x-2">
                <input
                  type="text"
                  value={commandInput}
                  onChange={(e) => setCommandInput(e.target.value)}
                  placeholder="Enter command..."
                  className="flex-1 bg-gray-100 dark:bg-gray-700 rounded px-4 py-2 text-sm"
                />
                <button
                  type="submit"
                  className="px-4 py-2 bg-green-600 rounded hover:bg-green-700 transition-colors"
                >
                  Execute
                </button>
              </form>
            </div>
          </div>
        </div>

        {/* Right Panel */}
        <div className="space-y-6">
          {/* Quick Actions - Clickable */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
            <div className="grid grid-cols-2 gap-3">
              {quickActions.map((action, index) => {
                const Icon = action.icon;
                return (
                  <button type="button"
                    key={index}
                    className="p-3 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg transition-colors flex flex-col items-center space-y-2"
                    onClick={() => router.push(action.href)}
                  >
                    <Icon className={`w-6 h-6 text-${action.color}-500`} />
                    <span className="text-xs text-center">{action.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Recent Activities - Clickable */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <div 
              className="flex items-center justify-between mb-4 cursor-pointer hover:text-blue-500 transition-colors"
              onClick={() => router.push('/audit')}
            >
              <h2 className="text-lg font-semibold">Recent Activities</h2>
              <ChevronRight className="w-4 h-4" />
            </div>
            <div className="space-y-3">
              <div 
                className="flex items-start space-x-3 p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer transition-colors"
                onClick={() => router.push('/audit')}
              >
                <UserCheck className="w-4 h-4 text-green-500 mt-1" />
                <div>
                  <p className="text-sm">User john.doe@company.com logged in</p>
                  <p className="text-xs text-gray-400">2 minutes ago</p>
                </div>
              </div>
              <div 
                className="flex items-start space-x-3 p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer transition-colors"
                onClick={() => router.push('/governance/policies')}
              >
                <FileCheck className="w-4 h-4 text-blue-500 mt-1" />
                <div>
                  <p className="text-sm">Policy PCI-DSS-2024 updated</p>
                  <p className="text-xs text-gray-400">15 minutes ago</p>
                </div>
              </div>
              <div 
                className="flex items-start space-x-3 p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer transition-colors"
                onClick={() => router.push('/devops/deployments')}
              >
                <Rocket className="w-4 h-4 text-purple-500 mt-1" />
                <div>
                  <p className="text-sm">Deployment to production completed</p>
                  <p className="text-xs text-gray-400">1 hour ago</p>
                </div>
              </div>
            </div>
            <button
              type="button"
              className="w-full mt-3 p-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded transition-colors text-sm"
              onClick={() => router.push('/audit')}
            >
              View All Activities
            </button>
          </div>

          {/* Cost Summary - Clickable */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <div 
              className="flex items-center justify-between mb-4 cursor-pointer hover:text-blue-500 transition-colors"
              onClick={() => router.push('/governance/cost')}
            >
              <h2 className="text-lg font-semibold flex items-center space-x-2">
                <DollarSign className="w-5 h-5 text-green-500" />
                <span>Cost Summary</span>
              </h2>
              <ChevronRight className="w-4 h-4" />
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Monthly Budget</span>
                <span className="text-sm font-bold">$50,000</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Current Spend</span>
                <span className="text-sm font-bold text-yellow-500">$42,341</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Savings This Month</span>
                <span className="text-sm font-bold text-green-500">$4,523</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-2">
                <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '85%' }}></div>
              </div>
              <button
                type="button"
                className="w-full mt-2 p-2 bg-blue-600 hover:bg-blue-700 rounded transition-colors text-sm"
                onClick={() => router.push('/governance/cost')}
              >
                View Cost Details
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Alert Detail Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setSelectedAlert(null)}>
          <div className="bg-gray-800 rounded-lg p-6 max-w-2xl w-full mx-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 className="text-xl font-bold mb-2">{selectedAlert.title}</h2>
                <div className="flex items-center space-x-3">
                  <span className={`px-2 py-1 text-xs rounded border ${getSeverityColor(selectedAlert.severity)}`}>
                    {selectedAlert.severity.toUpperCase()}
                  </span>
                  <span className="text-sm text-gray-400">{selectedAlert.source}</span>
                  <span className="text-sm text-gray-400">{selectedAlert.time}</span>
                </div>
              </div>
              <button type="button"
                onClick={() => setSelectedAlert(null)}
                className="text-gray-400 hover:text-white text-2xl"
              >
                ×
              </button>
            </div>
            
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-400 mb-2">Description</h3>
              <p className="text-gray-300">{selectedAlert.description}</p>
            </div>

            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-400 mb-2">Affected Resources</h3>
              <div className="bg-gray-900 rounded p-3">
                <code className="text-sm text-green-400">
                  /subscriptions/205b477d/resourceGroups/prod-rg/providers/Microsoft.Sql/servers/prod-sql-01
                </code>
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-400 mb-2">Recommended Actions</h3>
              <ol className="list-decimal list-inside space-y-2 text-sm text-gray-300">
                <li>Check database connection strings and network configuration</li>
                <li>Verify firewall rules and security groups</li>
                <li>Review recent changes to the database configuration</li>
                <li>Check Azure service health for ongoing issues</li>
              </ol>
            </div>

            <div className="flex space-x-3">
              <button
                type="button"
                className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors"
                onClick={() => {
                  toast({ title: 'Acknowledged', description: `${selectedAlert.title}` });
                  setSelectedAlert(null);
                }}
              >
                Acknowledge
              </button>
              <button
                type="button"
                className="px-4 py-2 bg-green-600 rounded hover:bg-green-700 transition-colors"
                onClick={() => {
                  toast({ title: 'Resolved', description: `${selectedAlert.title}` });
                  setSelectedAlert(null);
                }}
              >
                Mark Resolved
              </button>
              <button
                type="button"
                className="px-4 py-2 bg-orange-600 rounded hover:bg-orange-700 transition-colors"
                onClick={() => {
                  toast({ title: 'Escalated', description: `${selectedAlert.title}` });
                  router.push('/operations/notifications');
                }}
              >
                Escalate
              </button>
              <button
                type="button"
                className="px-4 py-2 bg-purple-600 rounded hover:bg-purple-700 transition-colors"
                onClick={() => router.push('/operations/automation')}
              >
                Run Playbook
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}